"""Graph utilities for sequence-to-atom-graph conversion and pooling.

Key functions:
  - sequence_to_graph:  aa sequence → PyG Data (atom graph)
  - atom_to_residue_map: aa sequence → [n_atoms] residue index per atom
  - atom_to_residue_pooling:  atom-level interaction map → residue-level matrix
"""

import torch
import numpy as np
from torch_geometric import data as DATA
from rdkit import Chem

from featurizer import MolGraphConvFeaturizer

# ── per-residue heavy-atom counts (RDKit MolFromSequence convention) ──
# backbone = N, CA, C, O  (+ OXT for C-terminal residue)
SIDE_CHAIN_ATOMS = {
    "A": 1,   # CB
    "C": 2,   # CB, SG
    "D": 4,   # CB, CG, OD1, OD2
    "E": 5,   # CB, CG, CD, OE1, OE2
    "F": 7,   # CB, CG, CD1, CD2, CE1, CE2, CZ
    "G": 0,
    "H": 6,   # CB, CG, ND1, CD2, CE1, NE2
    "I": 4,   # CB, CG1, CG2, CD1
    "K": 5,   # CB, CG, CD, CE, NZ
    "L": 4,   # CB, CG, CD1, CD2
    "M": 4,   # CB, CG, SD, CE
    "N": 4,   # CB, CG, OD1, ND2
    "P": 3,   # CB, CG, CD
    "Q": 5,   # CB, CG, CD, OE1, NE2
    "R": 7,   # CB, CG, CD, NE, CZ, NH1, NH2
    "S": 2,   # CB, OG
    "T": 3,   # CB, OG1, CG2
    "V": 3,   # CB, CG1, CG2
    "W": 10,  # CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
    "Y": 8,   # CB, CG, CD1, CD2, CE1, CE2, CZ, OH
}

_featurizer = None


def _get_featurizer() -> MolGraphConvFeaturizer:
    global _featurizer
    if _featurizer is None:
        _featurizer = MolGraphConvFeaturizer(use_edges=True)
    return _featurizer


def sequence_to_graph(seq: str) -> DATA.Data:
    """Convert an amino-acid sequence to a 2D topological atom graph.

    Uses RDKit MolFromSequence (no 3D structure needed) and the
    MolGraphConvFeaturizer to extract atom / bond features.

    Returns
    -------
    DATA.Data
        PyG Data object with .x (node features), .edge_index, .edge_attr.
    """
    featurizer = _get_featurizer()
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        raise ValueError(f"RDKit could not parse sequence: {seq}")
    graph_data = featurizer._featurize(mol)
    return DATA.Data(
        x=torch.Tensor(graph_data.node_features),
        edge_index=torch.LongTensor(graph_data.edge_index),
        edge_attr=torch.Tensor(graph_data.edge_features),
    )


def sequence_to_mol(seq: str):
    """Return the RDKit Mol object for a sequence (needed for TopKPooling)."""
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        raise ValueError(f"RDKit could not parse sequence: {seq}")
    return mol


def atom_to_residue_map(seq: str) -> torch.LongTensor:
    """Build a mapping from atom index → residue index.

    Returns
    -------
    torch.LongTensor  shape [n_atoms]
        residue_index[i] = which residue (0-indexed) atom i belongs to.
    """
    n = len(seq)
    residue_of_atom = []
    for pos, aa in enumerate(seq):
        n_side = SIDE_CHAIN_ATOMS.get(aa, 0)
        if pos == n - 1:   # C-terminal: extra OXT atom
            n_atoms = 5 + n_side
        else:
            n_atoms = 4 + n_side
        residue_of_atom.extend([pos] * n_atoms)
    return torch.LongTensor(residue_of_atom)


def atom_to_residue_pooling(
    interaction_map: torch.Tensor,
    pep_perm: torch.LongTensor,
    cdr3_perm: torch.LongTensor,
    pep_atom2res: torch.LongTensor,
    cdr3_atom2res: torch.LongTensor,
    pep_len: int,
    cdr3_len: int,
    mode: str = "max",
) -> torch.Tensor:
    """Pool atom-level interaction scores to residue-level.

    Parameters
    ----------
    interaction_map : [B, k_pep, k_cdr3, H_gcn]
        Cross-molecule attention output from MultiHeadAttention.
    pep_perm : [B * k]
        Indices of selected peptide atoms (flattened across batch).
    cdr3_perm : [B * k]
        Indices of selected CDR3 atoms.
    pep_atom2res : [n_pep_atoms]
        Mapping from peptide atom index → residue index.
    cdr3_atom2res : [n_cdr3_atoms]
        Mapping from CDR3 atom index → residue index.
    pep_len : int
        Number of peptide residues.
    cdr3_len : int
        Number of CDR3 residues.
    mode : str
        Pooling mode: "max", "mean", or "min".

    Returns
    -------
    torch.Tensor  [B, pep_len, cdr3_len]
        Residue-level spatial bias matrix D_spatial.
    """
    B, k, _, H_gcn = interaction_map.shape
    device = interaction_map.device

    # reduce feature dim: take mean ||·||₂ as scalar score per atom pair
    contact_scores = interaction_map.norm(p=2, dim=-1)  # [B, k, k]

    # reshape perms to [B, k]
    pep_perm_batch = pep_perm.view(B, k)
    cdr3_perm_batch = cdr3_perm.view(B, k)

    # map each selected atom → its residue
    pep_res = pep_atom2res.to(device)[pep_perm_batch]   # [B, k]
    cdr3_res = cdr3_atom2res.to(device)[cdr3_perm_batch]  # [B, k]

    # accumulate into [B, pep_len, cdr3_len] grid
    d_spatial = torch.full(
        (B, pep_len, cdr3_len),
        float("-inf") if mode == "max" else 0.0,
        device=device,
    )

    if mode == "max":
        d_spatial = torch.full((B, pep_len, cdr3_len), float("-inf"), device=device)
        for b in range(B):
            idx = pep_res[b] * cdr3_len + cdr3_res[b]
            d_spatial_flat = d_spatial[b].view(-1)
            d_spatial_flat.scatter_reduce_(
                0, idx, contact_scores[b].view(-1),
                reduce="amax", include_self=False,
            )
    elif mode == "mean":
        count = torch.zeros(B, pep_len, cdr3_len, device=device)
        for b in range(B):
            d_spatial_flat = d_spatial[b].view(-1)
            idx = pep_res[b] * cdr3_len + cdr3_res[b]
            d_spatial_flat.scatter_reduce_(
                0, idx, contact_scores[b].view(-1),
                reduce="sum", include_self=False,
            )
            ones = torch.ones_like(contact_scores[b].view(-1))
            count_flat = count[b].view(-1)
            count_flat.scatter_reduce_(
                0, idx, ones, reduce="sum", include_self=False,
            )
        d_spatial = d_spatial / (count + 1e-8)
    else:  # "min"
        d_spatial = torch.full((B, pep_len, cdr3_len), float("inf"), device=device)
        for b in range(B):
            idx = pep_res[b] * cdr3_len + cdr3_res[b]
            d_spatial_flat = d_spatial[b].view(-1)
            d_spatial_flat.scatter_reduce_(
                0, idx, contact_scores[b].view(-1),
                reduce="amin", include_self=False,
            )

    # fill -inf → large positive value (no contact → large distance penalty)
    if mode in ("max",):
        d_spatial = torch.where(
            torch.isinf(d_spatial),
            torch.tensor(1e6, device=device),
            d_spatial,
        )

    return d_spatial


def generate_graph_batch(seqs: list):
    """Batch-generate graphs and atom-to-residue maps for a list of sequences.

    Returns
    -------
    graphs : list[DATA.Data]
    mols : list[rdkit.Chem.Mol]
    atom2res : list[torch.LongTensor]
    """
    graphs = []
    mols = []
    atom2res = []
    for seq in seqs:
        mol = sequence_to_mol(seq)
        graph = sequence_to_graph(seq)
        a2r = atom_to_residue_map(seq)
        mols.append(mol)
        graphs.append(graph)
        atom2res.append(a2r)
    return graphs, mols, atom2res
