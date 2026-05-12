"""PDB structure dataset for 3D-aware TCR-peptide binding prediction.

Adapted from deepAntigen/deepAntigen/antigenTCR/load_dataset/load_structure.py
Uses TCR-ECHO's featurizer and graph utilities for consistency.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric import data as DATA

from featurizer import MolGraphConvFeaturizer


class PDBStructureDataset(Dataset):
    """Dataset that loads PDB-extracted peptide/CDR3 chains with 3D coordinates.

    Expected directory layout:
        data_dir/
          pdb_Extracted/
            {pdbid}_peptide.pdb
            {pdbid}_cdr3.pdb
            {pdbid}_pep.pkl      # backbone connectivity map
            {pdbid}_cdr3.pkl
          distance_matrix/
            {pdbid}.npy           # [N_pep_atoms, N_cdr3_atoms] capped at 30A

    The CSV at `csv_path` must have a 'pdbid' column.
    """

    def __init__(self, csv_path: str, data_dir: str = "",
                 cache_graphs: bool = True):
        """
        Parameters
        ----------
        csv_path : str  path to CSV with pdbid column
        data_dir : str  root directory containing pdb_Extracted/ and distance_matrix/
        cache_graphs : bool  if True, featurize graphs once and reuse
        """
        import pandas as pd
        meta = pd.read_csv(csv_path, header=0)
        pdbinfos = meta['pdbid']
        self.pdbids = [pi.split('_')[0] for pi in pdbinfos]

        if not data_dir:
            data_dir = os.path.dirname(csv_path.rstrip('/')) + '/'
        self.data_dir = data_dir

        # ── Load / featurize PDB molecules ────────────────────────────
        self.peptide_chems = {}
        self.cdr3_chems = {}
        self.peptide_graphs = {}
        self.cdr3_graphs = {}

        featurizer = MolGraphConvFeaturizer(use_edges=True)

        for pdbid in self.pdbids:
            # peptide
            pep_file = os.path.join(data_dir, 'pdb_Extracted', f'{pdbid}_peptide.pdb')
            pep_con = os.path.join(data_dir, 'pdb_Extracted', f'{pdbid}_pep.pkl')
            peptide_chem = Chem.MolFromPDBFile(pep_file)
            if peptide_chem is None:
                raise ValueError(f"Could not read PDB file: {pep_file}")
            peptide_chem = self._check_impossible_connection(peptide_chem)
            peptide_chem = self._add_CON(pep_con, peptide_chem)
            self.peptide_chems[pdbid] = peptide_chem

            # cdr3
            cdr3_file = os.path.join(data_dir, 'pdb_Extracted', f'{pdbid}_cdr3.pdb')
            cdr3_con = os.path.join(data_dir, 'pdb_Extracted', f'{pdbid}_cdr3.pkl')
            cdr3_chem = Chem.MolFromPDBFile(cdr3_file)
            if cdr3_chem is None:
                raise ValueError(f"Could not read PDB file: {cdr3_file}")
            cdr3_chem = self._check_impossible_connection(cdr3_chem)
            cdr3_chem = self._add_CON(cdr3_con, cdr3_chem)
            self.cdr3_chems[pdbid] = cdr3_chem

            if cache_graphs:
                pep_feat = featurizer._featurize(peptide_chem)
                self.peptide_graphs[pdbid] = DATA.Data(
                    x=torch.Tensor(pep_feat.node_features),
                    edge_index=torch.LongTensor(pep_feat.edge_index),
                    edge_attr=torch.Tensor(pep_feat.edge_features),
                )
                cdr3_feat = featurizer._featurize(cdr3_chem)
                self.cdr3_graphs[pdbid] = DATA.Data(
                    x=torch.Tensor(cdr3_feat.node_features),
                    edge_index=torch.LongTensor(cdr3_feat.edge_index),
                    edge_attr=torch.Tensor(cdr3_feat.edge_features),
                )

        self.featurizer = featurizer
        self._cache_graphs = cache_graphs

    def _check_impossible_connection(self, molecule):
        """Remove inter-residue bonds that RDKit incorrectly adds from PDB."""
        new_molecule = Chem.RWMol(molecule)
        for atom in molecule.GetAtoms():
            for neighbor in atom.GetNeighbors():
                n_res = neighbor.GetPDBResidueInfo().GetResidueNumber()
                c_res = atom.GetPDBResidueInfo().GetResidueNumber()
                if n_res != c_res:
                    new_molecule.RemoveBond(atom.GetIdx(), neighbor.GetIdx())
        return new_molecule.GetMol()

    def _add_CON(self, con_path, molecule):
        """Re-add correct peptide bonds from connectivity pickle."""
        if not os.path.exists(con_path):
            return molecule
        editable = Chem.EditableMol(molecule)
        with open(con_path, 'rb') as f:
            connect = pickle.load(f)
        for atomid1, atomid2 in connect.items():
            bond = molecule.GetBondBetweenAtoms(atomid1, atomid2)
            if bond is None:
                editable.AddBond(atomid1, atomid2, order=Chem.rdchem.BondType.SINGLE)
        new_molecule = editable.GetMol()
        return Chem.RemoveHs(new_molecule)

    def __len__(self):
        return len(self.pdbids)

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]

        distance_matrix = np.load(
            os.path.join(self.data_dir, 'distance_matrix', f'{pdbid}.npy')
        )
        distance_matrix = np.where(distance_matrix > 30, 30, distance_matrix)

        peptide_chem = self.peptide_chems[pdbid]
        cdr3_chem = self.cdr3_chems[pdbid]

        if self._cache_graphs:
            peptide_graph = self.peptide_graphs[pdbid]
            cdr3_graph = self.cdr3_graphs[pdbid]
        else:
            pep_feat = self.featurizer._featurize(peptide_chem)
            peptide_graph = DATA.Data(
                x=torch.Tensor(pep_feat.node_features),
                edge_index=torch.LongTensor(pep_feat.edge_index),
                edge_attr=torch.Tensor(pep_feat.edge_features),
            )
            cdr3_feat = self.featurizer._featurize(cdr3_chem)
            cdr3_graph = DATA.Data(
                x=torch.Tensor(cdr3_feat.node_features),
                edge_index=torch.LongTensor(cdr3_feat.edge_index),
                edge_attr=torch.Tensor(cdr3_feat.edge_features),
            )

        return (
            pdbid,
            pickle.dumps(peptide_chem),
            pickle.dumps(peptide_graph),
            pickle.dumps(cdr3_chem),
            pickle.dumps(cdr3_graph),
            distance_matrix,
        )


def collate_structure_batch(batch):
    """Collate function for PDBStructureDataset.

    Returns unpickled objects ready for model forward pass.
    """
    pdbids = [item[0] for item in batch]
    peptide_chems = [pickle.loads(item[1]) for item in batch]
    peptide_graphs = [pickle.loads(item[2]) for item in batch]
    cdr3_chems = [pickle.loads(item[3]) for item in batch]
    cdr3_graphs = [pickle.loads(item[4]) for item in batch]
    distance_matrixs = [item[5] for item in batch]

    return pdbids, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, distance_matrixs
