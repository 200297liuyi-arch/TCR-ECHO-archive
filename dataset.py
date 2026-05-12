"""TCR-Peptide dataset with pre-computed atom graphs.

CRITICAL: All RDKit graph construction happens in __init__, NOT in __getitem__.
This prevents CPU bottleneck during DataLoader multi-process training.
"""

import pickle
import torch
from torch.utils.data import Dataset
from graph_utils import sequence_to_graph, sequence_to_mol, atom_to_residue_map


class TCRPeptideDataset(Dataset):
    def __init__(self, df, tokenizer, atchley_map, cols,
                 mask_prob: float = 0.0,
                 tcr_max_len: int = 25,
                 pep_max_len: int = 15,
                 use_graph: bool = True):
        """
        Parameters
        ----------
        df : DataFrame  columns: TCR seq, peptide seq, label
        tokenizer : HuggingFace tokenizer (ESM)
        atchley_map : dict  amino acid → 5-dim Atchley factor vector
        cols : dict  keys 'tcr', 'peptide', 'label' mapping to df columns
        mask_prob : float  random masking probability during training
        tcr_max_len : int  fixed TCR sequence length (pad/truncate)
        pep_max_len : int  fixed peptide sequence length (pad/truncate)
        use_graph : bool  whether to build atom-level molecular graphs
        """
        print(f'mask_prob: {mask_prob}')

        self.seqs1 = df[cols['tcr']].astype(str).tolist()
        self.seqs2 = df[cols['peptide']].astype(str).tolist()
        self.labels = df[cols['label']].tolist()
        self.tokenizer = tokenizer
        self.atchley_map = atchley_map
        self.mask_prob = mask_prob
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        self.use_graph = use_graph

        # ── Pre-compute ALL graphs during __init__ (NOT in __getitem__) ─
        if use_graph:
            self._tcr_pkl = {}
            self._pep_pkl = {}
            self._tcr_mol_pkl = {}
            self._pep_mol_pkl = {}
            self._tcr_a2r = {}
            self._pep_a2r = {}

            unique_tcr = set(self.seqs1)
            unique_pep = set(self.seqs2)
            print(f'Pre-computing graphs for {len(unique_tcr)} unique TCR '
                  f'+ {len(unique_pep)} unique peptides...')

            for seq in unique_tcr:
                mol = sequence_to_mol(seq)
                graph = sequence_to_graph(seq)
                a2r = atom_to_residue_map(seq)
                self._tcr_pkl[seq] = pickle.dumps(graph)
                self._tcr_mol_pkl[seq] = pickle.dumps(mol)
                self._tcr_a2r[seq] = a2r

            for seq in unique_pep:
                mol = sequence_to_mol(seq)
                graph = sequence_to_graph(seq)
                a2r = atom_to_residue_map(seq)
                self._pep_pkl[seq] = pickle.dumps(graph)
                self._pep_mol_pkl[seq] = pickle.dumps(mol)
                self._pep_a2r[seq] = a2r

            print(f'  Graph pre-computation complete.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq1 = self.seqs1[idx]  # TCR
        seq2 = self.seqs2[idx]  # peptide
        label = self.labels[idx]

        # ── Track 1: ESM tokens ──────────────────────────────────────
        enc1_dict = self.tokenizer(
            seq1,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.tcr_max_len,
        )
        enc2_dict = self.tokenizer(
            seq2,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.pep_max_len,
        )
        input_ids1 = enc1_dict['input_ids'].squeeze(0)
        mask1 = enc1_dict['attention_mask'].squeeze(0)
        input_ids2 = enc2_dict['input_ids'].squeeze(0)
        mask2 = enc2_dict['attention_mask'].squeeze(0)

        # random masking (training only)
        if self.mask_prob > 0.0 and self.mask_token_id is not None:
            rand1 = torch.rand(input_ids1.shape)
            mask_idx1 = (rand1 < self.mask_prob) & mask1.bool()
            input_ids1[mask_idx1] = self.mask_token_id
            rand2 = torch.rand(input_ids2.shape)
            mask_idx2 = (rand2 < self.mask_prob) & mask2.bool()
            input_ids2[mask_idx2] = self.mask_token_id

        # ── Track 1: Atchley factors ─────────────────────────────────
        atchley_dim = len(next(iter(self.atchley_map.values())))
        zero_vec = [0.0] * atchley_dim

        tcr_factors = [self.atchley_map.get(aa, zero_vec)
                       for aa in seq1[:self.tcr_max_len]]
        while len(tcr_factors) < self.tcr_max_len:
            tcr_factors.append(zero_vec)
        at1 = torch.tensor(tcr_factors, dtype=torch.float)

        pep_factors = [self.atchley_map.get(aa, zero_vec)
                       for aa in seq2[:self.pep_max_len]]
        while len(pep_factors) < self.pep_max_len:
            pep_factors.append(zero_vec)
        at2 = torch.tensor(pep_factors, dtype=torch.float)

        # ── Track 2: pre-computed atom graphs (pure dict lookup) ─────
        if self.use_graph:
            return (
                input_ids1, mask1,
                input_ids2, mask2,
                at1, at2,
                torch.tensor(label, dtype=torch.long),
                self._tcr_pkl[seq1],             # pre-pickled bytes
                self._pep_pkl[seq2],
                self._tcr_mol_pkl[seq1],
                self._pep_mol_pkl[seq2],
                self._tcr_a2r[seq1],
                self._pep_a2r[seq2],
            )

        return (
            input_ids1, mask1,
            input_ids2, mask2,
            at1, at2,
            torch.tensor(label, dtype=torch.long),
        )


def collate_graph_batch(batch):
    """Custom collate: unpickle pre-serialized graphs + stack tensors."""
    if len(batch[0]) == 7:
        # no graphs
        inp1 = torch.stack([b[0] for b in batch])
        msk1 = torch.stack([b[1] for b in batch])
        inp2 = torch.stack([b[2] for b in batch])
        msk2 = torch.stack([b[3] for b in batch])
        at1 = torch.stack([b[4] for b in batch])
        at2 = torch.stack([b[5] for b in batch])
        lbl = torch.stack([b[6] for b in batch])
        return inp1, msk1, inp2, msk2, at1, at2, lbl

    # with graphs — unpickle pre-serialized bytes
    inp1 = torch.stack([b[0] for b in batch])
    msk1 = torch.stack([b[1] for b in batch])
    inp2 = torch.stack([b[2] for b in batch])
    msk2 = torch.stack([b[3] for b in batch])
    at1 = torch.stack([b[4] for b in batch])
    at2 = torch.stack([b[5] for b in batch])
    lbl = torch.stack([b[6] for b in batch])

    tcr_graphs = [pickle.loads(b[7]) for b in batch]
    pep_graphs = [pickle.loads(b[8]) for b in batch]
    tcr_mols = [pickle.loads(b[9]) for b in batch]
    pep_mols = [pickle.loads(b[10]) for b in batch]
    tcr_a2r = [b[11] for b in batch]
    pep_a2r = [b[12] for b in batch]

    return (
        inp1, msk1, inp2, msk2, at1, at2, lbl,
        tcr_graphs, pep_graphs, tcr_mols, pep_mols,
        tcr_a2r, pep_a2r,
    )
