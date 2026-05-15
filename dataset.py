"""TCR-Peptide dataset with pre-computed atom graphs.

CRITICAL: All RDKit graph construction happens in __init__, NOT in __getitem__.
Graphs stored as pickled bytes to avoid PyTorch IPC mmap issues with DataLoader workers.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from graph_utils import sequence_to_graph, sequence_to_mol, atom_to_residue_map


class TCRPeptideDataset(Dataset):
    def __init__(self, df, tokenizer, atchley_map, cols,
                 mask_prob: float = 0.0,
                 tcr_max_len: int = 25,
                 pep_max_len: int = 15,
                 use_graph: bool = True,
                 graph_cache_dir: str = None):
        print(f'mask_prob: {mask_prob}')

        self.seqs1 = df[cols['tcr']].astype(str).str.strip().str.rstrip(';').tolist()
        self.seqs2 = df[cols['peptide']].astype(str).str.strip().str.rstrip(';').tolist()
        self.labels = df[cols['label']].tolist()
        self.tokenizer = tokenizer
        self.atchley_map = atchley_map
        self.mask_prob = mask_prob
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        self.use_graph = use_graph
        self._valid_indices = None

        if use_graph:
            self._tcr_pkl = {}     # seq → pickled PyG Data bytes
            self._pep_pkl = {}
            self._tcr_mol_pkl = {} # seq → pickled rdkit Mol bytes
            self._pep_mol_pkl = {}
            self._tcr_a2r = {}
            self._pep_a2r = {}

            if graph_cache_dir and os.path.exists(graph_cache_dir):
                self._load_from_cache(graph_cache_dir)
            else:
                self._build_graphs()

            self._valid_indices = []
            for i in range(len(self.labels)):
                if (self.seqs1[i] in self._tcr_pkl
                        and self.seqs2[i] in self._pep_pkl):
                    self._valid_indices.append(i)
            if len(self._valid_indices) < len(self.labels):
                print(f'  Filtered to {len(self._valid_indices)} valid samples '
                      f'(dropped {len(self.labels) - len(self._valid_indices)})')
            else:
                self._valid_indices = None

            print(f'  Graph pre-computation complete.')

    def _load_from_cache(self, cache_dir):
        unique_tcr = set(self.seqs1)
        unique_pep = set(self.seqs2)
        loaded, missed = 0, 0

        for seq in unique_tcr:
            fname = os.path.join(cache_dir, f"tcr_{seq.replace('/', '_')}.pkl")
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    data = pickle.load(f)
                self._tcr_pkl[seq] = pickle.dumps(data["graph"])
                self._tcr_mol_pkl[seq] = pickle.dumps(data["mol"])
                self._tcr_a2r[seq] = data["a2r"]
                loaded += 1
            else:
                missed += 1

        for seq in unique_pep:
            fname = os.path.join(cache_dir, f"pep_{seq.replace('/', '_')}.pkl")
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    data = pickle.load(f)
                self._pep_pkl[seq] = pickle.dumps(data["graph"])
                self._pep_mol_pkl[seq] = pickle.dumps(data["mol"])
                self._pep_a2r[seq] = data["a2r"]
                loaded += 1
            else:
                missed += 1

        print(f'  Loaded {loaded} graphs from cache, {missed} missed')
        if missed > 0:
            print(f'  Building {missed} missing graphs...')
            self._build_graphs_missing(cache_dir)

    def _build_graphs(self):
        unique_tcr = set(self.seqs1) - set(self._tcr_pkl.keys())
        unique_pep = set(self.seqs2) - set(self._pep_pkl.keys())
        print(f'Building graphs for {len(unique_tcr)} unique TCR '
              f'+ {len(unique_pep)} unique peptides...')

        skip_tcr = 0
        for seq in unique_tcr:
            try:
                mol = sequence_to_mol(seq)
            except ValueError:
                skip_tcr += 1
                continue
            graph = sequence_to_graph(seq, mol=mol)
            a2r = atom_to_residue_map(seq)
            self._tcr_pkl[seq] = pickle.dumps(graph)
            self._tcr_mol_pkl[seq] = pickle.dumps(mol)
            self._tcr_a2r[seq] = a2r

        skip_pep = 0
        for seq in unique_pep:
            try:
                mol = sequence_to_mol(seq)
            except ValueError:
                skip_pep += 1
                continue
            graph = sequence_to_graph(seq, mol=mol)
            a2r = atom_to_residue_map(seq)
            self._pep_pkl[seq] = pickle.dumps(graph)
            self._pep_mol_pkl[seq] = pickle.dumps(mol)
            self._pep_a2r[seq] = a2r

        if skip_tcr or skip_pep:
            print(f'  Skipped {skip_tcr} TCR + {skip_pep} peptide(s) '
                  f'with unparseable sequences')

    def _build_graphs_missing(self, cache_dir):
        missing_tcr = [s for s in set(self.seqs1) if s not in self._tcr_pkl]
        missing_pep = [s for s in set(self.seqs2) if s not in self._pep_pkl]

        for seq in missing_tcr:
            try:
                mol = sequence_to_mol(seq)
                graph = sequence_to_graph(seq, mol=mol)
                a2r = atom_to_residue_map(seq)
            except (ValueError, Exception):
                continue
            self._tcr_pkl[seq] = pickle.dumps(graph)
            self._tcr_mol_pkl[seq] = pickle.dumps(mol)
            self._tcr_a2r[seq] = a2r
            fname = os.path.join(cache_dir, f"tcr_{seq.replace('/', '_')}.pkl")
            with open(fname, "wb") as f:
                pickle.dump({"graph": graph, "mol": mol, "a2r": a2r}, f)

        for seq in missing_pep:
            try:
                mol = sequence_to_mol(seq)
                graph = sequence_to_graph(seq, mol=mol)
                a2r = atom_to_residue_map(seq)
            except (ValueError, Exception):
                continue
            self._pep_pkl[seq] = pickle.dumps(graph)
            self._pep_mol_pkl[seq] = pickle.dumps(mol)
            self._pep_a2r[seq] = a2r
            fname = os.path.join(cache_dir, f"pep_{seq.replace('/', '_')}.pkl")
            with open(fname, "wb") as f:
                pickle.dump({"graph": graph, "mol": mol, "a2r": a2r}, f)

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.labels)

    def __getitem__(self, idx):
        if self._valid_indices is not None:
            idx = self._valid_indices[idx]
        seq1 = self.seqs1[idx]
        seq2 = self.seqs2[idx]
        label = self.labels[idx]

        enc1_dict = self.tokenizer(
            seq1, return_tensors='pt', padding='max_length',
            truncation=True, max_length=self.tcr_max_len,
        )
        enc2_dict = self.tokenizer(
            seq2, return_tensors='pt', padding='max_length',
            truncation=True, max_length=self.pep_max_len,
        )
        input_ids1 = enc1_dict['input_ids'].squeeze(0)
        mask1 = enc1_dict['attention_mask'].squeeze(0)
        input_ids2 = enc2_dict['input_ids'].squeeze(0)
        mask2 = enc2_dict['attention_mask'].squeeze(0)

        if self.mask_prob > 0.0 and self.mask_token_id is not None:
            rand1 = torch.rand(input_ids1.shape)
            mask_idx1 = (rand1 < self.mask_prob) & mask1.bool()
            input_ids1[mask_idx1] = self.mask_token_id
            rand2 = torch.rand(input_ids2.shape)
            mask_idx2 = (rand2 < self.mask_prob) & mask2.bool()
            input_ids2[mask_idx2] = self.mask_token_id

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

        if self.use_graph:
            return (
                input_ids1, mask1, input_ids2, mask2, at1, at2,
                torch.tensor(label, dtype=torch.long),
                self._tcr_pkl[seq1], self._pep_pkl[seq2],
                self._tcr_mol_pkl[seq1], self._pep_mol_pkl[seq2],
                self._tcr_a2r[seq1], self._pep_a2r[seq2],
            )

        return (
            input_ids1, mask1, input_ids2, mask2, at1, at2,
            torch.tensor(label, dtype=torch.long),
        )


def collate_graph_batch(batch):
    """Custom collate: unpickle pre-serialized graphs + stack tensors."""
    if len(batch[0]) == 7:
        inp1 = torch.stack([b[0] for b in batch])
        msk1 = torch.stack([b[1] for b in batch])
        inp2 = torch.stack([b[2] for b in batch])
        msk2 = torch.stack([b[3] for b in batch])
        at1 = torch.stack([b[4] for b in batch])
        at2 = torch.stack([b[5] for b in batch])
        lbl = torch.stack([b[6] for b in batch])
        return inp1, msk1, inp2, msk2, at1, at2, lbl

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
