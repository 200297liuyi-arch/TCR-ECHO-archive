"""Precompute atom graphs for deepAntigen paper's train.csv + test CSV."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from multiprocessing import Pool, cpu_count
from graph_utils import sequence_to_graph, sequence_to_mol, atom_to_residue_map

CACHE_DIR = 'datasets/echo/panpep/graph_cache'
CSV_FILES = [
    '/home/lyf/projects/deepAntigen/test_antigenTCR/Data/sequence/train.csv',
    '/home/lyf/projects/deepAntigen/test_antigenTCR/Data/sequence/zero-shot_sample.csv',
    '/home/lyf/projects/deepAntigen/test_antigenTCR/Data/sequence/covid19.csv',
]


def build_one(args):
    seq, kind = args
    fname = os.path.join(CACHE_DIR, f"{kind}_{seq.replace('/', '_')}.pkl")
    if os.path.exists(fname):
        return "cached", seq
    try:
        mol = sequence_to_mol(seq)
        graph = sequence_to_graph(seq, mol=mol)
        a2r = atom_to_residue_map(seq)
        with open(fname, "wb") as f:
            pickle.dump({"graph": graph, "mol": mol, "a2r": a2r}, f)
        return "built", seq
    except Exception as e:
        return "skip", f"{kind} {seq[:30]}: {e}"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"SKIP: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # handle BOM
        if df.columns[0].startswith('﻿'):
            df.columns = [c.lstrip('﻿') for c in df.columns]
        print(f"\n{csv_path}: {len(df)} rows")

        # Collect unique sequences
        tasks = set()
        peptide_col = 'peptide'
        tcr_col = 'binding_TCR'
        for _, row in df.iterrows():
            pep = str(row[peptide_col]).strip()
            tcr = str(row[tcr_col]).strip().rstrip(';')
            tasks.add((pep, 'pep'))
            tasks.add((tcr, 'tcr'))

        tasks = list(tasks)
        print(f"  Unique seqs: {len(tasks)}")

        cached, built, skipped = 0, 0, 0
        with Pool(min(32, cpu_count())) as pool:
            for status, detail in pool.imap_unordered(build_one, tasks, chunksize=50):
                if status == "cached": cached += 1
                elif status == "built": built += 1
                else:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  SKIP: {detail}")
        print(f"  Cached: {cached}, Built: {built}, Skipped: {skipped}")

    # Final count
    total = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')])
    print(f"\nTotal graphs in cache: {total}")


if __name__ == '__main__':
    main()
