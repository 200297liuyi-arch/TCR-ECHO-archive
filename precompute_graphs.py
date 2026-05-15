"""Pre-compute all molecular graphs (multiprocessing workers write files directly)."""

import pickle, os
from multiprocessing import Pool, cpu_count, Manager
import pandas as pd
from graph_utils import sequence_to_graph, sequence_to_mol, atom_to_residue_map


def build_and_save(args):
    """Build graph and save directly to disk. Returns (status, details)."""
    seq, kind, cache_dir = args
    fname = os.path.join(cache_dir, f"{kind}_{seq.replace('/', '_')}.pkl")
    if os.path.exists(fname):
        return "cached", seq
    try:
        mol = sequence_to_mol(seq)
        graph = sequence_to_graph(seq, mol=mol)
        a2r = atom_to_residue_map(seq)
        with open(fname, "wb") as f:
            pickle.dump({"graph": graph, "mol": mol, "a2r": a2r}, f)
        return "saved", seq
    except Exception as e:
        return "skip", f"{kind} {seq[:30]}: {e}"


def precompute(csv_path, cache_dir, tcr_col="binding_TCR", pep_col="peptide"):
    """Build graphs for all unique sequences in a CSV, saving to cache_dir."""
    df = pd.read_csv(csv_path)
    os.makedirs(cache_dir, exist_ok=True)

    tcr_seqs = set(df[tcr_col].astype(str).str.strip().str.rstrip(";"))
    pep_seqs = set(df[pep_col].astype(str).str.strip().str.rstrip(";"))

    tasks = []
    for s in tcr_seqs:
        tasks.append((s, "tcr", cache_dir))
    for s in pep_seqs:
        tasks.append((s, "pep", cache_dir))

    if not tasks:
        print(f"All graphs already cached → {cache_dir}")
        return

    # Cap workers at 32 to avoid memory pressure from 40× RDKit instances
    n_workers = min(cpu_count(), 32)
    print(f"Building {len(tasks)} graphs with {n_workers} workers → {cache_dir}")

    saved, skipped, cached = 0, 0, 0
    with Pool(n_workers) as pool:
        for status, detail in pool.imap_unordered(
            build_and_save, tasks, chunksize=50
        ):
            if status == "saved":
                saved += 1
            elif status == "skip":
                skipped += 1
                if skipped <= 10:
                    print(f"  SKIP {detail}")
            else:
                cached += 1
            total = saved + skipped + cached
            if total % 10000 == 0:
                print(f"  ... {total}/{len(tasks)} ({saved} new, {cached} cached, {skipped} skip)")

    print(f"Done: {saved} saved, {cached} cached, {skipped} skipped → {cache_dir}")


if __name__ == "__main__":
    base = "datasets/panpep"
    cache = "datasets/panpep/graph_cache"

    for split in ["train_joint", "val_joint", "majority_testing_dataset"]:
        path = os.path.join(base, f"{split}.csv")
        if not os.path.exists(path):
            print(f"SKIP {path} — not found")
            continue
        precompute(path, cache)
