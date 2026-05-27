"""Zero-shot evaluation: ESM-only model on zero_dataset.csv (unseen peptides).

The zero-shot set contains only positive binding pairs — we report prediction
statistics (mean prob, hit-rate@0.5) rather than AUC since there's only 1 class.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import Model
from dataset import TCRPeptideDataset
from utils import load_atchley, load_checkpoint

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = 'runs/esm_only'
ZERO_CSV = 'datasets/echo/panpep/zero_dataset.csv'
OUT_DIR = 'runs/esm_only'


def compute_preds(model, loader, device, use_graph=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            if use_graph:
                (inp1, msk1, inp2, msk2, at1, at2, labels,
                 tcr_graphs, pep_graphs, tcr_mols, pep_mols,
                 tcr_a2r, pep_a2r) = batch
                logits, _ = model(
                    inp1, msk1, inp2, msk2, at1, at2, labels,
                    tcr_graphs=tcr_graphs, pep_graphs=pep_graphs,
                    tcr_mols=tcr_mols, pep_mols=pep_mols,
                    tcr_a2r=tcr_a2r, pep_a2r=pep_a2r,
                )
            else:
                *inputs, labels = batch
                logits, _ = model(*inputs, labels=labels)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def main():
    # 1. Load model
    model, _, cfg = load_checkpoint(Model, CKPT_DIR, class_imbalance=0.5, device=DEVICE)
    model.eval()
    print(f"Loaded ESM-only checkpoint from {CKPT_DIR}")

    # 2. Load zero-shot data
    df = pd.read_csv(ZERO_CSV)
    print(f"Zero-shot samples: {len(df)}")
    pos = df['label'].sum()
    print(f"  Pos: {pos} ({100*pos/len(df):.1f}%), Neg: {len(df)-pos}")

    columns = cfg['dataset']['columns']

    # 3. Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/{cfg['esm']['encoder1']}"
    )
    atchley_map = load_atchley(cfg.get('atchley_path'))

    ds = TCRPeptideDataset(
        df, tokenizer, atchley_map, columns,
        use_graph=False,
    )
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 4. Predict
    preds, labels = compute_preds(model, loader, DEVICE)
    hit_rate = (preds >= 0.5).mean()

    print(f"\nZero-Shot Results (ESM-only) — {len(preds)} true binders:")
    print(f"  Mean prob:    {preds.mean():.4f}")
    print(f"  Median prob:  {np.median(preds):.4f}")
    print(f"  Std prob:     {preds.std():.4f}")
    print(f"  Min prob:     {preds.min():.4f}")
    print(f"  Max prob:     {preds.max():.4f}")
    print(f"  Hit-rate@0.5: {hit_rate:.4f}  ({hit_rate*100:.1f}% predicted as binder)")
    print(f"  Hit-rate@0.7: {(preds >= 0.7).mean():.4f}")
    print(f"  Hit-rate@0.9: {(preds >= 0.9).mean():.4f}")

    # 5. Per-peptide breakdown
    top_peps = df['peptide'].value_counts().head(10).index.tolist()
    print(f"\nPer-peptide mean prob (top 10 by frequency):")
    for pep in top_peps:
        mask_pep = df['peptide'] == pep
        p = preds[mask_pep.values]
        print(f"  {pep:15s}  n={len(p):4d}  mean={p.mean():.4f}  hit@0.5={(p>=0.5).mean():.4f}")

    # 6. Save
    out_df = pd.DataFrame({
        'peptide': df['peptide'],
        'tcr': df['binding_TCR'],
        'pred': preds,
        'label': labels.astype(int),
    })
    out_path = os.path.join(OUT_DIR, 'zero_shot_predictions.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")


if __name__ == '__main__':
    main()
