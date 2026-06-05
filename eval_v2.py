"""Quick evaluation script for V2 checkpoint."""
import os, sys, torch
import pandas as pd
import numpy as np
sys.path.insert(0, '/home/lyf/projects/TCR-ECHO')

from utils import load_checkpoint
from gcn_plugin import GCNPlugin
from dataset import TCRPeptideDataset, collate_graph_batch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_atchley
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

device = 'cuda:0'  # CUDA_VISIBLE_DEVICES=2 maps GPU2 → device 0
chk_dir = 'runs/gcn_joint_v2/'

model_cls = GCNPlugin
model, _, cfg = load_checkpoint(model_cls, chk_dir, 0.5, device=device)
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{cfg['esm']['encoder1']}")
atchley_map = load_atchley(cfg.get('atchley_path'))
use_graph = cfg.get('use_gcn', False)

def evaluate(df, ds, name, out_path):
    cols = cfg['dataset']['columns']
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'],
        collate_fn=collate_graph_batch if use_graph else None)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            (inp1, msk1, inp2, msk2, at1, at2, labels,
             tcr_graphs, pep_graphs, *_rest) = batch
            logits, _ = model(inp1, msk1, inp2, msk2, at1, at2, labels,
                              tcr_graphs=tcr_graphs, pep_graphs=pep_graphs)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    f1 = f1_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    print(f"{name}: AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")

    # Save with consistent columns: peptide, tcr, pred, label
    pep_col = cols['peptide']
    tcr_col = cols['tcr']
    lbl_col = cols['label']
    out = pd.DataFrame({
        'peptide': df[pep_col].values[:len(all_preds)],
        'tcr': df[tcr_col].values[:len(all_preds)],
        'pred': all_preds,
        'label': all_labels,
    })
    out.to_csv(out_path, index=False)

# ── Test set ──────────────────────────────────────────────────
print("=" * 60)
print("Test: PanPep majority_testing_dataset")
df_test = pd.read_csv(cfg['dataset']['test_csv'])
test_ds = TCRPeptideDataset(df_test, tokenizer, atchley_map, cfg['dataset']['columns'],
    use_graph=use_graph, graph_cache_dir=cfg.get('graph_cache_dir'))
evaluate(df_test, test_ds, "Test", os.path.join(chk_dir, 'test_predictions.csv'))

# ── Zero-shot ─────────────────────────────────────────────────
print("=" * 60)
print("Zero-shot: deepAntigen zero_test_paper")
z_csv = 'datasets/deepantigen/zero_test_paper.csv'
z_cols = {'peptide': 'peptide', 'tcr': 'binding_TCR', 'label': 'label'}
df_z = pd.read_csv(z_csv)
z_ds = TCRPeptideDataset(df_z, tokenizer, atchley_map, z_cols,
    use_graph=use_graph, graph_cache_dir=cfg.get('graph_cache_dir'))
evaluate(df_z, z_ds, "Zero-shot", os.path.join(chk_dir, 'zero_shot_predictions.csv'))
print("Done.")
