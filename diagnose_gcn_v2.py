"""GCN ablation: compare predictions with/without GCN, analyse gate distribution."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/lyf/projects/TCR-ECHO')

from utils import load_checkpoint
from gcn_plugin import GCNPlugin
from dataset import TCRPeptideDataset, collate_graph_batch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_atchley
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import pearsonr

device = 'cuda:0'
chk_dir = 'runs/gcn_joint_v2/'

model_cls = GCNPlugin
model, _, cfg = load_checkpoint(model_cls, chk_dir, 0.5, device=device)
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{cfg['esm']['encoder1']}")
atchley_map = load_atchley(cfg.get('atchley_path'))
use_graph = cfg.get('use_gcn', False)

# Load test data
import pandas as pd
df = pd.read_csv(cfg['dataset']['test_csv'])
ds = TCRPeptideDataset(df, tokenizer, atchley_map, cfg['dataset']['columns'],
    use_graph=use_graph, graph_cache_dir=cfg.get('graph_cache_dir'))
loader = DataLoader(ds, batch_size=64, collate_fn=collate_graph_batch)

model.eval()
all_logits_gcn, all_logits_no_gcn, all_gates, all_labels = [], [], [], []

with torch.no_grad():
    for batch in loader:
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        (inp1, msk1, inp2, msk2, at1, at2, labels,
         tcr_graphs, pep_graphs, *_rest) = batch

        # With GCN
        logits_gcn = model(inp1, msk1, inp2, msk2, at1, at2, labels=None,
                           tcr_graphs=tcr_graphs, pep_graphs=pep_graphs)
        # Without GCN
        logits_no_gcn = model(inp1, msk1, inp2, msk2, at1, at2, labels=None)

        # Capture gate values (hijack forward to get gate)
        from torch_geometric.data import Batch
        pep_batch = Batch.from_data_list(pep_graphs).to(device)
        tcr_batch = Batch.from_data_list(tcr_graphs).to(device)
        gcn_out = model.gcn(pep_batch, tcr_batch)
        F_gcn_val = gcn_out["gcn_feat"]

        # Re-run ESM part to get tcr_feat/pep_feat
        out1 = model.esm1(input_ids=inp1, attention_mask=msk1).last_hidden_state
        out2 = model.esm2(input_ids=inp2, attention_mask=msk2).last_hidden_state
        tcr_enc = out1[:, 1:, :]
        pep_enc = out2[:, 1:, :]
        tcr_att, pep_att = model.cross_attn(tcr_enc, pep_enc, at1, at2)
        tcr_pool = tcr_att.mean(dim=1)
        pep_pool = pep_att.mean(dim=1)
        tcr_feat = model.tcr_proj(tcr_pool)
        pep_feat = model.pep_proj(pep_pool)

        gate = torch.sigmoid(model.gate_gcn(torch.cat([tcr_feat, pep_feat], dim=-1)))
        all_gates.append(gate.cpu())

        all_logits_gcn.append(logits_gcn.cpu())
        all_logits_no_gcn.append(logits_no_gcn.cpu())
        all_labels.append(labels.cpu())

logits_gcn = torch.cat(all_logits_gcn).numpy()
logits_no_gcn = torch.cat(all_logits_no_gcn).numpy()
gates = torch.cat(all_gates).numpy()
labels_np = torch.cat(all_labels).numpy()

preds_gcn = 1 / (1 + np.exp(-logits_gcn))
preds_no_gcn = 1 / (1 + np.exp(-logits_no_gcn))

corr = np.corrcoef(preds_gcn, preds_no_gcn)[0, 1]
max_diff = np.max(np.abs(preds_gcn - preds_no_gcn))

print("=" * 60)
print("GCN Ablation Analysis")
print("=" * 60)
print(f"Prediction correlation (GCN vs No-GCN): {corr:.6f}")
print(f"Max absolute prediction difference:      {max_diff:.6f}")
print()

auc_gcn = roc_auc_score(labels_np, preds_gcn)
auc_no_gcn = roc_auc_score(labels_np, preds_no_gcn)
print(f"Test AUC with GCN:    {auc_gcn:.4f}")
print(f"Test AUC without GCN: {auc_no_gcn:.4f}")
print(f"Delta AUC:            {auc_gcn - auc_no_gcn:+.4f}")
print()

# Gate analysis
print("=" * 60)
print("Gate Distribution (sigmoid, per GCN dimension)")
print("=" * 60)
print(f"Gate mean:     {gates.mean():.4f}")
print(f"Gate std:      {gates.std():.4f}")
print(f"Gate min:      {gates.min():.4f}")
print(f"Gate max:      {gates.max():.4f}")
print()

# Per-dimension gate stats
dim_means = gates.mean(axis=0)
print(f"Per-dim gate mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
print(f"Top-5 dims: {np.argsort(dim_means)[-5:][::-1]} = {np.sort(dim_means)[-5:][::-1]}")
print(f"Bot-5 dims: {np.argsort(dim_means)[:5]} = {np.sort(dim_means)[:5]}")

# Gated GCN contribution magnitude
gated_norm = np.linalg.norm(gates * gcn_out["gcn_feat"].cpu().numpy(), axis=1).mean()
print(f"\nMean ||gate * gcn_feat||_2: {gated_norm:.4f}")

# Classifier weight analysis for GCN portion
clf_weight = model.classifier[0].weight.data.cpu().numpy()  # [512, 1152]
gcn_weight_norm = np.linalg.norm(clf_weight[:, 1024:])  # last 128 cols
esm_weight_norm = np.linalg.norm(clf_weight[:, :1024])
print(f"\nClassifier weight norm (ESM portion): {esm_weight_norm:.2f}")
print(f"Classifier weight norm (GCN portion): {gcn_weight_norm:.2f}")
print(f"Ratio GCN/ESM: {gcn_weight_norm / esm_weight_norm:.4f}")

# ── Zero-shot ablation ─────────────────────────────────────────
print()
print("=" * 60)
print("Zero-Shot GCN Ablation")
z_csv = 'datasets/deepantigen/zero_test_paper.csv'
df_z = pd.read_csv(z_csv)
z_cols = {'peptide': 'peptide', 'tcr': 'binding_TCR', 'label': 'label'}
z_ds = TCRPeptideDataset(df_z, tokenizer, atchley_map, z_cols,
    use_graph=use_graph, graph_cache_dir=cfg.get('graph_cache_dir'))
z_loader = DataLoader(z_ds, batch_size=64, collate_fn=collate_graph_batch)

z_preds_gcn, z_preds_no_gcn, z_labels = [], [], []
with torch.no_grad():
    for batch in z_loader:
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        (inp1, msk1, inp2, msk2, at1, at2, labels,
         tcr_graphs, pep_graphs, *_rest) = batch
        lg = model(inp1, msk1, inp2, msk2, at1, at2, labels=None,
                   tcr_graphs=tcr_graphs, pep_graphs=pep_graphs)
        ln = model(inp1, msk1, inp2, msk2, at1, at2, labels=None)
        z_preds_gcn.append(torch.sigmoid(lg).cpu())
        z_preds_no_gcn.append(torch.sigmoid(ln).cpu())
        z_labels.append(labels.cpu())

z_pg = torch.cat(z_preds_gcn).numpy()
z_pn = torch.cat(z_preds_no_gcn).numpy()
z_lb = torch.cat(z_labels).numpy()
print(f"Zero-shot AUC with GCN:    {roc_auc_score(z_lb, z_pg):.4f}")
print(f"Zero-shot AUC without GCN: {roc_auc_score(z_lb, z_pn):.4f}")
print(f"Delta AUC:                 {roc_auc_score(z_lb, z_pg) - roc_auc_score(z_lb, z_pn):+.4f}")
print(f"Prediction correlation:    {np.corrcoef(z_pg, z_pn)[0,1]:.4f}")
print("Done.")
