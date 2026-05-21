"""Tests for modified deepAntigen_Seq:
1. Sanity check — single forward+backward pass
2. Micro-batch overfitting — 32 samples, 100 steps
3. Metric computation — mini validation with printed metrics
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from deepAntigen_Seq.load_dataset.load_seq import pTCR_DataSet, collate
from deepAntigen_Seq.networks.pTCR_seq import DeepGCN
from deepAntigen_Seq.utils.model_utils import (
    compute_metrics, AverageMeter, set_optimizer, adjust_learning_rate,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Data: use train_joint.csv, tiny subset ──────────────────────────
csv_path = 'datasets/panpep/train_joint.csv'
cache_dir = 'datasets/panpep/graph_cache'
full_ds = pTCR_DataSet(csv_path, aug=False, test=True)

# ── Model args ──────────────────────────────────────────────────────
args = {
    'hidden_size': 128,
    'depth': 2,
    'k': 50,
    'heads': 4,
}

# ====================================================================
# TEST 1: Sanity Check — single forward + backward
# ====================================================================
print("\n" + "=" * 60)
print("TEST 1: Sanity Check (single forward + backward)")
print("=" * 60)

model = DeepGCN(args).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Single batch
loader = DataLoader(Subset(full_ds, range(4)), batch_size=4,
                    collate_fn=collate, shuffle=False)
_, _, _, labels, pep_graphs, cdr3_graphs = next(iter(loader))
pep_graphs = Batch.from_data_list(pep_graphs).to(device)
cdr3_graphs = Batch.from_data_list(cdr3_graphs).to(device)
labels = labels.to(device).float()

print(f"  peptide graph nodes: {pep_graphs.x.shape}")
print(f"  CDR3 graph nodes:    {cdr3_graphs.x.shape}")
print(f"  labels shape:         {labels.shape}")

logits = model(pep_graphs, cdr3_graphs)
print(f"  logits shape:         {logits.shape} -> [{logits[0].item():.4f}, {logits[1].item():.4f}, ...]")

loss = criterion(logits, labels)
print(f"  loss:                 {loss.item():.6f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check gradients flow to all parts
grad_ok = 0
grad_bad = 0
for name, p in model.named_parameters():
    if p.requires_grad:
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_ok += 1
        else:
            grad_bad += 1
            print(f"  WARNING: zero gradient at {name}")
print(f"  Gradients flowing:   {grad_ok}/{grad_ok+grad_bad} layers")
print(f"  GPU memory:           {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB" if device.type == 'cuda' else "")

# Free mem
del model, optimizer, pep_graphs, cdr3_graphs, labels, logits, loss
torch.cuda.empty_cache()

# ====================================================================
# TEST 2: Micro-batch Overfitting — 32 samples, 100 steps
# ====================================================================
print("\n" + "=" * 60)
print("TEST 2: Micro-batch Overfitting (32 samples, 100 steps)")
print("=" * 60)

model = DeepGCN(args).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tiny_loader = DataLoader(Subset(full_ds, range(32)), batch_size=32,
                         collate_fn=collate, shuffle=False)
_, _, _, labels_oh, pep_gs, cdr3_gs = next(iter(tiny_loader))
pep_gs = Batch.from_data_list(pep_gs).to(device)
cdr3_gs = Batch.from_data_list(cdr3_gs).to(device)
labels_oh = labels_oh.to(device).float()

print(f"  Batch: 32 samples, {labels_oh.sum().item():.0f} pos / {(1-labels_oh).sum().item():.0f} neg")
print(f"  {'Step':<8} {'Loss':<12}")
print(f"  {'-'*20}")

initial_loss = None
losses = []
for step in range(1, 101):
    logits = model(pep_gs, cdr3_gs)
    loss = criterion(logits, labels_oh)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step == 1:
        initial_loss = loss.item()
    if step % 20 == 0 or step == 1:
        print(f"  {step:<8} {loss.item():<12.6f}")

final_loss = losses[-1]
decay = (initial_loss - final_loss) / initial_loss * 100 if initial_loss else 0
print(f"\n  Initial loss: {initial_loss:.6f}")
print(f"  Final loss:   {final_loss:.6f}")
print(f"  Reduction:    {decay:.1f}%")

if decay > 90:
    print("  VERDICT: PASS — loss dropped >90%, gradients flow end-to-end")
elif decay > 50:
    print("  VERDICT: WEAK — loss dropped but not converging to 0, check architecture")
else:
    print("  VERDICT: FAIL — gradients likely broken at some layer")

del model, optimizer
torch.cuda.empty_cache()

# ====================================================================
# TEST 3: Metric Computation — mini validation
# ====================================================================
print("\n" + "=" * 60)
print("TEST 3: Metric Computation (mini validation)")
print("=" * 60)

# Use a slightly larger subset for meaningful metrics
val_loader = DataLoader(Subset(full_ds, range(64)), batch_size=32,
                        collate_fn=collate, shuffle=False)

model = DeepGCN(args).to(device)
model.eval()
all_logits = []
all_labels = []
with torch.no_grad():
    for _, _, _, labels, pep_gs, cdr3_gs in val_loader:
        pep_gs = Batch.from_data_list(pep_gs).to(device)
        cdr3_gs = Batch.from_data_list(cdr3_gs).to(device)
        logits = model(pep_gs, cdr3_gs)
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.numpy())

acc, auc, f1, prec, rec, auprc = compute_metrics(all_labels, all_logits)
print(f"  Samples:   {len(all_labels)}")
print(f"  Accuracy:  {acc:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  AUC-PR:    {auprc:.4f}")

# Check for NaN/Inf
all_ok = not (np.isnan([acc, auc, f1, prec, rec, auprc]).any() or
              np.isinf([acc, auc, f1, prec, rec, auprc]).any())
print(f"  NaN/Inf check: {'PASS' if all_ok else 'FAIL'}")

del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
