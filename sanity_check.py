"""Sanity Check: overfit 100 samples to verify the fused TCR-ECHO model.

Tests:
  1. Loss → 0   (focal + contrastive)
  2. Accuracy → 100%
  3. Gradient flow through SuperNodeExchange & cross-modal layers
  4. Activation statistics per GCN layer

Usage:
  python sanity_check.py
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── tiny data ─────────────────────────────────────────────────────────
N_SAMPLES = 100          # 50 pos + 50 neg
BATCH_SIZE = 8           # small: 650M ESM + GCN
N_EPOCHS = 200

# ── model (use cached 650M ESM — t6 unavailable offline) ─
ESM_NAME = "esm2_t33_650M_UR50D"

# ── LoRA (overfit mode: higher rank, more capacity) ────────────────────
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0        # OFF for sanity check

# Force offline mode — everything is cached
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ── GCN ────────────────────────────────────────────────────────────────
USE_GCN = True
GCN_HIDDEN = 64
GCN_DEPTH = 2
GCN_K = 8
GCN_HEADS = 4
GCN_IN_CHANNELS = 25

# ── training (overfit mode) ────────────────────────────────────────────
LR = 1e-3                 # high LR to force overfit
WEIGHT_DECAY = 0.0        # OFF
DROPOUT = 0.0             # OFF
MASK_PROB = 0.0           # OFF (no sequence masking)
FOCAL_GAMMA = 2.0
CONTRASTIVE_TEMP = 0.1
LAMBDA_ENC = 0.5
LAMBDA_INT = 2.0
SECOND_CONTRASTIVE = True

# ── gradient monitoring ────────────────────────────────────────────────
GRAD_MONITOR_EVERY = 50   # print gradient norms every N epochs

# ══════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print(" Loading data ...")
print("─" * 60)

df = pd.read_csv("datasets/Panpep_trainingData.csv")

# Balance: 50 pos + 50 neg
pos = df[df["label"] == 1].sample(N_SAMPLES // 2, random_state=42)
neg = df[df["label"] == 0].sample(N_SAMPLES // 2, random_state=42)
df_tiny = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

pos_count = (df_tiny["label"] == 1).sum()
neg_count = (df_tiny["label"] == 0).sum()
class_balance = neg_count / (pos_count + neg_count)
print(f"Tiny dataset: {len(df_tiny)} samples  (pos={pos_count}, neg={neg_count})")
print(f"Class balance (alpha): {class_balance:.3f}")

from dataset import TCRPeptideDataset, collate_graph_batch
from utils import load_atchley

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{ESM_NAME}")
atchley_map = load_atchley()

COLUMNS = {"tcr": "cdr3b", "peptide": "peptide", "label": "label"}

ds = TCRPeptideDataset(
    df_tiny, tokenizer, atchley_map, COLUMNS,
    mask_prob=MASK_PROB,
    use_graph=USE_GCN,
)

collate_fn = collate_graph_batch if USE_GCN else None
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ══════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print(" Building model ...")
print("─" * 60)

from model import Model

gcn_args = dict(
    hidden_size=GCN_HIDDEN,
    depth=GCN_DEPTH,
    k=GCN_K,
    heads=GCN_HEADS,
    in_channels=GCN_IN_CHANNELS,
)

# LoRA target layers for esm2_t33_650M
lora_layers = [32, 31, 30, 29, 28]

model = Model(
    esm1_name=ESM_NAME,
    esm2_name=ESM_NAME,
    use_lora=True,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=lora_layers,
    contrastive_temp=CONTRASTIVE_TEMP,
    lambda_enc=LAMBDA_ENC,
    lambda_int=LAMBDA_INT,
    classifier_hidden=128,
    dropout=DROPOUT,
    cross_attn_dropout=0.0,      # OFF for sanity
    focal_gamma=FOCAL_GAMMA,
    class_balance=class_balance,
    second_contrastive=SECOND_CONTRASTIVE,
    use_gcn=USE_GCN,
    gcn_args=gcn_args,
    gcn_freeze_encoder=False,    # train all GCN for sanity check
    random_init=False,
    lambda_gcn_aux=10.0,         # weight of GCN auxiliary loss (boosted for gradient flow)
).to(DEVICE)

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:    {total:,}")
print(f"Trainable:       {trainable:,}")
print(f"GCN params:      {sum(p.numel() for p in model.gcn.parameters()):,}")
print(f"  init_w_pep:    {sum(p.numel() for p in model.gcn.init_w_pep.parameters()):,}")
print(f"  layers[0]:     {sum(p.numel() for p in model.gcn.layers[0].parameters()):,}")
print(f"  topk_pep:      {sum(p.numel() for p in model.gcn.topk_pep.parameters()):,}")
print(f"  cross_attn:    {sum(p.numel() for p in model.gcn.peptide_cdr3_att.parameters()):,}")
print(f"Spatial proj:    {sum(p.numel() for p in model.gcn_spatial_proj.parameters()):,}")
print(f"Classifier:      {sum(p.numel() for p in model.classifier.parameters()):,}")

# ══════════════════════════════════════════════════════════════════════
#  Gradient / Activation hooks
# ══════════════════════════════════════════════════════════════════════

grad_log = {}
act_log = {}

def _make_grad_hook(name):
    def hook(module, grad_input, grad_output):
        # grad_output is a tuple of gradient tensors w.r.t. module outputs
        g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if g is not None:
            grad_log[name] = {
                "norm": g.norm().item(),
                "mean": g.mean().item(),
                "std": g.std().item(),
                "max": g.max().item(),
                "min": g.min().item(),
            }
    return hook

def _make_act_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            act_log[name] = {
                "norm": out.norm().item(),
                "mean": out.mean().item(),
                "std": out.std().item(),
                "max": out.max().item(),
                "min": out.min().item(),
            }
    return hook

# Register hooks on SuperNodeExchange in each GCN layer
for i, layer in enumerate(model.gcn.layers):
    layer.super_exchange.register_full_backward_hook(
        _make_grad_hook(f"layer_{i}/super_exchange")
    )
    layer.super_exchange.register_forward_hook(
        _make_act_hook(f"layer_{i}/super_exchange")
    )
    layer.tgcn_pep.register_full_backward_hook(
        _make_grad_hook(f"layer_{i}/tgcn_pep")
    )
    layer.tgcn_tcr.register_full_backward_hook(
        _make_grad_hook(f"layer_{i}/tgcn_tcr")
    )

# Also monitor the TopK outputs
model.gcn.topk_pep.register_full_backward_hook(
    _make_grad_hook("topk_pep")
)
model.gcn.topk_tcr.register_full_backward_hook(
    _make_grad_hook("topk_tcr")
)
model.gcn.peptide_cdr3_att.register_full_backward_hook(
    _make_grad_hook("peptide_cdr3_att")
)

# Monitor TopK score values (forward hook to check saturation)
def _topk_score_hook(module, inp, out):
    _, _, scores, _ = out
    if scores.numel() > 0:
        act_log["topk_score"] = {
            "norm": scores.norm().item(),
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "max": scores.max().item(),
            "min": scores.min().item(),
        }
model.gcn.topk_pep.register_forward_hook(_topk_score_hook)

# Monitor spatial projection
model.gcn_spatial_proj.register_full_backward_hook(
    _make_grad_hook("gcn_spatial_proj")
)

# Register forward hooks on TGCN GRU cells for activation monitoring
for i, layer in enumerate(model.gcn.layers):
    layer.gru_pep.register_forward_hook(
        _make_act_hook(f"layer_{i}/gru_pep")
    )
    layer.gru_tcr.register_forward_hook(
        _make_act_hook(f"layer_{i}/gru_tcr")
    )

# ══════════════════════════════════════════════════════════════════════
#  Optimizer
# ══════════════════════════════════════════════════════════════════════

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ══════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print(" Training (overfit sanity check)")
print("─" * 60)
print(f"{'Epoch':>5s} | {'Loss':>10s} | {'Focal':>8s} | {'L_enc':>8s} | {'L_int':>8s} | {'Acc':>8s} | {'GCN_grad':>10s} | {'Gate':>8s}")
print("─" * 75)

best_acc = 0.0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    grad_log.clear()
    act_log.clear()

    epoch_total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch]

        if USE_GCN:
            (inp1, msk1, inp2, msk2, at1, at2, labels,
             tcr_graphs, pep_graphs, tcr_mols, pep_mols,
             tcr_a2r, pep_a2r) = batch
            logits, loss = model(
                inp1, msk1, inp2, msk2, at1, at2, labels,
                tcr_graphs=tcr_graphs, pep_graphs=pep_graphs,
                tcr_mols=tcr_mols, pep_mols=pep_mols,
                tcr_a2r=tcr_a2r, pep_a2r=pep_a2r,
            )
        else:
            inp1, msk1, inp2, msk2, at1, at2, labels = batch
            logits, loss = model(inp1, msk1, inp2, msk2, at1, at2, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()

        epoch_total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = epoch_total_loss / len(loader)
    acc = correct / max(total, 1)

    if acc > best_acc:
        best_acc = acc

    # ── gradient / activation reporting ───────────────────────────
    do_report = (epoch == 1 or epoch % GRAD_MONITOR_EVERY == 0
                 or epoch == N_EPOCHS)

    if do_report:
        # aggregate GCN gradient norm
        gcn_grad_norms = []
        for key, val in grad_log.items():
            gcn_grad_norms.append(val["norm"])
        avg_grad_norm = sum(gcn_grad_norms) / max(len(gcn_grad_norms), 1)

        # gate activation (rho from SuperNodeExchange)
        gate_act = []
        for key, val in act_log.items():
            if "super_exchange" in key:
                gate_act.append(f"{key}: μ={val['mean']:.3f} σ={val['std']:.3f}")

        gate_str = "; ".join(gate_act[:2]) if gate_act else "n/a"
        score_str = ""
        if "topk_score" in act_log:
            s = act_log["topk_score"]
            score_str = f"score μ={s['mean']:.3f} σ={s['std']:.3f}"

        print(f"{epoch:5d} | {avg_loss:10.4f} | {'-':>8s} | {'-':>8s} | {'-':>8s} | "
              f"{acc:8.4f} | {avg_grad_norm:10.2e} | {score_str[:40]}")

    # ── convergence check ─────────────────────────────────────────
    if epoch == N_EPOCHS or (acc >= 0.99 and avg_loss < 0.1):
        print(f"\n>>> Convergence at epoch {epoch}: loss={avg_loss:.6f}, acc={acc:.4f}")
        break

# ══════════════════════════════════════════════════════════════════════
#  Final diagnostics
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" FINAL DIAGNOSTICS")
print("=" * 60)

print(f"\nBest accuracy: {best_acc:.4f}")
print(f"Final loss:    {avg_loss:.6f}")

# Gradient flow report
print(f"\n─ Gradient norms (from hooks) ─")
if grad_log:
    for name, stats in sorted(grad_log.items()):
        print(f"  {name:40s}  norm={stats['norm']:10.6f}  mean={stats['mean']:10.6f}  "
              f"max={stats['max']:10.6f}  min={stats['min']:10.6f}")
else:
    print("  (no gradient hooks captured — may need a forward pass)")

# Activation report
print(f"\n─ Activation statistics (from hooks) ─")
if act_log:
    for name, stats in sorted(act_log.items()):
        print(f"  {name:40s}  norm={stats['norm']:10.4f}  mean={stats['mean']:10.6f}  "
              f"std={stats['std']:10.6f}  max={stats['max']:10.6f}")
else:
    print("  (no activation hooks captured)")

# Parameter gradient check
print(f"\n─ Parameter gradients (direct check) ─")
for name, param in model.named_parameters():
    if param.grad is not None and "gcn" in name:
        gnorm = param.grad.norm().item()
        if gnorm > 1e-8:
            print(f"  {name:55s}  grad_norm={gnorm:.6e}")
        if gnorm == 0.0:
            print(f"  {name:55s}  grad_norm=0.0  ⚠ ZERO GRADIENT")
        if gnorm > 100:
            print(f"  {name:55s}  grad_norm={gnorm:.2e}  ⚠ EXPLODING GRADIENT")

# Sanity verdict
print("\n" + "=" * 60)
if best_acc >= 0.99 and avg_loss < 0.1:
    print(" ✅ SANITY CHECK PASSED — model can overfit 100 samples")
elif best_acc >= 0.90:
    print(" ⚠ SANITY CHECK PARTIAL — accuracy > 90% but not perfect; check capacity")
else:
    print(" ❌ SANITY CHECK FAILED — model cannot overfit 100 samples")
    print("    Check: gradients, loss design, data pipeline")
print("=" * 60)
