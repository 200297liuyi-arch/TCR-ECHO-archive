"""Diagnostic: trace which TopKPooling weights get gradients, and why overfit is weak."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from deepAntigen_Seq.load_dataset.load_seq import pTCR_DataSet, collate
from deepAntigen_Seq.networks.pTCR_seq import DeepGCN

device = torch.device('cuda')
csv_path = 'datasets/panpep/train_joint.csv'
full_ds = pTCR_DataSet(csv_path, aug=False, test=True)

args = {'hidden_size': 128, 'depth': 2, 'k': 10, 'heads': 4}

# ===========================================================================
# 1. PRECISELY identify which TopKPooling params get zero gradient
# ===========================================================================
print("=" * 60)
print("DIAGNOSTIC 1: Per-parameter gradient audit (depth=2, full forward)")
print("=" * 60)

model = DeepGCN(args).to(device)
criterion = torch.nn.BCEWithLogitsLoss()

loader = DataLoader(Subset(full_ds, range(4)), batch_size=4, collate_fn=collate, shuffle=False)
_, _, _, labels, pep_gs, cdr3_gs = next(iter(loader))
pep_gs = Batch.from_data_list(pep_gs).to(device)
cdr3_gs = Batch.from_data_list(cdr3_gs).to(device)
labels = labels.to(device).float()

logits = model(pep_gs, cdr3_gs)
loss = criterion(logits, labels)
loss.backward()

print(f"{'Layer':<55} {'Grad?':<8} {'|grad|':<12}")
print("-" * 75)
for name, p in model.named_parameters():
    has = "YES" if (p.grad is not None and p.grad.abs().sum() > 0) else "NO"
    gnorm = p.grad.abs().sum().item() if p.grad is not None else 0.0
    marker = " <--" if has == "NO" and 'top_K_pooling' in name else ""
    print(f"  {name:<52} {has:<8} {gnorm:<12.6f}{marker}")

# ===========================================================================
# 2. Is the TopK gradient blocked by the topk() sort or by architecture?
# ===========================================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: TopK score gradient isolation test")
print("=" * 60)

from deepAntigen_Seq.networks.top_k_pooling_seq import TopKPooling, topk

# Test: can gradients flow through score[perm] when perm comes from topk()?
x = torch.randn(3, 10, 128, requires_grad=True, device=device)  # 3 graphs, 10 atoms each
pool = TopKPooling(128, ratio=5).to(device)

# Explicit forward to check grad on weight
xx = x.view(-1, 128)  # [30, 128]
score = (xx * pool.weight).sum(dim=-1)  # [30]
score_t = pool.nonlinearity(score / pool.weight.norm(p=2, dim=-1))

# Simulate multi-graph batch
batch = torch.arange(3, device=device).repeat_interleave(10)
perm = topk(score_t, 5, batch)
x_top = xx[perm] * score_t[perm].view(-1, 1)
bz = 3
x_top_out = x_top.view(bz, 5, -1)

# Simple loss: MSE to target
target = torch.randn(3, 5, 128, device=device)
loss2 = torch.nn.functional.mse_loss(x_top_out, target)
loss2.backward()

print(f"  pool.weight.grad sum: {pool.weight.grad.abs().sum().item():.8f}" +
      (" (FLOWS)" if pool.weight.grad is not None and pool.weight.grad.abs().sum() > 0 else " (ZERO!)"))
print(f"  x.grad sum:           {x.grad.abs().sum().item():.8f}" +
      (" (FLOWS)" if x.grad is not None and x.grad.abs().sum() > 0 else " (ZERO!)"))

# ===========================================================================
# 3. Overfit with TOPK ONLY at the right depth — isolate the bottleneck
# ===========================================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Multi-layer GCN + TopK weight grad check (depth=1)")
print("=" * 60)

# Use depth=1 so the only TopKPooling IS called
args1 = {'hidden_size': 128, 'depth': 1, 'k': 10, 'heads': 4}
model1 = DeepGCN(args1).to(device)

loader1 = DataLoader(Subset(full_ds, range(4)), batch_size=4, collate_fn=collate, shuffle=False)
_, _, _, labels1, pep_gs1, cdr3_gs1 = next(iter(loader1))
pep_gs1 = Batch.from_data_list(pep_gs1).to(device)
cdr3_gs1 = Batch.from_data_list(cdr3_gs1).to(device)
labels1 = labels1.to(device).float()

logits1 = model1(pep_gs1, cdr3_gs1)
loss1 = criterion(logits1, labels1)
loss1.backward()

for name, p in model1.named_parameters():
    if 'top_K_pooling' in name:
        has = "YES" if (p.grad is not None and p.grad.abs().sum() > 0) else "NO"
        gnorm = p.grad.abs().sum().item() if p.grad is not None else 0.0
        print(f"  {name:<52} {has:<8} {gnorm:<12.6f}")

# ===========================================================================
# 4. Overfit with depth=1 to see if convergence improves
# ===========================================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Overfit test with depth=1 (32 samples, 200 steps)")
print("=" * 60)

model2 = DeepGCN(args1).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

tiny_loader = DataLoader(Subset(full_ds, range(32)), batch_size=32, collate_fn=collate, shuffle=False)
_, _, _, labels2, pep_gs2, cdr3_gs2 = next(iter(tiny_loader))
pep_gs2 = Batch.from_data_list(pep_gs2).to(device)
cdr3_gs2 = Batch.from_data_list(cdr3_gs2).to(device)
labels2 = labels2.to(device).float()

initial = None
for step in range(1, 201):
    logits2 = model2(pep_gs2, cdr3_gs2)
    loss2 = criterion(logits2, labels2)
    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()
    if step == 1:
        initial = loss2.item()
    if step % 50 == 0 or step == 1:
        print(f"  Step {step:>3}: loss={loss2.item():.6f}")

final = loss2.item()
print(f"\n  Initial: {initial:.6f}  Final: {final:.6f}  Drop: {(initial-final)/initial*100:.1f}%")

del model, model1, model2
torch.cuda.empty_cache()
