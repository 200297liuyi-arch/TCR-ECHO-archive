"""Diagnostic 2: isolate why overfit is weak. Test: no-TopK global pooling, higher LR, more steps."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn.dense.linear as pyg_linear

from deepAntigen_Seq.load_dataset.load_seq import pTCR_DataSet, collate
from deepAntigen_Seq.networks.top_k_pooling_seq import TopKPooling

device = torch.device('cuda')
csv_path = 'datasets/panpep/train_joint.csv'
full_ds = pTCR_DataSet(csv_path, aug=False, test=True)

# Make a minimal 32-sample batch
loader = DataLoader(Subset(full_ds, range(32)), batch_size=32, collate_fn=collate, shuffle=False)
_, _, _, labels, pep_gs, cdr3_gs = next(iter(loader))
pep_batch = Batch.from_data_list(pep_gs).to(device)
cdr3_batch = Batch.from_data_list(cdr3_gs).to(device)
labels32 = labels.to(device).float()
print(f"32 samples: {labels32.sum().item():.0f} pos / {(1-labels32).sum().item():.0f} neg")
print(f"Pep nodes: {pep_batch.x.shape[0]}, CDR3 nodes: {cdr3_batch.x.shape[0]}")

# ===========================================================================
# Test A: Keep TopK but crank LR way up
# ===========================================================================
class QuickTGCN(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr='add')
        self.message_w = pyg_linear.Linear(hidden+11, hidden)
        self.update_w = pyg_linear.Linear(2*hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
    def forward(self, x, ei, ea, ib):
        xu = self.propagate(ei, x=x, edge_attr=ea)
        return self.gru(x, xu)
    def message(self, x_j, edge_attr):
        return F.leaky_relu(self.message_w(torch.cat((x_j, edge_attr), dim=1)), 0.1)
    def update(self, inputs, x):
        return F.leaky_relu(self.update_w(torch.cat((inputs, x), dim=1)), 0.1)

class QuickMHA(nn.Module):
    def __init__(self, h, nh):
        super().__init__()
        self.h = h; self.nh = nh
        self.wc = nn.Linear(h, h*nh); self.wp = nn.Linear(h, h*nh)
    def forward(self, pep, cdr):
        B = pep.size(0)
        cs = self.wc(cdr).view(B,-1,self.nh,self.h).transpose(1,2)
        ps = self.wp(pep).view(B,-1,self.nh,self.h).transpose(1,2)
        scr = torch.matmul(ps, cs.transpose(-1,-2))/self.h
        scr = torch.mean(scr, dim=1)
        att = torch.softmax(scr.view(B,-1), dim=1).view(B, scr.shape[1], scr.shape[2]).unsqueeze(-1)
        return torch.sum((pep.unsqueeze(-3) + cdr.unsqueeze(-2)) * att, dim=(1,2))

class QuickModel(nn.Module):
    def __init__(self, h=128, d=2, k=10, nh=4, use_topk=True):
        super().__init__()
        self.use_topk = use_topk
        self.pep_init = pyg_linear.Linear(25, h)
        self.cdr_init = pyg_linear.Linear(25, h)
        self.pep_gcn = nn.ModuleList([QuickTGCN(h) for _ in range(d)])
        self.cdr_gcn = nn.ModuleList([QuickTGCN(h) for _ in range(d)])
        self.pep_bn = nn.ModuleList([BatchNorm(h) for _ in range(d)])
        self.cdr_bn = nn.ModuleList([BatchNorm(h) for _ in range(d)])
        if use_topk:
            self.pep_topk = TopKPooling(h, k)
            self.cdr_topk = TopKPooling(h, k)
        self.mha = QuickMHA(h, nh)
        self.out = nn.Linear(h, 1)

    def forward(self, pep_g, cdr_g):
        px = F.leaky_relu(self.pep_init(pep_g.x), 0.1)
        cx = F.leaky_relu(self.cdr_init(cdr_g.x), 0.1)
        for i in range(len(self.pep_gcn)):
            px = self.pep_bn[i](self.pep_gcn[i](px, pep_g.edge_index, pep_g.edge_attr, pep_g.batch))
            cx = self.cdr_bn[i](self.cdr_gcn[i](cx, cdr_g.edge_index, cdr_g.edge_attr, cdr_g.batch))
        if self.use_topk:
            px, _ = self.pep_topk(px, pep_g.batch)
            cx, _ = self.cdr_topk(cx, cdr_g.batch)
        else:
            # Global mean pooling per graph
            from torch_scatter import scatter_mean
            px = scatter_mean(px, pep_g.batch, dim=0)
            cx = scatter_mean(cx, cdr_g.batch, dim=0)
            B = int(pep_g.batch.max().item() + 1)
            px = px.view(B, 1, -1); cx = cx.view(B, 1, -1)
        x = self.mha(px, cx)
        return self.out(x).squeeze(-1)

# Test A: TopK, higher LR
print("\n=== Test A: TopK, lr=1e-2, 300 steps ===")
m = QuickModel(use_topk=True).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
losses = []
for s in range(1, 301):
    l = F.binary_cross_entropy_with_logits(m(pep_batch, cdr3_batch), labels32)
    opt.zero_grad(); l.backward(); opt.step()
    losses.append(l.item())
    if s == 1 or s % 100 == 0: print(f"  {s:>4}: {l.item():.6f}")
print(f"  Final: {losses[-1]:.6f} (from {losses[0]:.6f}, drop {(losses[0]-losses[-1])/losses[0]*100:.1f}%)")
del m; torch.cuda.empty_cache()

# Test B: No TopK (global pooling), same LR
print("\n=== Test B: Global mean pooling, lr=1e-2, 300 steps ===")
m = QuickModel(use_topk=False).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
losses2 = []
for s in range(1, 301):
    l = F.binary_cross_entropy_with_logits(m(pep_batch, cdr3_batch), labels32)
    opt.zero_grad(); l.backward(); opt.step()
    losses2.append(l.item())
    if s == 1 or s % 100 == 0: print(f"  {s:>4}: {l.item():.6f}")
print(f"  Final: {losses2[-1]:.6f} (from {losses2[0]:.6f}, drop {(losses2[0]-losses2[-1])/losses2[0]*100:.1f}%)")
del m; torch.cuda.empty_cache()

# Test C: No GCN at all, just MHA on raw atom features
print("\n=== Test C: Raw features + MHA, lr=1e-2, 300 steps ===")
class RawModel(nn.Module):
    def __init__(self, h=128, nh=4):
        super().__init__()
        self.pep_proj = nn.Linear(25, h)
        self.cdr_proj = nn.Linear(25, h)
        self.mha = QuickMHA(h, nh)
        self.out = nn.Linear(h, 1)
    def forward(self, pep_g, cdr_g):
        from torch_scatter import scatter_mean
        px = scatter_mean(self.pep_proj(pep_g.x), pep_g.batch, dim=0)
        cx = scatter_mean(self.cdr_proj(cdr_g.x), cdr_g.batch, dim=0)
        B = int(pep_g.batch.max().item()+1)
        x = self.mha(px.view(B,1,-1), cx.view(B,1,-1))
        return self.out(x).squeeze(-1)

m = RawModel().to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
losses3 = []
for s in range(1, 301):
    l = F.binary_cross_entropy_with_logits(m(pep_batch, cdr3_batch), labels32)
    opt.zero_grad(); l.backward(); opt.step()
    losses3.append(l.item())
    if s == 1 or s % 100 == 0: print(f"  {s:>4}: {l.item():.6f}")
print(f"  Final: {losses3[-1]:.6f} (from {losses3[0]:.6f}, drop {(losses3[0]-losses3[-1])/losses3[0]*100:.1f}%)")

# ===========================================================================
# Test D: Pure MLP on flattened atom features (bypass GCN+MHA)
# ===========================================================================
print("\n=== Test D: Pure MLP on global-mean-pooled atoms, lr=1e-2 ===")
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(50, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, pep_g, cdr_g):
        from torch_scatter import scatter_mean
        pp = scatter_mean(pep_g.x, pep_g.batch, dim=0)
        cp = scatter_mean(cdr_g.x, cdr_g.batch, dim=0)
        return self.net(torch.cat([pp, cp], dim=-1)).squeeze(-1)

m = MLPModel().to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
losses4 = []
for s in range(1, 301):
    l = F.binary_cross_entropy_with_logits(m(pep_batch, cdr3_batch), labels32)
    opt.zero_grad(); l.backward(); opt.step()
    losses4.append(l.item())
    if s == 1 or s % 100 == 0: print(f"  {s:>4}: {l.item():.6f}")
print(f"  Final: {losses4[-1]:.6f} (from {losses4[0]:.6f}, drop {(losses4[0]-losses4[-1])/losses4[0]*100:.1f}%)")

print("\n=== SUMMARY ===")
print(f"Test A (TopK+GCN+MHA):         {losses[0]:.4f} -> {losses[-1]:.4f}  ({(losses[0]-losses[-1])/losses[0]*100:.0f}%)")
print(f"Test B (NoTopK+GCN+MHA):       {losses2[0]:.4f} -> {losses2[-1]:.4f}  ({(losses2[0]-losses2[-1])/losses2[0]*100:.0f}%)")
print(f"Test C (Raw+MHA, no GCN):      {losses3[0]:.4f} -> {losses3[-1]:.4f}  ({(losses3[0]-losses3[-1])/losses3[0]*100:.0f}%)")
print(f"Test D (Pure MLP):             {losses4[0]:.4f} -> {losses4[-1]:.4f}  ({(losses4[0]-losses4[-1])/losses4[0]*100:.0f}%)")
