# Paper-Aligned GCN Independent Encoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deepAntigen paper's independent encoder architecture in gcn_components.py and simplify gcn_only_train.py to use it.

**Architecture:** Two independent `PaperEncoder`s (peptide, CDR3) with TGCN+BN layers, TopK only at final layer, cross-attention only at final MHA with sum(dim=(1,2)) output → Projector(128→64)→Classifier(64→1). No per-layer cross-modal exchange.

**Tech Stack:** PyTorch, PyTorch Geometric, same environment as existing TCR-ECHO (RTX 5090, Python 3.x, CUDA)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `gcn_components.py` | Modify | Add PaperTopKPooling, PaperEncoder, PaperAlignedDeepGCN; modify MHA with output_mode |
| `gcn_only_train.py` | Modify | Simplify GCNOnlyModel, update training config |

---

### Task 1: Add `paper_topk()` function and `PaperTopKPooling` to `gcn_components.py`

**Files:**
- Modify: `gcn_components.py` — add after the existing `topk()` function

- [ ] **Step 1: Add `paper_topk()` function**

Insert after line 316 (after the existing `topk()` function, before the PositionalEncoding class at line 322):

```python
def paper_topk(x, ratio, batch):
    """Top-k atom selection per molecule — paper-aligned minimal version.

    Returns only perm (no on_index, no per-mol k values).
    Handles molecules with < ratio atoms via min clamp.
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = num_nodes.new_full((num_nodes.size(0),), ratio)
    k = torch.min(k, num_nodes)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    return perm
```

- [ ] **Step 2: Add `PaperTopKPooling` class**

Insert after `paper_topk()`, before `PositionalEncoding`:

```python
class PaperTopKPooling(torch.nn.Module):
    """Paper-aligned TopK pooling — single learnable weight, no positional encoding."""

    def __init__(self, in_channels: int, ratio: int = 1, nonlinearity: Callable = torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.weight = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, batch):
        xx = x.unsqueeze(-1) if x.dim() == 1 else x
        score = (xx * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        perm = paper_topk(score, self.ratio, batch)
        x_top = xx[perm] * score[perm].view(-1, 1)
        bz = batch.max().item() + 1
        # Handle variable k: pad to ratio if needed
        k_list = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
        k_per_mol = torch.min(k_list, torch.full_like(k_list, self.ratio))
        if (k_per_mol != self.ratio).any():
            # Some molecules have < ratio atoms — pad to ratio
            x_parts, idx = [], 0
            for i in range(bz):
                ki = k_per_mol[i].item()
                need = self.ratio - ki
                if need > 0:
                    x_parts.append(torch.cat([
                        x_top[idx:idx+ki],
                        x_top[idx:idx+ki].new_zeros(need, x_top.shape[-1]),
                    ]))
                else:
                    x_parts.append(x_top[idx:idx+ki])
                idx += ki
            x_top = torch.stack(x_parts)
        else:
            x_top = x_top.view(bz, self.ratio, -1)
        return x_top, perm
```

- [ ] **Step 3: Verify syntax** — Run Python import check

```bash
cd /home/lyf/projects/TCR-ECHO && python -c "from gcn_components import paper_topk, PaperTopKPooling; print('OK')"
```

---

### Task 2: Add `PaperEncoder` to `gcn_components.py`

**Files:**
- Modify: `gcn_components.py` — add after PaperTopKPooling

- [ ] **Step 1: Add `PaperEncoder` class**

Insert after PaperTopKPooling class:

```python
class PaperEncoder(nn.Module):
    """Paper-aligned independent per-molecule GCN encoder.

    depth × (TGCN → BatchNorm), TopK only at last layer.
    No cross-modal interaction — peptide and CDR3 encoded separately.
    """

    def __init__(self, in_channels: int, hidden_channels: int, depth: int, k: int):
        super().__init__()
        self.init_w = pyg_linear.Linear(
            in_channels, hidden_channels, weight_initializer='kaiming_uniform'
        )
        self.GCN_Depth = depth
        self.gcn = nn.ModuleList([TGCN(hidden_channels) for _ in range(depth)])
        self.top_K_pooling = nn.ModuleList([
            PaperTopKPooling(hidden_channels, ratio=k) for _ in range(depth)
        ])
        self.bn_x = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(depth)])

    def forward(self, graphs):
        x, edge_index, edge_attr, ibatch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        x_l = F.leaky_relu(self.init_w(x), 0.1)
        for i in range(self.GCN_Depth):
            x_l = self.gcn[i](x_l, edge_index, edge_attr, ibatch)
            x_l = self.bn_x[i](x_l)
            if i == self.GCN_Depth - 1:
                fs, perm = self.top_K_pooling[i](x_l, batch=ibatch)
        return fs
```

- [ ] **Step 2: Verify import**

```bash
cd /home/lyf/projects/TCR-ECHO && python -c "from gcn_components import PaperEncoder; print('OK')"
```

---

### Task 3: Modify `MultiHeadAttention` to support paper's `sum` output mode

**Files:**
- Modify: `gcn_components.py:452-496` (MultiHeadAttention class)

- [ ] **Step 1: Add `output_mode` parameter**

Edit the MultiHeadAttention class:

```python
class MultiHeadAttention(nn.Module):
    """Cross-molecule attention on top-k atom features.

    output_mode:
      'full' — returns intermap * att [B, k, k, H] (current default)
      'sum'  — returns sum(intermap * att, dim=(1,2)) [B, H] (paper-aligned)
    """

    def __init__(self, hidden_size: int, n_heads: int, output_mode: str = 'full'):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.output_mode = output_mode
        self.W_CDR3 = nn.Linear(hidden_size, hidden_size * n_heads)
        self.W_Peptide = nn.Linear(hidden_size, hidden_size * n_heads)
        self.reset_param()

    def reset_param(self):
        nn.init.xavier_uniform_(self.W_CDR3.weight)
        nn.init.xavier_uniform_(self.W_Peptide.weight)

    def forward(self, peptide, cdr3):
        batch_size = peptide.size(0)

        cdr3_s = (
            self.W_CDR3(cdr3)
            .view(batch_size, -1, self.n_heads, self.hidden_size)
            .transpose(1, 2)
        )
        peptide_s = (
            self.W_Peptide(peptide)
            .view(batch_size, -1, self.n_heads, self.hidden_size)
            .transpose(1, 2)
        )

        scores = (
            torch.matmul(peptide_s, cdr3_s.transpose(-1, -2))
            / self.hidden_size
        )
        scores = torch.mean(scores, dim=1)
        scores_reshape = scores.view(scores.shape[0], -1)
        att = torch.softmax(scores_reshape, dim=1)
        att = att.view(scores.shape[0], scores.shape[1], scores.shape[2])
        att = att.unsqueeze(-1)

        intermap = peptide.unsqueeze(-3) + cdr3.unsqueeze(-2)
        if self.output_mode == 'sum':
            return torch.sum(intermap * att, dim=(1, 2))
        return intermap * att
```

This changes are:
1. Add `output_mode` parameter (default `'full'` — backward compatible)
2. In forward: `if output_mode == 'sum': return torch.sum(...)` at the end

- [ ] **Step 2: Verify backward compatibility**

```bash
cd /home/lyf/projects/TCR-ECHO && python -c "
from gcn_components import MultiHeadAttention
import torch
mha_full = MultiHeadAttention(128, 4, output_mode='full')
mha_sum = MultiHeadAttention(128, 4, output_mode='sum')
x = torch.randn(2, 20, 128)
out_full = mha_full(x, x)
out_sum = mha_sum(x, x)
assert out_full.shape == (2, 20, 20, 128), f'full shape: {out_full.shape}'
assert out_sum.shape == (2, 128), f'sum shape: {out_sum.shape}'
print('OK')
"
```

---

### Task 4: Add `PaperAlignedDeepGCN` to `gcn_components.py`

**Files:**
- Modify: `gcn_components.py` — add at end of file

- [ ] **Step 1: Add `PaperAlignedDeepGCN` class**

Append to `gcn_components.py`:

```python
# ══════════════════════════════════════════════════════════════════════
#  PaperAlignedDeepGCN  — paper's independent encoder architecture
# ══════════════════════════════════════════════════════════════════════

class PaperAlignedDeepGCN(nn.Module):
    """Paper-aligned GCN: 2 independent encoders + final MHA only.

    Architecture (pTCR_seq.py, Nature Communications 2025):
      peptide → PaperEncoder(depth=5, k=20) → [B, 20, 128]
      cdr3    → PaperEncoder(depth=5, k=20) → [B, 20, 128]
      MHA(sum output) → [B, 128]
      Projector(128→64) + ReLU + Dropout(0.2)
      Classifier(64→1) → logits

    No per-layer cross-modal exchange. Cross-attention only at final MHA.
    """

    def __init__(self, args: dict):
        super().__init__()
        HS = args['hidden_size']
        depth = args['depth']
        k = args['k']
        heads = args.get('heads', 4)
        in_channels = args.get('in_channels', 25)

        self.peptide_encoder = PaperEncoder(in_channels, HS, depth, k)
        self.cdr3_encoder = PaperEncoder(in_channels, HS, depth, k)
        self.peptide_cdr3_att = MultiHeadAttention(HS, heads, output_mode='sum')
        self.dropout = nn.Dropout(p=0.2)
        self.projector = pyg_linear.Linear(
            HS, int(0.5 * HS), weight_initializer='kaiming_uniform'
        )
        self.classier = pyg_linear.Linear(
            int(0.5 * HS), 1, weight_initializer='kaiming_uniform'
        )

    def forward(self, peptide_graphs, cdr3_graphs):
        peptide_fs = self.peptide_encoder(peptide_graphs)
        cdr3_fs = self.cdr3_encoder(cdr3_graphs)
        peptide_cdr3_intermap = self.peptide_cdr3_att(peptide_fs, cdr3_fs)
        proj = F.relu(self.dropout(self.projector(peptide_cdr3_intermap)))
        logits = self.classier(proj)
        return logits.squeeze(-1)
```

- [ ] **Step 2: Verify forward pass with dummy data**

```bash
cd /home/lyf/projects/TCR-ECHO && python -c "
from gcn_components import PaperAlignedDeepGCN
import torch
from torch_geometric.data import Data, Batch

gcn_args = {'hidden_size': 128, 'depth': 5, 'k': 20, 'heads': 4, 'in_channels': 25}
model = PaperAlignedDeepGCN(gcn_args)
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params:,}')

# Dummy graphs (2 molecules, 25 features each, > 20 atoms each)
g1 = Data(x=torch.randn(30, 25), edge_index=torch.randint(0, 30, (2, 60)),
          edge_attr=torch.randn(60, 11), batch=torch.zeros(30, dtype=torch.long))
g2 = Data(x=torch.randn(25, 25), edge_index=torch.randint(0, 25, (2, 50)),
          edge_attr=torch.randn(50, 11), batch=torch.zeros(25, dtype=torch.long))
batch1 = Batch.from_data_list([g1, g2])
batch2 = Batch.from_data_list([g2, g1])
out = model(batch1, batch2)
print(f'Output shape: {out.shape}, values: {out}')
print('OK')
"
```

---

### Task 5: Simplify `GCNOnlyModel` in `gcn_only_train.py`

**Files:**
- Modify: `gcn_only_train.py:65-97` (GCNOnlyModel class)

- [ ] **Step 1: Replace GCNOnlyModel with paper-aligned version**

Replace the entire `GCNOnlyModel` class (lines 65-97) with:

```python
class GCNOnlyModel(nn.Module):
    """Paper-aligned GCN-only model — independent encoders + MHA sum output."""

    def __init__(self, gcn_args):
        super().__init__()
        from gcn_components import PaperAlignedDeepGCN
        self.gcn = PaperAlignedDeepGCN(gcn_args)

    def forward(self, pep_graphs, tcr_graphs, pep_mols, tcr_mols):
        return self.gcn(pep_graphs, tcr_graphs)
```

Remove unused imports at the top if needed (no more `nn.Sequential`, `nn.LayerNorm` usage in GCNOnlyModel — though they may still be used elsewhere, so be careful).

Actually, just remove the class and replace. The imports stay as they are (F, nn etc. used elsewhere in the file).

- [ ] **Step 2: Update model instantiation to remove removed params**

Find `GCNOnlyModel(gcn_args)` call around line 194 and verify no extra args are passed:

```python
model = GCNOnlyModel(gcn_args).to(DEVICE)
```

(Should already be this — verify.)

- [ ] **Step 3: Update DataLoader batch_size to 64**

Change line 190-191 from `batch_size=64` → confirm it's already 64. If not:

```python
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_gcn, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_gcn, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_gcn, num_workers=0)
```

- [ ] **Step 4: Update training config**

Change in `main()`:
- Line 199: `weight_decay=1e-4` → `weight_decay=0` (paper config)
- Line 205: `patience = 60` → `patience = 150` (paper training is slow but steady)
- Line 202: `MultiStepLR(opt, milestones=[200, 400], gamma=0.5)` — verify this matches. Update comment if needed.

- [ ] **Step 5: Update log message to reflect paper config**

Change line 208-209 to reference paper config:

```python
logger.info(f'Training: max_epochs={MAX_EPOCHS}, patience={patience}, '
            f'batches/epoch≈{len(train_loader)} (paper-aligned independent encoder)')
```

- [ ] **Step 6: Verify full file loads without error**

```bash
cd /home/lyf/projects/TCR-ECHO && python -c "
import sys; sys.argv = ['gcn_only_train.py', '--help']
# Just test imports
exec(open('gcn_only_train.py').read().split('def main')[0])
print('Imports OK')
"
```

---

### Task 6: Stop old training process

- [ ] **Step 1: Find and stop the currently running training**

```bash
ps aux | grep gcn_only_train | grep -v grep
```

If running, kill it:

```bash
kill <PID>
```

Or if needed:

```bash
pkill -f gcn_only_train
```

- [ ] **Step 2: Note current best checkpoint for comparison**

```bash
ls -la /home/lyf/projects/TCR-ECHO/runs/gcn_only/best_model.pth 2>/dev/null && echo "v2 checkpoint exists" || echo "no v2 checkpoint"
```

---

### Task 7: Launch paper-aligned training

- [ ] **Step 1: Launch training**

```bash
cd /home/lyf/projects/TCR-ECHO
nohup python -u gcn_only_train.py > runs/gcn_only/training_paper.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Verify training started (wait 2 min, check log)**

```bash
sleep 120 && tail -20 /home/lyf/projects/TCR-ECHO/runs/gcn_only/training_paper.log
```

Expected:
- `Paper-aligned independent encoder` in log header
- Params ~1.5M (vs 2.35M for old architecture)
- train_loss starts decreasing from ~0.17-0.18
- Each epoch ~1.5 min (same as before, similar compute)

---

### Verification Checklist

- [ ] `PaperAlignedDeepGCN` forward pass works with real data (test loader)
- [ ] No NaN in first 5 epochs
- [ ] train_loss decreases (unlike old run's flat line)
- [ ] val AUC starts climbing above 0.5 within ~10 epochs
