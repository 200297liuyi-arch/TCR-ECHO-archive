# GCN Interaction Matrix as Attention Bias — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert GCN atom-level interaction features as additive bias into BidirectionalDualViewAttention's sequence logits (S_seq) before softmax.

**Architecture:** GCNPlugin gains a `_build_gcn_attn_bias()` method that projects interaction_map [B,k,k,128] → [B,k,k,8] via Linear, normalizes with `/√128 × τ`, then per-sample A2R-guided scatter_max maps atom-level features to residue-level [B,8,L_tcr,L_pep]. This bias tensor is passed to BidirectionalDualViewAttention.forward() and added to S_seq before softmax.

**Tech Stack:** PyTorch, torch_geometric (already in project)

---

## Files

| File | Action | Responsibility |
|------|--------|----------------|
| `gcn_plugin.py:97-104` | Modify | Add `gcn_bias_temp` param + `gcn_bias_proj` Linear(128,8) |
| `gcn_plugin.py:161-175` | Modify | Pass `tcr_a2r`/`pep_a2r` through GCN branch |
| `gcn_plugin.py:200-240` | Modify | Build bias after GCN forward, pass to cross_attn |
| `attentions.py:58` | Modify | `_forward_one_direction` accepts `gcn_bias=None` |
| `attentions.py:95-96` | Modify | Add bias to S_seq before softmax |
| `attentions.py:108` | Modify | `forward` accepts and passes `gcn_bias` to both directions |

---

### Task 1: Add GCN bias projection and temperature to GCNPlugin.__init__

**Files:**
- Modify: `gcn_plugin.py:97-104`

- [ ] **Step 1: Add new parameters after gcn_spatial_dropout**

```python
# In __init__, after line 104 (self.gcn_spatial_dropout):

# ── GCN attention bias: project interaction_map → attention logit space ──
self.gcn_bias_proj = nn.Linear(self.gcn_hidden, num_heads)  # 128 → 8
self.gcn_bias_temp = nn.Parameter(torch.tensor(0.1))
```

Placement: after `self.gcn_spatial_dropout = nn.Dropout(0.2)` (line 104), before the aux classifier head (line 107).

- [ ] **Step 2: Verify init**

```bash
cd /home/lyf/projects/TCR-ECHO && TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -c "
import yaml, torch
with open('configs/config_gcn.yaml') as f: cfg = yaml.safe_load(f)
from gcn_plugin import GCNPlugin
lora_p = cfg['lora']['presets'][cfg['esm']['encoder1']]
m = GCNPlugin(
    esm1_name=cfg['esm']['encoder1'], esm2_name=cfg['esm']['encoder2'],
    lora_r=lora_p['r'], lora_alpha=lora_p['alpha'], lora_dropout=lora_p['dropout'],
    lora_target_modules=lora_p['layers_to_transform'],
    contrastive_temp=cfg['contrastive']['temperature'],
    lambda_enc=cfg['contrastive']['lambda_enc'], lambda_int=cfg['contrastive']['lambda_int'],
    classifier_hidden=cfg['classifier_hidden'], dropout=cfg['training']['dropout'],
    focal_gamma=cfg['training']['focal_gamma'], class_balance=0.5,
    use_lora=cfg.get('use_lora', True), num_heads=8,
    cross_attn_dropout=cfg['training']['cross_attn_dropout'],
    second_contrastive=cfg['training'].get('second_contrastive', True),
    gcn_args=cfg['gcn'],
)
print(f'gcn_bias_proj: {m.gcn_bias_proj}')
print(f'gcn_bias_temp: {m.gcn_bias_temp.item():.4f}')
print('OK')
"
```

Expected: `gcn_bias_temp: 0.1000`, no errors.

- [ ] **Step 3: Commit**

```bash
git add gcn_plugin.py
git commit -m "feat: add gcn_bias_proj Linear(128→8) and gcn_bias_temp τ=0.1 to GCNPlugin"
```

---

### Task 2: Add _build_gcn_attn_bias() method to GCNPlugin

**Files:**
- Modify: `gcn_plugin.py` (new method after __init__, before forward)

- [ ] **Step 1: Add the method**

Insert `_build_gcn_attn_bias` after `__init__` (before `forward` at line 161):

```python
def _build_gcn_attn_bias(self, interaction_map, p_perm, c_perm,
                          p_valid, c_valid, tcr_a2r, pep_a2r,
                          L_tcr, L_pep):
    """Build residue-level attention bias from GCN atom-level interaction.

    interaction_map: [B, k, k, H]   atom-level cross-modal features (ghost-masked)
    p_perm:         [B, k]         peptide selected atom local indices
    c_perm:         [B, k]         TCR selected atom local indices
    p_valid:        [B, k]         1=real peptide atom, 0=ghost
    c_valid:        [B, k]         1=real TCR atom, 0=ghost
    tcr_a2r:        list[Tensor]   per-sample atom→residue mapping
    pep_a2r:        list[Tensor]
    L_tcr:          int            max TCR residues in batch
    L_pep:          int            max peptide residues in batch

    Returns: [B, num_heads, L_tcr, L_pep]
    """
    B, K, _, H = interaction_map.shape
    num_heads = self.gcn_bias_proj.out_features  # 8

    # 1. Project: [B, k, k, H] → [B, k, k, 8]
    feat = self.gcn_bias_proj(interaction_map)  # [B, k, k, 8]

    # 2. Scale: / sqrt(H) × τ
    feat = feat / (H ** 0.5) * self.gcn_bias_temp

    # 3. Per-sample A2R scatter_max into residue-level
    bias_list = []
    for i in range(B):
        tcr_map = tcr_a2r[i]   # [n_tcr_atoms] → residue_idx per atom
        pep_map = pep_a2r[i]   # [n_pep_atoms]

        tcr_res = tcr_map[c_perm[i].long()]   # [k] residue idx for selected TCR atoms
        pep_res = pep_map[p_perm[i].long()]   # [k] residue idx for selected peptide atoms

        pv = p_valid[i].bool()   # [k]
        cv = c_valid[i].bool()   # [k]
        pair_valid = cv[:, None] & pv[None, :]  # [k, k] valid atom pairs

        feat_i = feat[i]  # [k, k, 8]
        bias_i = torch.zeros(L_tcr, L_pep, num_heads,
                             device=feat.device, dtype=feat.dtype)

        # Only process valid atom pairs
        valid_indices = pair_valid.nonzero(as_tuple=False)  # [N_valid, 2]
        for idx in range(valid_indices.size(0)):
            a_tcr, a_pep = valid_indices[idx]
            r_tcr = tcr_res[a_tcr].item()
            r_pep = pep_res[a_pep].item()
            if r_tcr < L_tcr and r_pep < L_pep:
                bias_i[r_tcr, r_pep] = torch.max(
                    bias_i[r_tcr, r_pep], feat_i[a_tcr, a_pep]
                )

        bias_list.append(bias_i.permute(2, 0, 1))  # [8, L_tcr, L_pep]

    return torch.stack(bias_list)  # [B, 8, L_tcr, L_pep]
```

- [ ] **Step 2: Smoketest with synthetic data**

```bash
cd /home/lyf/projects/TCR-ECHO && TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -c "
import torch, yaml
with open('configs/config_gcn.yaml') as f: cfg = yaml.safe_load(f)
from gcn_plugin import GCNPlugin
lora_p = cfg['lora']['presets'][cfg['esm']['encoder1']]
m = GCNPlugin(
    esm1_name=cfg['esm']['encoder1'], esm2_name=cfg['esm']['encoder2'],
    lora_r=lora_p['r'], lora_alpha=lora_p['alpha'], lora_dropout=lora_p['dropout'],
    lora_target_modules=lora_p['layers_to_transform'],
    contrastive_temp=cfg['contrastive']['temperature'],
    lambda_enc=cfg['contrastive']['lambda_enc'], lambda_int=cfg['contrastive']['lambda_int'],
    classifier_hidden=cfg['classifier_hidden'], dropout=cfg['training']['dropout'],
    focal_gamma=cfg['training']['focal_gamma'], class_balance=0.5,
    use_lora=cfg.get('use_lora', True), num_heads=8,
    cross_attn_dropout=cfg['training']['cross_attn_dropout'],
    gcn_args=cfg['gcn'],
)
B, K, H = 2, cfg['gcn']['k'], cfg['gcn']['hidden_size']
imap = torch.randn(B, K, K, H)
p_perm = torch.randint(0, 50, (B, K))
c_perm = torch.randint(0, 100, (B, K))
p_valid = torch.ones(B, K)
c_valid = torch.ones(B, K)
c_valid[0, -2:] = 0  # ghost TCR atoms in sample 0
p_valid[1, -3:] = 0  # ghost pep atoms in sample 1
tcr_a2r = [torch.randint(0, 30, (200,)), torch.randint(0, 25, (180,))]
pep_a2r = [torch.randint(0, 8, (60,)), torch.randint(0, 6, (55,))]
bias = m._build_gcn_attn_bias(imap, p_perm, c_perm, p_valid, c_valid,
                               tcr_a2r, pep_a2r, L_tcr=30, L_pep=8)
print(f'Bias shape: {bias.shape}')  # expected: [2, 8, 30, 8]
print(f'Ghost TCR atoms excluded: {bias[0, :, c_perm[0,-2:], :].abs().sum():.4f}')
print('OK')
"
```

Expected: `Bias shape: torch.Size([2, 8, 30, 8])`, ghost atoms contribute zero.

- [ ] **Step 3: Commit**

```bash
git add gcn_plugin.py
git commit -m "feat: add _build_gcn_attn_bias() — project + scale + A2R scatter_max"
```

---

### Task 3: Wire gcn_bias through GCNPlugin.forward() → cross_attn()

**Files:**
- Modify: `gcn_plugin.py:200-240`

- [ ] **Step 1: Build bias after GCN forward, pass to cross_attn**

In `forward()`, after the GCN output extraction (around line 209), build the bias tensor. Then modify the `self.cross_attn()` call.

Replace the block starting at `gcn_out = self.gcn(pep_batch, tcr_batch, pep_mols, tcr_mols)` through the cross_attn call:

```python
            gcn_out = self.gcn(pep_batch, tcr_batch, pep_mols, tcr_mols)

            interaction_map = gcn_out["interaction_map"]  # [B, k, k, H_gcn]
            joint_mask = gcn_out["joint_mask"]            # [B, k, k, 1]

            # ── Build GCN attention bias ──────────────────────────────
            gcn_bias = self._build_gcn_attn_bias(
                interaction_map,
                gcn_out["p_perm"], gcn_out["c_perm"],
                gcn_out["p_valid"], gcn_out["c_valid"],
                tcr_a2r, pep_a2r,
                L_tcr=tcr_enc.size(1), L_pep=pep_enc.size(1),
            )

            B_gcn, K, _, H_gcn_local = interaction_map.shape
```

And replace the `self.cross_attn(...)` call:

```python
        tcr_att, pep_att = self.cross_attn(
            tcr_enc, pep_enc, atchley1, atchley2,
            gcn_bias=gcn_bias if F_spatial is not None else None,
        )
```

- [ ] **Step 2: Run existing training for 2 epochs to verify no crash**

```bash
cd /home/lyf/projects/TCR-ECHO && timeout 600 env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py --config configs/config_gcn.yaml 2>&1 | head -80
```

Expected: Epoch 1 and 2 complete without errors. Training loss decreases.

- [ ] **Step 3: Commit**

```bash
git add gcn_plugin.py
git commit -m "feat: wire gcn_bias through forward() → cross_attn()"
```

---

### Task 4: Modify BidirectionalDualViewAttention to accept gcn_bias

**Files:**
- Modify: `attentions.py:58,95-96,108-141`

- [ ] **Step 1: Add gcn_bias to _forward_one_direction signature and S_seq**

In `_forward_one_direction`, add `gcn_bias=None` parameter and apply it to S_seq before softmax:

```python
    def _forward_one_direction(self, q_seq, kv_seq, atc_q, atc_kv,
                                q_proj, k_proj, v_proj, out_proj,
                                transpose_U=False,
                                gcn_bias=None):
```

After `S_seq = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)` (line 83), add:

```python
        # View 1: sequence attention
        S_seq = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,Lq,Lkv]

        # Add GCN physical interaction bias (if provided)
        if gcn_bias is not None:
            S_seq = S_seq + gcn_bias
```

The rest of the method after S_seq stays unchanged.

- [ ] **Step 2: Modify forward() to accept and pass gcn_bias**

Change `forward` signature:

```python
    def forward(self, tcr_enc, pep_enc, atchley1, atchley2, *, gcn_bias=None):
```

Update both `_forward_one_direction` calls to pass `gcn_bias`:

```python
        # Direction 1: TCR → Peptide
        tcr_att = self._forward_one_direction(
            q_seq=tcr_enc, kv_seq=pep_enc,
            atc_q=atchley1, atc_kv=atchley2,
            q_proj=self.q_proj_t, k_proj=self.k_proj_p, v_proj=self.v_proj_p,
            out_proj=self.out_proj_t,
            transpose_U=False,
            gcn_bias=gcn_bias,
        )

        # Direction 2: Peptide → TCR  (U^T for symmetric scoring)
        pep_att = self._forward_one_direction(
            q_seq=pep_enc, kv_seq=tcr_enc,
            atc_q=atchley2, atc_kv=atchley1,
            q_proj=self.q_proj_p, k_proj=self.k_proj_t, v_proj=self.v_proj_t,
            out_proj=self.out_proj_p,
            transpose_U=True,
            gcn_bias=gcn_bias,
        )
```

GCN interaction_map: pep atoms (queries) × tcr atoms (keys) → after scatter, bias is [L_tcr, L_pep]. Direction 1 (TCR→Pep) matches directly. Direction 2 (Pep→TCR) needs transpose because S_seq is [L_pep, L_tcr]:

```python
        # Direction 2: Peptide → TCR  (transpose bias to match [L_pep, L_tcr])
        pep_att = self._forward_one_direction(
            ...,
            gcn_bias=gcn_bias.transpose(-1, -2) if gcn_bias is not None else None,
        )
```

- [ ] **Step 3: Run shape sanity test**

```bash
cd /home/lyf/projects/TCR-ECHO && TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -c "
import torch, yaml
with open('configs/config_gcn.yaml') as f: cfg = yaml.safe_load(f)
from gcn_plugin import GCNPlugin
from dataset import TCRPeptideDataset, collate_graph_batch
from utils import load_atchley
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

DEVICE = 'cuda:1'
lora_p = cfg['lora']['presets'][cfg['esm']['encoder1']]
m = GCNPlugin(
    esm1_name=cfg['esm']['encoder1'], esm2_name=cfg['esm']['encoder2'],
    lora_r=lora_p['r'], lora_alpha=lora_p['alpha'], lora_dropout=lora_p['dropout'],
    lora_target_modules=lora_p['layers_to_transform'],
    contrastive_temp=cfg['contrastive']['temperature'],
    lambda_enc=cfg['contrastive']['lambda_enc'], lambda_int=cfg['contrastive']['lambda_int'],
    classifier_hidden=cfg['classifier_hidden'], dropout=cfg['training']['dropout'],
    focal_gamma=cfg['training']['focal_gamma'], class_balance=0.5,
    use_lora=cfg.get('use_lora', True), num_heads=8,
    cross_attn_dropout=cfg['training']['cross_attn_dropout'],
    gcn_args=cfg['gcn'],
).to(DEVICE)

df = pd.read_csv('datasets/echo/panpep/zero_test_paper.csv').head(4)
tokenizer = AutoTokenizer.from_pretrained(f\"facebook/{cfg['esm']['encoder1']}\")
atchley_map = load_atchley(cfg.get('atchley_path'))
ds = TCRPeptideDataset(df, tokenizer, atchley_map, cfg['dataset']['columns'], use_graph=True)
ldr = DataLoader(ds, batch_size=4, collate_fn=collate_graph_batch)

batch = next(iter(ldr))
batch = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch]
(inp1,msk1,inp2,msk2,at1,at2,lbl,g1,g2,m1,m2,a2r1,a2r2) = batch

with torch.no_grad():
    logits, loss = m(inp1,msk1,inp2,msk2,at1,at2,lbl,
                     tcr_graphs=g1, pep_graphs=g2, tcr_mols=m1, pep_mols=m2,
                     tcr_a2r=a2r1, pep_a2r=a2r2)
print(f'Forward OK: logits={logits.shape}, loss={loss.item():.4f}')
print(f'gcn_bias_temp: {m.gcn_bias_temp.item():.4f}')
print(f'Gradient check: {m.gcn_bias_temp.grad is None}')  # True: still 0.1
"
```

Expected: `Forward OK: logits=torch.Size([4, 1]), loss=...`

- [ ] **Step 4: Commit**

```bash
git add attentions.py
git commit -m "feat: add gcn_bias to BidirectionalDualViewAttention — additive bias on S_seq before softmax"
```

---

### Task 5: Verify gradient flow and backward compatibility

**Files:**
- Test: manual verification (no new test file)

- [ ] **Step 1: Verify gradient flows through gcn_bias_temp**

```bash
cd /home/lyf/projects/TCR-ECHO && TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -c "
import torch, yaml, pandas as pd
with open('configs/config_gcn.yaml') as f: cfg = yaml.safe_load(f)
from gcn_plugin import GCNPlugin
from dataset import TCRPeptideDataset, collate_graph_batch
from utils import load_atchley
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

DEVICE = 'cuda:1'
lora_p = cfg['lora']['presets'][cfg['esm']['encoder1']]
m = GCNPlugin(
    esm1_name=cfg['esm']['encoder1'], esm2_name=cfg['esm']['encoder2'],
    lora_r=lora_p['r'], lora_alpha=lora_p['alpha'], lora_dropout=lora_p['dropout'],
    lora_target_modules=lora_p['layers_to_transform'],
    contrastive_temp=cfg['contrastive']['temperature'],
    lambda_enc=cfg['contrastive']['lambda_enc'], lambda_int=cfg['contrastive']['lambda_int'],
    classifier_hidden=cfg['classifier_hidden'], dropout=cfg['training']['dropout'],
    focal_gamma=cfg['training']['focal_gamma'], class_balance=0.5,
    use_lora=cfg.get('use_lora', True), num_heads=8,
    cross_attn_dropout=cfg['training']['cross_attn_dropout'],
    gcn_args=cfg['gcn'],
).to(DEVICE)

df = pd.read_csv('datasets/echo/panpep/zero_test_paper.csv').head(4)
tokenizer = AutoTokenizer.from_pretrained(f\"facebook/{cfg['esm']['encoder1']}\")
atchley_map = load_atchley(cfg.get('atchley_path'))
ds = TCRPeptideDataset(df, tokenizer, atchley_map, cfg['dataset']['columns'], use_graph=True)
ldr = DataLoader(ds, batch_size=4, collate_fn=collate_graph_batch)
batch = next(iter(ldr))
batch = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch]
(inp1,msk1,inp2,msk2,at1,at2,lbl,g1,g2,m1,m2,a2r1,a2r2) = batch

# Track temp parameter
temp_before = m.gcn_bias_temp.item()
logits, loss = m(inp1,msk1,inp2,msk2,at1,at2,lbl,
                 tcr_graphs=g1, pep_graphs=g2, tcr_mols=m1, pep_mols=m2,
                 tcr_a2r=a2r1, pep_a2r=a2r2)
loss.backward()
print(f'gcn_bias_temp grad: {m.gcn_bias_temp.grad.item():.6f}')
print(f'gcn_bias_proj.weight grad norm: {m.gcn_bias_proj.weight.grad.norm().item():.6f}')
print(f'Gradient flows through gcn_bias: OK')
"
```

Expected: Non-zero gradients for both `gcn_bias_temp` and `gcn_bias_proj.weight`.

- [ ] **Step 2: Verify ESM-only mode is unaffected**

```bash
cd /home/lyf/projects/TCR-ECHO && TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -c "
import torch, yaml, pandas as pd
with open('configs/config_esm_only.yaml') as f: cfg = yaml.safe_load(f)
from model import Model
from dataset import TCRPeptideDataset
from utils import load_atchley
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

DEVICE = 'cuda:1'
lora_p = cfg['lora']['presets'][cfg['esm']['encoder1']]
m = Model(
    esm1_name=cfg['esm']['encoder1'], esm2_name=cfg['esm']['encoder2'],
    lora_r=lora_p['r'], lora_alpha=lora_p['alpha'], lora_dropout=lora_p['dropout'],
    lora_target_modules=lora_p['layers_to_transform'],
    contrastive_temp=cfg['contrastive']['temperature'],
    lambda_enc=cfg['contrastive']['lambda_enc'], lambda_int=cfg['contrastive']['lambda_int'],
    classifier_hidden=cfg['classifier_hidden'], dropout=cfg['training']['dropout'],
    focal_gamma=cfg['training']['focal_gamma'], class_balance=0.5,
    use_lora=cfg.get('use_lora', True), num_heads=8,
    cross_attn_dropout=cfg['training']['cross_attn_dropout'],
).to(DEVICE)

df = pd.read_csv('datasets/echo/panpep/zero_test_paper.csv').head(4)
tokenizer = AutoTokenizer.from_pretrained(f\"facebook/{cfg['esm']['encoder1']}\")
atchley_map = load_atchley(cfg.get('atchley_path'))
ds = TCRPeptideDataset(df, tokenizer, atchley_map, cfg['dataset']['columns'], use_graph=False)
ldr = DataLoader(ds, batch_size=4)
batch = next(iter(ldr))
batch = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch]
inp1,msk1,inp2,msk2,at1,at2,lbl = batch
logits, loss = m(inp1,msk1,inp2,msk2,at1,at2,lbl)
print(f'ESM-only forward OK: logits={logits.shape}, loss={loss.item():.4f}')
"
```

Expected: `ESM-only forward OK`, no errors.

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "verify: gradient flow through gcn_bias confirmed, ESM-only unaffected"
```
