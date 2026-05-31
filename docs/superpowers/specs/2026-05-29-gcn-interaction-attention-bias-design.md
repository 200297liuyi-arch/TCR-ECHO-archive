# GCN Interaction Matrix as Attention Bias

## Goal
Insert GCN atom-level interaction features as an additive bias into BidirectionalDualViewAttention's sequence logits (S_seq), so attention heads can learn from physical interaction patterns before softmax.

## Architecture

### Data Flow
```
GCN.forward()
  │
  ├── interaction_map  [B, k, k, 128]    atom-level cross-modal interaction
  ├── p_perm           [B, k]             peptide selected atom local indices
  ├── c_perm           [B, k]             TCR selected atom local indices
  ├── p_valid          [B, k]             1=real atom, 0=ghost pad
  ├── c_valid          [B, k]
  │
  ▼
GCNPlugin._build_gcn_attn_bias()
  │
  ├── 1. Project: Linear(128, 8)(interaction_map) → [B, k, k, 8]
  ├── 2. Scale:   / sqrt(128) × τ                    τ init=0.1, learnable
  ├── 3. Scatter: per-sample, a2r-guided, scatter_max → [B, 8, L_tcr, L_pep]
  │
  ▼
BidirectionalDualViewAttention.forward(gcn_bias=bias)
  │
  ├── S_seq = Q @ K^T / √d_k            [B, 8, L_tcr, L_pep]
  ├── S_seq = S_seq + gcn_bias           additive bias before softmax
  ├── attn_seq = softmax(S_seq)          softmax sees combined logits
  ├── S_bio = A_tcr @ U @ A_pep^T        biophysical view (unchanged)
  ├── attn_bio = softmax(S_bio)
  └── attn = (1-ρ)·attn_seq + ρ·attn_bio  interpolate views
```

## Key Design Decisions

### 1. Project First, Scatter Later
`Linear(128, 8)` on feature dim before A2R scatter. The two ops commute (Linear on feature axis, scatter on spatial axes) but scatter on 8 channels is 16× cheaper than 128.

### 2. Scatter Max (configurable)
`scatter_op='max'` default. When k=20, 3-5 atoms may map to the same residue pair. `max` preserves sharp contact peaks; `mean` dilutes them. Consistent with existing max+avg pooling in `F_spatial`.

### 3. Numerical Normalization: `/√d + Learnable Temperature`

```
S_gcn_bias = Linear(128, 8)(interaction_map) / sqrt(128) * τ
where τ = learnable scalar, initialized to 0.1
```

S_seq is scaled by `1/√d_k ≈ 1/12.65`. Dividing by sqrt(128) ≈ 11.3 puts GCN bias in the same initial range. `τ=0.1` starts the GCN bias conservatively at ~1% of S_seq magnitude, preventing it from dominating the pre-trained sequence attention. The model can increase τ as it learns to trust the physical signal.

**Why not LayerNorm:** scatter produces a sparse matrix — most residue pairs have no atom mapping and are zero. LayerNorm's `(x-μ)/σ` would inflate silent positions into non-zero noise, creating phantom attention paths.

**Why not tanh:** unnecessary nonlinearity. `/√d + τ` provides a fully linear gradient path; tanh would introduce saturation when pre-activation values grow large during training.

### 4. Ghost Atom Handling
Ghost (zero-padded) atoms have near-zero interaction values through the MHA + joint_mask chain. `p_valid`/`c_valid` masks guide scatter placement only — real atoms go to their residue bins, ghost positions are excluded from the scatter reduction. No gradient path is interrupted.

## Code Changes

| File | Change |
|------|--------|
| `gcn_plugin.py` | New `_build_gcn_attn_bias(...)` method with `self.gcn_bias_temp` (nn.Parameter, init=0.1). Called after GCN forward, bias passed to `self.cross_attn()`. |
| `attentions.py` | `forward()` accepts optional `gcn_bias: Tensor = None`. Added to `S_seq` before softmax when provided. |

## Edge Cases

- **ESM-only mode**: `gcn_bias=None`, attention behavior unchanged
- **No atoms for a residue**: bias=0 at that position, attention falls back to pure sequence
- **Ghost atoms**: excluded from scatter via valid masks, contribute nothing

## Non-Goals
- Does not modify the biophysical view (S_bio) or the mix parameter (ρ)
- Does not change GCN architecture or training config
- Does not add new loss terms
