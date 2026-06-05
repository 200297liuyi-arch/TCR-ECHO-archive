# Findings & Decisions

## Phase 13: SeqAlignedGCN — Architecture Alignment (2026-06-04)

### GCN Bias Dead Path Confirmed (2026-06-03)
- τ=0.227 vs τ=0: predictions 100% identical (corr=1.0000, max|Δ|=3.5e-4)
- 95.97% of predictions change, but max change = 0.035 percentage points
- GCN attention bias contributes absolutely nothing to the model
- Root cause: 1-bit label supervises 128-dim × 100 atom-pair interaction — signal diluted 12,800×

### deepAntigen_Seq vs TCR-ECHO DeepGCN: Three Architecture Gaps (2026-06-04)
| Component | deepAntigen_Seq (pTCR_seq.py) | TCR-ECHO DeepGCN (old) |
|-----------|------------------------------|-------------------------|
| Cross-modal interaction | Final MHA only (independent encoders) | Every layer via SuperNodeExchange |
| GRU placement | Inside TGCN (message→update→GRUCell) | End of layer (LocalMP→SuperNode→GRU) |
| Max depth | 5 (stable, no cross-talk during encoding) | 2 (≥3 causes gradient collapse) |
| TopK scoring | 1 weight (128 params) | 3-layer MLP (74K params) |
| MHA output | `sum(dim=(1,2))` → [B,H] | Full [B,k,k,H] interaction_map |

### SeqAlignedGCN Design (2026-06-04)
- 2 independent PaperEncoders (TGCN×5, BN, PaperTopKPooling@last)
- MHA(full output) only at the end
- 5% node dropout during training (paper's augmentation)
- Returns same dict interface as DeepGCN → drop-in replacement
- Params: 1,641,472 (d=5) vs old DeepGCN 1,053,184 (d=2)
  - Same depth: SeqAlignedGCN d=2 = 739,840 — 30% smaller

### Design Decisions (new)
| Decision | Rationale |
|----------|-----------|
| Independent encoders (no per-layer cross-modal) | Aligns with deepAntigen_Seq; enables depth=5 without gradient collapse |
| TGCN (GRU inside) instead of LocalMP+external GRU | deepAntigen_Seq places GRU immediately after message+update — matches paper |
| PaperTopKPooling (1 weight) instead of 3-layer MLP for TopK | Paper uses simple scoring; 3-layer MLP was 74K params of overkill |
| Keep output_mode='full' (not paper's sum) | TCR-ECHO fusion needs full interaction_map for GCN bias + F_spatial paths |
| Return dict (matching DeepGCN interface) | Drop-in replacement — gcn_plugin.py changed 1 line |

## Phase 12: GCN Bias Training — Reproducible (seed=42) (2026-06-02~03)

### Training Results
- 195/200 epochs (early stop) — best val AUC **0.7646** (ep175)
- **Majority Test**: AUC **0.8390**, Acc **0.7742**, F1 **0.7562** (+2.95pp vs prev run)
- **Zero-Shot**: AUC 0.8050, F1 0.7075
- Three LR reductions: 5e-5 → 2.5e-5 (ep113) → 1.25e-5 → 6.25e-6 (ep188)
- Higher test AUC but lower val AUC than previous no-seed run

### Reproducibility Fixes
- `set_seed(seed)` in utils.py: random, numpy, torch.manual_seed, cuda.manual_seed_all, cudnn.deterministic
- Validation always from training data (15% stratified by label)
- `random_seed: 42` in all config YAMLs
- Prior runs had NO seed — results were not reproducible

### Design Decisions (new)
| Decision | Rationale |
|----------|-----------|
| set_seed before data split AND model init | Both split randomness and weight init must be controlled |
| Val always from train CSV (stratified) | Pre-split val_csv caused inconsistency; 15% stratified guarantees label balance |
| cudnn.deterministic=True | May slow training slightly but required for exact reproducibility |

## Phase 11: Residual Gating + Scheduler — Gate Collapse Root Cause & Fix (2026-05-28~29)

### Gate Collapse Discovery (2026-05-28)
- **Root cause**: Language gate learned W_tcr≈0.14, W_pep≈0.22 — aggressively suppressing ESM features
- Physics gate W_phys≈0.61 stayed open — model was replacing language with physics, not fusing
- This explains why fusion AUC (0.765) was far below ESM-only (0.837)

### Residual Gating Implementation
- Language gate: `feat * W` → `feat * (1 + W)` — guarantees 100% signal floor regardless of gate value
- Physics gate: kept multiplicative (W_phys=0.61 was healthy, didn't need protection)
- Language gate final Linear bias initialized to 1.5 → initial gate ~0.82, effective multiplier ~1.82×

### Overfitting Prevention (2026-05-28)
- **weight_decay 1e-4 → 5e-4**: stronger baseline regularization
- **ReduceLROnPlateau**: factor=0.5, patience=10, min_lr=1e-6
  - LR 5e-5 → 2.5e-5 (ep94) → 1.25e-5 (ep124)
  - Each reduction brought a fresh burst of AUC improvement
  - Final best at epoch 153 with LR=1.25e-5
- **early_stopping patience 10→20**: gives scheduler more room to work

### Impact Analysis (2026-05-29)
- **Broke through previous ceiling**: 0.7664 (ep153) vs 0.7647 (prev run ep148)
- The breakthrough came from fine-tuning at extremely low LR (1.25e-5) over 39 epochs
- Without scheduler, the previous run kept using 5e-5 LR and overfit after its peak
- The combination of measures (residual gating + scheduler + higher wd) gave ~0.002 AUC improvement

### Epoch-by-Epoch Comparison (Current vs Previous)
| Stage | Current Run | Previous Run | Advantage |
|-------|:--:|:--:|:--:|
| Epoch 25 | 0.7216 | 0.7150 | +0.007 |
| Epoch 50 | 0.7409 | 0.7393 | +0.002 |
| Epoch 75 | 0.7546 | 0.7495 | +0.005 |
| Epoch 100 | 0.7591 | 0.7519 | +0.007 |
| Epoch 125 | 0.7618 | 0.7612 | +0.001 |
| Epoch 153 | **0.7664** | 0.7636 | **+0.003** |

### Design Decisions (new)
| Decision | Rationale |
|----------|-----------|
| Residual language gate | Guarantees fusion model floor ≥ ESM-only by preserving language signal |
| Bias-init 1.5 on gate output | Gives language branch a training "head start" against GCN gradient competition |
| Physics gate unchanged (multiplicative) | W_phys=0.61 was healthy; residual would unnecessarily weaken its modulation |
| ReduceLROnPlateau > StepLR | Epoch of convergence unknown a priori; adaptive decay matches real signal |
| wd 5e-4 | FocalLoss has implicit regularization; 1e-4 was too conservative, 5e-4 provides real L2 |

## Paper-Aligned Independent Encoder — 10-Fold CV Results (2026-05-27)

### Architecture Decision: Independent Encoders > Per-Layer SuperNodeExchange
- Per-layer cross-modal exchange (SuperNodeExchange) at depth=5 → gradient stagnation (train_loss barely moves)
- Paper's 2 independent TGCN Encoders + final MHA only → stable convergence at depth=5
- Epoch 1 val_auc jumped from 0.50 (old) to 0.66 (paper-aligned) — architecture was the bottleneck
- Implemented in `gcn_components.py`: PaperEncoder (TGCN×5→BN→TopK@last), PaperAlignedDeepGCN

### GCN-Only 10-Fold CV Performance (zero_test_paper.csv, 1714 samples)
- Mean test AUC: 0.7499 ± 0.024
- Best: 0.7824, Worst: 0.7095
- Between paper's COVID-19 (0.71) and Gao (0.84) zero-shot results
- ESM-only baseline (0.8148) still superior — GCN provides complementary signal, not standalone

### Training Dynamics
- FocalLoss(γ=2) inherently slows convergence (down-weights easy samples)
- SGD+momentum converges but slowly — paper compensates with 700 epochs
- Step LR@200,400 helps break plateaus
- ~80 sec/epoch on RTX 5090, early stopping at patience=60 typically around epoch 200-500
- Mean epochs to convergence: ~350

## deepAntigen vs TCR-ECHO GCN Systematic Comparison (2026-05-25)

### Source Code Analysis
Compared `/home/lyf/projects/deepAntigen/deepAntigen/antigenTCR/networks/pTCR_seq.py` (94 lines) vs TCR-ECHO `gcn_components.py` (730 lines) + `gcn_only_train.py`.

### 3 Critical Differences Explaining Performance Gap

**1. Loss Function: FocalLoss(γ=2) vs BCEWithLogitsLoss**
- Original: `FocalLoss(gamma=2, reduction='sum')` — easy samples get weight (1-p_t)^2. Confident correct prediction (p=0.95) → weight=0.0025.
- TCR-ECHO: `BCEWithLogitsLoss(reduction='sum')` — all samples equal weight.
- Impact: FocalLoss is a built-in regularizer; BCE forces model to overfit easy samples.

**2. Classifier Capacity: 8K vs 197K (24× difference)**
- Original: `MHA-output [B,128] → Projector(128→64) → ReLU → Dropout(0.2) → Classifier(64→2) → Softmax` = ~8,320 params.
- TCR-ECHO: `F_spatial [B,256] → spatial_proj(256→256×2) → LayerNorm → Dropout(0.3) → Classifier(256→256→1)` = ~197K params.
- Impact: GCN already extracts rich features; large classifier head is overfitting backdoor.

**3. Encoder Architecture: Independent vs Per-Layer Cross-Modal**
- Original: 2 completely independent `Encoder` modules (peptide_encoder, cdr3_encoder). Each: 5× TGCN(GRU) → BN. Cross-modal interaction ONLY at final MHA. TopK pooling only at last layer.
- TCR-ECHO: Per-layer `CrossModalGCNLayer`: LocalMP → SuperNodeExchange(MHA-pool+gate+cross-proj) → GRU → BN. Cross-modal dialogue at EVERY layer.
- Impact: SuperNodeExchange creates richer features but also more overfitting risk at depth=5.

### Architecture Details (Original)
| Component | Original deepAntigen | TCR-ECHO |
|-----------|---------------------|----------|
| Per-layer local agg | TGCN (message_w + update_w + GRUCell) | LocalMessagePassing (pure MLP, no GRU) |
| Per-layer GRU | Inside TGCN | At end of CrossModalGCNLayer (after LocalMP + SuperNodeExchange) |
| Cross-modal | Final MHA only | Every layer via SuperNodeExchange |
| Atom→super-node | scatter_mean (N/O-biased TopK) | MHA-weighted pooling (_attn_pool) |
| MHA output | `sum(intermap * att, dim=(1,2))` → [B,128] | Full interaction_map [B,k,k,H] → masked max/avg pool → [B,256] |
| Node dropout | 5% in dataset __getitem__ | 5% in DeepGCN.forward() |
| GCN dropout | Dropout(0.2) on [B,128] MHA output | Dropout(0.2) on [B,k,k,H] interaction_map |
| Optimizer | SGD lr=1e-4, momentum=0.9, wd=0 | SGD lr=1e-4, momentum=0.9, wd=1e-4 |
| LR schedule | Step@200,400 ×0.5 | MultiStepLR@200,400 ×0.5 |
| Epochs | 700 | 700 |
| Val split | 10% (StratifiedKFold) | 10% (random split, seed=42) |

### P0 Fix v1 Results (2026-05-23~24)
- dropout_atom activated + weight_decay=1e-4
- BCE loss, LR=2e-4, classifier 256, val 5%
- Best val AUC: 0.7630 (ep163), Test AUC: 0.7515
- Early stopping at ep309 (patience=150)
- **Overfitting persisted**: train_loss dropped 68% while val_auc stagnated 146 epochs

### P0 Fix v2 (2026-05-25) — Training in Progress
- FocalLoss(γ=2) + classifier 256→64 (16.5K params) + LR 1e-4 + val 10%
- Total params reduced from 2,400,257 → 2,350,721
- patience=60

### Design Decisions (new)
| Decision | Rationale |
|----------|-----------|
| FocalLoss > BCE for GCN-only | Paper's core regularizer — down-weights easy samples |
| Classifier 256→64 | Align with paper's tiny head (8K); GCN features are already rich |
| LR 2e-4→1e-4 | Paper config; slower convergence reduces overfitting |
| Val 5%→10% | Paper's StratifiedKFold 10%; more stable val_auc with 6K samples |
| Keep weight_decay=1e-4 | Paper has wd=0 but FocalLoss provides implicit regularization; wd as extra guard |
| Keep SuperNodeExchange | Intentional architecture improvement over paper; regularization fixes should suffice |
<!--
  WHAT: Knowledge base for TCR-ECHO. Stores all discoveries, decisions, and research.
  WHY: Context windows are limited. This file is external memory — persistent and unlimited.
  WHEN: Update after ANY discovery. Follow the 2-Action Rule.
-->

## deepAntigen Paper Analysis (2026-05-22)

### Paper Reference
Que, Xue, Wang et al. "Identifying T cell antigen at the atomic level with graph convolutional network." Nature Communications 16, 5171 (2025).

### Paper Architecture vs Ours
| Component | Paper deepAntigen | TCR-ECHO GCN |
|-----------|------------------|--------------|
| Encoder structure | 2 independent Encoders, no cross-talk during encoding | Per-layer SuperNodeExchange (cross-modal dialogue every layer) |
| Cross-molecule interaction | Only at final MHA | At every GCN layer + final MHA |
| GRU placement | Inside TGCN (after local MP) | At end of CrossModalGCNLayer (after local MP + global context) |
| MHA output | sum(dim=(1,2)) → [B,H] | Full [B,k,k,H] interaction map → masked pool |
| MHA attention | Per-row softmax + mask in softmax | Flattened softmax + external joint_mask |

### Paper Training Config (test_antigenTCR/config_seq.ini)
- depth=5, k=20, heads=4, hidden=128
- Optimizer: SGD, lr=1e-4, momentum=0.9, weight_decay=0
- batch_size=32, epochs=700
- Step LR decay @200,400 (×0.5)
- Training data: 62,446 samples (31,223 pos/neg, balanced)
- Data sources: IEDB, VDJdb, PIRD, McPAS-TCR, ImmuneCODE, NeoTCR

### Paper Reported Performance (TCR Zero-Shot)
- COVID-19 dataset: AUROC=0.71, AUPR=0.75
- PanPep baseline: AUROC=0.51 (!!)
- Gao et al. dataset: AUROC≈0.84
- Our GCN-only: 0.7445 (better than PanPep 0.51, between paper's 0.71 and 0.84)

### Key Finding: MHA Softmax — Flattened > Per-Row for Our Architecture
- Per-row softmax (paper-aligned): Test AUC 0.7036
- Flattened softmax (original): Test AUC 0.7445
- Flattened softmax works better with full interaction map + masked pool
- Per-row softmax is designed for sum(dim=(1,2)) output, not compatible with our architecture

### Key Finding: Depth=5 + Per-Layer Cross-Modal Exchange = Gradient Collapse
- Paper's 5 independent layers work because encoders don't cross-talk
- Our per-layer SuperNodeExchange at depth=5 → gradient collapse (train_loss→0.001)
- depth=2 is stable with our architecture; deeper layers need gradient stabilization

### Key Finding: Paper's Training Data ≠ PanPep
- Paper compiled 62k TCR pairs from 6 databases
- Located at `/home/lyf/projects/deepAntigen/test_antigenTCR/Data/sequence/train.csv`
- Our graph cache built for this data (+31,300 new graphs)
- COVID-19 test set (1.1M pairs) also available but not yet evaluated

### Design Decision: Our GCN Architecture is Fundamentally Different
Our per-layer SuperNodeExchange is an intentional improvement, not a bug. Trade-offs:
- **Pros**: Richer cross-modal information flow, better gradient pathways
- **Cons**: Limits max depth (2 works, 5 crashes), harder to train from scratch
- **Verdict**: Keep it — ESM+GCN fusion expects rich per-layer features

## GCN Architecture Alignment with deepAntigen_Seq (2026-05-21)

### Discovery: 3 Architectural Discrepancies vs Original
After component-by-component comparison with the original deepAntigen_Seq framework:

### Fix 1: Double GRU → Single GRU per layer
**Symptom**: TGCN had internal `GRUCell` + CrossModalGCNLayer had another `GRUCell` = 2 GRUs per layer.
**Original**: Only 1 GRU at step ④ (end of layer). Step ① local message passing is pure MLP (no GRU).
**Fix**: Replaced `TGCN` → `LocalMessagePassing` in `CrossModalGCNLayer`. `LocalMessagePassing` was already defined but unused — it's the pure-MLP version of local aggregation. Layer-end GRU retained as the sole temporal smoothing per layer.
**Files**: `gcn_components.py` (CrossModalGCNLayer)

### Fix 2: scatter_mean → MHA-Weighted Atom Pooling
**Symptom**: `SuperNodeExchange` used `scatter_mean` for super-node creation — all atoms equal weight.
**Original**: "利用多头注意力机制对原子特征加权，传递给超级节点" — within-graph MHA pooling: learnable query attends to N atom keys, producing weighted-sum super node. (Cross-graph MHA is degenerate with 1 key — Linear is correct there.)
**Fix**: Added `_attn_pool()` method with learnable query `attn_q_pep`/`attn_q_tcr`, key/value projections, per-molecule softmax, weighted sum via `scatter_add`. Verified gradient flow (attn_q_pep grad norm: 16.49).
**Files**: `gcn_components.py` (SuperNodeExchange)

### Fix 3: Missing Graph Augmentation (Node Dropout)
**Symptom**: No graph data augmentation during training.
**Original**: "训练阶段以 5% 概率随机丢弃节点及相连边"
**Fix**: Added `DeepGCN._node_dropout()` — random 5% node masking, edge filtering (both endpoints must survive), index remapping for surviving nodes. Applied in `forward()` during training only.
**Files**: `gcn_components.py` (DeepGCN)

### Design Decisions (new)
| Decision | Rationale |
|----------|-----------|
| LocalMessagePassing (not TGCN) for per-layer local aggregation | Aligns with deepAntigen step ① — pure MLP, GRU only at step ④ |
| MHA-weighted atom→super-node pooling | Aligns with deepAntigen step ② — learnable importance weighting of atoms |
| 5% node dropout during training | Aligns with deepAntigen graph augmentation — regularization for atom graphs |
| Cross-graph exchange using Linear (not MHA) | Retained — with 1 super-node/molecule, MHA is degenerate; Linear is correct |

## Requirements
- Fuse ESM-2 protein language model (Track 1) with deepAntigen atom-level GCN (Track 2)
- Predict TCR-peptide binding — binary classification
- Pipeline: joint pretraining → structure fine-tuning → full evaluation
- Environment: RTX 5090 ×3 (32GB each), offline, PyTorch 2.x

## Phase 2: Joint Pretraining Results (2026-05-14/15)
- 50 epochs, ~9 min/epoch
- Best val AUC: 0.7294 (epoch 42)
- **Test AUC: 0.7914** (majority_testing_dataset, 5,074 samples)
- Checkpoint: `runs/gcn_joint/best_model.pth` (5.4 GB)

## Phase 3: 133-Fold LOGO CV Results (2026-05-18)
- 95/133 folds triggered early stopping (patience=80), 38 folds ran full 200 epochs
- Best val loss: min=0.0000, max=0.3708, mean=0.090
- Checkpoints: `runs/structure/{pdb}/atom-level_parameters.pt` (133 files)

## Phase 3: Critical Bug Discoveries (2026-05-16)

### Bug 1: Cross-Fold Contamination (Round 1)
(see previous version)

### Bug 2-7: (Rounds 2-3)
(see previous version for dual scheduler, dead params, MHA spatial blindness, NaN, averaging paradox, gradient conflict)

### Bug 8: Non-Standard Focal Loss Alpha (Round 3)
(see previous version)

## Phase 3: 2026-05-17/18 Discoveries

### Bug 17: Cross-Graph Atom Misalignment (2026-05-17)
**Symptom**: Without this fix, training ran but results were silently wrong — atoms from Mol 0 appearing in Mol 1's interaction map.

**Root cause**: `TopKPooling.forward()` tail-padded with `torch.cat([real, zeros])` then `reshape(B, k, H)`. When molecules have different N/O atom counts (k[i] = min(10, n_NO_atoms)), reshape steals atoms across molecule boundaries. Mol 0 gets 5 atoms + 5 from Mol 1. Mol 1 gets 3 remaining + 7 from Mol 2.

**Fix**: Per-graph `torch.split(k_list)` → independent zero-pad → `torch.stack`. Each molecule gets its own k atoms in its own row. `valid_mask [B,k]` (1=real, 0=ghost) returned alongside tensors.

**Files**: `gcn_components.py` (topk, TopKPooling.forward, DeepGCN.forward)

### Bug 18: Ghost Atom Leakage Through MHA + Gate (2026-05-17)
Padded positions have zero features but non-zero bias through `W_CDR3/W_Peptide` → participate in MultiHeadAttention softmax → steal attention weight from real atoms. Spatial gate `σ(p)·σ(c)` doesn't help because padded `topk_scores=0` → `σ(0)=0.5` → gate half-open.

**Fix**: `joint_mask = (p_valid ⊗ c_valid).unsqueeze(-1)` → `[B,k,k,1]`. Injected at 3 layers:
1. DeepGCN.forward: `interaction_map *= joint_mask` (ghost features → 0)
2. Stage 2 forward: `interaction_map *= joint_mask` (redundant safety)
3. Focal loss: filter `valid_pairs = joint_mask.view(-1).bool()`, only real pairs contribute

### Bug 19: Gradient Collapse from Mean Reduction (2026-05-17)
**Symptom**: Stage 2 loss = 0.008 with `reduction='mean'` vs 1.30 with `sum/B`. Training stuck.

**Root cause**: `reduction='mean'` → per-parameter gradient ≈ avg_pair_grad ≈ 1e-6. Adam's ε=1e-8 → sqrt(v+ε) ≈ 1e-4 (ε-dominated). Effective step = lr × g/sqrt(v+ε) ≈ 1e-3 × 0.01 = 1e-5. Training doesn't move.

**Fix**: `reduction='sum'` over real pairs, then `/batch_size`. Gradient = (N_valid/B) × avg_pair_grad ≈ 90×. After ~100-step Adam warmup, v_t >> ε, effective step ≈ lr.

**Insight**: Adam is scale-invariant in steady state (`m/sqrt(v)` ≈ sign), but ONLY when gradient magnitude >> √ε (≈ 1e-4). Below that, ε dominates and effective step shrinks proportionally.

### Bug 20: Scheduler Absolute Threshold Blindness (2026-05-17)
**Symptom**: Fold 2 val ≈ 0.003, scheduler `threshold=0.01` absolute. Requires val improvement > 3× val itself. Scheduler sees permanent "no improvement" → hacks LR → death spiral.

**Fix**: `threshold_mode='rel'`, `threshold=0.001` (0.1% relative). Adapts to any loss scale — 0.1% of 0.003 = 3e-6, easily achievable. `factor=0.75` instead of 0.5 — gentler LR decay prevents overshoot.

### Bug 21: Collapse Counter-Reset Attack (2026-05-17)
**Symptom**: Fold 2 epoch 40: train=0.004, val=0.003 → model collapsed to single-class prediction. But val < min_val → saved as "best" → patience counter reset to 0 → early stopping disabled.

**Fix**: Collapse detector with FOLD-ADAPTIVE threshold: `train_loss < 0.01 * initial_train_loss`. Fold 0: threshold=0.013, Fold 2: threshold=0.005. Collapsed epochs are ineligible for saving AND count as non-improvement (counter++).

### Bug 22: LR Floor Too Deep (2026-05-17)
**Symptom**: `min_lr = 1e-7`, `factor=0.5`, `patience=30` → 31 reductions × 30 = 930 epochs minimum to trigger. Stage 2 only has 200 epochs.

**Fix**: Replaced LR-floor-trigger with patience-based early stopping: counter increments on each non-improvement epoch, resets on genuine improvement. `patience=80`, collapse-guard prevents counter reset.

### Bug 23: Checkpoint Save Filter Too Broad (2026-05-17/18)
**Symptom**: Intended ~5MB checkpoints, actual 62MB. `requires_grad=True` filter catches `cross_attn` (BidirectionalDualViewAttention), `classifier`, `gcn_aux_head` — these are trainable in the full model but NOT in Stage 2 optimizer scope.

**Root cause**: `freeze_encoder()` only freezes GCN encoder + init projections. `cross_attn` and `classifier` belong to the ESM track, not the GCN track. They have `requires_grad=True` (LoRA params are trainable) but are never in Stage 2's forward path.

**Fix**: Filter by `stage2_opt.param_groups` parameter IDs instead of `requires_grad`. PENDING.

## Technical Decisions (cumulative)
| Decision | Rationale |
|----------|-----------|
| LoRA on ESM, not full fine-tune | 650M model too large; LoRA r=8 gives ~2M trainable params |
| Dual ESM encoders (not shared) | TCR and peptide are different biochemical domains |
| gcn_aux_head auxiliary loss | Fixes GCN gradient starvation |
| SuperNodeExchange MHA → Linear | 1 super-node: softmax(1×1) ≡ 1.0 |
| Precompute graphs + disk cache | 51k unique seqs; build once, load instantly |
| LOGO CV for structure fine-tuning | 133 folds, standard deepAntigen protocol |
| Independent 2-class atom_contact_head | `cat([-x,+x])` forced anti-correlation |
| focal_alpha=0.995 (standard form) | 22:1 pos:neg weight ratio |
| Per-fold fresh checkpoint load | Eliminates cross-fold contamination |
| Stage 2: ReduceLROnPlateau only | Dual schedulers fight for LR control |
| Stage 2: freeze_encoder (not freeze_topk) | TopK+MHA jointly fine-tuned with contact head |
| Sigmoid-mul spatial gate | No NaN risk, no averaging paradox |
| Distance MSE disabled | Gradient conflict with Focal on interaction_map |
| Per-graph split/pad/stack | Eliminates cross-graph atom misalignment |
| joint_mask at 3 layers | Full-link ghost atom elimination |
| `reduction='sum'/B` | Restores Adam-compatible gradient scale |
| `threshold_mode='rel'` | Scheduler works at any loss magnitude |
| Collapse guard (adaptive) | Prevents collapsed models from resetting early stopping |
| Patience-based early stopping | Reliable trigger within 200-epoch budget |
| `weight_decay=1e-6` | Prevents L2 dominance in late Focal training |

### Architecture Change: TopK All-Atom Selection (2026-05-19)
**Change**: TopKPooling now scores and selects from ALL atoms (previously N/O atoms only).
- Removed `generate_O_N()` helper function and `from itertools import accumulate` import
- Rewrote `topk()`: per-molecule sort → take top-k directly by score → local→global index conversion
- k_i = min(ratio, n_atoms_i) — handles molecules with fewer atoms than ratio
- `on_index` now lists all atoms (0..total_atoms-1) instead of N/O atoms only
- Zero-padding logic preserved for molecules with < ratio total atoms
- Files: `gcn_components.py` (topk, TopKPooling, DeepGCN)

### Architecture Change: Spatial Feature Aggregation (2026-05-19)
**Problem**: `sum(dim=(1,2))` destroyed k×k spatial topology — all fine-grained pairing info mixed into one vector. Single Linear(128→1280) was 10× sparse expansion.

**Fix 1 — Masked Max/Avg Pooling**:
- Flatten interaction_map [B,10,10,128] → [B,100,128]
- Masked Max: `masked_fill(-1e9)` → ghost atoms at -inf can't be selected
- Masked Avg: `(×mask).sum() / valid_count` → correct mean, not diluted by 84/100 ghosts
- cat(max, avg) → [B,256]

**Fix 2 — Residual GCN Projection**:
- 256→512→256 MLP + Identity shortcut + LayerNorm
- Identity shortcut: 256→256 with 0 extra params, pure gradient pass-through
- Files: `model.py`

### Architecture Change: Cross-Modal Gated Fusion (2026-05-19)
**Problem**: Simple cat gave ESM 67% of fusion dim (2560/3840); classifier Linear(3840→512) ≈ 2M params.

**Fix — Four-component gated fusion**:
1. **ESM Projection**: Linear(1280→512) + LayerNorm for both TCR and peptide
2. **Decoupled Language Gate**: cat(tcr_proj, pep_proj) [1024] → MLP(1024→256→1024) → Sigmoid → split(512,512) → W_tcr, W_pep. Each gets independent per-dimension weights informed by joint context.
3. **Cross-Modal Physics Gate**: ctx_lang=(tcr+pep)/2 → Linear(512→128) → cat(F_spatial, ctx_lang_small) [384] → MLP(384→64→256) → Sigmoid → W_phys. Language context directly shapes physics gating.
4. **Balanced Fusion**: cat(gated_tcr, gated_pep, gated_phys) → [B,1280], classifier input down from 3840
- Files: `model.py`

### Architecture Change: Loss Weight Cosine Annealing (2026-05-19)
**Problem**: Fixed λ_gcn_aux=1.0 over-emphasized GCN aux task throughout training; fixed λ_int=2.0 dominated language space.

**Fix**:
- `model.forward()` accepts `lambda_gcn_aux_override`, `lambda_int_override` (None = use stored value)
- `train.py`: `cosine_anneal(epoch, total, start, end)` per epoch
- Config: `loss_annealing` YAML section with start/end/schedule per weight
- λ_gcn_aux: 1.0→0.1 (GCN cold starts, then yields to focal loss)
- λ_int: 2.0→0.5 (early contrastive focus, later classifier gets more signal)
- Annealed weights logged to wandb
- Files: `model.py`, `train.py`, `configs/config_gcn.yaml`

## Technical Decisions (cumulative)
| Decision | Rationale |
|----------|-----------|
| LoRA on ESM, not full fine-tune | 650M model too large; LoRA r=8 gives ~2M trainable params |
| Dual ESM encoders (not shared) | TCR and peptide are different biochemical domains |
| gcn_aux_head auxiliary loss | Fixes GCN gradient starvation |
| SuperNodeExchange MHA → Linear | 1 super-node: softmax(1×1) ≡ 1.0 |
| Precompute graphs + disk cache | 51k unique seqs; build once, load instantly |
| LOGO CV for structure fine-tuning | 133 folds, standard deepAntigen protocol |
| Independent 2-class atom_contact_head | `cat([-x,+x])` forced anti-correlation |
| focal_alpha=0.995 (standard form) | 22:1 pos:neg weight ratio |
| Per-fold fresh checkpoint load | Eliminates cross-fold contamination |
| Stage 2: ReduceLROnPlateau only | Dual schedulers fight for LR control |
| Stage 2: freeze_encoder (not freeze_topk) | TopK+MHA jointly fine-tuned with contact head |
| Sigmoid-mul spatial gate | No NaN risk, no averaging paradox |
| Distance MSE disabled | Gradient conflict with Focal on interaction_map |
| Per-graph split/pad/stack | Eliminates cross-graph atom misalignment |
| joint_mask at 3 layers | Full-link ghost atom elimination |
| `reduction='sum'/B` | Restores Adam-compatible gradient scale |
| `threshold_mode='rel'` | Scheduler works at any loss magnitude |
| Collapse guard (adaptive) | Prevents collapsed models from resetting early stopping |
| Patience-based early stopping | Reliable trigger within 200-epoch budget |
| `weight_decay=1e-6` | Prevents L2 dominance in late Focal training |
| **TopK all-atom selection** | N/O constraint excluded C,S,H atoms from attention; all-atom gives model full chemical vocabulary |
| **Masked Max/Avg pool + residual MLP** | sum destroyed spatial topology; Max captures strongest contact, Avg preserves background, residual MLP gives smooth manifold learning |
| **Decoupled language gate** | Shared W_lang on summed TCR+pep forced identical filtering; cat→split gives independent TCR/PEP gates |
| **Cross-modal physics gate** | Self-gating was blind to language context; ctx_lang injection makes it truly cross-modal |
| **ESM proj 1280→512 + LayerNorm** | Raw 1280-dim dominated fusion (67% vs 33%); 512+512+256 balances modalities |
| **Identity shortcut in GCN proj** | 256→256 Linear shortcut was redundant; identity preserves gradient with 0 params |
| **Cosine annealing λ_gcn_aux 1.0→0.1** | Fixed λ over-emphasized aux task late in training; annealing lets focal loss dominate after GCN cold start |
| **Cosine annealing λ_int 2.0→0.5** | Fixed λ dominated language feature space; decay gives classifier more signal later |
- Phase 2 checkpoint: `runs/gcn_joint/best_model.pth` (5.4 GB)
- Graph cache: `datasets/panpep/graph_cache/` (64,524 pickle files)
- PDB data: `datasets/pdb_structure/` (133 structures)
- Phase 3 checkpoints: `runs/structure/{pdb}/atom-level_parameters.pt` (133 files, 8GB)
- Config (Phase 2): `configs/config_gcn.yaml`
- Config (Phase 3): `configs/config_structure.yaml`
- Log (Phase 3): `runs/structure/training.log`
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
