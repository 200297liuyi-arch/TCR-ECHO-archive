# Progress Log

## Session: 2026-05-28~29 — Phase 2 Joint Training v2 (Residual Gating + Scheduler)

### Gate Collapse Root Cause Analysis
- **Status:** complete
- Loaded best checkpoint from Phase 2 v1 (AUC 0.7647)
- Analyzed gate statistics on 30 validation batches (~1920 samples)
- **W_tcr=0.14, W_pep=0.22, W_phys=0.61** — language gate collapsed, physics gate healthy
- Concluded: model was replacing ESM features with GCN, not fusing

### Architecture Fixes
- **Status:** complete
- `gcn_plugin.py`: gate_lang refactored from nn.Sequential to named layers
- `gcn_plugin.py`: language gating changed to residual `feat * (1 + W)`
- `gcn_plugin.py`: gate_lang_fc2.bias initialized to 1.5
- `gcn_plugin.py`: physics gating unchanged (multiplicative, default init)
- `configs/config_gcn.yaml`: weight_decay 1e-4 → 5e-4
- `configs/config_gcn.yaml`: early_stopping patience 10 → 20
- `configs/config_gcn.yaml`: added scheduler section (ReduceLROnPlateau)
- `train.py`: added scheduler creation + step logic + LR logging

### Phase 2 v2 Training (2026-05-28~29, ~26h, ongoing)
- **Status:** training at epoch 155/200
- Config: bs=64, lr=5e-5, wd=5e-4, patience=20, ReduceLROnPlateau(factor=0.5, patience=10)
- GPU 1 (RTX 5090), ~9 min/epoch, 23.6 GB VRAM
- **Best val AUC: 0.7664 (epoch 153)** — broke previous ceiling of 0.7647!
- LR schedule: 5e-5 (ep1-93) → 2.5e-5 (ep94-124) → 1.25e-5 (ep125+)
- Each LR reduction brought renewed AUC improvement

### AUC Progression
| Epoch | AUC | LR | Note |
|-------|-----|----|------|
| 1 | 0.6432 | 5e-5 | start |
| 25 | 0.7216 | 5e-5 | leading prev run by +0.007 |
| 63 | 0.7506 | 5e-5 | broke 0.75 |
| 83 | 0.7585 | 5e-5 | peak at 5e-5 |
| 94 | 0.7539 | **2.5e-5** | 1st LR reduction |
| 107 | 0.7615 | 2.5e-5 | recovery |
| 114 | 0.7628 | 2.5e-5 | peak at 2.5e-5 |
| 125 | 0.7618 | **1.25e-5** | 2nd LR reduction |
| 148 | 0.7638 | 1.25e-5 | grinding upward |
| 151 | 0.7654 | 1.25e-5 | broke 0.765 |
| **153** | **0.7664** | 1.25e-5 | **NEW CEILING** |
| 155 | 0.7641 | 1.25e-5 | slight pullback |

### Files changed (2026-05-28~29)
| File | Action |
|------|--------|
| `gcn_plugin.py` | Residual gating + bias init for language gate |
| `train.py` | +ReduceLROnPlateau scheduler + LR logging + train_auc progress |
| `configs/config_gcn.yaml` | wd 5e-4, patience 20, scheduler, bs=64 |
| `utils.py` | Fixed load_checkpoint GCNPlugin compatibility |
| `.claude/task_plan.md` | Phase 11 added |
| `.claude/findings.md` | Gate analysis + residual gating findings |
| `.claude/progress.md` | This update |

## Session: 2026-05-25~27 — Paper-Aligned Independent Encoder Architecture

### Paper-Aligned Architecture Implementation
- **Status:** complete
- Replaced per-layer SuperNodeExchange with 2 independent Encoders (paper's pTCR_seq.py)
- New classes in `gcn_components.py`: PaperTopKPooling, PaperEncoder, PaperAlignedDeepGCN
- MultiHeadAttention extended with `output_mode='sum'` — returns `sum(dim=(1,2))` [B,H]
- GCNOnlyModel simplified: independent encoders → MHA → Projector(128→64)→Classifier(64→1)
- Training config: SGD lr=1e-4 wd=0, FocalLoss(γ=2, reduction='sum'), bs=64, patience=60

### 10-Fold CV Results (2026-05-27 00:01 complete, ~32.5h on 3 GPUs)

| Stat | val AUC | test AUC |
|------|---------|----------|
| Mean | 0.7493 | 0.7499 |
| Median | 0.7471 | 0.7515 |
| Best | 0.7642 | 0.7824 |
| Worst | 0.7390 | 0.7095 |

### Key Changes
- `gcn_only_train.py`: added --fold/--n-folds/--gpu/--lr args, StratifiedKFold, per-fold output
- `run_cv.sh`: auto-launch 10 folds in batches of 3, wait + aggregate
- Patience fixed: 150→60 (was incorrectly changed during paper refactoring)

### Files changed
| File | Action |
|------|--------|
| `gcn_components.py` | +PaperTopKPooling, +PaperEncoder, +PaperAlignedDeepGCN, MHA output_mode |
| `gcn_only_train.py` | Simplified GCNOnlyModel, argparse, StratifiedKFold, wd=0, patience=60 |
| `run_cv.sh` | Created — parallel fold launcher |
| `docs/superpowers/specs/2026-05-25-gcn-paper-aligned-encoder-design.md` | Created |
| `docs/superpowers/plans/2026-05-25-gcn-paper-aligned-encoder-plan.md` | Created |
| `.claude/task_plan.md` | Phase 10 added |
| `.claude/findings.md` | Updated |
| `.claude/progress.md` | This update |

## Session: 2026-05-25 — deepAntigen Source Comparison & GCN v2 Fix

### Systematic Comparison with Original deepAntigen
- **Status:** complete
- Read original `pTCR_seq.py` (94 lines), `load_seq.py`, `model_utils.py`, `config_seq.ini`
- Identified 3 critical gaps: FocalLoss, classifier capacity, encoder architecture
- Original uses 2 independent Encoders (no per-layer cross-modal), FocalLoss(γ=2), 8K classifier
- TCR-ECHO uses per-layer SuperNodeExchange, BCE loss, 197K classifier

### P0 Fix v2 Implementation
- **Status:** complete
- `gcn_only_train.py`: BCEWithLogitsLoss → FocalLoss(γ=2, reduction='sum')
- `gcn_only_train.py`: classifier_hidden 256→64 (16.5K params vs 197K)
- `gcn_only_train.py`: LR 2e-4→1e-4, val_split 5%→10%, patience 150→60
- Total params: 2,400,257 → 2,350,721
- Verified: FocalLoss down-weights easy samples (Focal 0.09 vs BCE 0.99 on test input)

### P0 Fix v1 Recap (2026-05-23~24)
- dropout_atom activated + weight_decay=1e-4, BCE loss
- Result: val AUC 0.7630, test AUC 0.7515, early stop ep309
- Overfitting not solved — led to v2 fixes

### Files changed (2026-05-25)
| File | Action |
|------|--------|
| `gcn_only_train.py` | FocalLoss class, classifier 64, LR 1e-4, val 10%, patience 60 |
| `.claude/task_plan.md` | Phase 9 added |
| `.claude/findings.md` | Full comparison analysis + architecture table |
| `.claude/progress.md` | This update |

### Current Training (launched 2026-05-25 12:27)
- FocalLoss(γ=2), 16.5K classifier, LR=1e-4, wd=1e-4, val=10%, patience=60
- Train 56,201 / Val 6,245 / Test 1,714
- Monitoring: cron job every 30 min

## Session: 2026-05-22 — deepAntigen Paper Analysis & GCN-Only Benchmarking

### ESM-only Zero-Shot Evaluation
- **Status:** complete
- Evaluated on `zero_test_paper.csv` (1,714 samples, 857+/857-)
- **Result: Test AUC 0.8148, Acc 0.7474, F1 0.7149**
- Only 2.2% below majority test AUC (0.8371) — strong generalization

### GCN-Only Training & Paper Analysis
- **Status:** complete (analysis), training pending
- **GCN-only baseline (depth=2, k=10, focal loss, PanPep data)**: Test AUC 0.7445 on zero_test_paper.csv
- Got worse when aligning MHA to per-row softmax (0.7036) — flattened softmax + focal loss + masked pool is the winning combo

### deepAntigen Paper Deep-Dive
- **Status:** complete
- Paper: Que, Xue, Wang et al., Nature Communications 16, 5171 (2025)
- Paper config: depth=5, k=20, hidden=128, SGD lr=1e-4, bs=32, epochs=700, momentum=0.9
- Paper TCR zero-shot: 0.71 (COVID-19), ~0.84 (Gao dataset), PanPep baseline=0.51
- Paper training data: 62,446 samples (balanced) from IEDB+VDJdb+PIRD+McPAS+ImmuneCODE+NeoTCR

### Key Finding: Architecture Incompatibility
- Paper's deepAntigen: **2 independent encoders**, cross-attention only at final MHA
- Our GCN: **per-layer SuperNodeExchange** (cross-modal dialogue at every layer)
- depth=5 with per-layer cross-modal exchange → gradient collapse (train_loss→0.001)
- Our architecture works at depth=2, but paper's 5-layer config is incompatible
- Our per-layer cross-modal exchange is a design improvement over paper, but limits max depth

### Original Paper Training Data Located
- Found at `/home/lyf/projects/deepAntigen/test_antigenTCR/Data/sequence/train.csv` (62,446 samples)
- Test: `zero-shot_sample.csv` (1,714 samples, same as zero_test_paper.csv)
- Graphs precomputed for paper data: +31,300 new graphs added to cache
- `precompute_paper_graphs.py` created for this purpose

### Files changed (2026-05-22)
| File | Action |
|------|--------|
| `zero_shot_eval.py` | **Created** — ESM-only zero-shot evaluation script |
| `gcn_only_train.py` | **Created** — standalone GCN training script, supports paper data |
| `gcn_components.py` | MHA softmax: flattened→per-row→reverted to flattened (flattened wins) |
| `utils.py` | Fixed `load_checkpoint` to not pass `use_gcn`/`gcn_args` to Model |
| `precompute_paper_graphs.py` | **Created** — builds graphs for paper's training data |
| `.claude/progress.md` | This update |
| `.claude/findings.md` | Paper analysis + architecture findings |
| `.claude/task_plan.md` | Updated phases |

### Performance Summary
| Model | Train Data | Zero-Shot Test AUC | Zero-Shot Acc |
|-------|-----------|-------------------|---------------|
| ESM-only | PanPep 52k | 0.8148 | 0.7474 |
| GCN-only (depth=2) | PanPep 52k | 0.7445 | 0.6919 |
| GCN-only (depth=5) | Paper 62k | ❌ collapsed | ❌ |

### Pending
- [ ] GCN-only training with paper data (stable depth=2 config) — training crashed, need fix
- [ ] ESM+GCN joint training (gcn_plugin.py)
- [ ] COVID-19 zero-shot evaluation (paper's 1.1M test set)

## Session: 2026-05-21 — GCN Architecture Alignment with deepAntigen_Seq

### GCN Architecture Fixes (vs original deepAntigen_Seq framework)
- **Status:** complete
- Identified 3 discrepancies via component-by-component comparison:

**Fix 1: Double GRU → Single GRU per layer**
- Problem: TGCN had internal GRU + CrossModalGCNLayer had another GRU = 2 GRUs/layer
- Original: only 1 GRU at step ④ (end of layer), local MP is pure MLP
- Fix: replaced `TGCN` with `LocalMessagePassing` (already defined, unused) in `CrossModalGCNLayer`
- `LocalMP`: pure MLP neighbor aggregation, no GRU inside

**Fix 2: scatter_mean → MHA-weighted atom pooling**
- Problem: `SuperNodeExchange` used `scatter_mean` to create super nodes
- Original: "利用多头注意力机制对原子特征加权" — MHA-weighted pooling per molecule
- Fix: added `_attn_pool()` with learnable query vectors (`attn_q_pep`, `attn_q_tcr`), key/value projections (`attn_k`, `attn_v`), per-molecule softmax, weighted sum
- Gradient path verified

**Fix 3: Added 5% node dropout augmentation**
- Original: "训练阶段以 5% 概率随机丢弃节点及相连边"
- Fix: added `DeepGCN._node_dropout()` — randomly drops nodes + filters incident edges + remaps indices
- Applied in `DeepGCN.forward()` when `self.training=True`

### Files changed (2026-05-21)
| File | Action |
|------|--------|
| `gcn_components.py` | 3 fixes: LocalMP replaces TGCN, MHA pooling replaces scatter_mean, +node dropout |

## Session: 2026-05-21 — Modular Refactoring + ESM-only Fixes

### ESM-only Performance Debug & Fix (vs ECHO-deepantigen reference)
- **Status:** complete
- Identified 3 critical issues by comparing with ECHO-deepantigen source:
  1. Cross-attention mix-then-normalize → normalize-then-mix
  2. ESM projection bottleneck (1280→512) removed
  3. Dropout 0.08→0.3, added cross_attn_dropout=0.3
- Added early stopping (patience=10)
- Aligned training params: batch_size=32, epochs=200, wd=1e-4
- **Result: Test AUC 0.7868→0.8371 (+5.0%), F1 0.6882→0.7519 (+6.4%)**
- Training: 91 epochs (early stop at 90), ~5 min/epoch

### Modular Refactoring: Model + GCNPlugin
- **Status:** complete
- Goal: separate ECHO (language) and deepAntigen (physics) into two modules
- Approach: Base Class + Plugin (option 2)
- `model.py`: pure ESM-only, all GCN code/imports removed (260 lines)
- `gcn_plugin.py`: GCNPlugin(Model) — inherits ESM track, adds GCN components
- `train.py`: dynamic `ModelClass = GCNPlugin if use_graph else Model`
- `dataset.py`, `utils.py`: unchanged

### Files changed (2026-05-21)
| File | Action |
|------|--------|
| `model.py` | Removed GCN imports, params, init blocks, forward branches |
| `gcn_plugin.py` | **Created** — GCNPlugin(Model) with all GCN logic |
| `train.py` | Dynamic ModelClass, fixed cross_attn_dropout path |
| `attentions.py` | normalize-then-mix fix |
| `configs/config_esm_only.yaml` | dropout, epochs, bs, wd aligned |
| `configs/config_gcn.yaml` | same alignment |
| `CLAUDE.md` | Full rewrite |

## Session: 2026-05-19 — Architecture Refactoring (4 major optimizations)

### Optimization 1: TopK All-Atom Selection
- **Status:** complete
- Removed N/O-only constraint in `gcn_components.py`
- Deleted `generate_O_N()` and `from itertools import accumulate`
- Rewrote `topk()`: per-molecule sort → top-k directly → local→global index conversion
- k_i = min(ratio, n_atoms_i), handles molecules with fewer atoms than ratio
- `on_index` now full atom list (0..total_atoms-1), compatible with Stage 1 Pearson loss

### Optimization 2: Spatial Feature Aggregation
- **Status:** complete
- Problem: `sum(dim=(1,2))` destroyed k×k spatial topology; single Linear(128→1280) was sparse
- Fix 1: flatten [B,10,10,128] → [B,100,128] → masked Max/Avg pool
  - Max: `masked_fill(-1e9)` → ghost atoms can't win
  - Avg: `×mask` → sum / valid_count → correct mean, not diluted by ghosts
  - cat(max, avg) → F_spatial_raw [B,256]
- Fix 2: GCN projection → residual MLP 256→512→256 + LayerNorm + Identity shortcut
- Files: `model.py`

### Optimization 3: Cross-Modal Gated Fusion
- **Status:** complete
- Problem: simple cat gave ESM 67% of fusion dim; classifier over-parameterized (3840×512)
- Fix 1: ESM Projections — Linear(1280→512) + LayerNorm for both tcr and pep
- Fix 2: Decoupled language gate — cat(tcr_proj, pep_proj) [1024] → MLP(1024→256→1024) → Sigmoid → split → W_tcr, W_pep (independent per-dimension weights)
- Fix 3: Cross-modal physics gate — ctx_lang(512→128) + F_spatial(256) → MLP(384→64→256) → Sigmoid → W_phys
- Fix 4: gated_tcr ⊙ W_tcr, gated_pep ⊙ W_pep, gated_phys ⊙ W_phys → cat [1280]
- Fix 5: classifier input 3840→1280, params ~2.0M→~0.66M
- Fusion dimensions: 512+512+256=1280 (balanced)
- Files: `model.py`

### Optimization 4: Loss Weight Cosine Annealing
- **Status:** complete
- Problem: fixed λ_gcn_aux=1.0 over-emphasized aux task; fixed λ_int=2.0 dominated language space
- Fix 1: model.forward() now accepts `lambda_gcn_aux_override`, `lambda_int_override`
- Fix 2: cosine_anneal() function in train.py, called per epoch
- Fix 3: config section `loss_annealing` with start/end/schedule per weight
- λ_gcn_aux: 1.0 → 0.1 (GCN cold start → yields to focal loss)
- λ_int: 2.0 → 0.5 (contrastive focus → classifier gets more signal)
- Annealed weights logged to wandb for monitoring
- Files: `model.py`, `train.py`, `configs/config_gcn.yaml`

### Files Modified (2026-05-19)
| File | Changes |
|------|---------|
| `gcn_components.py` | topk() rewrite, removed generate_O_N/accumulate, updated comments |
| `model.py` | spatial agg, GCN proj, gated fusion, ESM proj, loss overrides |
| `train.py` | cosine_anneal(), per-epoch weight compute, pass to model, wandb log |
| `configs/config_gcn.yaml` | loss_annealing section |
| `CLAUDE.md` | Full architecture rewrite, updated state |
| `.claude/task_plan.md` | Phase 5 added, decisions table expanded |
| `.claude/progress.md` | This file |
| `.claude/findings.md` | Architecture changes + decisions |

## Session: 2026-05-17 — Cross-Graph Padding & Gradient Restoration

### Phase 3: Critical Architecture Fixes
- **Status:** complete (133-fold LOGO CV completed 2026-05-18)

### Bug 17: Cross-Graph Atom Misalignment (2026-05-17)
- **Root cause**: `TopKPooling.forward()` used `torch.cat([real, tail_pad]) → reshape(B, k, H)`. When molecules have different N/O counts, reshape steals atoms from adjacent molecules. Mol 0's row contains Mol 1's atoms.
- **Fix**: Per-graph `torch.split(k_list)` → independent zero-pad → `torch.stack`. Each molecule's atoms stay in their own row. `valid_mask [B, k]` marks real vs ghost positions.
- Files: `gcn_components.py` (topk, TopKPooling, DeepGCN)

### Bug 18: Ghost Atom Leakage (2026-05-17)
- Root cause: Padded positions in `[B,k]` tensors have zero features but non-zero bias through MHA and atom_contact_head. They participate in attention softmax, stealing weight from real atoms.
- **Fix**: `joint_mask = (p_valid ⊗ c_valid).unsqueeze(-1)` → `[B,k,k,1]`. Injected at 3 layers: DeepGCN (interaction_map *= joint_mask), Stage 2 forward, and Focal loss filter.
- Files: `gcn_components.py`, `train_structure.py`, `model.py`

### Bug 19: Gradient Collapse from `reduction='mean'` (2026-05-17)
- **Symptom**: Stage 2 loss 0.008 with `reduction='mean'` vs 1.30 with sum/B. Adam's ε floor (1e-8) dominates sqrt(v) for mean-scale gradients, shrinking effective step ~100×.
- **Fix**: `reduction='sum'` over real pairs only, then `/batch_size`. Restores ~90× gradient scale. Adam step ≈ lr × 0.9 after warmup.
- Files: `train_structure.py` (finetune_stage2, evaluate_stage2), `model.py` (_compute_structure_loss)

### Bug 20: Scheduler Threshold Blindness (2026-05-17)
- **Symptom**: Fold 2 val_loss ≈ 0.003. Scheduler `threshold=0.01` absolute — requires improvement > 0.01 which is 3× val_loss itself. Scheduler sees no improvement, cuts LR → death spiral.
- **Fix**: `threshold_mode='rel'`, `threshold=0.001` (0.1% relative). Adapts to any loss scale. Also `factor=0.75` (gentler than 0.5).
- Files: `train_structure.py`

### Bug 21: Collapsed Model Resets Patience Counter (2026-05-17)
- **Symptom**: Fold 2 epoch 40: train=0.0037 val=0.0033 → model collapsed but val < min_val → saved as "best" → counter reset → early stopping defeated.
- **Fix**: Collapse guard — `train_loss < 0.01 * initial_train_loss` (fold-adaptive). Collapsed epochs are ineligible for saving AND count as non-improvement.
- Files: `train_structure.py`

### Bug 22: LR Floor Unreachable (2026-05-17)
- **Symptom**: `min_lr = stage2_lr * 1e-4 = 1e-7`. With factor=0.5 and patience=30: needs 10 reductions × 30 = 300 epochs to trigger. Stage 2 only has 200 epochs.
- **Fix**: Replaced with patience-based early stopping: counter=0, patience=80. Every epoch without val improvement → counter++. Counter hits 80 → break. Collapse guard prevents counter reset.
- Files: `train_structure.py`

### Bug 23: Checkpoint Size Bloat (2026-05-17)
- **Symptom**: Intended ~5MB per checkpoint, actual 62MB. Lightweight filter (`requires_grad=True`) catches cross_attn, classifier, gcn_aux_head — these are trainable in the full model but NOT in the Stage 2 optimizer scope.
- **Fix pending**: Filter by `stage2_opt.param_groups` parameter IDs instead of global `requires_grad`.
- Files: `train_structure.py` (save logic)

### Architecture Diff: 2026-05-15 → 2026-05-18
| Component | Old (2026-05-15) | Final (2026-05-18) |
|-----------|-----------------|---------------------|
| TopK padding | cat+reshape (misalignment) | split→pad→stack + valid_mask |
| Ghost atoms | Leaked through MHA+gate | joint_mask [B,k,k,1] at 3 layers |
| Stage 2 loss reduction | sum (tail-padded) | sum/B (real pairs only) |
| Scheduler threshold | abs=0.01 | rel=0.001 |
| Scheduler factor | 0.5 | 0.75 |
| Early stopping | LR floor (unreachable) | patience=80 + collapse guard |
| weight_decay (Stage 2) | 1e-4 | 1e-6 |
| Checkpoint | 5.4 GB (full model) | ~5MB intended, 62MB actual (bug 23) |

### Phase 3: 133-Fold LOGO CV Results (2026-05-18)
- **Completed**: All 133 folds trained
- **Early stopping**: 95/133 folds triggered (patience=80), 38/133 ran full 200 epochs
- **Best val loss**: min=0.0000, max=0.3708, mean=0.090
- **Checkpoints**: 133 PDB directories, 8GB total
- **Log**: `runs/structure/training.log`

### Files Modified (cumulative 2026-05-17/18)
| File | Changes |
|------|---------|
| `gcn_components.py` | topk returns k; TopKPooling split/pad/stack + valid_mask; DeepGCN joint_mask |
| `train_structure.py` | sum/B reduction; rel threshold; patience early stopping; collapse guard; wd=1e-6; lightweight save |
| `model.py` | _compute_structure_loss Stage 2: joint_mask + sum/B |
| `configs/config_structure.yaml` | (unchanged this session) |
| `CLAUDE.md`, `task_plan.md`, `progress.md`, `findings.md` | Documentation sync |

## Session: 2026-05-16 — Critical Bug Hunt & Architecture Hardening
(see previous version for full details — 4 rounds of refactoring)

## Session: 2026-05-14/15 — Phase 2 Complete
(see previous version for full details — joint pretraining AUC 0.7914)

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Architecture refactoring complete, ready for Phase 2 re-training |
| Where am I going? | Re-train Phase 2 with optimized model → Phase 4 zero-shot evaluation |
| What's the goal? | Train dual-track ESM-2 + GCN TCR-peptide binding predictor with balanced modalities |
| What have I learned? | Gated fusion prevents modality dominance; masked pooling preserves spatial signals; cosine annealing helps cold-start |
| What have I done? | 4 architecture optimizations: all-atom TopK, masked pool aggregation, gated fusion, loss annealing |
