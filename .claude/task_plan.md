# Task Plan: TCR-ECHO Dual-Model Fusion
<!--
  WHAT: Roadmap for fusing ESM-2 language model with deepAntigen GCN for TCR-peptide binding prediction.
  WHY: Complex multi-phase ML project spanning 50+ tool calls. This file keeps goals fresh.
  WHEN: Created 2026-05-13. Update after each phase completes.
-->

## Goal
Train a dual-track TCR-peptide binding predictor fusing ESM-2 protein language model (Track 1) with deepAntigen atom-level GCN (Track 2), achieving SOTA binding prediction on majority and zero-shot peptide sets.

## Current Phase
Phase 13: SeqAlignedGCN — 完全对齐 deepAntigen_Seq 原文架构 (2026-06-04) — IMPLEMENTED

### Phase 13: SeqAlignedGCN — 完全对齐 deepAntigen_Seq 原文 (2026-06-04) — COMPLETE ✅
- [x] **Root cause discovery**: GCN attention bias has ZERO effect (τ=0.227 vs τ=0: prediction correlation=1.0000)
- [x] **deepAntigen source analysis**: atom-level model uses PDB 3D structures for supervision — we only have 1-bit labels
- [x] **deepAntigen_Seq vs TCR-ECHO DeepGCN comparison**:
  - 跨模态交互：原文独立编码器仅末层 MHA，我们每层 SuperNodeExchange
  - GRU 位置：原文 TGCN 内嵌 GRU，我们层末 GRU (LocalMP→SuperNode→GRU)
  - 深度：原文 depth=5，我们 depth=2（≥3 梯度坍塌）
- [x] `SeqAlignedGCN` 实现：2 个独立 PaperEncoder(depth=5, k=20) + 末层 MHA(full output)
- [x] `PaperEncoder.forward()` 扩展为返回 dict（features + perm_local + valid_mask）
- [x] `PaperTopKPooling.forward()` 返回局部索引和 valid_mask
- [x] `gcn_plugin.py` 切换：`DeepGCN` → `SeqAlignedGCN`
- [x] 配置更新：depth 2→5, k=20
- [x] 参数量：1.64M (vs 旧 DeepGCN d=2: 1.05M)
- **Status:** implemented, ready for training

### Phase 12: GCN Bias Training — Reproducible (seed=42) + Diagnostic (2026-06-02~03) — COMPLETE ✅
- [x] `set_seed()` added to `utils.py` (random + numpy + torch + cudnn)
- [x] `train.py`, `gcn_only_train.py` call `set_seed()` on entry
- [x] Val split always from train CSV (15% stratified by label, configurable `random_seed: 42`)
- [x] **Training**: 195/200 epochs (early stop), val AUC **0.7646** (ep175)
- [x] **Test**: AUC **0.8390**, Acc 0.7742, F1 **0.7562** (+2.95pp vs prev GCN Bias F1)
- [x] **Zero-shot**: AUC 0.8050, F1 0.7075
- [x] **τ ablation diagnostic** (2026-06-03):
  - τ=0.227 vs τ=0: predictions 100% identical (correlation=1.0000, max|Δ|=3.5e-4)
  - GCN bias path contributes NOTHING — model learned to ignore it
  - 95.97% predictions changed by τ, but max change 0.035 percentage points
- [x] **deepAntigen source analysis**:
  - Atom-level (`pTCR_atom.py`): 2-stage fine-tuning with PDB distance matrix supervision
  - Seq-level (`pTCR_seq.py`): `sum(dim=(1,2))` + FocalLoss, no 3D structures
  - Both use identical TGCN backbone — difference is supervision, not architecture
- **Status:** complete — GCN Bias confirmed dead path, root cause understood

### Phase 11: Phase 2 Joint Training v2 — Residual Gating + Scheduler (2026-05-28~29) — COMPLETE ✅
- [x] Root cause: language gate collapse (W_tcr≈0.14, W_pep≈0.22) → ESM features suppressed
- [x] Fix 1: Residual gating — `gated_tcr = tcr_feat * (1 + W_tcr)` guarantees 100% signal floor
- [x] Fix 2: Language gate bias init to 1.5 → initial gate ~0.82, effective multiplier ~1.82×
- [x] Fix 3: weight_decay 1e-4 → 5e-4
- [x] Fix 4: Add ReduceLROnPlateau scheduler (factor=0.5, patience=10, min_lr=1e-6)
- [x] Fix 5: early_stopping patience 10→20
- [x] Fix 6: batch_size 32→64 (~4 min/epoch vs ~10 min)
- [x] **Epoch 153/200**: AUC **0.7664** — BROKE previous ceiling of 0.7647!
- [x] Evaluate best checkpoint on majority + zero-shot test sets
- **Status:** complete

### Phase 10: Paper-Aligned Independent Encoder + 10-Fold CV (2026-05-25~27) — COMPLETE ✅
- [x] **Root cause confirmed**: per-layer SuperNodeExchange incompatible with depth=5 → train_loss stuck
- [x] **Paper-aligned architecture implemented** (2026-05-25):
  - `PaperTopKPooling`, `PaperEncoder`, `PaperAlignedDeepGCN` added to `gcn_components.py`
  - `MultiHeadAttention` extended with `output_mode='sum'` (paper's `sum(dim=(1,2))`)
  - `GCNOnlyModel` simplified: independent encoders + MHA → Projector(128→64)→Classifier(64→1)
  - Params: 1,649,793 (vs old 2,350,721, -30%)
- [x] **Training config**: SGD lr=1e-4 wd=0 momentum=0.9, FocalLoss(γ=2, reduction='sum'), bs=64, patience=60, step LR@200,400
- [x] **10-fold CV on 3 GPUs** (2026-05-25 15:32 ~ 2026-05-27 00:01, ~32.5h total)
  - Mean val AUC: 0.7493, Mean test AUC: 0.7499
  - Best test AUC: 0.7824 (fold 5), Best val AUC: 0.7642 (fold 4)
  - All folds triggered early stopping (no NaN, no collapse)
- **Status:** complete — GCN-only mean test AUC 0.75, between paper's COVID (0.71) and Gao (0.84)

### Phase 9: GCN-Only Architecture Alignment (2026-05-24~25) — COMPLETE ✅
- [x] Root cause analysis: systematic comparison with deepAntigen original `pTCR_seq.py`
- [x] P0 fix v1: dropout_atom + weight_decay → val AUC 0.7630, test AUC 0.7515 (overfitting)
- [x] P0 fix v2: FocalLoss + small classifier → train_loss stuck at 0.69 AUC — architecture bottleneck confirmed
- **Status:** superseded by Phase 10 (paper-aligned encoder)

### Phase 8: GCN-Only Benchmarking & Paper Analysis (2026-05-22~23) — COMPLETE ✅
- [x] ESM-only zero-shot evaluation: Test AUC 0.8148
- [x] GCN-only baseline (depth=2, PanPep data): Test AUC 0.7445
- [x] MHA softmax comparison: flattened (0.7445) > per-row (0.7036)
- [x] deepAntigen paper analysis — architecture, config, training data all differ
- [x] Paper training data located + graphs precomputed
- [x] **GCN-only v1 (FocalLoss, lr=1e-4, gamma=0.1, bs=32)**: 415 epochs, best val_auc=0.6194 — poor
- [x] **Root cause analysis**: compared with working reference `ECHO-deepantigen/deepAntigen_Seq`
  - Loss: FocalLoss→BCEWithLogitsLoss(reduction='sum')
  - lr: 1e-4→2e-4, lr_decay: 0.1→0.5
  - bs: 32→64, removed grad clipping
  - Architecture difference: reference uses independent TGCN encoders + end-only cross-attention; we use cross-modal SuperNodeExchange at every layer
- [x] **GCN-only v2 (BCE+lr fix)**: 214 epochs, best val_auc=**0.7657** (epoch 153), stopped at overfitting plateau
- [x] Datasets reorganized: `datasets/echo/` (ECHO) + `datasets/deepantigen/` (paper)
- [ ] COVID-19 zero-shot evaluation (optional, 1.1M pairs)
- **Status:** complete — GCN-only ceiling ~0.77, ESM-only 0.81 remains superior

## Phases

### Phase 0: Bug Fixing
- [x] Fix `structure_losses.py`: `view` → `reshape` for non-contiguous tensors
- [x] Fix `model.py`: Stage 2 `_compute_structure_loss` 1D logits crash
- [x] Fix `train_structure.py`: missing optimizer in `finetune_stage1/2`
- [x] Fix `train_structure.py`: Stage 2 optimizer param scope (frozen GCN)
- [x] Fix `model.py`: unused `fusion_gcn` parameter
- [x] Fix `dataset.py`: strip trailing `;` and filter bad amino acids (O, X)
- **Status:** complete

### Phase 1: GCN Gradient Fix
- [x] Root cause: `F_spatial` ignored by classifier → GCN receives zero gradient
- [x] Solution: `gcn_aux_head` — auxiliary GCN-only classifier with focal loss
- [x] Add `lambda_gcn_aux` hyperparameter to `Model.__init__`
- [x] Sanity check (100 samples): 100% accuracy, all GCN gradients flow
- **Status:** complete

### Phase 2: Joint Pretraining
- [x] Config: `config_gcn.yaml` — ESM LoRA + GCN full params + λ_gcn_aux=1.0
- [x] Data: Panpep_trainingData.csv → 52,562 train / 9,276 val
- [x] Graph cache: 64,524 precomputed graphs (32 workers, 30s)
- [x] DataLoader optimizations: batch_size=128, num_workers=8, GPU 98%
- [x] Run 50 epochs on RTX 5090 (completed 2026-05-15 02:30)
- [x] Best val AUC: 0.7294 (epoch 42)
- [x] **Test AUC: 0.7914, Accuracy: 0.7262, F1: 0.7115** (majority_testing_dataset)
- **Status:** complete

### Phase 3: Structure Fine-Tuning
- [x] Fix checkpoint loading: `model_state` key compatibility
- [x] Config `echo_pretrained` → `runs/gcn_joint/best_model.pth`
- [x] Stage 1: Pearson correlation regression (works, TopK contact rate 0.97%→9.86%)
- [x] Fix: replace `cat([-x,+x])` with `atom_contact_head` (deepAntigen-aligned)
- [x] **Refactoring 2026-05-16: Zero-Leak Architecture**
  - [x] Per-fold fresh checkpoint load — no weight/optimizer/LR state carryover
  - [x] `del` + `gc.collect()` + `torch.cuda.empty_cache()` per fold
  - [x] `copy.deepcopy(dataset)` per fold — PyG graph pointer isolation
  - [x] Multi-branch `atom_contact_head`: contact (128→256→128→64→2) + distance (128→128→1)
  - [x] `lambda_distance=5.0` auxiliary MSE loss
- [x] **Refactoring 2026-05-16: Stage Isolation & Scheduler Fix**
  - [x] Stage 1→2 hard boundary: `del stage1_opt/sched` + `torch.cuda.empty_cache()`
  - [x] Stage 1: fixed LR 1e-4 (CosineAnnealingLR removed)
  - [x] Stage 2: ReduceLROnPlateau only
  - [x] Stage 2: `set_stage(2)` → `freeze_encoder()` → TopK+MHA jointly fine-tuned
  - [x] Stage 2 optimizer: atom_contact_head + TopK+MHA only
  - [x] `gcn_spatial_proj` reverted to `Linear(128→1280)` — matches Phase 2 checkpoint
- [x] **Refactoring 2026-05-17: Performance & Architecture Hardening**
  - [x] `PDBStructureDataset` full pre-cache: `_process_single_item()` in `__init__`, `__getitem__` O(1)
  - [x] `generate_contact_labels`/`generate_mask` GPU vectorization
  - [x] Stage 1 joint_scores: Python loop → single `torch.mm` + broadcast mask
  - [x] `evaluate_stage2()`: `@torch.no_grad()`, `model.eval()`, ReduceLROnPlateau on val_loss
  - [x] Focal Loss α=0.995 standard semantics, sigmoid-mul spatial gate, Distance MSE disabled
- [x] **Refactoring 2026-05-17: Cross-Graph Padding & Gradient Scale (this session)**
  - [x] TopKPooling: `cat+reshape` → per-graph `split→pad→stack` + `valid_mask [B,k]`
  - [x] DeepGCN: `joint_mask [B,k,k,1]` injected into interaction_map, ghost atoms zeroed
  - [x] Stage 2 loss: `reduction='sum'/batch_size` — restores gradient scale ~90× vs mean
  - [x] Scheduler: `threshold_mode='rel'`, `threshold=0.001`, `factor=0.75`
  - [x] Early stopping: patience counter with collapse guard (`train < 0.01×initial`)
  - [x] Lightweight checkpoint: only `requires_grad=True` params saved
  - [x] `weight_decay` reduced to 1e-6 for Stage 2
- [x] **133-fold LOGO CV completed 2026-05-18**
  - [x] 95/133 folds stopped early (patience=80), 38/133 ran full 200 epochs
  - [x] Best val loss range: 0.0000–0.3708 (mean 0.090)
  - [x] Checkpoints in `runs/structure/{pdb}/atom-level_parameters.pt` (133 total, 8GB)
- [ ] Evaluate structure-fine-tuned model on zero-shot peptides
- **Status:** 133-fold training complete, evaluation pending

### Phase 4: Full Evaluation
- [x] Benchmark on majority_testing_dataset.csv (seen peptides) → AUC 0.7914
- [ ] Benchmark on zero_dataset.csv (unseen peptides)
- [ ] Compare: ESM-only vs ESM+GCN spatial bias
- [ ] Ablation: λ_gcn_aux sweep
- **Status:** partial (majority test done)

### Phase 5: Architecture Refactoring (2026-05-19) — COMPLETE
- [x] **5a. TopK all-atom selection** — removed N/O constraint in `gcn_components.py`
- [x] **5b. Spatial aggregation optimization** — `model.py`
- [x] **5c. GCN projection upgrade** — `model.py`
- [x] **5d. Cross-Modal Gated Fusion** — `model.py`
- [x] **5e. Loss weight cosine annealing** — `model.py`, `train.py`, `config_gcn.yaml`
- **Status:** complete

### Phase 6: ESM-Only Baseline (2026-05-20) — COMPLETE ✅
- [x] Create `configs/config_esm_only.yaml` — pure language track, use_gcn=false
- [x] **DEBUG 2026-05-20**: Identified 3 critical issues vs ECHO-deepantigen reference
  1. Cross-attention: mix-then-normalize → normalize-then-mix
  2. ESM projections removed for ESM-only (2560-dim classifier input)
  3. Dropout 0.08→0.3, cross_attn_dropout=0.3, weight_decay 5e-5→1e-4
- [x] **Train 91 epochs (early stop at 90), ~5 min/epoch, 13.9 GB VRAM**
- [x] **BEST RESULT: val AUC 0.7671, Test AUC 0.8371, Acc 0.7730, F1 0.7519**
- [x] Checkpoint: `runs/esm_only/best_model.pth`

### Phase 7: Modular Refactoring (2026-05-21) — COMPLETE ✅
- [x] Clean `model.py`: remove all GCN code/imports, pure ESM-only (260 lines)
- [x] Create `gcn_plugin.py`: GCNPlugin(Model) with Track 2 physics
  - Inherits ESM encoders, cross_attn, loss functions via super().__init__()
  - Adds: AtomDeepGCN, projections (1280→512), gated fusion, aux head
  - Replaces classifier (1280-dim input)
- [x] Modify `train.py`: dynamic `ModelClass = GCNPlugin if use_graph else Model`
- [x] `dataset.py`, `utils.py`, `attentions.py`: unchanged
- **Status:** complete

### Known Defect
- [ ] **Checkpoint save filter leaks params**: `cross_attn`, `classifier`, `gcn_aux_head` not in Stage 2 optimizer but `requires_grad=True` → saved (62MB vs expected 5MB). Fix: filter by `stage2_opt.param_groups` IDs.

## Key Questions
1. Does spatial gate (sigmoid-mul) improve Stage 2 contact prediction convergence?
2. Does joint TopK+MHA fine-tuning in Stage 2 improve over frozen TopK?
3. Does structure fine-tuning improve zero-shot peptide generalization?
4. Can distance regression be re-enabled after contact classification converges?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| LoRA r=8 on ESM layers [32-28] | Memory efficiency — 650M model frozen, ~2M trainable params |
| GCN full-param training in Phase 2 | Joint pretraining requires learning GCN from scratch |
| λ_gcn_aux=1.0 | Maintains gradient flow without dominating main loss |
| Dual ESM-2 encoders (not weight-tied) | TCR and peptide are different domains |
| LOGO CV for structure fine-tuning | Standard deepAntigen protocol: 133 folds |
| Independent 2-class atom_contact_head | `cat([-x,+x])` forced anti-correlation, crippling asymmetric decisions |
| focal_alpha=0.995 | Standard Focal Loss: 22:1 pos:neg weight ratio |
| Per-fold fresh checkpoint load | Eliminates cross-fold weight/optimizer/LR state leakage |
| `copy.deepcopy(dataset)` per fold | Severs PyG graph object pointer chains across folds |
| Stage 2: ReduceLROnPlateau only | CosineAnnealingLR + ReduceLROnPlateau on same optimizer causes LR cliff-drop |
| Stage 2: freeze_encoder (not freeze_topk) | TopK+MHA jointly fine-tuned with atom_contact_head |
| Sigmoid-mul spatial gate | Eliminates `.abs()` NaN risk, addition "averaging" paradox |
| Distance MSE disabled | Gradient conflict with Focal loss on shared interaction_map |
| Per-graph split→pad→stack in TopKPooling | `cat+reshape` caused cross-graph atom misalignment |
| `reduction='sum'/batch_size` in Stage 2 | Restores gradient scale; Adam ε floor would dominate mean-scale gradients |
| `threshold_mode='rel'` for scheduler | Absolute threshold=0.01 blinded on folds with val<0.01 |
| Collapse guard: train<0.01×initial | Relative threshold adapts to each fold's loss scale |
| weight_decay=1e-6 for Stage 2 | Focal loss converges rapidly; L2 regularization dominates late training |
| **TopK all-atom selection** | N/O constraint excluded C,S,H atoms from attention; all-atom gives model full chemical vocabulary |
| **Flatten + masked Max/Avg pool** | sum(dim=(1,2)) destroyed k×k spatial topology; Max captures strongest contact, Avg preserves background |
| **GCN 256→512→256 residual MLP** | Single Linear(128→1280) was 10× sparse expansion; residual MLP gives smooth manifold learning |
| **Decoupled language gate** | Shared W_lang on summed TCR+pep forced identical filtering; cat→split gives independent TCR/PEP gates |
| **Cross-modal physics gate** | Self-gating (F_spatial → W_phys) was blind to language context; ctx_lang injection makes it truly cross-modal |
| **ESM proj 1280→512 + LayerNorm** | Raw 1280-dim dominated fusion (67% vs 33%); 512=512+256 balances modalities |
| **Identity shortcut in GCN proj** | 256→256 Linear shortcut was redundant 65K params; identity preserves gradient perfectly |
| **Cosine annealing λ_gcn_aux 1.0→0.1** | Fixed λ=1.0 over-emphasized aux task late in training; annealing lets focal loss dominate after cold start |
| **Cosine annealing λ_int 2.0→0.5** | Fixed λ=2.0 dominated language feature space; decay gives classifier more signal later |
| **set_seed(42) everywhere** | Prior runs had no random seed — results were not reproducible; seed now configurable via `random_seed` in YAML |
| **Val always from train CSV (15% stratified)** | Eliminates pre-split val_csv inconsistency; guarantees val/test distribution gap is controlled |
| **SeqAlignedGCN replacing DeepGCN** | Per-layer SuperNodeExchange limits depth to 2 and adds 164K params of cross-modal overhead; independent encoders enable depth=5 matching paper |
| **Ablation: GCN bias path is dead** | τ=0 prediction = τ=0.227 prediction (correlation=1.0); 1-bit label can't supervise 128×100 atom-pair features — signal diluted 12,800× |

## Errors Encountered (cumulative)
| # | Error | Attempt | Resolution |
|---|-------|---------|------------|
| 24 | GCN overfitting (val AUC plateaus ~ep150, train_loss continues -60%) | 3 | (1) dropout_atom + wd, (2) FocalLoss + small classifier + low LR + 10% val |
|---|-------|---------|------------|
| 1 | `roc_auc_score` on 0 samples (val set empty) | 1 | Regenerated `val_joint.csv` from 85/15 split |
| 2 | `NameError: name 'os' is not defined` | 1 | Added `import os` to dataset.py |
| 3 | Graph precomputation ~90min single-thread | 1 | Multiprocessing Pool(32) → 30s |
| 4 | Checkpoint `model_state` vs `model` key | 1 | Fallback logic in `_run_structure_training` |
| 5 | Stage 2 loss stuck at 5.6 | 1 | `cat([-x,+x])` → independent `atom_contact_head` |
| 6 | Cross-fold loss 阴跌 (0.88→0.03) | 2 | Per-fold fresh checkpoint + del/gc/cuda_cache + deepcopy |
| 7 | Stage 2 loss plateau at 3.1 | 2 | Multi-branch head (10× capacity) + focal_alpha→0.995 |
| 8 | Dual scheduler LR cliff-drop | 1 | Remove CosineAnnealingLR from Stage 2 |
| 9 | Dead params in Stage 2 optimizer | 1 | Remove classifier/cross_attn/gcn_spatial_proj |
| 10 | `.abs()` NaN risk + non-differentiable at 0 | 1 | Eliminate temperature; sigmoid-mul gate |
| 11 | Addition "averaging" (15-15=0→0.5) | 1 | Sigmoid-bound → multiplication = logical AND |
| 12 | Focal/MSE gradient conflict | 1 | Disable distance MSE branch |
| 13 | Non-standard Focal alpha semantics | 1 | Fixed to standard: α_t=α for pos, 1-α for neg |
| 14 | DataLoader pickle bottleneck (840×) | 1 | Pre-cache all items as tuples in `__init__` |
| 15 | GPU vectorized loss causes training collapse | 1 | Reverted to CPU-vectorized versions |
| 16 | `gcn_spatial_proj` documentation error | 1 | Reverted to `Linear(128→1280)` — ESM hidden=1280 |
| 17 | Cross-graph atom misalignment | 1 | `cat+reshape` → per-graph split/pad/stack |
| 18 | Ghost atoms leak through spatial gate | 1 | `joint_mask [B,k,k,1]` injected in DeepGCN + loss filter |
| 19 | Mean reduction causes gradient collapse | 1 | `reduction='sum'/batch_size` — ~90× gradient restoration |
| 20 | Scheduler threshold blindness (abs 0.01 on val~0.003) | 1 | `threshold_mode='rel'`, threshold=0.001 |
| 21 | Collapsed model resets patience counter | 1 | Collapse guard: train<0.01×initial ineligible for saving |
| 22 | LR floor unreachable within 200 epochs | 1 | Replaced with patience-based early stopping (patience=80) |
| 23 | Checkpoint 62MB instead of 5MB | 0 | Save filter leaks cross_attn/classifier — not yet fixed |

## Notes
- Graph cache: `datasets/panpep/graph_cache/` (64,524 pickle files)
- Phase 2 checkpoint: `runs/gcn_joint/best_model.pth` (5.4 GB)
- Phase 3 checkpoints: `runs/structure/{pdb}/atom-level_parameters.pt` (133 files, 8GB total)
- Phase 3 log: `runs/structure/training.log`
