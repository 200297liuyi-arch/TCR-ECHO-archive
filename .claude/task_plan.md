# Task Plan: TCR-ECHO Dual-Model Fusion
<!--
  WHAT: Roadmap for fusing ESM-2 language model with deepAntigen GCN for TCR-peptide binding prediction.
  WHY: Complex multi-phase ML project spanning 50+ tool calls. This file keeps goals fresh.
  WHEN: Created 2026-05-13. Update after each phase completes.
-->

## Goal
Train a dual-track TCR-peptide binding predictor fusing ESM-2 protein language model (Track 1) with deepAntigen atom-level GCN (Track 2), achieving SOTA binding prediction on majority and zero-shot peptide sets.

## Current Phase
Architecture Refactoring (2026-05-19) → Phase 2 re-training with optimized model

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

### Phase 6: ESM-Only Baseline (2026-05-20) — COMPLETE
- [x] Create `configs/config_esm_only.yaml` — pure language track, use_gcn=false
- [x] Train 50 epochs on Panpep data (~3 min/epoch, 9.3 GB VRAM)
- [x] **Test AUC: 0.7868** — within 0.0046 of old ESM+GCN (0.7914)
- [x] Establishes: ESM language model carries ~99.4% of old model's predictive power
- [x] Confirms: GCN track was severely underutilized in old architecture (sum pool + cat fusion)

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

## Errors Encountered (cumulative)
| # | Error | Attempt | Resolution |
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
