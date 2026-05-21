# Progress Log
<!--
  WHAT: Session log — chronological record of what was done, when, and what happened.
  WHY: Answers "What have I done?" in the 5-Question Reboot Test. Helps resume after breaks.
  WHEN: Update after completing each phase or encountering errors.
-->

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
