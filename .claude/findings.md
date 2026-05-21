# Findings & Decisions
<!--
  WHAT: Knowledge base for TCR-ECHO. Stores all discoveries, decisions, and research.
  WHY: Context windows are limited. This file is external memory — persistent and unlimited.
  WHEN: Update after ANY discovery. Follow the 2-Action Rule.
-->

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
