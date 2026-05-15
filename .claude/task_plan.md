# Task Plan: TCR-ECHO Dual-Model Fusion
<!--
  WHAT: Roadmap for fusing ESM-2 language model with deepAntigen GCN for TCR-peptide binding prediction.
  WHY: Complex multi-phase ML project spanning 50+ tool calls. This file keeps goals fresh.
  WHEN: Created 2026-05-13. Update after each phase completes.
-->

## Goal
Train a dual-track TCR-peptide binding predictor fusing ESM-2 protein language model (Track 1) with deepAntigen atom-level GCN (Track 2), achieving SOTA binding prediction on majority and zero-shot peptide sets.

## Current Phase
Phase 2 — Joint Pretraining (training launched 2026-05-14)

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
- [x] Test: majority_testing_dataset.csv (5,230 seen peptides)
- [x] Graph cache: 64,524 precomputed graphs (32 workers, 30s via multiprocessing)
- [x] Dataset loads from cache (train: 51,022 graphs, val: 9,402, 0 missed)
- [x] Safety: empty dataloader check in train.py + empty preds guard in utils.py
- [x] Per-epoch logging with GPU info (PID, util, VRAM, temp) added to train.py
- [x] DataLoader optimizations: num_workers=8, pin_memory=True, non_blocking=True, persistent_workers=True, prefetch_factor=4
- [x] batch_size: 6→32→128→64→128 (final: 128, VRAM 11.2/32 GB, GPU util 98%)
- [ ] Run 50 epochs on RTX 5090 ← about to relaunch with bs=128
- [ ] Monitor GCN gradient health each epoch
- [ ] Save best checkpoint by val AUC
- **Status:** in_progress (optimizing throughput)

### Phase 3: Structure Fine-Tuning
- [ ] Stage 1: PDB data → train TopK pooling with Pearson correlation loss
- [ ] Stage 2: Freeze GCN, train spatial_proj + classifier with contact loss
- [ ] Evaluate on zero-shot peptide set
- **Status:** pending

### Phase 4: Full Evaluation
- [ ] Benchmark on majority_testing_dataset.csv (seen peptides)
- [ ] Benchmark on zero_dataset.csv (unseen peptides)
- [ ] Compare: ESM-only vs ESM+GCN spatial bias
- [ ] Ablation: λ_gcn_aux sweep
- **Status:** pending

## Key Questions
1. Is λ_gcn_aux=1.0 optimal for joint training, or does it need tuning?
2. Does GCN gradient health remain stable across 50 epochs?
3. Can batch_size be increased beyond 6 given 6.4 GB / 32 GB used?
4. Do 52k training samples per epoch make training time impractical (~2-4h/epoch)?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| LoRA r=8 on ESM layers [32-28] | Memory efficiency — 650M model frozen, only ~2M trainable params |
| GCN full-param training in Phase 2 | Joint pretraining requires learning GCN from scratch |
| λ_gcn_aux=1.0 | Sanity check: λ=1 maintains gradient flow without dominating main loss |
| batch_size=6 | Conservative; 6.4/32 GB used suggests room to increase |
| Graph cache via multiprocessing | 51k graphs: single-threaded ~90min → 32 workers = 30s (180× speedup) |
| Worker-direct disk write pattern | Eliminates main-process serialization bottleneck |
| Dual ESM-2 encoders (not weight-tied) | TCR and peptide are different domains; separate encoding beneficial |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `roc_auc_score` on 0 samples (val set empty) | 1 | `val_joint.csv` was 0 rows; regenerated from `Panpep_trainingData.csv` 85/15 split |
| `roc_auc_score` on 0 samples (val set empty) | 2 | Added empty-preds guard in `utils.py:compute_metrics` + empty-loader check in `train.py` |
| `NameError: name 'os' is not defined` in dataset.py | 1 | Added `import os` at top of dataset.py (used `os.path.exists` in `__init__`) |
| Graph precomputation single-threaded ~90min | 1 | Multiprocessing Pool(32) — workers write files directly to disk |
| Old precompute script cached/executed instead of new | 1 | Cleaned cache dir, verified file content, relaunched |
| `skills` field not recognized in settings.json | 1 | Replaced with `extraKnownMarketplaces` + `enabledPlugins` (plugin system) |

## Notes
- Graph cache at `datasets/panpep/graph_cache/` — 64,524 pickle files, 30s load time
- Training log at `runs/gcn_joint/training.log` — no per-batch output (silent during epoch)
- GPU 0: 6.4 GB / 32 GB, first epoch est. 2-4 hours (8760 batches × ~1-2s each)
- Superpowers + planning-with-files plugins configured but need session restart
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions (attention manipulation)
- Log ALL errors — they help avoid repetition
