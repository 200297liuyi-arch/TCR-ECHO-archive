# Progress Log
<!--
  WHAT: Session log — chronological record of what was done, when, and what happened.
  WHY: Answers "What have I done?" in the 5-Question Reboot Test. Helps resume after breaks.
  WHEN: Update after completing each phase or encountering errors.
-->

## Session: 2026-05-13 (continued 2026-05-14)

### Phase 2: Joint Pretraining — Launch
- **Status:** in_progress
- **Started:** 2026-05-13 11:07 (first attempt) / 2026-05-14 11:21 (current launch)
- Actions taken:
  - Diagnosed empty val set crash (`val_joint.csv` had 0 rows; regenerated from 85/15 split on `Panpep_trainingData.csv`)
  - Added safety guard in `utils.py:compute_metrics` for 0-sample predictions
  - Added empty-dataloader check in `train.py:train_one` before epoch loop
  - Diagnosed `NameError: name 'os' is not defined` — added `import os` to dataset.py
  - Discovered graph precomputation takes ~90 min single-threaded (51k unique TCR sequences)
  - Built multiprocessing `precompute_graphs.py`: Pool(32), workers write files directly to disk
  - Precomputed 64,524 graphs in ~30 seconds (0 failures across all 3 datasets)
  - Modified `dataset.py` to support `graph_cache_dir` — loads graphs from pickle cache <1s
  - Added `graph_cache_dir` to `configs/config_gcn.yaml`
  - Fixed `extraKnownMarketplaces` + `enabledPlugins` in `~/.claude/settings.json` (was invalid `skills` field)
  - Rewrote `.claude/task_plan.md`, `.claude/findings.md`, `.claude/progress.md` in planning-with-files format
  - Launched 50-epoch training (PID 964366, GPU 0, 6.4/32 GB)
- Files created/modified:
  - `precompute_graphs.py` (created) — multiprocessing graph precomputation with disk cache
  - `datasets/panpep/graph_cache/` (created) — 64,524 pickle files
  - `dataset.py` (modified) — graph_cache_dir, os import, lazy load from cache
  - `utils.py` (modified) — empty-predictions guard in compute_metrics
  - `train.py` (modified) — empty-dataloader check, cache_dir passthrough
  - `configs/config_gcn.yaml` (modified) — graph_cache_dir field
  - `~/.claude/settings.json` (modified) — plugin system config
  - `.claude/task_plan.md` (rewritten) — planning-with-files template
  - `.claude/findings.md` (rewritten) — planning-with-files template
  - `.claude/progress.md` (rewritten) — planning-with-files template

### Phase 2: Training Monitor
- **Status:** in_progress
- **Started:** 2026-05-14 11:21
- Observations:
  - Dataset loads from cache: train 51,022 graphs + val 9,402 graphs, 0 missed
  - ESM-2 650M ×2 loaded with LoRA r=8, GCN full params
  - Initial: batch_size=6, GPU 6.4 GB, util 21-36%, ~2-4h/epoch
  - After optimization: batch_size=128, GPU 11.2 GB, util **98%**, ~6.5 min/epoch
  - AUC progress: 0.6782(bs=128)→0.6889(bs=128, reached before shared memory crash)
- Pending:
  - Relaunch with bs=128 + pickle-bytes dataset (fixes mmap shared memory OOM)
  - Confirm 98% GPU util stable across epochs
  - Complete full 50-epoch run
  - Monitor GCN gradient health

### Phase 2: GPU Utilization Optimization (2026-05-14 afternoon)
- **Root cause of low GPU util (19-36%)**: num_workers=0 (default) — single-threaded tokenization; batch_size=6 too small; no pin_memory/non_blocking
- **Optimizations applied:**
  1. batch_size: 6 → 16 → 32 → 128 (VRAM: 6.4→7.1→11.2 GB)
  2. DataLoader: num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
  3. `.to(device, non_blocking=True)` for async GPU transfer
  4. Per-epoch `print` with PID + GPU utilization/memory/temperature via nvidia-smi
- **Result**: GPU util 12-53% → **98-99%**, epoch time 2-4h → **6.5 min** (~20-40× speedup)
- **Shared memory OOM** (bs=128): PyG Data objects stored directly caused mmap IPC overflow. Fixed by keeping pickled bytes in dataset, unpickling in collate_fn.
- **Final config for relaunch**: bs=128, pickle-bytes dataset, 8 workers, all optimizations active

## Test Results
<!-- Update as training progresses -->
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Dataset load (train) | cache dir | 51,022 graphs loaded | 51,022 loaded, 0 missed | ✓ |
| Dataset load (val) | cache dir | 9,402 graphs loaded | 9,402 loaded, 0 missed | ✓ |
| Graph precompute (3 datasets) | 32 workers | All graphs built, 0 failures | 64,524 files, 0 skipped | ✓ |
| ESM model load | offline HF cache | Model loads on GPU | 6.4 GB GPU, weights ok | ✓ |
| Training epoch 1 | 52,562 samples, 50 epochs | Running (no crash) | In progress | ⟳ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-13 11:50 | `ValueError: 0 sample(s)` in roc_auc_score | 1 | Regenerated val_joint.csv from 85/15 split |
| 2026-05-13 11:50 | `ValueError: 0 sample(s)` in roc_auc_score | 2 | Added empty-preds guard in utils.py |
| 2026-05-14 11:21 | `NameError: name 'os' is not defined` | 1 | Added `import os` to dataset.py |
| 2026-05-14 11:21 | Graph precompute ~90 min single-threaded | 1 | Multiprocessing Pool(32), worker-direct writes → 30s |
| 2026-05-14 11:21 | Old precompute script output (stale .pyc?) | 2 | Cleaned cache dir, verified file, relaunched |
| 2026-05-13 ~15:00 | `skills` field not recognized | 1 | Replaced with `extraKnownMarketplaces` + `enabledPlugins` |
| 2026-05-14 13:14 | `shape '[6, 10, -1]' invalid for input of size 7552` in TopKPooling | 1 | `tcr_CAFF` has only 9 N/O atoms < k=10; padded `x_top`+`perm` to fixed size |
| 2026-05-13 ~15:00 | `Skill` tool returns "Unknown skill" | 1 | Superpowers needs session restart after settings fix |
| 2026-05-14 17:30 | `RuntimeError: unable to mmap` in DataLoader IPC | 1 | PyG Data objects returned from __getitem__ overflowed /dev/shm via mmap; reverted to pickle bytes storage + unpickle in collate_fn |
| 2026-05-14 ~17:00 | GPU utilization only 12-53% | 3 | ① batch_size 6→128 ② num_workers=8 + pin_memory + non_blocking + persistent_workers ③ per-epoch logging — result: 98% GPU util |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 2 — Joint Pretraining (bs=128, GPU 98% util, about to relaunch) |
| Where am I going? | Phase 3 (Structure Fine-Tuning) → Phase 4 (Full Evaluation) |
| What's the goal? | Train dual-track ESM-2 + GCN TCR-peptide binding predictor |
| What have I learned? | Graph cache = 180× speedup; GCN gradients flow at λ=1.0; bs=128 + DataLoader opts → GPU 98% |
| What have I done? | Fixed bugs, built cache, optimized throughput 20-40×, ready for 50-epoch run |

---
*Update after completing each phase or encountering errors*
