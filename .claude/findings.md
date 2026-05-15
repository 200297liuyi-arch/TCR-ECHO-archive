# Findings & Decisions
<!--
  WHAT: Knowledge base for TCR-ECHO. Stores all discoveries, decisions, and research.
  WHY: Context windows are limited. This file is external memory — persistent and unlimited.
  WHEN: Update after ANY discovery. Follow the 2-Action Rule.
-->

## Requirements
- Fuse ESM-2 protein language model (Track 1) with deepAntigen atom-level GCN (Track 2)
- Predict TCR-peptide binding — binary classification
- Train on Panpep (52,562 train / 9,276 val), eval on majority + zero-shot test sets
- Pipeline: joint pretraining → structure fine-tuning → full evaluation
- Environment: RTX 5090 ×3 (32GB each), offline, PyTorch 2.x

## Research Findings

### ESM-2 650M Memory Footprint & Batch Size Scaling (2026-05-14)
- Single ESM-2 650M in fp16: ~1.3 GB; two copies (esm1 + esm2): ~2.6 GB
- With LoRA r=8: optimizer states ~0.1 GB (only LoRA params trained)
- GCN + classifier + activation memory: ~3-4 GB
- VRAM scaling: bs=6→6.4GB, bs=32→7.1GB, bs=64→8.2GB, bs=128→11.2GB
- 32 GB card has room for bs=256+ but IPC mmap limits hit at bs=128 with raw Data objects
- bs=128 achieves **98% GPU utilization** on RTX 5090
- Epoch time: bs=6→2-4h, bs=128→**6.5 min** (~20-40× throughput improvement)

### Graph Precomputation Performance (2026-05-14)
- 51,022 unique sequences (50,814 TCR + 208 peptides) — single-threaded: ~90 min
- Multiprocessing Pool(32), workers write directly to disk: 30 seconds (all three datasets)
- Speedup: ~180× — eliminated main-process IPC serialization bottleneck
- 64,524 total cache files: train 51,022 + val 9,402 + test 5,074 (with dedup), 0 failures
- Dataset load from cache: <1 second (was 14+ minutes)

### GCN Gradient Starvation in Late Fusion (Phase 1)
- **Discovery**: Sanity check (100 samples, λ=0): 91% accuracy but ALL GCN gradients = 0
- **Root cause**: `cat(tcr_pool[1280], pep_pool[1280], F_spatial[1280])` → ESM 2560-d dominates F_spatial 1280-d; classifier ignores GCN path
- **Solution**: `gcn_aux_head` — auxiliary GCN-only classifier (`F_spatial_raw → MLP → logit`) with focal loss, weighted by λ_gcn_aux
- **Verification (λ=10)**: 100% accuracy, all GCN layer gradients flow (norm 0.01–1.5)
- **Verification (λ=1)**: MultiHeadAttention grad 0.08, TGCN grad 0.01–0.05 — sufficient
- **Decision**: λ=1.0 for joint pretraining

### TopK Gradient Asymmetry (Phase 1)
- **Observation**: `topk_pep` gradients (0.07–0.21) ≈ 1000× stronger than `topk_tcr` (5×10⁻⁵)
- **Hypothesis**: TCR sequences produce more atoms → k=10 from larger pool is noisier → smaller per-atom gradient
- **Impact**: TCR atom selection underfits; future work: separate k values or gradient rebalancing

### Activation Norm Progression (Phase 1)
- Layer 0: GRU norms 25–52, SuperNodeExchange norm 17 (well-behaved)
- Layer 1: GRU norms 219–281, SuperNodeExchange norm 184 (growing but stable)
- No exploding activations across 300 epochs in sanity check

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| LoRA on ESM, not full fine-tune | 650M model too large; LoRA r=8 gives ~2M trainable params |
| Dual ESM encoders (not shared) | TCR and peptide are different biochemical domains |
| GCN full-param training in Phase 2 | Joint pretraining requires learning GCN from scratch |
| gcn_aux_head auxiliary loss | Fixes gradient starvation; verified gradients flow at λ=1.0 |
| SuperNodeExchange MHA → Linear | 1 super-node: softmax(1×1) ≡ 1.0; MHA degenerates to Linear |
| Precompute graphs + disk cache | 51k unique seqs; build once, load instantly |
| Workers write files directly | Eliminates IPC serialization bottleneck in multiprocessing |
| batch_size=128 final | Scaled 6→16→32→128; 11.2/32 GB VRAM, 98% GPU util; pickle bytes to avoid mmap IPC OOM |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Empty validation set (0 samples → roc_auc_score crash) | Regenerated val_joint.csv from 85/15 split; added safety guards in utils.py + train.py |
| `NameError: name 'os' is not defined` in dataset.py | Added `import os` at top of dataset.py |
| Graph precomputation single-threaded ~90 min | Multiprocessing Pool(32) with worker-direct disk writes → 30s |
| Old precompute script cached/executed | Cleaned cache dir, verified file content on disk, relaunched |
| `skills` field not in Claude Code settings schema | Replaced with `extraKnownMarketplaces` + `enabledPlugins` |
| Superpowers plugin not loaded this session | Settings change requires session restart to take effect |
| Training log silent during epoch | No per-batch logging in `train_one`; first output after epoch 1 validation |
| TopKPooling reshape crash (N/O < k) | `tcr_CAFF` has 34 atoms but only 9 N/O; `topk` clamped k→9; `reshape(bz,10,-1)` failed. Fix: zero-pad to `bz*k` in `TopKPooling.forward` |
| Multiprocessing main-process bottleneck (102% CPU) | Worker → main IPC serialization was bottleneck; fixed by workers writing files directly |
| GPU utilization 12-53% (bs=6/32) | Added num_workers=8, pin_memory, non_blocking, persistent_workers, prefetch_factor=4; bs increased 6→128; result: 98% GPU util |
| DataLoader mmap OOM (bs=128 + raw Data objects) | Returning PyG Data objects from __getitem__ caused mmap fd overflow in IPC; fixed by storing pickled bytes + unpickling in collate_fn |
| Per-epoch output missing (training log silent) | train.py had no print() in training loop — only wandb logging; added print with timestamp, PID, GPU util/VRAM/temp |

## Code Review Fixes
| Fix | File | Root Cause |
|-----|------|------------|
| `view()` → `reshape()` | structure_losses.py | Non-contiguous tensor after `.expand()` / `.transpose()` |
| 2-class logit dimension | model.py | `flat_map.mean(dim=-1)` produced scalar, not 2-class |
| Missing optimizer param | train_structure.py | `model.optimizer` never set on Model |
| Frozen param optimizer | train_structure.py | Stage 2 built optimizer from frozen GCN params |
| `aux_logits` NameError risk | model.py | Variable unbound when `use_gcn=False` |
| Dead `positive_loss` code | model.py | Computed but never used |
| MHA degeneracy (1 super-node) | gcn_components.py | softmax(1×1) ≡ 1.0, equivalent to Linear |
| Double MolFromSequence | graph_utils.py + dataset.py | `sequence_to_mol()` then `sequence_to_graph()` each parsed |
| CPU-GPU sync in perm indices | gcn_components.py | O(B×k) `.item()` calls per forward |

## Resources
- ESM-2 cache: `~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/`
- Graph cache: `datasets/panpep/graph_cache/` (64,524 pickle files)
- Config: `configs/config_gcn.yaml`
- Log: `runs/gcn_joint/training.log`
- Checkpoint dir: `runs/best_model_gcn_joint/`
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
- GPU: RTX 5090 ×3 (GPU 0 used for training)

## Visual/Browser Findings
<!-- CRITICAL: Update after every 2 view/browser operations -->
- nvidia-smi (epoch 1): GPU 0: 6.4 GB / 32 GB, 21-36% util, 42°C
- pstree: Python process ~80 threads (DataLoader workers + wandb service)
- System: 251 GB RAM, 22 GB used, 227 GB available, 40 CPU cores
