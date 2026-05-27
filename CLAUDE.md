# TCR-ECHO: Dual-Model Fusion for TCR-Peptide Binding Prediction

## Environment
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
- GPU: RTX 5090 x3, 32GB each
- No internet — models from HF cache at `~/.cache/huggingface/hub/`
- ESM model cached: `esm2_t33_650M_UR50D` only

## Modular Architecture (2026-05-21 refactored)

```
model.py           # ECHO base: pure ESM-2 language model (260 lines)
gcn_plugin.py      # GCNPlugin(Model): extends base with Track 2 physics
train.py           # Dynamic ModelClass selection by config
```

### Track 1 (Language): ESM-2 — `model.py`
- ESM-2 ×2 (esm2_t33_650M_UR50D, not weight-tied) with LoRA r=8
- BidirectionalDualViewAttention — **normalize-then-mix** (aligned with ECHO paper)
- Mean pooling → tcr_pool [B,1280], pep_pool [B,1280]
- Direct cat: [B,2560] → Classifier (no projection bottleneck)
- Loss: Focal(γ=2.0) + Contrastive L_enc(λ=0.5) + Contrastive L_int(λ=2.0)

### Track 2 (Physics): Atom-level GCN — `gcn_plugin.py`
- **GCNPlugin(Model)** inherits all ESM components via super().__init__()
- Adds: AtomDeepGCN → TopK all-atom → MHA → interaction_map → masked pool → F_spatial
- ESM Projections (1280→512) + LayerNorm — only in GCN mode
- Cross-Modal Gated Fusion: decoupled language gate + cross-modal physics gate
- GCN Aux Loss (λ=1.0→0.1 cosine annealed)
- Replaces classifier: 1280-dim input (512+512+256)

### Classifier
- **ESM-only**: Linear(2560→512) + ReLU + Dropout(0.3) + Linear(512→1)
- **ESM+GCN**: Linear(1280→512) + ReLU + Dropout(0.3) + Linear(512→1)

## Current State (2026-05-21)

### ESM-Only Baseline (FIXED)
| Metric | Old (buggy) | New (fixed) | Δ |
|--------|------------|------------|---|
| Best val AUC | 0.7293 | 0.7671 | +3.8% |
| **Test AUC** | 0.7868 | **0.8371** | **+5.0%** |
| Test Accuracy | 0.7159 | 0.7730 | +5.7% |
| Test F1 | 0.6882 | 0.7519 | +6.4% |
| Epochs trained | 50 | 91 (early stop) | — |
| Epoch time | ~3 min | ~5 min | bs=32 |
| Checkpoint | — | `runs/esm_only/best_model.pth` | — |

### 3 Critical Fixes (aligned with ECHO-deepantigen reference)
1. **Cross-attention normalize-then-mix**: softmax each view separately, then interpolate
2. **ESM projection removed**: full 2560-dim classifier input (no 1280→512 bottleneck)
3. **Dropout alignment**: 0.08→0.3, added cross_attn_dropout=0.3

### Modular Refactoring (2026-05-21)
- `model.py`: pure ESM-only, no GCN imports/dependencies
- `gcn_plugin.py`: GCNPlugin(Model) — all GCN code isolated in one file
- `train.py`: dynamic class selection → `ModelClass = GCNPlugin if use_graph else Model`

### Pending
- [ ] Phase 2 re-training with GCNPlugin (config_gcn.yaml ready)
- [ ] GCN-only training with paper data (depth=2, crashed, need fix)

### Zero-Shot Evaluation (2026-05-22)
| Model | Train Data | Test AUC | Test Acc | Test F1 |
|-------|-----------|----------|----------|---------|
| **ESM-only** | PanPep 52k | **0.8148** | **0.7474** | **0.7149** |
| GCN-only (depth=2) | PanPep 52k | 0.7445 | 0.6919 | 0.6637 |
| GCN-only (paper-aligned) | Paper 62k | ❌ collapsed | ❌ | ❌ |

### deepAntigen Paper Reference (Nature Communications 2025)
- Paper TCR zero-shot: 0.71 (COVID), 0.84 (Gao), PanPep baseline 0.51
- Our GCN-only 0.74 > PanPep baseline 0.51 (paper confirms PanPep is weak on zero-shot)
- Paper training data: `test_antigenTCR/Data/sequence/train.csv` (62,446 balanced)

### Phase 2 Results (Old Architecture, pre-refactor)
| Metric | Value |
|--------|-------|
| Best val AUC | 0.7294 |
| Test AUC | 0.7914 |
| Checkpoint | `runs/gcn_joint/best_model.pth` |

## Launch Commands
```bash
# ESM-only baseline
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_esm_only.yaml > runs/esm_only/training.log 2>&1 &

# Phase 2: Joint Pretraining (GCNPlugin)
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_gcn.yaml > runs/gcn_joint/training.log 2>&1 &
```

## Planning Files
- `.claude/task_plan.md` — full task plan with phases
- `.claude/progress.md` — progress log with issues encountered
- `.claude/findings.md` — all bugs found & fixed, architecture insights

## Bugs Fixed (cumulative, 15 items)
1. gcn_aux_head — GCN zero gradients
2. view→reshape — non-contiguous tensor crash
3. SuperNodeExchange MHA→Linear — degenerate with 1 super-node
4. Double MolFromSequence — shared mol object
5. Vectorized perm local indices — CPU-GPU sync removal
6. Data: strip `;`, filter bad AA (O,X)
7. focal_alpha 0.7→0.9→0.995
8. TopK all-atom selection
9. Masked spatial aggregation + residual MLP
10. Cross-Modal Gated Fusion
11. Loss weight cosine annealing
12. **Cross-attention normalize-then-mix** (2026-05-20) — critical perf fix
13. **ESM projection bottleneck removed** (2026-05-20) — critical perf fix
14. **Dropout alignment** (2026-05-20): 0.08→0.3
15. **Early stopping** (2026-05-20): patience=10
16. **Modular refactoring** (2026-05-21): Model base + GCNPlugin
