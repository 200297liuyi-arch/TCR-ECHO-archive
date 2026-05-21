# TCR-ECHO: Dual-Model Fusion for TCR-Peptide Binding Prediction

## Environment
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
- GPU: RTX 5090 x3, 32GB each
- No internet — models from HF cache at `~/.cache/huggingface/hub/`
- ESM model cached: `esm2_t33_650M_UR50D` only

## Architecture (2026-05-19 refactored)
Dual-track model fusing:

### Track 1 (Language): ESM-2
- ESM-2 ×2 (esm2_t33_650M_UR50D, not weight-tied) with LoRA r=8
- BidirectionalDualViewAttention (sequence + Atchley biophysical dual-view)
- **Normalize-then-mix**: softmax(S_seq) and softmax(S_bio) separately, then interpolate
- Mean pooling → tcr_pool [B,1280], pep_pool [B,1280]
- **ESM Projection (GCN mode only)**: Linear(1280→512) + LayerNorm → tcr_proj, pep_proj
- **ESM-only mode**: NO projection — full 1280-dim fed directly to classifier (2560-dim input)

### Track 2 (Physics): Atom-level GCN
- RDKit Mol → atom graphs (25-dim node, 11-dim edge features)
- DeepGCN (depth=2, hidden=128): TGCN → SuperNodeExchange → GRU → BatchNorm per layer
- **TopKPooling**: all-atom scoring (was N/O-only), k=10, per-graph split/pad/stack + valid_mask
- MultiHeadAttention → interaction_map [B,10,10,128]
- **Spatial Aggregation**: flatten [B,100,128] → masked Max/Avg pool → F_spatial_raw [B,256]
- **GCN Projection**: 256→512→256 residual MLP + LayerNorm → F_spatial [B,256]

### Cross-Modal Gated Fusion (replaces simple cat)
- **Decoupled Language Gate**: cat(tcr_proj, pep_proj) [1024] → MLP(1024→256→1024) → Sigmoid → split → W_tcr, W_pep
- **Cross-Modal Physics Gate**: ctx_lang(512→128) + F_spatial(256) → MLP(384→64→256) → Sigmoid → W_phys
- **Gated**: gated_tcr ⊙ W_tcr, gated_pep ⊙ W_pep, gated_phys ⊙ W_phys
- cat(gated_tcr, gated_pep, gated_phys) → [B, 1280] → classifier

### Classifier
- **ESM-only**: Linear(2560→512) + ReLU + Dropout(0.3) + Linear(512→1) → logit
- **ESM+GCN**: Linear(1280→512) + ReLU + Dropout(0.3) + Linear(512→1) → logit

### Loss Functions
- Focal Loss (γ=2.0, α=class_balance): main classifier
- Contrastive L_enc (λ=0.5): raw ESM embeddings
- Contrastive L_int (λ=2.0→0.5 cosine annealed): pooled cross-attn features
- GCN Aux Loss (λ=1.0→0.1 cosine annealed): auxiliary GCN-only head
- **Loss Annealing**: cosine decay over training, configurable per-weight

## Key Parameters
- ESM: LoRA r=8, layers [32,31,30,29,28]
- GCN: hidden=128, depth=2, k=10, heads=4
- ESM proj: 1280→512
- GCN proj: 256→512→256 (identity residual)
- Fusion: 512+512+256=1280 (was 3840)
- focal_alpha: 0.995 (standard Focal Loss)

## Current State (2026-05-20) — 3 Critical Fixes Applied
- **Phase 2**: Complete — 50 epochs, val AUC 0.7294, test AUC 0.7914 (old architecture)
- **Phase 3**: Removed — structure fine-tuning degraded binding prediction
- **Architecture Refactoring (2026-05-19)**: 4 major optimizations applied
- **ESM-only baseline (2026-05-20, OLD)**: 50 epochs, val AUC 0.7293, **test AUC 0.7868**
  - Identified 3 critical issues vs ECHO-deepantigen reference code:
  1. **Cross-attention bug**: mix-then-normalize → normalize-then-mix (aligned with ECHO paper)
  2. **ESM projection bottleneck**: removed 1280→512 projections for ESM-only, full 2560-dim classifier
  3. **Dropout too low**: 0.08→0.3 classifier, added cross_attn_dropout=0.3
  - Also: added early stopping (patience=10), batch_size=32, epochs=200, wd=1e-4
- **Re-training ESM-only baseline with fixes**

### Phase 2 Results (Old Architecture)
| Metric | Value |
|--------|-------|
| Best val AUC | 0.7294 (epoch 42) |
| Test AUC | 0.7914 (majority_testing_dataset) |
| Test Accuracy | 0.7262 |
| Test F1 | 0.7115 |
| Epoch time | ~9 min |
| Checkpoint | `runs/gcn_joint/best_model.pth` (5.4 GB) |

### ESM-Only Baseline (New Architecture, no GCN)
| Metric | Value |
|--------|-------|
| Best val AUC | 0.7293 (epoch 46) |
| **Test AUC** | **0.7868** |
| Test Accuracy | 0.7159 |
| Test F1 | 0.6882 |
| Epoch time | ~3 min |
| Checkpoint | `runs/esm_only/best_model.pth` |

## Planning Files
- `.claude/task_plan.md` — full task plan with phases
- `.claude/progress.md` — progress log with issues encountered
- `.claude/findings.md` — all bugs found & fixed, architecture insights

## Launch Commands
```bash
# Phase 2: Joint Pretraining (with refactored architecture)
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_gcn.yaml > runs/gcn_joint/training.log 2>&1 &
```

## Bugs Fixed & Changes (cumulative)
1. `gcn_aux_head` added — GCN had zero gradients (late fusion ignored F_spatial)
2. `view`->`reshape` in structure_losses.py — non-contiguous tensor crash
3. SuperNodeExchange MHA -> Linear — MHA degenerate with 1 super-node
4. Double MolFromSequence — shared mol object
5. Vectorized perm local indices — removed CPU-GPU sync loop
6. Data: strip `;`, filter bad AA (O,X)
7. focal_alpha 0.7->0.9->0.995 — progressive up-weight of positive class
8. TopK all-atom selection — removed N/O-only constraint
9. Spatial aggregation: sum → flatten + masked Max/Avg pool + residual MLP
10. Cross-Modal Gated Fusion: cat → decoupled language gate + cross-modal physics gate
11. Loss weight cosine annealing — λ_gcn_aux 1.0→0.1, λ_int 2.0→0.5
12. **Cross-attention normalize-then-mix** (2026-05-20) — softmax before interpolation, not after
13. **ESM projection bottleneck removed** (2026-05-20) — ESM-only uses full 2560-dim, not 1024
14. **Dropout alignment** (2026-05-20): 0.08→0.3, added cross_attn_dropout=0.3
15. **Early stopping** (2026-05-20): added to training loop (patience=10)
