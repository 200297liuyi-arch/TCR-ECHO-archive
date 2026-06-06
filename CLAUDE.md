# TCR-ECHO: Dual-Model Fusion for TCR-Peptide Binding Prediction

## Environment
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
- GPU: RTX 5090 x3, 32GB each
- No internet — models from HF cache at `~/.cache/huggingface/hub/`
- ESM model cached: `esm2_t33_650M_UR50D` only

## Architecture Evolution

```
V1 (GCN Bias):  ESM proj(1280→512) + DeepGCN + bias + aux head + complex gating
                → bias dead (τ ablation), aux useless, ~500K wasted params

V2 (Sum+Gate):  ESM proj(1280→512) + SeqAlignedGCN + Linear gate + direct concat
                → clean, working, but ZS regression due to ESM projection bottleneck

V3 (Raw+Alpha): ESM raw 2560-dim + SeqAlignedGCN + scalar α gate + direct concat
                → fixing V2's ZS regression, training in progress
```

### Track 1 (Language): ESM-2 — `model.py`
- ESM-2 ×2 (esm2_t33_650M_UR50D, not weight-tied) with LoRA r=8
- BidirectionalDualViewAttention — **normalize-then-mix** (aligned with ECHO paper)
- Mean pooling → tcr_pool [B,1280], pep_pool [B,1280]
- Loss: Focal(γ=2.0) + Contrastive L_enc(λ=0.5) + Contrastive L_int(λ=2.0)

### Track 2 (Physics): Atom-level GCN — `gcn_plugin.py`
- **GCNPlugin(Model)** inherits all ESM components via super().__init__()
- **SeqAlignedGCN** (depth=5, k=20, heads=4, HS=128): 2 independent PaperEncoders + final MHA(sum)
- Fully aligned with deepAntigen Seq (Nature Communications 2025)

### V3 Details (current, training)
```python
# GCNPlugin forward data flow:
ESM ×2 → DualViewAttn → mean pool → tcr_pool, pep_pool [B,1280]  # raw, no projection
GCN → SeqAlignedGCN → MHA(sum) → gcn_feat [B,128]
gate = σ(α)  # scalar, stateless — no ESM coupling → avoids distribution shift
fused = cat([tcr_pool, pep_pool, gate * gcn_feat]) → [B,2688]
Classifier: Linear(2688→512) → ReLU → Dropout(0.3) → Linear(512→1)
```
- Only 1 extra parameter vs ESM-only (α scalar)

### V2 Details (previous)
```python
# Key difference vs V3: ESM projection 1280→512 (caused ZS info loss)
tcr_feat = Linear(1280→512) + LayerNorm(tcr_pool)
gate = σ(Linear(1024→128)([tcr_feat, pep_feat]))  # ESM-coupled gate
fused = cat([tcr_feat, pep_feat, gate * gcn_feat]) → [B,1152]
```

### Classifier
- **ESM-only**: Linear(2560→512) + ReLU + Dropout(0.3) + Linear(512→1)
- **V2**: Linear(1152→512) + ReLU + Dropout(0.3) + Linear(512→1)
- **V3**: Linear(2688→512) + ReLU + Dropout(0.3) + Linear(512→1)

## Current State (2026-06-05)

### Performance Summary (all models, unified test set)
| Model | Val AUC | Test AUC | Test Acc | ZS AUC | ZS Acc | ZS F1 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| ESM-only | 0.7671 | 0.8371 | 0.7730 | **0.8148** | **0.7474** | **0.7149** |
| GCN Joint v2 | 0.7664 | 0.8364 | 0.7717 | 0.8076 | — | 0.6975 |
| GCN Bias (seed=42) | 0.7646 | **0.8390** | **0.7742** | 0.8050 | 0.7351 | 0.7075 |
| V2 (Sum+Gate) | **0.7698** | 0.8382 | 0.7723 | 0.8092 | 0.7392 | 0.7125 |
| V3 (Raw+Alpha) | — | — | — | — | — | — |

### V3 — Training in progress
- PID: 4051834 | GPU 2 | Config: `configs/config_gcn_v3.yaml`
- Log: `runs/gcn_joint_v3/training.log` | Checkpoint: `runs/gcn_joint_v3/best_model.pth`
- Hypothesis: removing ESM projection (1280→512) + stateless scalar gate fixes ZS regression

### V2 Zero-Shot Root Cause Analysis (2026-06-05)
Three issues identified:
1. **ESM projection bottleneck** (high): Linear(1280→512) drops 60% of pretrained feature space, losing ZS-critical information. ESM-only's 2560-dim classifier outperforms V2's 1152-dim.
2. **Gate coupling amplifies distribution shift** (medium): `σ(Linear(ESM_feat))` modulates GCN; on unseen peptides, ESM shift → gate shift → GCN modulation shift → classifier confusion.
3. **GCN encoder overfits to training domain** (medium): SeqAlignedGCN(depth=5) learns atom patterns specific to PanPep 52K; zero-shot peptides from different chemical space produce noisy features.

V3 addresses all three: raw ESM (fixes #1), scalar gate (fixes #2), and by eliminating ESM→gate coupling, GCN noise (#3) becomes less damaging since gate can't amplify distribution shift.

### GCN Bias — Confirmed Dead Path (V1 τ ablation)
- τ=0.227 vs τ=0: predictions 100% identical (corr=1.0, max|Δ|=0.00035)
- GCN attention bias contributes nothing — model learned to ignore it
- Root cause: 1-bit label supervises 128dim×100 atom pairs → signal diluted 12,800×

### ESM-Only Baseline (FIXED)
| Metric | Old (buggy) | New (fixed) | Δ |
|--------|------------|------------|---|
| Best val AUC | 0.7293 | 0.7671 | +3.8% |
| **Test AUC** | 0.7868 | **0.8371** | **+5.0%** |
| Test Accuracy | 0.7159 | 0.7730 | +5.7% |
| Test F1 | 0.6882 | 0.7519 | +6.4% |
| Epochs trained | 50 | 91 (early stop) | — |
| Checkpoint | — | `runs/esm_only/best_model.pth` | — |

### 3 Critical Fixes (aligned with ECHO-deepantigen reference)
1. **Cross-attention normalize-then-mix**: softmax each view separately, then interpolate
2. **ESM projection removed**: full 2560-dim classifier input (no 1280→512 bottleneck)
3. **Dropout alignment**: 0.08→0.3, added cross_attn_dropout=0.3

## Launch Commands
```bash
# ESM-only baseline
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_esm_only.yaml > runs/esm_only/training.log 2>&1 &

# V3: Raw ESM + scalar gate (current)
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_gcn_v3.yaml > runs/gcn_joint_v3/training.log 2>&1 &
```

## Planning Files
- `.claude/task_plan.md` — full task plan with phases
- `.claude/progress.md` — progress log with issues encountered
- `.claude/findings.md` — all bugs found & fixed, architecture insights

## Bugs Fixed (cumulative, 17 items)
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
13. **ESM projection bottleneck removed** (2026-05-20) — critical perf fix (for ESM-only)
14. **Dropout alignment** (2026-05-20): 0.08→0.3
15. **Early stopping** (2026-05-20): patience=10
16. **Modular refactoring** (2026-05-21): Model base + GCNPlugin
17. **F1 pos_label bug** (2026-06-05): GCN Bias ZS F1 incorrectly recorded as 0.7562 (pos_label=0), actual value 0.7075

## Changelog
- 2026-06-05: **V3 launched** — raw ESM (no projection) + scalar gate, targeting ZS AUC ≥0.8148
- 2026-06-05: V2 evaluated — Val AUC 0.7698 (best), but ZS regression (0.8092 vs ESM-only 0.8148). Root cause analysis: ESM projection bottleneck + gate coupling + GCN ZS noise.
- 2026-06-04: V2 architecture refactor — deleted GCN bias, switch to MHA(sum), lightweight gate. 5 files changed, -277 lines net.
- 2026-06-03: GCN Bias training completed — Test AUC 0.8390, ZS F1 bug discovered later.
