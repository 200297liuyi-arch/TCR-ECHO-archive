# TCR-ECHO: Dual-Model Fusion for TCR-Peptide Binding Prediction

## Environment
- Python: `/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python`
- GPU: RTX 5090 ×3, 32GB each
- No internet — models from HF cache at `~/.cache/huggingface/hub/`
- ESM model cached: `esm2_t33_650M_UR50D` only

## Architecture
Dual-track model fusing:
- **Track 1 (Language)**: ESM-2 ×2 → BidirectionalDualViewAttention → cross-attention
- **Track 2 (Physics)**: DeepGCN (TGCN → SuperNodeExchange → GRU) per layer → TopKPooling → MultiHeadAttention → F_spatial
- Late fusion: `cat(tcr_pool, pep_pool, F_spatial)` → MLP classifier
- **Auxiliary head**: `gcn_aux_head` (F_spatial_raw → logit) with lambda_gcn_aux weight
- Contrastive losses: L_enc (embeddings) + L_int (pooled)

## Key Parameters
- ESM: LoRA r=8, layers [32,31,30,29,28]
- GCN: hidden=128, depth=2, k=10, heads=4
- gcn_freeze_encoder: false (joint pretraining), lambda_gcn_aux: 1.0

## Current State (2026-05-13)
- **Phase 2**: Joint pretraining — ready to launch
- Data: 52,562 train / 9,276 val (from Panpep_trainingData.csv)
- Test: majority_testing_dataset.csv (5,230 seen peptides)
- Config: `configs/config_gcn.yaml`
- Log: `runs/gcn_joint/training.log`

## Planning Files
- `.claude/task_plan.md` — full task plan with phases
- `.claude/progress.md` — progress log with issues encountered
- `.claude/findings.md` — all bugs found & fixed, architecture insights

## Launch Training
```bash
cd /home/lyf/projects/TCR-ECHO
nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline \
  /home/lyf/miniconda3/envs/tcr-echo-5090/bin/python -u train.py \
  --config configs/config_gcn.yaml > runs/gcn_joint/training.log 2>&1 &
```

## Bugs Fixed & Why
1. `gcn_aux_head` added — GCN had zero gradients (late fusion ignored F_spatial)
2. `view`→`reshape` in structure_losses.py — non-contiguous tensor crash
3. SuperNodeExchange MHA → Linear — MHA degenerate with 1 super-node
4. Double MolFromSequence — shared mol object
5. Vectorized perm local indices — removed CPU-GPU sync loop
6. Data: strip `;`, filter bad AA (O,X)
