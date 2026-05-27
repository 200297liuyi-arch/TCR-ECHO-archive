# GCN Paper-Aligned Independent Encoder Architecture

## Goal
Implement deepAntigen paper's (Nature Communications 2025) independent encoder architecture for GCN-only training, replacing the per-layer SuperNodeExchange architecture that is incompatible with depth=5.

## Architecture

### PaperAlignedDeepGCN
```
PaperEncoder(peptide) ──→ pep_fs [B,20,128]
                              │
                              ├──→ MultiHeadAttention → sum(dim=(1,2)) → [B,128]
                              │
PaperEncoder(cdr3) ─────→ cdr_fs [B,20,128]
                              │
                              ↓
                    Projector(128→64) + ReLU + Dropout(0.2)
                              ↓
                    Classifier(64→1) → logits
```

### PaperEncoder (per-molecule, independent)
```
init_w(25→128) + LeakyReLU
for i in 0..depth-1:
    TGCN(hidden=128, with GRUCell) → BatchNorm
    if i == depth-1: PaperTopKPooling(ratio=20) → [B, 20, 128]
return fs
```

### PaperTopKPooling — paper's simple version
- Single weight [1, H] projection: score = (x * weight).sum(dim=-1)
- tanh normalization: score / ||weight||
- topk by score, no valid_mask, no positional encoding
- Assumes all molecules have >= k atoms

## Key Differences from Current Architecture
| | Current | Paper-Aligned |
|---|---|---|
| Encoders | Shared CrossModalGCNLayer per layer | 2 independent Encoders |
| Cross-modal | Every layer via SuperNodeExchange | Final MHA only |
| Per-layer unit | LocalMP + SuperNodeExchange + GRU | TGCN (has internal GRU) |
| TopK timing | After all layers (once) | At last encoder layer |
| MHA output | Full [B,k,k,H] + masked pool | sum(dim=(1,2)) → [B,H] |
| Classifier | 256→64→1 (~16K params) | 128→64→1 (~8K params) |

## Training Config
- Optimizer: SGD, lr=1e-4, momentum=0.9, weight_decay=0
- Loss: FocalLoss(γ=2, reduction='sum')
- batch_size=64, epochs=700
- LR schedule: MultiStepLR @200,400 ×0.5
- patience=150

## Files Changed
- `gcn_components.py`: Add PaperTopKPooling, PaperEncoder, PaperAlignedDeepGCN
- `gcn_only_train.py`: Simplified GCNOnlyModel
- Existing DeepGCN, CrossModalGCNLayer, SuperNodeExchange preserved for ESM+GCN joint training

## Files NOT Changed
- `model.py`, `gcn_plugin.py`, `train.py`, `dataset.py`, `utils.py`
