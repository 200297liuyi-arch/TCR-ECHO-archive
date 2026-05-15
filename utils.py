import os
import torch
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def save_checkpoint(model, optimizer, cfg, best_metric, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': cfg,
        'best_metric': best_metric,
    }
    path = os.path.join(out_dir, 'best_model.pth')
    torch.save(checkpoint, path)
    with open(os.path.join(out_dir, 'best_params.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)


def load_checkpoint(Model, checkpoint_dir, class_imbalance, device='cuda'):
    path = os.path.join(checkpoint_dir, 'best_model.pth')
    import torch.serialization
    from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar])

    chk = torch.load(path, map_location=device, weights_only=False)
    cfg = chk['config']
    lora_preset = cfg['lora']['presets'][cfg['esm']['encoder1']]
    use_lora = cfg.get('use_lora', True)
    use_gcn = cfg.get('use_gcn', False)
    gcn_args = cfg.get('gcn', None)

    model_params = {
        'esm1_name':            cfg['esm']['encoder1'],
        'esm2_name':            cfg['esm']['encoder2'],
        'use_lora':             use_lora,
        'lora_r':               lora_preset['r'],
        'lora_alpha':           lora_preset['alpha'],
        'lora_dropout':         lora_preset['dropout'],
        'lora_target_modules':  lora_preset['layers_to_transform'],
        'contrastive_temp':     cfg['contrastive']['temperature'],
        'lambda_enc':           cfg['contrastive']['lambda_enc'],
        'lambda_int':           cfg['contrastive']['lambda_int'],
        'classifier_hidden':    cfg['classifier_hidden'],
        'dropout':              cfg['training']['dropout'],
        'focal_gamma':          cfg['training']['focal_gamma'],
        'class_balance':        class_imbalance,
        'second_contrastive':   cfg['training'].get('second_contrastive', True),
        'use_gcn':              use_gcn,
        'gcn_args':             gcn_args,
        'gcn_freeze_encoder':   cfg.get('gcn_freeze_encoder', True),
        'fusion_gcn':           cfg.get('fusion_gcn', True),
        'lambda_gcn_aux':       cfg.get('lambda_gcn_aux', 1.0),
    }

    model = Model(**model_params).to(device)
    model.load_state_dict(chk['model_state'])
    model.eval()

    optimizer = None
    return model, optimizer, cfg


def compute_metrics(model, loader, device='cpu', use_graph=False, return_preds=False):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            if use_graph:
                (inp1, msk1, inp2, msk2, at1, at2, labels,
                 tcr_graphs, pep_graphs, tcr_mols, pep_mols,
                 tcr_a2r, pep_a2r) = batch
                logits, _ = model(
                    inp1, msk1, inp2, msk2, at1, at2, labels,
                    tcr_graphs=tcr_graphs, pep_graphs=pep_graphs,
                    tcr_mols=tcr_mols, pep_mols=pep_mols,
                    tcr_a2r=tcr_a2r, pep_a2r=pep_a2r,
                )
            else:
                *inputs, labels = batch
                logits, _ = model(*inputs, labels=labels)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(all_labels) == 0:
        empty = {'auc': 0.5, 'accuracy': 0.0, 'f1': 0.0}
        if return_preds:
            return empty, [], []
        return empty, [], []

    auc = roc_auc_score(all_labels, all_preds)
    preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    if return_preds:
        return {'auc': auc, 'accuracy': acc, 'f1': f1}, all_preds, all_labels
    return {'auc': auc, 'accuracy': acc, 'f1': f1}, all_preds, all_labels


def load_atchley(atchley_path=None):
    """Return Atchley factor dict. Uses built-in values if path not given."""
    return {
        'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
        'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
        'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
        'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
        'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
        'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
        'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
        'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
        'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
        'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
        'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
        'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
        'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
        'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
        'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
        'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
        'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
        'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
    }
