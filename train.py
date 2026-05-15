import os
import yaml
import torch
import wandb
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import Model
from dataset import TCRPeptideDataset, collate_graph_batch
from utils import save_checkpoint, compute_metrics, load_atchley, load_checkpoint
from train_structure import train_structure_pipeline


def train_one(cfg, epochs, run_name_suffix=""):
    run = wandb.init(
        project=cfg['wandb']['project'],
        config=cfg,
        name=f"{cfg.get('run_name', 'exp')}{run_name_suffix}_{cfg['wandb']['run']}"
    )

    # ── Data ────────────────────────────────────────────────────────
    df_train = pd.read_csv(cfg['dataset']['train_csv'])
    if cfg['dataset'].get('val_csv'):
        df_val = pd.read_csv(cfg['dataset']['val_csv'])
        df_test = pd.read_csv(cfg['dataset']['test_csv'])
    else:
        df_test = pd.read_csv(cfg['dataset']['test_csv'])
        train_peps = set(df_train[cfg['dataset']['columns']['peptide']].unique())
        mask = df_test[cfg['dataset']['columns']['peptide']].isin(train_peps)
        df_val = df_test[~mask]
        df_test = df_test[mask]

    pos = (df_train['label'] == 1).sum()
    neg = (df_train['label'] == 0).sum()
    class_imbalance = neg / (pos + neg) if (pos + neg) > 0 else 0.5

    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/{cfg['esm']['encoder1']}"
    )
    atchley_map = load_atchley(cfg.get('atchley_path'))

    use_graph = cfg.get('use_gcn', False)

    train_ds = TCRPeptideDataset(
        df_train, tokenizer, atchley_map,
        cfg['dataset']['columns'],
        mask_prob=cfg['dataset'].get('mask_prob', 0.0),
        use_graph=use_graph,
        graph_cache_dir=cfg.get('graph_cache_dir'),
    )
    val_ds = TCRPeptideDataset(
        df_val, tokenizer, atchley_map,
        cfg['dataset']['columns'],
        use_graph=use_graph,
        graph_cache_dir=cfg.get('graph_cache_dir'),
    )

    collate_fn = collate_graph_batch if use_graph else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'],
        shuffle=True, collate_fn=collate_fn,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'],
        collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ── Model ───────────────────────────────────────────────────────
    lora_preset = cfg['lora']['presets'][cfg['esm']['encoder1']]

    gcn_args = cfg.get('gcn', None)
    model_params = {
        'esm1_name':            cfg['esm']['encoder1'],
        'esm2_name':            cfg['esm']['encoder2'],
        'use_lora':             cfg['use_lora'],
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
        'second_contrastive':   cfg['training']['second_contrastive'],
        'use_gcn':              use_graph,
        'gcn_args':             gcn_args,
        'gcn_freeze_encoder':   cfg.get('gcn_freeze_encoder', True),
        'fusion_gcn':           cfg.get('fusion_gcn', True),
        'lambda_gcn_aux':       cfg.get('lambda_gcn_aux', 1.0),
        'cross_attn_dropout':   cfg.get('cross_attn_dropout', 0.1),
    }

    model = Model(**model_params).to(cfg.get('device', 'cpu'))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            f"Train loader has 0 batches ({len(train_ds)} samples loaded "
            f"from {cfg['dataset']['train_csv']}). Check data file.")
    if len(val_loader) == 0:
        raise RuntimeError(
            f"Val loader has 0 batches ({len(val_ds)} samples loaded "
            f"from {cfg['dataset']['val_csv']}). Check data file.")

    best_auc = 0
    for epoch in range(epochs):
        # ── train ───────────────────────────────────────────────────
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            batch = [b.to(cfg.get('device', 'cpu'), non_blocking=True) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            if use_graph:
                (inp1, msk1, inp2, msk2, at1, at2, labels,
                 tcr_graphs, pep_graphs, tcr_mols, pep_mols,
                 tcr_a2r, pep_a2r) = batch
                logits, loss = model(
                    inp1, msk1, inp2, msk2, at1, at2, labels,
                    tcr_graphs=tcr_graphs, pep_graphs=pep_graphs,
                    tcr_mols=tcr_mols, pep_mols=pep_mols,
                    tcr_a2r=tcr_a2r, pep_a2r=pep_a2r,
                )
            else:
                inp1, msk1, inp2, msk2, at1, at2, labels = batch
                logits, loss = model(inp1, msk1, inp2, msk2, at1, at2, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / max(len(losses), 1)

        # ── validation ──────────────────────────────────────────────
        val_metrics, _, _ = compute_metrics(
            model, val_loader, device=cfg.get('device', 'cpu'),
            use_graph=use_graph,
        )
        run.log({
            'train_loss': avg_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'epoch': epoch,
        })

        now = __import__('datetime').datetime.now().strftime('%H:%M:%S')
        metric_str = ' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())
        import subprocess, os
        try:
            smi = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=5)
            gpu = smi.stdout.strip().replace('\n', ' | ')
        except:
            gpu = 'N/A'
        print(f'[{now}] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {metric_str} | PID={os.getpid()} | GPU: {gpu}',
              flush=True)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_checkpoint(model, optimizer, cfg, best_auc,
                            cfg['training']['output_dir'])

    run.finish()
    return best_auc


def main(config, structure_mode=False):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    if structure_mode:
        _run_structure_training(cfg, device)
        return

    best_auc = train_one(cfg, epochs=cfg['training']['epochs'])

    # ── test evaluation ─────────────────────────────────────────────
    class_imbalance = 0.5
    model, _, run_cfg = load_checkpoint(
        Model, cfg['training']['output_dir'], class_imbalance,
        device=cfg.get('device', 'cpu'),
    )
    df_test = pd.read_csv(run_cfg['dataset']['test_csv'])
    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/{run_cfg['esm']['encoder1']}"
    )
    atchley_map = load_atchley(run_cfg.get('atchley_path'))
    use_graph = run_cfg.get('use_gcn', False)

    test_ds = TCRPeptideDataset(
        df_test, tokenizer, atchley_map,
        run_cfg['dataset']['columns'],
        use_graph=use_graph,
        graph_cache_dir=run_cfg.get('graph_cache_dir'),
    )
    collate_fn = collate_graph_batch if use_graph else None
    test_loader = DataLoader(
        test_ds, batch_size=cfg['training']['batch_size'],
        collate_fn=collate_fn,
    )

    test_metrics, preds, labels = compute_metrics(
        model, test_loader, device=cfg.get('device', 'cpu'),
        use_graph=use_graph, return_preds=True,
    )

    out_df = pd.DataFrame({'pred': preds, 'label': labels})
    out_df.to_csv(
        os.path.join(cfg['training']['output_dir'], 'test_predictions.csv'),
        index=False,
    )
    print("Final test metrics:", test_metrics)


def _run_structure_training(cfg, device):
    """Run two-stage fine-tuning on PDB structure data."""
    import torch

    # ── Build model with structure support ────────────────────────
    lora_preset = cfg['lora']['presets'][cfg['esm']['encoder1']]
    gcn_args = cfg.get('gcn', None)
    class_balance = 0.5  # default for structure mode

    model_params = {
        'esm1_name': cfg['esm']['encoder1'],
        'esm2_name': cfg['esm']['encoder2'],
        'use_lora': cfg['use_lora'],
        'lora_r': lora_preset['r'],
        'lora_alpha': lora_preset['alpha'],
        'lora_dropout': lora_preset['dropout'],
        'lora_target_modules': lora_preset['layers_to_transform'],
        'contrastive_temp': cfg['contrastive']['temperature'],
        'lambda_enc': cfg['contrastive']['lambda_enc'],
        'lambda_int': cfg['contrastive']['lambda_int'],
        'classifier_hidden': cfg['classifier_hidden'],
        'dropout': cfg['training'].get('dropout', 0.1),
        'focal_gamma': cfg['training']['focal_gamma'],
        'class_balance': class_balance,
        'use_gcn': True,
        'gcn_args': gcn_args,
        'gcn_freeze_encoder': cfg.get('gcn_freeze_encoder', True),
        'lambda_gcn_aux':   cfg.get('lambda_gcn_aux', 1.0),
        'use_structure': True,
        'contact_threshold': cfg.get('structure_training', {}).get(
            'contact_threshold', 5.0
        ),
    }

    model = Model(**model_params).to(device)

    # ── Load pre-trained ECHO weights (Phase 1+2) ─────────────────
    echo_ckpt = cfg.get('echo_pretrained', '')
    if echo_ckpt and os.path.exists(echo_ckpt):
        state = torch.load(echo_ckpt, map_location=device)
        model.load_state_dict(state.get('model', state), strict=False)
        print(f"Loaded ECHO pretrained weights from {echo_ckpt}")

    # ── Load deepAntigen GCN weights (optional) ────────────────────
    gcn_ckpt = cfg.get('gcn_pretrained', '')
    if gcn_ckpt and os.path.exists(gcn_ckpt):
        state = torch.load(gcn_ckpt, map_location=device)
        gcn_state = state.get('model', state)
        # Only load GCN-related keys
        gcn_keys = {k: v for k, v in gcn_state.items()
                     if k.startswith('gcn.') or k.startswith('peptide_encoder.')
                     or k.startswith('cdr3_encoder.')}
        model.load_state_dict(gcn_keys, strict=False)
        print(f"Loaded GCN pretrained weights from {gcn_ckpt}")

    # ── Merge structure args from config ──────────────────────────
    struct_args = cfg.get('structure_training', {})
    struct_args['pdb_csv'] = cfg['structure']['pdb_csv']
    struct_args['data_dir'] = cfg['structure'].get('data_dir', '')
    struct_args['k'] = gcn_args['k']
    struct_args.setdefault('save_dir', 'runs/structure/')

    train_structure_pipeline(model, device, struct_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ECHO model")
    parser.add_argument(
        "--config", type=str, default="params/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--structure", action="store_true",
        help="Run two-stage 3D structure fine-tuning mode",
    )
    args = parser.parse_args()
    main(args.config, structure_mode=args.structure)
