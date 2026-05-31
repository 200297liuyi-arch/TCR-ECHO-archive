import os
import math
import yaml
import torch
import wandb
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import Model
from gcn_plugin import GCNPlugin
from dataset import TCRPeptideDataset, collate_graph_batch
from utils import save_checkpoint, compute_metrics, load_atchley, load_checkpoint


def cosine_anneal(epoch, total_epochs, start_val, end_val):
    """Cosine annealing from start_val → end_val over total_epochs."""
    if total_epochs <= 1:
        return start_val
    progress = epoch / max(total_epochs - 1, 1)
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))


def train_one(cfg, epochs, run_name_suffix=""):
    run = wandb.init(
        project=cfg['wandb']['project'],
        config=cfg,
        name=f"{cfg.get('run_name', 'exp')}{run_name_suffix}_{cfg['wandb']['run']}"
    )

    # ── Data ────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    df_full = pd.read_csv(cfg['dataset']['train_csv'])
    if cfg['dataset'].get('val_csv'):
        df_train, df_val = df_full, pd.read_csv(cfg['dataset']['val_csv'])
        df_test = pd.read_csv(cfg['dataset']['test_csv'])
    else:
        val_split = cfg['dataset'].get('val_split', 0.15)
        df_train, df_val = train_test_split(
            df_full,
            test_size=val_split,
            stratify=df_full[cfg['dataset']['columns']['label']],
            random_state=42,
        )
        df_test = pd.read_csv(cfg['dataset']['test_csv'])

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

    # shared params for both Model and GCNPlugin
    shared_params = {
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
        'cross_attn_dropout':   cfg['training'].get('cross_attn_dropout', 0.1),
    }

    if use_graph:
        gcn_args = cfg.get('gcn', None)
        model_params = {
            **shared_params,
            'gcn_args':             gcn_args,
            'gcn_freeze_encoder':   cfg.get('gcn_freeze_encoder', True),
            'lambda_gcn_aux':       cfg.get('lambda_gcn_aux', 1.0),
        }
        model = GCNPlugin(**model_params).to(cfg.get('device', 'cpu'))
    else:
        model = Model(**shared_params).to(cfg.get('device', 'cpu'))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    sched_cfg = cfg['training'].get('scheduler', {})
    scheduler = None
    if sched_cfg.get('name') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get('mode', 'max'),
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 10),
            min_lr=sched_cfg.get('min_lr', 1e-6),
        )

    if len(train_loader) == 0:
        raise RuntimeError(
            f"Train loader has 0 batches ({len(train_ds)} samples loaded "
            f"from {cfg['dataset']['train_csv']}). Check data file.")
    if len(val_loader) == 0:
        raise RuntimeError(
            f"Val loader has 0 batches ({len(val_ds)} samples loaded "
            f"from {cfg['dataset']['val_csv']}). Check data file.")

    # ── Loss annealing config ─────────────────────────────────────
    anneal_cfg = cfg.get('loss_annealing', {})
    gcn_aux_cfg = anneal_cfg.get('lambda_gcn_aux', {})
    lint_cfg = anneal_cfg.get('lambda_int', {})

    patience = cfg['training']['early_stopping']['patience']
    best_auc = 0
    no_improve = 0
    for epoch in range(epochs):
        # ── Compute annealed loss weights ──────────────────────────
        lambda_gcn_aux_now = lambda_int_now = None
        if gcn_aux_cfg.get('schedule') == 'cosine':
            lambda_gcn_aux_now = cosine_anneal(
                epoch, epochs, gcn_aux_cfg['start'], gcn_aux_cfg['end'])
        if lint_cfg.get('schedule') == 'cosine':
            lambda_int_now = cosine_anneal(
                epoch, epochs, lint_cfg['start'], lint_cfg['end'])

        # ── train ───────────────────────────────────────────────────
        model.train()
        losses = []
        running_logits, running_labels = [], []
        n_batches = len(train_loader)
        for bi, batch in enumerate(train_loader):
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
                    lambda_gcn_aux_override=lambda_gcn_aux_now,
                    lambda_int_override=lambda_int_now,
                )
            else:
                inp1, msk1, inp2, msk2, at1, at2, labels = batch
                logits, loss = model(inp1, msk1, inp2, msk2, at1, at2, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            running_logits.append(logits.detach().cpu())
            running_labels.append(labels.detach().cpu())
            if bi % 100 == 0:
                from sklearn.metrics import roc_auc_score as _ra
                all_l = torch.cat(running_logits).sigmoid().numpy()
                all_y = torch.cat(running_labels).numpy()
                try:
                    train_auc = _ra(all_y, all_l)
                except ValueError:
                    train_auc = 0.5
                print(f'  [train] batch {bi}/{n_batches} loss={loss.item():.4f} train_auc={train_auc:.4f}', flush=True)
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
            **({'lambda_gcn_aux': lambda_gcn_aux_now} if lambda_gcn_aux_now is not None else {}),
            **({'lambda_int': lambda_int_now} if lambda_int_now is not None else {}),
        })

        # Step LR scheduler
        lr_now = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
            lr_now = optimizer.param_groups[0]['lr']
            run.log({'lr': lr_now, 'epoch': epoch})

        now = __import__('datetime').datetime.now().strftime('%H:%M:%S')
        metric_str = ' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())
        import subprocess, os
        try:
            smi = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=5)
            gpu = smi.stdout.strip().replace('\n', ' | ')
        except:
            gpu = 'N/A'
        print(f'[{now}] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {metric_str} | lr={lr_now:.2e} | PID={os.getpid()} | GPU: {gpu}',
              flush=True)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_checkpoint(model, optimizer, cfg, best_auc,
                            cfg['training']['output_dir'])
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    run.finish()
    return best_auc


def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    best_auc = train_one(cfg, epochs=cfg['training']['epochs'])

    # ── test evaluation ─────────────────────────────────────────────
    class_imbalance = 0.5
    # peek at checkpoint config to determine model class
    import torch as _torch
    chk_path = os.path.join(cfg['training']['output_dir'], 'best_model.pth')
    chk = _torch.load(chk_path, map_location='cpu', weights_only=False)
    run_cfg = chk['config']
    use_graph = run_cfg.get('use_gcn', False)
    ModelClass = GCNPlugin if use_graph else Model
    del chk, _torch

    model, _, run_cfg = load_checkpoint(
        ModelClass, cfg['training']['output_dir'], class_imbalance,
        device=cfg.get('device', 'cpu'),
    )
    df_test = pd.read_csv(run_cfg['dataset']['test_csv'])
    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/{run_cfg['esm']['encoder1']}"
    )
    atchley_map = load_atchley(run_cfg.get('atchley_path'))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ECHO model")
    parser.add_argument(
        "--config", type=str, default="params/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
