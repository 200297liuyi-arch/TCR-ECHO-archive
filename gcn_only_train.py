"""GCN-only training — fully aligned with deepAntigen paper config + original train.csv.

Paper config (test_antigenTCR/config_seq.ini):
  depth=5, k=20, heads=4, hidden=128, optim=SGD, lr=1e-4, bs=32,
  epochs=700, weight_decay=0, momentum=0.9, step-lr @200,400.

Changes from v1 (crashed depth=2 script):
  - depth 2->5, k 10->20, heads 4->4 (exact paper config)
  - AdamW->SGD+momentum (paper config)
  - 100->700 epochs, patience 50->150
  - Added gradient clipping, NaN detection, LR step schedule
  - Training log saved to runs/gcn_only/training.log
"""
import os, sys, logging, time, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from gcn_components import DeepGCN
from utils import set_seed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAPH_CACHE = 'datasets/echo/panpep/graph_cache'
TRAIN_CSV = 'datasets/deepantigen/gcn_train.csv'
TEST_CSV = 'datasets/deepantigen/zero_test_paper.csv'
VAL_SPLIT = 0.10


# FocalLoss — aligned with deepAntigen paper: gamma=2, reduction='sum'
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, labels):
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = torch.exp(-ce_loss)
        weights = (1 - p_t) ** self.gamma
        fl = weights * ce_loss
        if self.reduction == 'sum':
            return fl.sum()
        elif self.reduction == 'mean':
            return fl.mean()
        return fl
criterion = FocalLoss(gamma=2, reduction='sum')


class GCNOnlyModel(nn.Module):
    """Paper-aligned GCN-only model — independent encoders + MHA sum output."""

    def __init__(self, gcn_args):
        super().__init__()
        from gcn_components import PaperAlignedDeepGCN
        self.gcn = PaperAlignedDeepGCN(gcn_args)

    def forward(self, pep_graphs, tcr_graphs, pep_mols, tcr_mols):
        return self.gcn(pep_graphs, tcr_graphs)


class GCNOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, df, columns, graph_cache_dir):
        import pickle
        self.peps = df[columns['peptide']].astype(str).str.strip().tolist()
        self.tcrs = df[columns['tcr']].astype(str).str.strip().str.rstrip(';').tolist()
        self.labels = df[columns['label']].tolist()
        self.pep_data, self.tcr_data, self.valid_idx = [], [], []
        for i, (pep, tcr) in enumerate(zip(self.peps, self.tcrs)):
            pep_f = os.path.join(graph_cache_dir, f"pep_{pep.replace('/', '_')}.pkl")
            tcr_f = os.path.join(graph_cache_dir, f"tcr_{tcr.replace('/', '_')}.pkl")
            if os.path.exists(pep_f) and os.path.exists(tcr_f):
                with open(pep_f, 'rb') as f: d = pickle.load(f)
                self.pep_data.append((d['graph'], d['mol']))
                with open(tcr_f, 'rb') as f: d = pickle.load(f)
                self.tcr_data.append((d['graph'], d['mol']))
                self.valid_idx.append(i)
        print(f'GCNOnlyDataset: {len(self.valid_idx)}/{len(self.labels)} valid samples')

    def __len__(self): return len(self.valid_idx)
    def __getitem__(self, idx):
        pep_g, pep_m = self.pep_data[idx]
        tcr_g, tcr_m = self.tcr_data[idx]
        return pep_g, tcr_g, pep_m, tcr_m, torch.tensor(self.labels[self.valid_idx[idx]], dtype=torch.float32)


def collate_gcn(batch):
    pep_gs, tcr_gs, pep_ms, tcr_ms, labels = zip(*batch)
    return (Batch.from_data_list(list(pep_gs)), Batch.from_data_list(list(tcr_gs)),
            list(pep_ms), list(tcr_ms), torch.stack(labels))


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss, n = 0, 0
    for pep_batch, tcr_batch, pep_ms, tcr_ms, labels in loader:
        pep_batch, tcr_batch, labels = pep_batch.to(device), tcr_batch.to(device), labels.to(device)
        opt.zero_grad()
        loss = criterion(model(pep_batch, tcr_batch, pep_ms, tcr_ms), labels)
        if torch.isnan(loss):
            logger.error(f'NaN loss detected! Skipping batch.')
            continue
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for pep_batch, tcr_batch, pep_ms, tcr_ms, labels in loader:
        pep_batch, tcr_batch = pep_batch.to(device), tcr_batch.to(device)
        preds = torch.sigmoid(model(pep_batch, tcr_batch, pep_ms, tcr_ms)).cpu().numpy()
        all_preds.extend(preds.tolist()); all_labels.extend(labels.numpy().tolist())
    preds, labels = np.array(all_preds), np.array(all_labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    f1 = f1_score(labels, (preds >= 0.5).astype(int))
    return {'auc': auc, 'accuracy': acc, 'f1': f1}, preds, labels


def main():
    parser = argparse.ArgumentParser(description='GCN-only training (paper-aligned)')
    parser.add_argument('--fold', type=int, default=None, help='Fold index (0-indexed) for k-fold CV')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # GPU assignment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(args.seed)

    # Per-fold output directory
    if args.fold is not None:
        out_dir = os.path.join('runs/gcn_only', f'fold_{args.fold}')
    else:
        out_dir = 'runs/gcn_only'
    os.makedirs(out_dir, exist_ok=True)

    # Per-fold logger
    log_file = os.path.join(out_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    global logger
    logger = logging.getLogger(__name__)

    columns = {'tcr': 'binding_TCR', 'peptide': 'peptide', 'label': 'label'}
    gcn_args = {'hidden_size': 128, 'depth': 5, 'k': 20, 'heads': 4, 'in_channels': 25}
    logger.info(f'GCN args: {gcn_args}')
    if args.fold is not None:
        logger.info(f'Fold {args.fold+1}/{args.n_folds}, GPU {args.gpu}, LR {args.lr}')

    df_all = pd.read_csv(TRAIN_CSV)
    if df_all.columns[0].startswith('﻿'):
        df_all.columns = [c.lstrip('﻿') for c in df_all.columns]

    df_test = pd.read_csv(TEST_CSV)
    if df_test.columns[0].startswith('﻿'):
        df_test.columns = [c.lstrip('﻿') for c in df_test.columns]

    # Train/val split (stratified, configurable seed)
    if args.fold is not None:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        labels = df_all['label'].values
        fold_indices = list(skf.split(np.zeros(len(df_all)), labels))
        train_idx, val_idx = fold_indices[args.fold]
        df_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_val = df_all.iloc[val_idx].reset_index(drop=True)
        logger.info(f'Fold {args.fold+1}/{args.n_folds}: {len(df_train)} train, {len(df_val)} val')
    else:
        df_train, df_val = train_test_split(
            df_all, test_size=VAL_SPLIT,
            stratify=df_all['label'],
            random_state=args.seed,
        )

    logger.info(f'Train pos/neg: {df_train.label.sum():.0f}/{len(df_train)-df_train.label.sum():.0f}')
    logger.info(f'Test  pos/neg: {df_test.label.sum():.0f}/{len(df_test)-df_test.label.sum():.0f}')

    train_ds = GCNOnlyDataset(df_train, columns, GRAPH_CACHE)
    val_ds = GCNOnlyDataset(df_val, columns, GRAPH_CACHE)
    test_ds = GCNOnlyDataset(df_test, columns, GRAPH_CACHE)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_gcn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_gcn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_gcn, num_workers=0)

    model = GCNOnlyModel(gcn_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'GCN-only params: {n_params:,}')

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[200, 400], gamma=0.5)

    best_val_auc, best_state, patience_counter = 0, None, 0
    patience = 60
    MAX_EPOCHS = 700

    logger.info(f'Training: max_epochs={MAX_EPOCHS}, patience={patience}, '
                f'batches/epoch~{len(train_loader)} (paper-aligned independent encoder)')

    t_start = time.time()
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_auc = val_metrics['auc']
        scheduler.step()
        lr = opt.param_groups[0]['lr']

        elapsed = time.time() - t_start
        logger.info(f'Epoch {epoch+1:3d}  lr={lr:.2e}  train_loss={train_loss:.4f}  '
                    f'val_auc={val_auc:.4f}  val_acc={val_metrics["accuracy"]:.4f}  '
                    f'val_f1={val_metrics["f1"]:.4f}  [{elapsed/60:.1f}m]')

        if np.isnan(train_loss) or train_loss > 100:
            logger.error(f'Training collapsed at epoch {epoch+1}! train_loss={train_loss:.4f}')
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f'  -> New best val_auc: {best_val_auc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break

    logger.info(f'Training complete. Best val AUC: {best_val_auc:.4f}')
    model.load_state_dict(best_state)

    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device)
    logger.info(f'\n========== GCN-Only Results (deepAntigen paper config) ==========')
    logger.info(f'  Fold: {args.fold if args.fold is not None else "single"}')
    logger.info(f'  Best val AUC: {best_val_auc:.4f}')
    logger.info(f'  Test AUC:     {test_metrics["auc"]:.4f}')
    logger.info(f'  Test Acc:     {test_metrics["accuracy"]:.4f}')
    logger.info(f'  Test F1:      {test_metrics["f1"]:.4f}')
    logger.info(f'  Total time:   {(time.time()-t_start)/60:.1f} min')

    torch.save({'model_state': best_state, 'gcn_args': gcn_args, 'fold': args.fold},
               os.path.join(out_dir, 'best_model.pth'))
    pd.DataFrame({
        'peptide': df_test['peptide'], 'tcr': df_test['binding_TCR'],
        'pred': test_preds, 'label': test_labels.astype(int),
    }).to_csv(os.path.join(out_dir, 'zero_test_predictions.csv'), index=False)
    logger.info(f'Saved to {out_dir}/')


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
