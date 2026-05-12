"""Two-stage fine-tuning for 3D structural learning.

Adapted from deepAntigen/deepAntigen/antigenTCR/run_antigenTCR_atom.py

Stage 1: Fine-tune Top-K pooling layers with Pearson correlation loss.
         Freezes ESM2 + TGCN encoder layers, trains TopK scoring network.
Stage 2: Fine-tune classifier + cross-attention with contact focal loss.
         Freezes encoder + TopK, trains attention + projector + classifier.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from sklearn.model_selection import KFold

from structure_dataset import PDBStructureDataset, collate_structure_batch
from structure_losses import (
    NegativePearsonCorrelationLossWithMask,
    WeightedFocalLoss,
    generate_contact_labels,
    generate_mask,
)


class AverageMeter:
    """Track running average of values."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_bn(m):
    """Keep BatchNorm in eval mode during fine-tuning (deepAntigen convention)."""
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_learning_rate(lr, epochs, epoch, cosine=True, lr_decay_rate=0.5):
    """Cosine annealing or step decay."""
    if cosine:
        eta_min = lr * (lr_decay_rate ** 3)
        return eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / epochs)) / 2
    else:
        return lr


def finetune_stage1(model, train_loader, optimizer, device, args):
    """Stage 1: Fine-tune TopK layers with Pearson correlation loss.

    Freezes ESM2 encoders + TGCN layers.
    Only TopK pooling layers and cross-attention are trainable.

    Parameters
    ----------
    model : Model (TCR-ECHO)
    train_loader : DataLoader  yields (pdbids, pep_chems, pep_graphs, tcr_chems, tcr_graphs, dist_mats)
    device : torch.device
    args : dict  training hyperparameters

    Returns
    -------
    avg_loss : float
    """
    model.set_stage(1)
    model.train()
    model.apply(fix_bn)

    pearson_criterion = NegativePearsonCorrelationLossWithMask().to(device)
    losses = AverageMeter()

    for _, pep_chems, pep_graphs, tcr_chems, tcr_graphs, dist_mats in train_loader:
        pep_batch = Batch.from_data_list(pep_graphs).to(device)
        tcr_batch = Batch.from_data_list(tcr_graphs).to(device)

        # In stage 1, we primarily train the GCN (Track 2).
        # Use a simplified forward that only runs GCN + structure loss.
        gcn_out = model.gcn(pep_batch, tcr_batch, pep_chems, tcr_chems)

        p_scores = gcn_out["p_scores"]
        c_scores = gcn_out["c_scores"]
        p_on_indexs = gcn_out.get("p_indexs", None)
        c_on_indexs = gcn_out.get("c_indexs", None)

        if p_scores is None or c_scores is None:
            continue

        p_scores_exp = torch.exp(p_scores)
        c_scores_exp = torch.exp(c_scores)
        joint_scores = torch.mm(
            p_scores_exp.unsqueeze(0).T, c_scores_exp.unsqueeze(0)
        )

        distances, mask = generate_mask(
            dist_mats,
            p_on_indexs if p_on_indexs is not None else [],
            c_on_indexs if c_on_indexs is not None else [],
            device,
        )
        loss = pearson_criterion(joint_scores, distances, mask)
        losses.update(loss.item(), len(dist_mats))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def finetune_stage2(model, train_loader, optimizer, device, args):
    """Stage 2: Fine-tune classifier + spatial projection with contact focal loss.

    Freezes encoder + TopK layers. Interaction map comes from frozen GCN,
    then flows through trainable gcn_spatial_proj for gradient propagation.

    Parameters
    ----------
    model : Model
    train_loader : DataLoader
    optimizer : torch.optim.Optimizer
    device : torch.device
    args : dict

    Returns
    -------
    avg_loss : float
    """
    model.set_stage(2)
    model.train()
    model.apply(fix_bn)

    classify_criterion = WeightedFocalLoss(
        alpha=args.get('focal_alpha', 0.7),
        gamma=2,
        reduction='sum',
    ).to(device)
    losses = AverageMeter()
    k = args['k']

    # Use the first Linear in gcn_spatial_proj as trainable contact head
    contact_proj = model.gcn_spatial_proj[0]  # nn.Linear(H_gcn, H_esm)

    for _, pep_chems, pep_graphs, tcr_chems, tcr_graphs, dist_mats in train_loader:
        pep_batch = Batch.from_data_list(pep_graphs).to(device)
        tcr_batch = Batch.from_data_list(tcr_graphs).to(device)

        gcn_out = model.gcn(pep_batch, tcr_batch, pep_chems, tcr_chems)

        p_perm = gcn_out["p_perm"]
        c_perm = gcn_out["c_perm"]
        interaction_map = gcn_out["interaction_map"]    # [B, k, k, H_gcn]

        # Generate contact labels from distance matrices
        labels = generate_contact_labels(dist_mats, p_perm, c_perm, k,
                                         threshold=args.get('contact_threshold', 5.0))
        labels = labels.to(device)

        # Route through trainable contact_proj so gradients flow
        B, K, _, H = interaction_map.shape
        flat_map = interaction_map.view(B * K * K, H)               # [B*K*K, H_gcn]
        pair_features = contact_proj(flat_map)                      # [B*K*K, H_esm]
        pair_scores = pair_features.mean(dim=-1).view(-1, 1)         # [B*K*K, 1]
        pair_logits = torch.cat([-pair_scores, pair_scores], dim=-1) # [B*K*K, 2]
        flat_labels = labels.view(-1)

        loss = classify_criterion(pair_logits, flat_labels)
        losses.update(loss.item() / len(dist_mats), len(dist_mats))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def train_structure_pipeline(model, device, args):
    """Run the full two-stage fine-tuning pipeline.

    Parameters
    ----------
    model : Model  TCR-ECHO model with use_gcn=True
    device : torch.device
    args : dict  see config_structure.yaml for expected keys

    Expected args keys:
        pdb_csv : str
        data_dir : str
        batch_size : int
        stage1_epochs : int
        stage2_epochs : int
        lr : float
        weight_decay : float
        cosine_annealing : bool
        k : int
        contact_threshold : float
        save_dir : str
    """
    dataset = PDBStructureDataset(args['pdb_csv'], args.get('data_dir', ''))

    kf = KFold(n_splits=len(dataset), shuffle=True, random_state=0)

    for train_index, val_index in kf.split(dataset):
        pdb = dataset.pdbids[val_index[0]]
        print(f"\n=== Fold: val PDB = {pdb} ===")

        train_ds = Subset(dataset, train_index)
        val_ds = Subset(dataset, val_index)
        batch_size = args.get('batch_size', 4)

        # ── Stage 1: Fine-tune TopK ──────────────────────────────────
        print("Stage 1: Fine-tuning TopK pooling layers...")
        train_loader = DataLoader(
            train_ds, shuffle=True, batch_size=batch_size,
            collate_fn=collate_structure_batch, pin_memory=True, drop_last=True,
        )

        # Setup optimizer for stage 1
        # Only TopK + cross-attention params are trainable after set_stage(1)
        model.set_stage(1)
        stage1_opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.gcn.parameters()),
            lr=args['lr'],
            weight_decay=args.get('weight_decay', 1e-4),
        )

        for epoch in range(args.get('stage1_epochs', 200)):
            lr_epoch = adjust_learning_rate(
                args['lr'], args['stage1_epochs'], epoch,
                cosine=args.get('cosine_annealing', True),
            )
            for pg in stage1_opt.param_groups:
                pg['lr'] = lr_epoch

            loss_val = finetune_stage1(model, train_loader, stage1_opt, device, args)
            if epoch % 20 == 0 or epoch == args['stage1_epochs'] - 1:
                print(f"  Stage1 Epoch {epoch}: Loss={loss_val:.4f}")

        # ── Stage 2: Fine-tune classifier ────────────────────────────
        print("Stage 2: Fine-tuning classifier layers...")
        train_loader = DataLoader(
            train_ds, shuffle=True, batch_size=batch_size,
            collate_fn=collate_structure_batch, pin_memory=True, drop_last=True,
        )

        model.set_stage(2)
        # Stage 2: GCN is frozen, train classifier + spatial projection + cross-attention
        stage2_trainable = (
            list(model.gcn_spatial_proj.parameters())
            + list(model.classifier.parameters())
            + list(model.cross_attn.parameters())
        )
        stage2_opt = optim.Adam(
            stage2_trainable,
            lr=args['lr'],
            weight_decay=args.get('weight_decay', 1e-4),
        )

        min_loss = float('inf')
        for epoch in range(args.get('stage2_epochs', 200)):
            lr_epoch = adjust_learning_rate(
                args['lr'], args['stage2_epochs'], epoch,
                cosine=args.get('cosine_annealing', True),
            )
            for pg in stage2_opt.param_groups:
                pg['lr'] = lr_epoch

            loss_val = finetune_stage2(model, train_loader, stage2_opt, device, args)
            if loss_val < min_loss:
                min_loss = loss_val
                save_dir = args.get('save_dir', 'runs/structure/')
                os.makedirs(os.path.join(save_dir, pdb), exist_ok=True)
                save_path = os.path.join(save_dir, pdb, 'atom-level_parameters.pt')
                state = {
                    'model': model.state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'min_loss': min_loss,
                }
                torch.save(state, save_path)
            if epoch % 20 == 0 or epoch == args['stage2_epochs'] - 1:
                print(f"  Stage2 Epoch {epoch}: Loss={loss_val:.4f}")

        print(f"Best model saved with loss={min_loss:.4f}")
