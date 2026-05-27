"""Verify GCN-only model forward pass + FocalLoss integration."""
import os, sys
sys.path.insert(0, '/home/lyf/projects/TCR-ECHO')
os.chdir('/home/lyf/projects/TCR-ECHO')

import pickle
import torch
from torch_geometric.data import Batch
from gcn_components import DeepGCN
from gcn_only_train import GCNOnlyModel, FocalLoss, collate_gcn

CACHE_DIR = 'datasets/echo/panpep/graph_cache'

def test_model_forward():
    """Load 2 cached graphs and verify forward pass produces correct shapes."""
    # Find any 2 cached pep/tcr pairs
    pep_files = sorted(os.listdir(os.path.join(CACHE_DIR,)))
    pep_f = os.path.join(CACHE_DIR, pep_files[0])
    tcr_f = os.path.join(CACHE_DIR, pep_files[-1])

    with open(pep_f, 'rb') as f: pep_d = pickle.load(f)
    with open(tcr_f, 'rb') as f: tcr_d = pickle.load(f)

    pep_g, pep_m = pep_d['graph'], pep_d['mol']
    tcr_g, tcr_m = tcr_d['graph'], tcr_d['mol']

    gcn_args = {'hidden_size': 128, 'depth': 5, 'k': 20, 'heads': 4, 'in_channels': 25}
    model = GCNOnlyModel(gcn_args, classifier_hidden=64)
    model.eval()

    # Create batch of 2
    pep_batch = Batch.from_data_list([pep_g, pep_g])
    tcr_batch = Batch.from_data_list([tcr_g, tcr_g])

    with torch.no_grad():
        logits = model(pep_batch, tcr_batch, [pep_m, pep_m], [tcr_m, tcr_m])

    assert logits.shape == (2,), f"Expected (2,) got {logits.shape}"
    print(f"  PASS: forward shape {logits.shape}, values {logits.tolist()}")
    return logits


def test_focal_loss_integration():
    """FocalLoss computes valid loss from model output."""
    logits = torch.tensor([2.0, -1.0, 0.5, -2.5])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    criterion = FocalLoss(gamma=2, reduction='sum')
    loss = criterion(logits, labels)

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    # Focal should be < BCE for the same input
    bce = torch.nn.BCEWithLogitsLoss(reduction='sum')(logits, labels)
    assert loss.item() < bce.item(), f"Focal {loss:.4f} should be < BCE {bce:.4f}"
    print(f"  PASS: FocalLoss {loss:.4f} < BCE {bce:.4f}")


def test_classifier_param_count():
    """Classifier should have ~17K params (256→64→1), not ~197K (256→256→1)."""
    gcn_args = {'hidden_size': 128, 'depth': 5, 'k': 20, 'heads': 4, 'in_channels': 25}
    model = GCNOnlyModel(gcn_args, classifier_hidden=64)

    classifier_params = sum(p.numel() for n, p in model.named_parameters()
                           if 'classifier' in n)
    total_params = sum(p.numel() for p in model.parameters())

    assert classifier_params < 20000, f"Classifier too large: {classifier_params}"
    print(f"  PASS: classifier {classifier_params:,} params, total {total_params:,}")
    return classifier_params


if __name__ == '__main__':
    print("Verifying GCN-only model changes...")
    test_focal_loss_integration()
    test_classifier_param_count()
    test_model_forward()
    print("All verification tests passed.")
