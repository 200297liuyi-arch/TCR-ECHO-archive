"""Sanity test: FocalLoss vs BCEWithLogitsLoss behavior."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss aligned with deepAntigen paper. gamma=2, reduction='sum'."""
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


def test_focal_downweights_confident_correct():
    """Easy correct predictions should contribute much less than hard wrong ones."""
    labels = torch.tensor([1.0, 1.0, 0.0])
    # Sample 0: very confident and correct (logit=+3 → p≈0.95)
    # Sample 1: uncertain (logit=+0.1 → p≈0.525)
    # Sample 2: very confident and correct (logit=-3 → p≈0.047)
    logits = torch.tensor([3.0, 0.1, -3.0])

    fl = FocalLoss(gamma=2, reduction='sum')
    bce = nn.BCEWithLogitsLoss(reduction='sum')

    fl_val = fl(logits, labels).item()
    bce_val = bce(logits, labels).item()

    # BCE treats all samples equally by CE magnitude
    # Focal should be lower than BCE because confident correct samples are down-weighted
    assert fl_val < bce_val, f"Focal {fl_val:.4f} should be < BCE {bce_val:.4f}"
    print(f"  PASS: Focal {fl_val:.4f} < BCE {bce_val:.4f}")


def test_focal_gamma_increases_penalty_gap():
    """Higher gamma should further down-weight easy samples."""
    labels = torch.tensor([1.0, 1.0])
    logits = torch.tensor([3.0, 0.1])  # confident correct vs uncertain

    fl_g0 = FocalLoss(gamma=0, reduction='sum')(logits, labels).item()
    fl_g2 = FocalLoss(gamma=2, reduction='sum')(logits, labels).item()
    fl_g5 = FocalLoss(gamma=5, reduction='sum')(logits, labels).item()

    # gamma=0 should equal BCE
    bce = nn.BCEWithLogitsLoss(reduction='sum')(logits, labels).item()
    assert abs(fl_g0 - bce) < 1e-4, f"gamma=0 focal {fl_g0:.4f} should ≈ BCE {bce:.4f}"

    # Higher gamma → lower loss (easy samples down-weighted more)
    assert fl_g5 < fl_g2, f"gamma=5 ({fl_g5:.4f}) should be < gamma=2 ({fl_g2:.4f})"
    print(f"  PASS: gamma=0≈BCE {fl_g0:.4f}, gamma=2 {fl_g2:.4f}, gamma=5 {fl_g5:.4f}")


def test_focal_reductions():
    """sum and mean reductions should differ by batch size."""
    labels = torch.ones(8)
    logits = torch.randn(8) * 2

    fl_sum = FocalLoss(gamma=2, reduction='sum')(logits, labels).item()
    fl_mean = FocalLoss(gamma=2, reduction='mean')(logits, labels).item()

    assert abs(fl_sum / 8 - fl_mean) < 1e-4, f"sum/8={fl_sum/8:.4f} should ≈ mean={fl_mean:.4f}"
    print(f"  PASS: sum/8={fl_sum/8:.4f} ≈ mean={fl_mean:.4f}")


if __name__ == '__main__':
    print("Testing FocalLoss...")
    test_focal_downweights_confident_correct()
    test_focal_gamma_increases_penalty_gap()
    test_focal_reductions()
    print("All tests passed.")
