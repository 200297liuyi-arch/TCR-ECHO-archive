"""Structure-aware loss functions from deepAntigen for 3D contact learning.

Adapted from deepAntigen/deepAntigen/antigenTCR/utils/model_utils.py
and run_antigenTCR_atom.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativePearsonCorrelationLossWithMask(nn.Module):
    """Negative Pearson correlation loss with mask for valid atom pairs.

    Minimizes: 1 + PearsonCorrelation(predicted_scores, true_distances)
    computed only over masked (valid) atom pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A, B, mask):
        """Compute negative Pearson correlation.

        Parameters
        ----------
        A : Tensor  joint predicted scores, any shape
        B : Tensor  true inter-atom distances, same shape as A
        mask : Tensor  binary mask, 1=valid, 0=ignore

        Returns
        -------
        loss : scalar  (1 + correlation), minimized at correlation = -1
        """
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        mask_flat = mask.reshape(-1)

        A_masked = A_flat[mask_flat != 0]
        B_masked = B_flat[mask_flat != 0]

        if A_masked.numel() == 0:
            return torch.tensor(0.0, device=A.device)

        mean_A = torch.mean(A_masked)
        mean_B = torch.mean(B_masked)

        cov_AB = torch.mean((A_masked - mean_A) * (B_masked - mean_B))
        std_A = torch.std(A_masked)
        std_B = torch.std(B_masked)

        correlation = cov_AB / (std_A * std_B + 1e-8)
        return 1 + correlation


class WeightedFocalLoss(nn.Module):
    """Focal loss with class weighting for binary contact classification.

    Adapted from deepAntigen's WeightedFocalLoss.
    """

    def __init__(self, alpha=0.7, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        """Compute weighted focal loss.

        Parameters
        ----------
        logits : Tensor  raw logits for each class, or binary logits
        labels : Tensor  integer class labels

        Returns
        -------
        loss : scalar
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        alpha_factor = torch.where(labels == 1, 1.0, 1 - self.alpha)
        weighted_ce = alpha_factor * ce_loss
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        weights = (1 - pt) ** self.gamma
        fl = weights * weighted_ce

        if self.reduction == 'sum':
            return fl.sum()
        elif self.reduction == 'mean':
            return fl.mean()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def generate_contact_labels(distance_matrix, p_perm, c_perm, k, threshold=5.0):
    """Generate binary contact labels from distance matrices and TopK perms.

    A contact is defined as atom-pair distance < threshold Angstroms.

    Parameters
    ----------
    distance_matrix : list of np.ndarray  one per sample, [N_pep_atoms, N_tcr_atoms]
    p_perm : Tensor  [B*k] peptide atom indices (local within molecule)
    c_perm : Tensor  [B*k] TCR atom indices
    k : int  number of top-k atoms per molecule
    threshold : float  contact distance cutoff in Angstroms

    Returns
    -------
    labels : LongTensor  [B, k, k]  binary contact labels
    """
    device = p_perm.device
    B = len(distance_matrix)
    labels = []

    for i in range(B):
        p_idx = p_perm[i * k:(i + 1) * k].detach().cpu().numpy()
        c_idx = c_perm[i * k:(i + 1) * k].detach().cpu().numpy()
        dist_mat = distance_matrix[i]
        # Extract submatrix at top-k positions
        sub_dist = dist_mat[np.ix_(p_idx, c_idx)]
        contact = (sub_dist < threshold).astype(np.int64)
        labels.append(torch.from_numpy(contact))

    return torch.stack(labels).to(device)


def generate_mask(distance_matrixs, p_on_indexs, c_on_indexs, device):
    """Generate valid atom-pair mask and distance tensor for Pearson loss.

    Parameters
    ----------
    distance_matrixs : list of np.ndarray
    p_on_indexs : list of int  N/O atom indices for peptide
    c_on_indexs : list of int  N/O atom indices for TCR
    device : torch.device

    Returns
    -------
    distances : Tensor  [total_N, total_O]
    mask : Tensor  [total_N, total_O]  binary mask
    """
    total_row = 0
    total_col = 0
    for dm in distance_matrixs:
        nrow, ncol = dm.shape
        total_row += nrow
        total_col += ncol

    distances = torch.zeros((total_row, total_col))
    current_row = 0
    current_col = 0
    for dm in distance_matrixs:
        nrow, ncol = dm.shape
        distances[current_row:current_row + nrow,
                  current_col:current_col + ncol] = torch.from_numpy(dm).float()
        current_row += nrow
        current_col += ncol

    # Subset to N/O atoms only
    p_on_tensor = torch.tensor(p_on_indexs, dtype=torch.long)
    c_on_tensor = torch.tensor(c_on_indexs, dtype=torch.long)
    distances = distances[p_on_tensor.unsqueeze(-1), c_on_tensor.unsqueeze(0)]
    mask = torch.where(distances > 0, 1, 0)

    return distances.to(device), mask.to(device)
