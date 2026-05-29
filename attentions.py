import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalDualViewAttention(nn.Module):
    """Bi-directional Dual-View Cross-Attention (Phase 2).

    Shared learnable parameters across both directions:
      - U [H,5,5]  — biophysical compatibility bilinear form
      - mix_param   — blend ratio between sequence & physicochemical views

    Direction 1 (TCR → Peptide):
      S_seq = Q_tcr @ K_pep^T / √d
      S_bio = A_tcr @ U @ A_pep^T

    Direction 2 (Peptide → TCR):
      S_seq = Q_pep @ K_tcr^T / √d
      S_bio = A_pep @ U.T @ A_tcr^T   (transposed U for symmetry)

    Fusion (shared ρ):
      ρ = (tanh(mix_param) + 1) / 2   ∈ [0, 1]
      combined = (1-ρ) * S_seq + ρ * S_bio
    """

    def __init__(self, dim, num_heads, dropout=0.1, enable_monitoring=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # QKV projections — separate per direction
        self.q_proj_t = nn.Linear(dim, dim)  # TCR as query
        self.k_proj_p = nn.Linear(dim, dim)  # Peptide as key
        self.v_proj_p = nn.Linear(dim, dim)  # Peptide as value
        self.q_proj_p = nn.Linear(dim, dim)  # Peptide as query
        self.k_proj_t = nn.Linear(dim, dim)  # TCR as key
        self.v_proj_t = nn.Linear(dim, dim)  # TCR as value

        self.out_proj_t = nn.Linear(dim, dim)
        self.out_proj_p = nn.Linear(dim, dim)

        # ── shared across both directions ────────────────────────────
        # bilinear form for biophysical compatibility
        self.U = nn.Parameter(torch.randn(num_heads, 5, 5) * 0.02)

        # shared mix parameter: blend sequence ↔ physicochemical views
        self.mix_param = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)

        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.register_buffer("rho_history", torch.zeros(100))
            self.history_idx = 0

    def _forward_one_direction(self, q_seq, kv_seq, atc_q, atc_kv,
                                q_proj, k_proj, v_proj, out_proj,
                                transpose_U=False,
                                gcn_bias=None):
        """Single cross-attention direction.

        q_seq: query sequence embedding   [B, Lq, D]
        kv_seq: key/value sequence embedding [B, Lkv, D]
        atc_q: Atchley factors for query  [B, Lq, 5]
        atc_kv: Atchley factors for key   [B, Lkv, 5]
        transpose_U: if True, use U^T for reverse direction symmetry
        """
        B, Lq, _ = q_seq.shape
        _, Lkv, _ = kv_seq.shape

        if atc_q.size(1) > Lq:
            atc_q = atc_q[:, :Lq, :]
        if atc_kv.size(1) > Lkv:
            atc_kv = atc_kv[:, :Lkv, :]

        # QKV
        q = q_proj(q_seq).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(kv_seq).view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(kv_seq).view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)

        # View 1: sequence attention
        S_seq = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,Lq,Lkv]

        # Add GCN physical interaction bias (if provided)
        if gcn_bias is not None:
            S_seq = S_seq + gcn_bias

        # View 2: biophysical attention
        S_bio = torch.zeros_like(S_seq)
        U_eff = self.U.transpose(-1, -2) if transpose_U else self.U
        for h in range(self.num_heads):
            U_h = U_eff[h]  # [5, 5]
            head_bias = torch.einsum("bip,pq,bjq->bij", atc_q, U_h, atc_kv)
            S_bio[:, h] = head_bias  # [B, Lq, Lkv]

        # Normalize each attention separately, then interpolate
        # (ECHO paper eq: avoids one view dominating softmax via scale differences)
        rho = (torch.tanh(self.mix_param) + 1) / 2  # [0, 1]
        attn_seq = F.softmax(S_seq, dim=-1)
        attn_bio = F.softmax(S_bio, dim=-1)
        attn = self.dropout(attn_seq * (1 - rho) + attn_bio * rho)

        if self.enable_monitoring:
            with torch.no_grad():
                self.rho_history[self.history_idx] = rho.item()
                self.history_idx = (self.history_idx + 1) % 100
        out = attn @ v  # [B, H, Lq, Dh]
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        return out_proj(out)

    def forward(self, tcr_enc, pep_enc, atchley1, atchley2, *, gcn_bias=None):
        """Bi-directional cross-attention.

        Parameters
        ----------
        tcr_enc : [B, L_tcr, D]
        pep_enc : [B, L_pep, D]
        atchley1 : [B, L_tcr, 5]
        atchley2 : [B, L_pep, 5]

        Returns
        -------
        tcr_att : [B, L_tcr, D]   TCR attended over Peptide
        pep_att : [B, L_pep, D]   Peptide attended over TCR
        """
        # Direction 1: TCR → Peptide
        tcr_att = self._forward_one_direction(
            q_seq=tcr_enc, kv_seq=pep_enc,
            atc_q=atchley1, atc_kv=atchley2,
            q_proj=self.q_proj_t, k_proj=self.k_proj_p, v_proj=self.v_proj_p,
            out_proj=self.out_proj_t,
            transpose_U=False,
            gcn_bias=gcn_bias,
        )

        # Direction 2: Peptide → TCR  (U^T for symmetric scoring, transpose bias)
        pep_att = self._forward_one_direction(
            q_seq=pep_enc, kv_seq=tcr_enc,
            atc_q=atchley2, atc_kv=atchley1,
            q_proj=self.q_proj_p, k_proj=self.k_proj_t, v_proj=self.v_proj_t,
            out_proj=self.out_proj_p,
            transpose_U=True,
            gcn_bias=gcn_bias.transpose(-1, -2) if gcn_bias is not None else None,
        )

        return tcr_att, pep_att
