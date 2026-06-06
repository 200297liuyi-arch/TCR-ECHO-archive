"""GCNPlugin — extends Model with Track 2 (Physics) atom-level GCN.

TCR-ECHO V3: remove ESM projection bottleneck.
  - Raw ESM features (2560-dim, no projection) — preserves zero-shot info
  - SeqAlignedGCN (independent encoders + final-MHA sum)
  - Scalar GCN gate (learnable α, stateless) — avoids distribution-shift coupling
  - Direct concat fusion: [tcr_raw, pep_raw, α·gcn] → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from model import Model, focal_loss, flexible_peptide_contrastive
from gcn_components import SeqAlignedGCN


class GCNPlugin(Model):
    """Dual-track model: ESM-2 (Track 1) + Atom-level GCN (Track 2).

    V3: raw ESM (no projection) + scalar gate (no ESM→gate coupling).
    """

    def __init__(
        self,
        esm1_name: str,
        esm2_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list,
        contrastive_temp: float,
        lambda_enc: float,
        lambda_int: float,
        classifier_hidden: int,
        dropout: float,
        focal_gamma: float,
        class_balance: float,
        use_lora: bool = False,
        num_heads: int = 8,
        enable_monitoring: bool = True,
        cross_attn_dropout: float = 0.1,
        second_contrastive: bool = True,
        random_init: bool = False,
        # ── GCN-specific ────────────────────────────────────────────
        gcn_args: dict = None,
        gcn_freeze_encoder: bool = True,
    ):
        super().__init__(
            esm1_name=esm1_name,
            esm2_name=esm2_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            contrastive_temp=contrastive_temp,
            lambda_enc=lambda_enc,
            lambda_int=lambda_int,
            classifier_hidden=classifier_hidden,
            dropout=dropout,
            focal_gamma=focal_gamma,
            class_balance=class_balance,
            use_lora=use_lora,
            num_heads=num_heads,
            enable_monitoring=enable_monitoring,
            cross_attn_dropout=cross_attn_dropout,
            second_contrastive=second_contrastive,
            random_init=random_init,
        )

        hidden_dim = getattr(
            self.esm1.config, 'hidden_size',
            getattr(self.esm1, 'embed_dim', None)
        )

        # ── Track 2: SeqAlignedGCN (deepAntigen Seq architecture) ─────
        if gcn_args is None:
            gcn_args = dict(
                hidden_size=128, depth=5, k=20, heads=4,
                in_channels=25,
            )
        self.gcn = SeqAlignedGCN(gcn_args)
        self.gcn_hidden = gcn_args["hidden_size"]

        if gcn_freeze_encoder:
            self.gcn.freeze_encoder()

        # ── Scalar GCN gate: learnable α, stateless — no ESM coupling ──
        self.gcn_alpha = nn.Parameter(torch.tensor(0.0))

        # ── Classifier: 1280(tcr) + 1280(pep) + 128(gcn) = 2688 ─────
        fusion_dim = hidden_dim * 2 + self.gcn_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        inp1, mask1,
        inp2, mask2,
        atchley1, atchley2,
        labels,
        tcr_graphs=None,
        pep_graphs=None,
        lambda_int_override=None,
        **kwargs,
    ):
        # ══════════════════════════════════════════════════════════════
        #  Track 1 (Language): ESM encoding
        # ══════════════════════════════════════════════════════════════
        if self.esm1_name.startswith("esmc"):
            out1 = self.esm1(sequence_tokens=inp1).embeddings
            out2 = self.esm2(sequence_tokens=inp2).embeddings
        else:
            out1 = self.esm1(
                input_ids=inp1, attention_mask=mask1
            ).last_hidden_state
            out2 = self.esm2(
                input_ids=inp2, attention_mask=mask2
            ).last_hidden_state

        tcr_enc = out1[:, 1:, :]
        pep_enc = out2[:, 1:, :]

        loss_enc = flexible_peptide_contrastive(
            pep_enc, tcr_enc, labels, temp=self.contrastive_temp
        )

        # ══════════════════════════════════════════════════════════════
        #  Track 2 (Physics): SeqAlignedGCN → gcn_feat [B, 128]
        # ══════════════════════════════════════════════════════════════
        F_gcn = None

        if tcr_graphs is not None:
            pep_batch = Batch.from_data_list(pep_graphs).to(tcr_enc.device)
            tcr_batch = Batch.from_data_list(tcr_graphs).to(tcr_enc.device)
            gcn_out = self.gcn(pep_batch, tcr_batch)
            F_gcn = gcn_out["gcn_feat"]  # [B, 128]

        # ══════════════════════════════════════════════════════════════
        #  Track 1: Cross-Attention + Pooling
        # ══════════════════════════════════════════════════════════════
        tcr_att, pep_att = self.cross_attn(
            tcr_enc, pep_enc, atchley1, atchley2,
        )

        tcr_pool = tcr_att.mean(dim=1)   # [B, 1280]
        pep_pool = pep_att.mean(dim=1)   # [B, 1280]

        if self.second_contrastive:
            loss_int = flexible_peptide_contrastive(
                pep_pool, tcr_pool, labels, temp=self.contrastive_temp
            )

        # ══════════════════════════════════════════════════════════════
        #  Scalar GCN gate + fusion (raw ESM, no projection)
        # ══════════════════════════════════════════════════════════════
        if F_gcn is not None:
            gate = torch.sigmoid(self.gcn_alpha)
            gated_gcn = gate * F_gcn
            fused = torch.cat([tcr_pool, pep_pool, gated_gcn], dim=-1)
        else:
            gcn_placeholder = torch.zeros(
                tcr_pool.size(0), self.gcn_hidden,
                device=tcr_pool.device, dtype=tcr_pool.dtype,
            )
            fused = torch.cat([tcr_pool, pep_pool, gcn_placeholder], dim=-1)

        logits = self.classifier(self.dropout(fused)).squeeze(-1)

        # ══════════════════════════════════════════════════════════════
        #  Loss (focal + contrastive)
        # ══════════════════════════════════════════════════════════════
        if labels is not None:
            _lambda_int = lambda_int_override if lambda_int_override is not None else self.lambda_int

            loss_focal = focal_loss(
                logits, labels,
                gamma=self.focal_gamma,
                alpha=self.class_balance,
            )
            if self.second_contrastive:
                total_loss = (
                    loss_focal
                    + self.lambda_enc * loss_enc
                    + _lambda_int * loss_int
                )
            else:
                total_loss = loss_focal + self.lambda_enc * loss_enc

            return logits, total_loss
        return logits
