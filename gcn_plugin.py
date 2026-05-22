"""GCNPlugin — extends Model with Track 2 (Physics) atom-level GCN.

Inherits ESM-2 language track from Model, adds:
  - AtomDeepGCN encoder
  - TopK all-atom pooling + MHA → interaction_map
  - Masked spatial aggregation + residual MLP
  - ESM projections (1280→512) + Cross-Modal Gated Fusion
  - GCN auxiliary classifier
  - Cosine-annealed loss weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from model import Model, focal_loss, flexible_peptide_contrastive
from gcn_components import DeepGCN as AtomDeepGCN


class GCNPlugin(Model):
    """Dual-track model: ESM-2 (Track 1) + Atom-level GCN (Track 2).

    Inherits ESM encoders, DualViewAttn, and loss functions from Model.
    Adds GCN encoder, projections, gates, and overrides forward().
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
        lambda_gcn_aux: float = 1.0,
    ):
        # ── Inherit ESM track from Model ───────────────────────────────
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

        # ESM hidden_dim (set by Model.__init__)
        hidden_dim = getattr(
            self.esm1.config, 'hidden_size',
            getattr(self.esm1, 'embed_dim', None)
        )

        # ── Track 2: Atom-level GCN encoder ───────────────────────────
        if gcn_args is None:
            gcn_args = dict(
                hidden_size=128, depth=2, k=10, heads=4,
                in_channels=25,
            )
        self.gcn = AtomDeepGCN(gcn_args)
        self.gcn_hidden = gcn_args["hidden_size"]
        self.gcn_k = gcn_args["k"]

        if gcn_freeze_encoder:
            self.gcn.freeze_encoder()

        # ── GCN spatial projection: 256→512→256 residual MLP ──────────
        spatial_in = self.gcn_hidden * 2  # 256
        spatial_mid = 512
        self.gcn_spatial_proj = nn.ModuleDict({
            "fc1": nn.Linear(spatial_in, spatial_mid),
            "fc2": nn.Linear(spatial_mid, spatial_in),
            "norm": nn.LayerNorm(spatial_in),
        })
        self.gcn_spatial_dropout = nn.Dropout(0.2)

        # ── GCN auxiliary classifier ──────────────────────────────────
        self.gcn_aux_head = nn.Sequential(
            nn.Linear(spatial_in, self.gcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gcn_hidden // 2, 1),
        )

        # ── ESM Projections: 1280→512 (for balanced fusion) ───────────
        proj_dim = 512
        self.tcr_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.pep_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ── Cross-Modal Gating ────────────────────────────────────────
        # Decoupled language gate
        lang_gate_in = proj_dim * 2  # 1024
        lang_gate_mid = 256
        self.gate_lang = nn.Sequential(
            nn.Linear(lang_gate_in, lang_gate_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_gate_mid, lang_gate_in),
        )

        # Cross-modal physics gate: language context → physics gate
        self.ctx_lang_proj = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
        )
        phys_gate_in = self.gcn_hidden * 2 + 128  # 256 + 128 = 384
        phys_gate_mid = 64
        self.gate_phys = nn.Sequential(
            nn.Linear(phys_gate_in, phys_gate_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phys_gate_mid, self.gcn_hidden * 2),
        )

        # ── Replace classifier: 1280-dim input (512+512+256) ─────────
        fusion_dim = proj_dim + proj_dim + self.gcn_hidden * 2  # 512+512+256=1280
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        # ── GCN loss weight ───────────────────────────────────────────
        self.lambda_gcn_aux = lambda_gcn_aux

    def forward(
        self,
        inp1, mask1,
        inp2, mask2,
        atchley1, atchley2,
        labels,
        tcr_graphs=None,
        pep_graphs=None,
        tcr_mols=None,
        pep_mols=None,
        tcr_a2r=None,
        pep_a2r=None,
        lambda_gcn_aux_override=None,
        lambda_int_override=None,
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
        #  Track 2 (Physics): Atom GCN → F_spatial
        # ══════════════════════════════════════════════════════════════
        F_spatial = None

        if tcr_graphs is not None:
            pep_batch = Batch.from_data_list(pep_graphs).to(tcr_enc.device)
            tcr_batch = Batch.from_data_list(tcr_graphs).to(tcr_enc.device)

            gcn_out = self.gcn(pep_batch, tcr_batch, pep_mols, tcr_mols)

            interaction_map = gcn_out["interaction_map"]  # [B, k, k, H_gcn]
            joint_mask = gcn_out["joint_mask"]            # [B, k, k, 1]

            B_gcn, K, _, H_gcn_local = interaction_map.shape
            flat_map = interaction_map.view(B_gcn, K * K, H_gcn_local)
            flat_mask = joint_mask.view(B_gcn, K * K, 1)

            # Masked Max Pooling
            max_map = flat_map.masked_fill(flat_mask == 0, -1e9)
            max_feat, _ = max_map.max(dim=1)

            # Masked Avg Pooling
            sum_feat = (flat_map * flat_mask).sum(dim=1)
            valid_count = flat_mask.sum(dim=1).clamp(min=1e-6)
            avg_feat = sum_feat / valid_count

            F_spatial_raw = torch.cat([max_feat, avg_feat], dim=-1)

            # Residual MLP
            x = F.relu(self.gcn_spatial_proj["fc1"](
                self.gcn_spatial_dropout(F_spatial_raw)))
            x = self.gcn_spatial_proj["fc2"](self.gcn_spatial_dropout(x))
            F_spatial = self.gcn_spatial_proj["norm"](x + F_spatial_raw)

            aux_logits = self.gcn_aux_head(F_spatial_raw).squeeze(-1)

        # ══════════════════════════════════════════════════════════════
        #  Track 1: Cross-Attention + Pooling
        # ══════════════════════════════════════════════════════════════
        tcr_att, pep_att = self.cross_attn(
            tcr_enc, pep_enc, atchley1, atchley2,
        )

        tcr_pool = tcr_att.mean(dim=1)
        pep_pool = pep_att.mean(dim=1)

        if self.second_contrastive:
            loss_int = flexible_peptide_contrastive(
                pep_pool, tcr_pool, labels, temp=self.contrastive_temp
            )

        # ── ESM projection: 1280 → 512 ────────────────────────────────
        tcr_feat = self.tcr_proj(tcr_pool)
        pep_feat = self.pep_proj(pep_pool)

        # ══════════════════════════════════════════════════════════════
        #  Cross-Modal Gated Fusion
        # ══════════════════════════════════════════════════════════════
        if F_spatial is not None:
            # Decoupled language gating
            h_lang = torch.cat([tcr_feat, pep_feat], dim=-1)
            W_joint = torch.sigmoid(self.gate_lang(h_lang))
            W_tcr, W_pep = torch.split(W_joint, 512, dim=-1)

            gated_tcr = W_tcr * tcr_feat
            gated_pep = W_pep * pep_feat

            # Cross-modal physics gating
            ctx_lang = (tcr_feat + pep_feat) / 2
            ctx_lang_small = self.ctx_lang_proj(ctx_lang)
            phys_input = torch.cat([F_spatial, ctx_lang_small], dim=-1)
            W_phys = torch.sigmoid(self.gate_phys(phys_input))
            gated_phys = W_phys * F_spatial

            fused = torch.cat([gated_tcr, gated_pep, gated_phys], dim=-1)
        else:
            fused = torch.cat([tcr_feat, pep_feat], dim=-1)

        logits = self.classifier(self.dropout(fused)).squeeze(-1)

        # ══════════════════════════════════════════════════════════════
        #  Loss
        # ══════════════════════════════════════════════════════════════
        if labels is not None:
            _lambda_int = lambda_int_override if lambda_int_override is not None else self.lambda_int
            _lambda_gcn_aux = lambda_gcn_aux_override if lambda_gcn_aux_override is not None else self.lambda_gcn_aux

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

            if F_spatial is not None:
                gcn_aux_loss = focal_loss(
                    aux_logits, labels,
                    gamma=self.focal_gamma,
                    alpha=self.class_balance,
                )
                total_loss = total_loss + _lambda_gcn_aux * gcn_aux_loss

            return logits, total_loss
        return logits
