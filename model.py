import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import EsmModel, EsmConfig
try:
    from esm.models.esmc import ESMC
except ImportError:
    ESMC = None

from attentions import BidirectionalDualViewAttention
from gcn_components import DeepGCN as AtomDeepGCN


class Model(nn.Module):
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
        # ── Track 2 (Physics) args ─────────────────────────────────
        use_gcn: bool = False,
        gcn_args: dict = None,
        gcn_freeze_encoder: bool = True,
        # ── Structure-aware training ───────────────────────────────
        use_structure: bool = False,
        contact_threshold: float = 5.0,
        fusion_gcn: bool = True,  # kept for checkpoint compatibility
    ):
        super().__init__()

        # ══════════════════════════════════════════════════════════════
        #  Track 1 (Language): ESM-2 encoders
        # ══════════════════════════════════════════════════════════════
        self.esm1_name = esm1_name
        self.esm2_name = esm2_name

        if random_init:
            cfg = EsmConfig.from_pretrained(f'facebook/{esm1_name}')
            self.esm1 = EsmModel(cfg)
            self.esm2 = EsmModel(cfg)
        else:
            if esm1_name.startswith("esmc"):
                if ESMC is None:
                    raise ImportError(
                        "ESMC model requested but esm.models.esmc not available"
                    )
                self.esm1 = ESMC.from_pretrained(esm1_name)
            else:
                self.esm1 = EsmModel.from_pretrained(f'facebook/{esm1_name}')
            if esm2_name.startswith("esmc"):
                if ESMC is None:
                    raise ImportError(
                        "ESMC model requested but esm.models.esmc not available"
                    )
                self.esm2 = ESMC.from_pretrained(esm2_name)
            else:
                self.esm2 = EsmModel.from_pretrained(f'facebook/{esm2_name}')

        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                bias='none',
                lora_dropout=lora_dropout,
                target_modules=(
                    ['attn.layernorm_qkv.1'] if 'esmc' in esm1_name
                    else ['attention.self.key', 'attention.self.value']
                ),
                layers_to_transform=lora_target_modules,
            )
            self.esm1 = get_peft_model(self.esm1, peft_cfg)
            self.esm2 = get_peft_model(self.esm2, peft_cfg)
            for name, param in self.esm1.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
            for name, param in self.esm2.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False

        hidden_dim = getattr(
            self.esm1.config, 'hidden_size',
            getattr(self.esm1, 'embed_dim', None)
        )

        # ══════════════════════════════════════════════════════════════
        #  Track 1 (Language): Dual-View Cross-Attention
        #    View 1: sequence attention  (Q·K^T / √d)
        #    View 2: Atchley bias        (atc1 @ U @ atc2^T)
        #  Single module handles both directions in one call:
        #    Direction 1: TCR(Query) attends to Peptide(Key/Value) → new TCR
        #    Direction 2: Peptide(Query) attends to TCR(Key/Value) → new Peptide
        # ══════════════════════════════════════════════════════════════
        self.cross_attn = BidirectionalDualViewAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=cross_attn_dropout,
            enable_monitoring=enable_monitoring,
        )
        self.second_contrastive = second_contrastive

        # ══════════════════════════════════════════════════════════════
        #  Track 2 (Physics): CrossModalGCN
        #    Sequence → RDKit Mol → atom graph
        #    L-layer GCN (local MP → super-node exchange → GRU)
        #    Top-K → MultiHeadAttention → F_spatial
        # ══════════════════════════════════════════════════════════════
        self.use_gcn = use_gcn
        if use_gcn:
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

            # project F_spatial from GCN dim to ESM dim for late fusion
            self.gcn_spatial_proj = nn.Sequential(
                nn.Linear(self.gcn_hidden, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
        else:
            self.gcn = None

        # ══════════════════════════════════════════════════════════════
        #  Late Fusion & Classifier
        #    cat(tcr_pool, pep_pool, F_spatial)  →  MLP  →  logit
        # ══════════════════════════════════════════════════════════════
        fusion_dim = hidden_dim * 2                     # tcr_pool + pep_pool
        if use_gcn:
            fusion_dim += hidden_dim                    # + F_spatial

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        # ══════════════════════════════════════════════════════════════
        #  Loss hyper-params
        # ══════════════════════════════════════════════════════════════
        self.focal_gamma = focal_gamma
        self.class_balance = class_balance
        self.contrastive_temp = contrastive_temp
        self.lambda_enc = lambda_enc
        self.lambda_int = lambda_int

        # ── Structure-aware training ───────────────────────────────
        self.use_structure = use_structure
        self.contact_threshold = contact_threshold
        self.stage = 0  # 0=normal, 1=stage1(TopK), 2=stage2(classifier)
        self._last_gcn_out = None  # cache GCN output for structure loss

    def set_stage(self, stage: int):
        """Set training stage for two-stage fine-tuning.

        0 = normal (train all)
        1 = stage1 (freeze encoder, train TopK)
        2 = stage2 (freeze encoder+TopK, train classifier)
        """
        self.stage = stage
        if stage == 1 and self.gcn is not None:
            self.gcn.freeze_encoder()
        elif stage == 2 and self.gcn is not None:
            self.gcn.freeze_topk()

    def forward(
        self,
        inp1, mask1,
        inp2, mask2,
        atchley1, atchley2,
        labels,
        # ── Track 2 inputs (optional, used when use_gcn=True) ────────
        tcr_graphs=None,
        pep_graphs=None,
        tcr_mols=None,
        pep_mols=None,
        tcr_a2r=None,      # kept for compatibility, unused
        pep_a2r=None,      # kept for compatibility, unused
        # ── Structure training inputs ──────────────────────────────
        distance_matrix=None,  # [B, N_pep_atoms, N_tcr_atoms] or list
    ):
        # ══════════════════════════════════════════════════════════════
        #  Track 1 (Language): 3.1 序列特征提取
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

        tcr_enc = out1[:, 1:, :]   # [B, L_tcr, H]
        pep_enc = out2[:, 1:, :]   # [B, L_pep, H]

        # 3.3 注：第一个对比学习损失 L_c1
        loss_enc = flexible_peptide_contrastive(
            pep_enc, tcr_enc, labels, temp=self.contrastive_temp
        )

        # ══════════════════════════════════════════════════════════════
        #  Track 2 (Physics): 2.1-2.3 分子图 → F_spatial
        # ══════════════════════════════════════════════════════════════
        F_spatial = None

        if self.use_gcn and tcr_graphs is not None:
            # 2.1 图结构初始化 (done in dataset, graphs passed in)
            # 2.2 跨分子门控 GCN
            pep_batch = Batch.from_data_list(pep_graphs).to(tcr_enc.device)
            tcr_batch = Batch.from_data_list(tcr_graphs).to(tcr_enc.device)

            gcn_out = self.gcn(pep_batch, tcr_batch, pep_mols, tcr_mols)
            self._last_gcn_out = gcn_out  # cache for structure loss access

            # 2.3 全局特征聚合: MHA权重图谱 → 加权求和 → F_spatial
            interaction_map = gcn_out["interaction_map"]  # [B, k, k, H_gcn]
            F_spatial_raw = interaction_map.sum(dim=(1, 2))  # [B, H_gcn]
            F_spatial = self.gcn_spatial_proj(F_spatial_raw)  # [B, H]

        # ══════════════════════════════════════════════════════════════
        #  Track 1 (Language): 3.3 Dual-View Cross-Attention
        #    View 1: sequence  |  View 2: Atchley bias
        # ══════════════════════════════════════════════════════════════
        tcr_att, pep_att = self.cross_attn(
            tcr_enc, pep_enc, atchley1, atchley2,
        )

        # 3.4 序列降维: Mean Pooling
        tcr_pool = tcr_att.mean(dim=1)   # [B, H]
        pep_pool = pep_att.mean(dim=1)   # [B, H]

        # 3.4 注：第二个对比学习损失 L_c2
        if self.second_contrastive:
            loss_int = flexible_peptide_contrastive(
                pep_pool, tcr_pool, labels, temp=self.contrastive_temp
            )

        # ══════════════════════════════════════════════════════════════
        #  4. Late Fusion & Output
        # ══════════════════════════════════════════════════════════════
        if self.use_gcn and F_spatial is not None:
            fused = torch.cat([tcr_pool, pep_pool, F_spatial], dim=-1)
        else:
            fused = torch.cat([tcr_pool, pep_pool], dim=-1)

        logits = self.classifier(self.dropout(fused)).squeeze(-1)  # [B]

        # ══════════════════════════════════════════════════════════════
        #  Loss
        # ══════════════════════════════════════════════════════════════
        if labels is not None:
            loss_focal = focal_loss(
                logits, labels,
                gamma=self.focal_gamma,
                alpha=self.class_balance,
            )
            if self.second_contrastive:
                total_loss = (
                    loss_focal
                    + self.lambda_enc * loss_enc
                    + self.lambda_int * loss_int
                )
            else:
                total_loss = loss_focal + self.lambda_enc * loss_enc

            # ── Structure-aware losses (two-stage fine-tuning) ─────
            structure_loss = None
            if (self.use_structure and distance_matrix is not None
                    and self._last_gcn_out is not None):
                structure_loss = self._compute_structure_loss(
                    self._last_gcn_out, distance_matrix
                )
                if structure_loss is not None:
                    total_loss = total_loss + structure_loss

            return logits, total_loss
        return logits

    def _compute_structure_loss(self, gcn_out, distance_matrix):
        """Compute structure-aware losses for two-stage fine-tuning.

        Stage 1: Negative Pearson Correlation loss on TopK joint scores.
        Stage 2: Weighted Focal loss on atom-pair contact predictions.
        """
        p_scores = gcn_out.get("p_scores")
        c_scores = gcn_out.get("c_scores")
        p_on_indexs = gcn_out.get("p_indexs", None)
        c_on_indexs = gcn_out.get("c_indexs", None)
        p_perm = gcn_out.get("p_perm")
        c_perm = gcn_out.get("c_perm")
        interaction_map = gcn_out.get("interaction_map")

        if p_scores is None or c_scores is None:
            return None

        try:
            from structure_losses import (
                NegativePearsonCorrelationLossWithMask,
                WeightedFocalLoss,
                generate_contact_labels,
                generate_mask,
            )
        except ImportError:
            return None

        if self.stage == 1:
            # Stage 1: Pearson correlation between joint scores and distances
            p_scores_exp = torch.exp(p_scores)
            c_scores_exp = torch.exp(c_scores)
            joint_scores = torch.mm(
                p_scores_exp.unsqueeze(0).T, c_scores_exp.unsqueeze(0)
            )
            distances, mask = generate_mask(
                distance_matrix,
                p_on_indexs if p_on_indexs is not None else [],
                c_on_indexs if c_on_indexs is not None else [],
                p_scores.device,
            )
            criterion = NegativePearsonCorrelationLossWithMask().to(p_scores.device)
            return criterion(joint_scores, distances, mask)

        elif self.stage == 2:
            # Stage 2: Contact prediction loss
            k = interaction_map.shape[1]
            labels = generate_contact_labels(
                distance_matrix, p_perm, c_perm, k,
                threshold=self.contact_threshold,
            )
            labels = labels.to(interaction_map.device)
            criterion = WeightedFocalLoss(alpha=0.7, gamma=2, reduction='sum')
            # interaction_map: [B, k, k, H] — average over hidden dim, produce 2-class logits
            B, K = interaction_map.shape[0], interaction_map.shape[1]
            flat_map = interaction_map.view(B * K * K, -1)
            pair_logits = flat_map.mean(dim=-1).view(-1, 1)   # [B*K*K, 1]
            pair_logits = torch.cat([-pair_logits, pair_logits], dim=-1)  # [B*K*K, 2]
            return criterion(pair_logits, labels.view(-1))

        return None


# ══════════════════════════════════════════════════════════════════════
#  Loss utilities
# ══════════════════════════════════════════════════════════════════════

def flexible_peptide_contrastive(pmhc, tcr, labels, temp=0.2, mode='pooled'):
    batch_size = pmhc.shape[0]
    device = pmhc.device

    pmhc_is_sequence = pmhc.dim() == 3
    tcr_is_sequence = tcr.dim() == 3

    if mode == 'pooled':
        pmhc_pooled = pmhc.mean(dim=1) if pmhc_is_sequence else pmhc
        tcr_pooled = tcr.mean(dim=1) if tcr_is_sequence else tcr
        pmhc_norm = F.normalize(pmhc_pooled, p=2, dim=1)
        tcr_norm = F.normalize(tcr_pooled, p=2, dim=1)
        sim = torch.matmul(pmhc_norm, tcr_norm.T) / temp
    elif mode == 'residue_wise':
        if not (pmhc_is_sequence and tcr_is_sequence):
            raise ValueError("Residue-wise mode requires sequence dims")
        L_p = pmhc.shape[1]
        L_t = tcr.shape[1]
        pmhc_flat = pmhc.reshape(-1, pmhc.shape[-1])
        tcr_flat = tcr.reshape(-1, tcr.shape[-1])
        pmhc_norm = F.normalize(pmhc_flat, p=2, dim=1).view(batch_size, L_p, -1)
        tcr_norm = F.normalize(tcr_flat, p=2, dim=1).view(batch_size, L_t, -1)
        sim = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                residue_sim = torch.matmul(pmhc_norm[i], tcr_norm[j].T) / temp
                sim[i, j] = residue_sim.mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)

    numer = exp_sim.diagonal()
    denom = exp_sim.sum(dim=1)
    eps = 1e-8
    losses = -torch.log((numer + eps) / (denom + eps))

    positive_anchor_mask = labels.bool()
    if positive_anchor_mask.sum() > 0:
        positive_loss = losses[positive_anchor_mask].mean()
    else:
        positive_loss = torch.tensor(0.0, device=device)

    return losses.mean()


def focal_loss(logits, labels, gamma, alpha):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction='none'
    )
    p_t = labels * probs + (1 - labels) * (1 - probs)
    alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()
