import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from transformers import EsmModel, EsmConfig
try:
    from esm.models.esmc import ESMC
except ImportError:
    ESMC = None

from attentions import BidirectionalDualViewAttention


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
        #  Fusion: cat(tcr_pool, pep_pool) → [B, 2560]
        # ══════════════════════════════════════════════════════════════
        fusion_dim = hidden_dim * 2  # 1280+1280=2560

        # ══════════════════════════════════════════════════════════════
        #  Classifier
        # ══════════════════════════════════════════════════════════════
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

    def forward(
        self,
        inp1, mask1,
        inp2, mask2,
        atchley1, atchley2,
        labels,
        lambda_int_override=None,
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
        #  Track 1 (Language): 3.3 Dual-View Cross-Attention
        #    View 1: sequence  |  View 2: Atchley bias
        # ══════════════════════════════════════════════════════════════
        tcr_att, pep_att = self.cross_attn(
            tcr_enc, pep_enc, atchley1, atchley2,
        )

        # 3.4 序列降维: Mean Pooling
        tcr_pool = tcr_att.mean(dim=1)   # [B, H]
        pep_pool = pep_att.mean(dim=1)   # [B, H]

        # 3.4 注：第二个对比学习损失 L_c2 (on raw 1280-dim features)
        if self.second_contrastive:
            loss_int = flexible_peptide_contrastive(
                pep_pool, tcr_pool, labels, temp=self.contrastive_temp
            )

        # ── Fusion: direct cat, full 2560-dim ─────────────────────────
        fused = torch.cat([tcr_pool, pep_pool], dim=-1)  # [B, 2560]

        logits = self.classifier(self.dropout(fused)).squeeze(-1)  # [B]

        # ══════════════════════════════════════════════════════════════
        #  Loss
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
