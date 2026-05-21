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
        # ── Auxiliary GCN loss ─────────────────────────────────────
        lambda_gcn_aux: float = 1.0,
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
        #    L-layer GCN → TopK (all-atom) → MHA → interaction_map
        #    Flatten + masked Max/Avg pool → MLP+residual → F_spatial
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

            # F_spatial projection: 256→512→256 with identity residual + LayerNorm
            spatial_in = self.gcn_hidden * 2  # 256
            spatial_mid = 512
            self.gcn_spatial_proj = nn.ModuleDict({
                "fc1": nn.Linear(spatial_in, spatial_mid),
                "fc2": nn.Linear(spatial_mid, spatial_in),
                "norm": nn.LayerNorm(spatial_in),
            })
            self.gcn_spatial_dropout = nn.Dropout(0.2)

            # auxiliary head: GCN-only classifier — forces gradient through GCN
            self.gcn_aux_head = nn.Sequential(
                nn.Linear(spatial_in, self.gcn_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.gcn_hidden // 2, 1),
            )

        else:
            self.gcn = None

        # ══════════════════════════════════════════════════════════════
        #  ESM Projections & Feature Dimension
        #    GCN mode: 1280→512 projections + gated fusion (balanced modalities)
        #    ESM-only: NO projection — full 2560-dim fed to classifier
        # ══════════════════════════════════════════════════════════════
        proj_dim = 512
        if use_gcn:
            self.tcr_proj = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            )
            self.pep_proj = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            )
        else:
            self.tcr_proj = None
            self.pep_proj = None

        # ══════════════════════════════════════════════════════════════
        #  Cross-Modal Gating (only when GCN is active)
        # ══════════════════════════════════════════════════════════════
        if use_gcn:
            # Decoupled language gating: joint context → independent TCR/PEP gates
            lang_gate_in = proj_dim * 2  # 1024 = cat(tcr_proj, pep_proj)
            lang_gate_mid = 256
            self.gate_lang = nn.Sequential(
                nn.Linear(lang_gate_in, lang_gate_mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lang_gate_mid, lang_gate_in),
            )

            # Cross-modal injection: ctx_lang ↓ → physics gate
            self.ctx_lang_proj = nn.Sequential(
                nn.Linear(proj_dim, 128),
                nn.ReLU(),
            )
            phys_gate_in = self.gcn_hidden * 2 + 128  # F_spatial(256) + ctx_lang(128)
            phys_gate_mid = 64
            self.gate_phys = nn.Sequential(
                nn.Linear(phys_gate_in, phys_gate_mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(phys_gate_mid, self.gcn_hidden * 2),
            )

            fusion_dim = proj_dim + proj_dim + self.gcn_hidden * 2  # 512+512+256
        else:
            fusion_dim = hidden_dim * 2  # 1280+1280=2560 — full ESM embeddings

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
        self.lambda_gcn_aux = lambda_gcn_aux

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
        # ── Dynamic loss weight overrides (cosine annealing) ────────
        lambda_gcn_aux_override=None,
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
        #  Track 2 (Physics): 2.1-2.3 分子图 → F_spatial
        # ══════════════════════════════════════════════════════════════
        F_spatial = None

        if self.use_gcn and tcr_graphs is not None:
            # 2.1 图结构初始化 (done in dataset, graphs passed in)
            # 2.2 跨分子门控 GCN
            pep_batch = Batch.from_data_list(pep_graphs).to(tcr_enc.device)
            tcr_batch = Batch.from_data_list(tcr_graphs).to(tcr_enc.device)

            gcn_out = self.gcn(pep_batch, tcr_batch, pep_mols, tcr_mols)

            # 2.3 空间特征聚合: flatten → masked max/avg pool → F_spatial
            interaction_map = gcn_out["interaction_map"]  # [B, k, k, H_gcn]
            joint_mask = gcn_out["joint_mask"]            # [B, k, k, 1]

            B_gcn, K, _, H_gcn_local = interaction_map.shape
            # Flatten pair dimension: [B, k*k, H]
            flat_map = interaction_map.view(B_gcn, K * K, H_gcn_local)
            flat_mask = joint_mask.view(B_gcn, K * K, 1)  # [B, k*k, 1]

            # Masked Max Pooling: ghost pairs → -inf, never selected
            max_map = flat_map.masked_fill(flat_mask == 0, -1e9)
            max_feat, _ = max_map.max(dim=1)               # [B, H_gcn]

            # Masked Avg Pooling: divide only by valid pair count
            sum_feat = (flat_map * flat_mask).sum(dim=1)   # [B, H_gcn]
            valid_count = flat_mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            avg_feat = sum_feat / valid_count              # [B, H_gcn]

            F_spatial_raw = torch.cat([max_feat, avg_feat], dim=-1)  # [B, 2*H_gcn]

            # Multi-layer MLP with identity residual + LayerNorm
            x = F.relu(self.gcn_spatial_proj["fc1"](
                self.gcn_spatial_dropout(F_spatial_raw)))
            x = self.gcn_spatial_proj["fc2"](self.gcn_spatial_dropout(x))
            F_spatial = self.gcn_spatial_proj["norm"](x + F_spatial_raw)  # [B, 256]

            # auxiliary GCN-only prediction — forces gradient through GCN
            aux_logits = self.gcn_aux_head(F_spatial_raw).squeeze(-1)  # [B]

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

        # ── ESM projection / direct pass ──────────────────────────────
        if self.use_gcn:
            tcr_feat = self.tcr_proj(tcr_pool)  # [B, 512]
            pep_feat = self.pep_proj(pep_pool)  # [B, 512]
        else:
            tcr_feat = tcr_pool   # [B, 1280] — no bottleneck for ESM-only
            pep_feat = pep_pool   # [B, 1280]

        # ══════════════════════════════════════════════════════════════
        #  4. Cross-Modal Gated Fusion & Output
        # ══════════════════════════════════════════════════════════════
        if self.use_gcn and F_spatial is not None:
            # Decoupled language gating: joint context → independent TCR/PEP gates
            h_lang = torch.cat([tcr_feat, pep_feat], dim=-1)  # [B, 1024]
            W_joint = torch.sigmoid(self.gate_lang(h_lang))    # [B, 1024]
            W_tcr, W_pep = torch.split(W_joint, 512, dim=-1)

            gated_tcr = W_tcr * tcr_feat  # [B, 512]
            gated_pep = W_pep * pep_feat  # [B, 512]

            # Cross-modal physics gating: language context → physics gate
            ctx_lang = (tcr_feat + pep_feat) / 2               # [B, 512]
            ctx_lang_small = self.ctx_lang_proj(ctx_lang)      # [B, 128]
            phys_input = torch.cat([F_spatial, ctx_lang_small], dim=-1)  # [B, 384]
            W_phys = torch.sigmoid(self.gate_phys(phys_input))  # [B, 256]
            gated_phys = W_phys * F_spatial                     # [B, 256]

            fused = torch.cat([gated_tcr, gated_pep, gated_phys], dim=-1)  # [B, 1280]
        else:
            fused = torch.cat([tcr_feat, pep_feat], dim=-1)  # [B, 2560] ESM-only

        logits = self.classifier(self.dropout(fused)).squeeze(-1)  # [B]

        # ══════════════════════════════════════════════════════════════
        #  Loss
        # ══════════════════════════════════════════════════════════════
        if labels is not None:
            # Dynamic weight overrides (cosine annealing from train loop)
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

            if self.use_gcn and F_spatial is not None:
                gcn_aux_loss = focal_loss(
                    aux_logits, labels,
                    gamma=self.focal_gamma,
                    alpha=self.class_balance,
                )
                total_loss = total_loss + _lambda_gcn_aux * gcn_aux_loss

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
