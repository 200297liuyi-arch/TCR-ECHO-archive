"""GCN components for atom-level graph processing with cross-modal exchange.

Architecture per layer (aligned with deepAntigen_Seq):
  ① LocalMessagePassing (MLP-based, no GRU) — local chemical neighbourhood
  ② SuperNodeExchange — MHA-weighted atom→super-node pooling + gated cross-molecule dialogue
  ③ GRU update — combine (prev_state, LocalMP_output + global_context)  ← only GRU per layer
  + BatchNorm after each layer

Graph augmentation: 5% node dropout during training (deepAntigen step ①).

After all L layers:
  TopKPooling (all-atom scoring) → MultiHeadAttention → interaction_map [B,k,k,H]

Key insight: unlike deepAntigen's TCR model (independent encoders, cross-attn
only at the end), here peptide and TCR atoms engage in gated dialogue at EVERY
layer. Each atom perceives both its local chemical neighbourhood AND the other
molecule's global state throughout the encoding process.
"""

import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.nn.dense.linear as pyg_linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add, scatter_mean


# ══════════════════════════════════════════════════════════════════════
#  LocalMessagePassing  — pure neighbour aggregation (no GRU)
# ══════════════════════════════════════════════════════════════════════

class LocalMessagePassing(MessagePassing):
    """Aggregate neighbor messages, return updated node features (no GRU).

    Message:  MLP(concat(node_j, edge_attr))
    Aggregate: sum
    Update:   MLP(concat(aggregated, node_self))
    """

    def __init__(self, hidden_channels: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.message_w = pyg_linear.Linear(
            hidden_channels + 11, hidden_channels,
            weight_initializer="kaiming_uniform"
        )
        self.update_w = pyg_linear.Linear(
            2 * hidden_channels, hidden_channels,
            weight_initializer="kaiming_uniform"
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        T = self.message_w(torch.cat((x_j, edge_attr), dim=1))
        return F.leaky_relu(T, 0.1)

    def update(self, inputs: Tensor, x) -> Tensor:
        output = self.update_w(torch.cat((inputs, x), dim=1))
        return F.leaky_relu(output, 0.1)


# ══════════════════════════════════════════════════════════════════════
#  TGCN  — Topological GCN with GRU state update (deepAntigen design)
# ══════════════════════════════════════════════════════════════════════

class TGCN(MessagePassing):
    """GCN layer with GRU cell for temporal smoothing across layers.

    For each node:
      message: MLP(concat(x_j, edge_attr))
      aggregate: sum
      update: GRU(x_0, aggregated_message)

    This is deepAntigen's TGCN (pTCR_seq.py), replacing LocalMessagePassing.
    """

    def __init__(self, hidden_channels: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.message_w = pyg_linear.Linear(
            hidden_channels + 11, hidden_channels,
            weight_initializer="kaiming_uniform"
        )
        self.update_w = pyg_linear.Linear(
            2 * hidden_channels, hidden_channels,
            weight_initializer="kaiming_uniform"
        )
        self.GRU_x = nn.GRUCell(hidden_channels, hidden_channels)

    def forward(self, x_0, edge_index, edge_attr, ibatch):
        x_u = self.propagate(edge_index, x=x_0, edge_attr=edge_attr, size=None)
        x_out = self.GRU_x(x_0, x_u)
        return x_out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        T = self.message_w(torch.cat((x_j, edge_attr), dim=1))
        return F.leaky_relu(T, 0.1)

    def update(self, inputs: Tensor, x) -> Tensor:
        output = self.update_w(torch.cat((inputs, x), dim=1))
        return F.leaky_relu(output, 0.1)


# ══════════════════════════════════════════════════════════════════════
#  SuperNodeExchange  — gated cross-molecule dialogue
# ══════════════════════════════════════════════════════════════════════

class SuperNodeExchange(nn.Module):
    """MHA-weighted atom→super-node pooling, then gated cross-attention.

    1. MHA-weighted pooling: learnable query attends to all atoms per molecule,
       producing weighted-sum super nodes (replaces scatter_mean, aligns with
       deepAntigen's "利用多头注意力机制对原子特征加权，传递给超级节点").
    2. Gate:  σ(W_g · [s_pep, s_tcr])   controls how much foreign info enters
    3. Cross-projection between s_pep and s_tcr (Linear, since 1 super-node
       per molecule makes MHA degenerate to Linear projection).
    4. Gated residual update: s_new = s + gate ⊙ cross_output
    5. Broadcast s_new back to each atom: s_new[batch]
    """

    def __init__(self, hidden_channels: int, n_heads: int = 4):
        super().__init__()
        H = hidden_channels
        # ── MHA atom→super-node pooling ────────────────────────────
        self.attn_q_pep = nn.Parameter(torch.randn(1, H) / math.sqrt(H))
        self.attn_q_tcr = nn.Parameter(torch.randn(1, H) / math.sqrt(H))
        self.attn_k = pyg_linear.Linear(
            H, H, weight_initializer="kaiming_uniform"
        )
        self.attn_v = pyg_linear.Linear(
            H, H, weight_initializer="kaiming_uniform"
        )
        self.scale = math.sqrt(H)

        # ── gate for cross-molecule exchange ────────────────────────
        self.gate_linear = pyg_linear.Linear(
            2 * H, H, weight_initializer="kaiming_uniform"
        )
        # Linear (not MHA): with 1 super-node per molecule, softmax
        # over 1 key is always 1.0 — MHA degenerates to Linear.
        self.cross_proj = pyg_linear.Linear(
            H, H, weight_initializer="kaiming_uniform"
        )

    def _attn_pool(self, x, batch, q):
        """Per-molecule MHA-weighted pooling: q attends to atoms → weighted sum.

        Args:
            x: [N, H] atom features
            batch: [N] molecule index per atom
            q: [1, H] learnable query
        Returns:
            s: [B, H] super node per molecule
        """
        K = self.attn_k(x)  # [N, H]
        V = self.attn_v(x)  # [N, H]

        # Attention scores per atom: q @ K_i
        scores = (q * K).sum(dim=-1) / self.scale  # [N]

        # Per-molecule softmax (numerically stable)
        n_mols = batch.max().item() + 1
        s_max = torch.full((n_mols,), float('-inf'), device=x.device)
        s_max.scatter_reduce_(0, batch, scores, reduce='amax', include_self=False)
        scores_shifted = scores - s_max[batch]
        scores_exp = torch.exp(scores_shifted)
        z = scatter_add(scores_exp, batch, dim=0)  # [B]
        attn_w = scores_exp / (z[batch] + 1e-8)     # [N]

        # Weighted sum: Σ α_i · V_i
        s = scatter_add(attn_w.unsqueeze(-1) * V, batch, dim=0)  # [B, H]
        return s

    def forward(self, pep_x, pep_batch, tcr_x, tcr_batch):
        """
        Returns
        -------
        pep_global_ctx : [n_pep_atoms, H]  broadcast super-node context
        tcr_global_ctx : [n_tcr_atoms, H]
        s_pep_new : [B, H]
        s_tcr_new : [B, H]
        """
        # 1. MHA-weighted atom→super-node pooling (deepAntigen step ②)
        s_pep = self._attn_pool(pep_x, pep_batch, self.attn_q_pep)  # [B, H]
        s_tcr = self._attn_pool(tcr_x, tcr_batch, self.attn_q_tcr)  # [B, H]

        # 2. gate: how much cross-molecule info to incorporate
        gate_pep = torch.sigmoid(
            self.gate_linear(torch.cat([s_pep, s_tcr], dim=-1))
        )
        gate_tcr = torch.sigmoid(
            self.gate_linear(torch.cat([s_tcr, s_pep], dim=-1))
        )

        # 3. cross-molecule projection: s_pep ← tcr info, s_tcr ← pep info
        s_pep_ctx = self.cross_proj(s_tcr)  # [B, H] — tcr context for pep
        s_tcr_ctx = self.cross_proj(s_pep)  # [B, H] — pep context for tcr

        # 4. gated residual update
        s_pep_new = s_pep + gate_pep * s_pep_ctx
        s_tcr_new = s_tcr + gate_tcr * s_tcr_ctx

        # 5. broadcast back to every atom
        pep_global_ctx = s_pep_new[pep_batch]  # [n_pep_atoms, H]
        tcr_global_ctx = s_tcr_new[tcr_batch]  # [n_tcr_atoms, H]

        return pep_global_ctx, tcr_global_ctx, s_pep_new, s_tcr_new


# ══════════════════════════════════════════════════════════════════════
#  CrossModalGCNLayer  — one full iteration of the GCN loop
# ══════════════════════════════════════════════════════════════════════

class CrossModalGCNLayer(nn.Module):
    """One complete GCN layer: LocalMP → Atom→SuperNode(MHA) → CrossExchange → GRU.

    Per-layer computation flow (aligned with deepAntigen_Seq):

      ① LocalMessagePassing (MLP-based, no GRU)
         ┌─ message: MLP(concat(neighbor_j, edge_attr))
         ├─ aggregate: sum over neighbors
         └─ update:   MLP(concat(aggregated, self))  ← pure MLP, no GRU

      ② SuperNodeExchange (MHA-weighted pooling + gated cross-molecule dialogue)
         ┌─ Atom→SuperNode: MHA-weighted pooling (replaces scatter_mean)
         ├─ Gate  = σ(W_g · [s_pep, s_tcr])            ← learned filter
         ├─ Cross-attention: s_pep (query) ⇄ s_tcr (key/value)
         └─ Broadcast: updated super node → every atom

      ③ GRU update (deepAntigen step ④ — the only GRU per layer)
         new_state = GRUCell(prev_state, LocalMP_output + global_context)
    """

    def __init__(self, hidden_channels: int, n_heads: int = 4):
        super().__init__()
        # ① LocalMessagePassing — MLP neighbour aggregation (no GRU, aligns with deepAntigen)
        self.local_mp_pep = LocalMessagePassing(hidden_channels)
        self.local_mp_tcr = LocalMessagePassing(hidden_channels)

        # ② super node: MHA-weighted atom pooling + gated cross-molecule exchange
        self.super_exchange = SuperNodeExchange(hidden_channels, n_heads)

        # ③ GRU: the only temporal smoothing per layer (deepAntigen step ④)
        self.gru_pep = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru_tcr = nn.GRUCell(hidden_channels, hidden_channels)

    def forward(self, pep_x, pep_edge_index, pep_edge_attr, pep_batch,
                tcr_x, tcr_edge_index, tcr_edge_attr, tcr_batch):
        # ① Local message passing — local chemical neighbourhood (no GRU)
        pep_local = self.local_mp_pep(pep_x, pep_edge_index, pep_edge_attr)
        tcr_local = self.local_mp_tcr(tcr_x, tcr_edge_index, tcr_edge_attr)

        # ② MHA-weighted super node creation + gated cross-molecule exchange
        pep_global, tcr_global, s_pep, s_tcr = self.super_exchange(
            pep_local, pep_batch, tcr_local, tcr_batch
        )

        # ③ GRU update: fuse prev_state with (LocalMP_output + global_context)
        pep_x_new = self.gru_pep(pep_x, pep_local + pep_global)
        tcr_x_new = self.gru_tcr(tcr_x, tcr_local + tcr_global)

        return pep_x_new, tcr_x_new, s_pep, s_tcr


# ══════════════════════════════════════════════════════════════════════
#  Top-K pooling helpers  (all-atom selection)
# ══════════════════════════════════════════════════════════════════════

def topk(x, ratio, chems, batch):
    """Select top-k atoms per molecule by score. All atoms participate.

    Returns:
        perm: [sum(k_i)] global flat indices of selected atoms
        on_index: list of all atom global indices (0..total_atoms-1)
        k: [B] tensor, k_i = min(ratio, n_atoms_i)
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    # Dense score matrix [B, max_num_nodes] — ghost positions = -inf, sort to end
    idx_flat = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    dense_idx = (idx_flat - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[dense_idx] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    # Sort descending per molecule — real atoms score > -inf, come first
    _, perm_local = dense_x.sort(dim=-1, descending=True)  # [B, max_num_nodes]

    # k_i = min(ratio, n_atoms_i) — handles molecules with < ratio atoms
    k = torch.full((batch_size,), ratio, device=x.device, dtype=num_nodes.dtype)
    k = torch.min(k, num_nodes)

    # Take top-k per molecule → global un-padded indices
    perm_parts = []
    for i in range(batch_size):
        ki = k[i].item()
        mol_topk = perm_local[i, :ki] + cum_num_nodes[i]
        perm_parts.append(mol_topk)
    perm = torch.cat(perm_parts)

    # All-atom global index list (replaces N/O-only on_index)
    total_atoms = int(num_nodes.sum().item())
    on_index = list(range(total_atoms))

    return perm.long(), on_index, k


def paper_topk(x, ratio, batch):
    """Top-k atom selection per molecule — paper-aligned minimal version.

    Returns only perm (no on_index, no per-mol k values).
    Handles molecules with < ratio atoms via min clamp.
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = num_nodes.new_full((num_nodes.size(0),), ratio)
    k = torch.min(k, num_nodes)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    return perm


class PaperTopKPooling(torch.nn.Module):
    """Paper-aligned TopK pooling — single learnable weight, no positional encoding."""

    def __init__(self, in_channels: int, ratio: int = 1, nonlinearity: Callable = torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.weight = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, batch):
        xx = x.unsqueeze(-1) if x.dim() == 1 else x
        score = (xx * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        perm = paper_topk(score, self.ratio, batch)
        x_top = xx[perm] * score[perm].view(-1, 1)
        bz = batch.max().item() + 1
        # Handle variable k: pad to ratio if needed
        k_list = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
        k_per_mol = torch.min(k_list, torch.full_like(k_list, self.ratio))
        if (k_per_mol != self.ratio).any():
            # Some molecules have < ratio atoms — pad to ratio
            x_parts, idx = [], 0
            for i in range(bz):
                ki = k_per_mol[i].item()
                need = self.ratio - ki
                if need > 0:
                    x_parts.append(torch.cat([
                        x_top[idx:idx+ki],
                        x_top[idx:idx+ki].new_zeros(need, x_top.shape[-1]),
                    ]))
                else:
                    x_parts.append(x_top[idx:idx+ki])
                idx += ki
            x_top = torch.stack(x_parts)
        else:
            x_top = x_top.view(bz, self.ratio, -1)
        return x_top, perm


# ══════════════════════════════════════════════════════════════════════
#  PaperEncoder  — independent per-molecule GCN (paper architecture)
# ══════════════════════════════════════════════════════════════════════

class PaperEncoder(nn.Module):
    """Paper-aligned independent per-molecule GCN encoder.

    depth x (TGCN -> BatchNorm), TopK only at last layer.
    No cross-modal interaction - peptide and CDR3 encoded separately.
    """

    def __init__(self, in_channels: int, hidden_channels: int, depth: int, k: int):
        super().__init__()
        self.init_w = pyg_linear.Linear(
            in_channels, hidden_channels, weight_initializer='kaiming_uniform'
        )
        self.GCN_Depth = depth
        self.gcn = nn.ModuleList([TGCN(hidden_channels) for _ in range(depth)])
        self.top_K_pooling = nn.ModuleList([
            PaperTopKPooling(hidden_channels, ratio=k) for _ in range(depth)
        ])
        self.bn_x = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(depth)])

    def forward(self, graphs):
        x, edge_index, edge_attr, ibatch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        x_l = F.leaky_relu(self.init_w(x), 0.1)
        for i in range(self.GCN_Depth):
            x_l = self.gcn[i](x_l, edge_index, edge_attr, ibatch)
            x_l = self.bn_x[i](x_l)
            if i == self.GCN_Depth - 1:
                fs, perm = self.top_K_pooling[i](x_l, batch=ibatch)
        return fs


# ══════════════════════════════════════════════════════════════════════
#  PositionalEncoding
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, in_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, in_channels, 2).float()
            * (-math.log(10000.0) / in_channels)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, ibatch):
        num_nodes = scatter_add(torch.ones_like(ibatch), ibatch, dim=0)
        index = torch.cat([torch.arange(num) for num in num_nodes])
        x = x.unsqueeze(1)
        x = x + self.pe[index, :]
        x = self.dropout(x)
        x = x.squeeze(1)
        return x


# ══════════════════════════════════════════════════════════════════════
#  TopKPooling  — all-atom top-k selection by learned score
# ══════════════════════════════════════════════════════════════════════

class TopKPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: int = 1,
        nonlinearity: Callable = torch.tanh,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.nonlinearity = nonlinearity

        self.pos_emb = PositionalEncoding(in_channels)

        self.layer_atom = pyg_linear.Linear(
            in_channels, 256, weight_initializer="kaiming_uniform"
        )
        self.layer_atom2 = pyg_linear.Linear(
            256, 128, weight_initializer="kaiming_uniform"
        )
        self.layer_atom3 = pyg_linear.Linear(
            128, 64, weight_initializer="kaiming_uniform"
        )
        self.weight_atom = nn.Parameter(torch.Tensor(1, 64))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_atom)

    def forward(self, x, chems, batch):
        x = self.pos_emb(x, batch)
        xx = x.unsqueeze(-1) if x.dim() == 1 else x
        xx_t = F.leaky_relu(self.layer_atom(xx), 0.1)
        xx_t = F.leaky_relu(self.layer_atom2(xx_t), 0.1)
        xx_t = F.leaky_relu(self.layer_atom3(xx_t), 0.1)
        score = (xx_t * self.weight_atom).sum(dim=-1)
        score = self.nonlinearity(score / self.weight_atom.norm(p=2, dim=-1))
        perm, on_index, k_per_mol = topk(score, self.ratio, chems, batch)
        # perm: [sum(k_i)] flat global atom indices
        # k_per_mol: [B] k_i = min(ratio, n_atoms_i) per molecule
        topk_scores_flat = score[perm]                        # [sum(k_i)]
        x_top_flat = xx[perm] * topk_scores_flat.unsqueeze(-1)

        bz = batch.max().item() + 1
        k_list = k_per_mol.tolist()

        # ── Global → local atom index per molecule ───────────────────
        num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
        cum_atoms = torch.cat([num_nodes.new_zeros(1),
                               num_nodes.cumsum(0)[:-1]], dim=0)  # [B]

        # ── Per-graph split → independent zero-pad → stack ───────────
        # Each molecule gets its own k atoms (no cross-graph contamination)
        x_splits = torch.split(x_top_flat, k_list)
        score_splits = torch.split(topk_scores_flat, k_list)
        perm_splits = torch.split(perm, k_list)

        x_padded, score_padded, perm_padded, valid_list = [], [], [], []
        ratio = self.ratio
        feat_dim = x_top_flat.shape[-1]
        for i in range(bz):
            ki = k_list[i]
            need = ratio - ki
            # convert global perm → local (within molecule i)
            local_perm = perm_splits[i] - cum_atoms[i].item()
            if need > 0:
                x_padded.append(torch.cat([
                    x_splits[i],
                    x_splits[i].new_zeros(need, feat_dim),
                ]))
                score_padded.append(torch.cat([
                    score_splits[i],
                    score_splits[i].new_zeros(need),
                ]))
                perm_padded.append(torch.cat([
                    local_perm,
                    local_perm.new_zeros(need),
                ]))
                valid_list.append(torch.cat([
                    torch.ones(ki, device=x.device),
                    torch.zeros(need, device=x.device),
                ]))
            else:
                x_padded.append(x_splits[i])
                score_padded.append(score_splits[i])
                perm_padded.append(local_perm)
                valid_list.append(torch.ones(ratio, device=x.device))

        x_top = torch.stack(x_padded)            # [B, k, H]
        topk_scores = torch.stack(score_padded)  # [B, k]
        perm_local = torch.stack(perm_padded)    # [B, k]  local indices per molecule
        valid_mask = torch.stack(valid_list)     # [B, k]  1=real, 0=pad

        return x_top, perm_local, score[on_index], on_index, topk_scores, valid_mask


# ══════════════════════════════════════════════════════════════════════
#  MultiHeadAttention  — cross-molecule interaction after TopK
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Cross-molecule attention on top-k atom features.

    output_mode:
      'full' — returns intermap * att [B, k, k, H] (current default)
      'sum'  — returns sum(intermap * att, dim=(1,2)) [B, H] (paper-aligned)
    """

    def __init__(self, hidden_size: int, n_heads: int, output_mode: str = 'full'):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.output_mode = output_mode
        self.W_CDR3 = nn.Linear(hidden_size, hidden_size * n_heads)
        self.W_Peptide = nn.Linear(hidden_size, hidden_size * n_heads)
        self.reset_param()

    def reset_param(self):
        nn.init.xavier_uniform_(self.W_CDR3.weight)
        nn.init.xavier_uniform_(self.W_Peptide.weight)

    def forward(self, peptide, cdr3):
        batch_size = peptide.size(0)

        cdr3_s = (
            self.W_CDR3(cdr3)
            .view(batch_size, -1, self.n_heads, self.hidden_size)
            .transpose(1, 2)
        )
        peptide_s = (
            self.W_Peptide(peptide)
            .view(batch_size, -1, self.n_heads, self.hidden_size)
            .transpose(1, 2)
        )

        scores = (
            torch.matmul(peptide_s, cdr3_s.transpose(-1, -2))
            / self.hidden_size
        )
        scores = torch.mean(scores, dim=1)
        scores_reshape = scores.view(scores.shape[0], -1)
        att = torch.softmax(scores_reshape, dim=1)
        att = att.view(scores.shape[0], scores.shape[1], scores.shape[2])
        att = att.unsqueeze(-1)

        intermap = peptide.unsqueeze(-3) + cdr3.unsqueeze(-2)
        if self.output_mode == 'sum':
            return torch.sum(intermap * att, dim=(1, 2))
        return intermap * att


# ══════════════════════════════════════════════════════════════════════
#  DeepGCN  — joint cross-modal GCN with per-layer SuperNodeExchange
# ══════════════════════════════════════════════════════════════════════

class DeepGCN(nn.Module):
    """Cross-modal GCN for TCR–peptide atom-level interaction.

    Architecture (aligned with deepAntigen_Seq):

      Graph augmentation: 5% node dropout during training
      Initial projection: 25-dim one-hot → HS (LeakyReLU)

      For each of L layers:
        ① LocalMessagePassing (MLP-based, no GRU — deepAntigen step ①)
        ② SuperNodeExchange (MHA-weight atom→super-node + gated cross-attention)
        ③ GRU update (prev_state → new_state, the only GRU per layer — step ④)
        + BatchNorm after each layer

      After L layers:
        TopKPooling (all-atom scoring) → select k key atoms per molecule
        MultiHeadAttention        → interaction map [B, k, k, H]

    Key design: unlike deepAntigen's TCR model (independent encoders),
    here peptide and TCR atoms engage in gated dialogue at every GCN layer.
    Each atom perceives both its local chemical neighbourhood and the
    other molecule's global state throughout the encoding process.
    """

    def __init__(self, args: dict):
        super().__init__()
        K = args["k"]
        HS = args["hidden_size"]
        depth = args["depth"]
        n_heads = args.get("heads", 4)
        in_channels = args.get("in_channels", 25)

        # ── initial projection: atom features → hidden space ─────────
        self.init_w_pep = pyg_linear.Linear(
            in_channels, HS, weight_initializer="kaiming_uniform"
        )
        self.init_w_tcr = pyg_linear.Linear(
            in_channels, HS, weight_initializer="kaiming_uniform"
        )

        # ── stacked cross-modal GCN layers ───────────────────────────
        self.depth = depth
        self.layers = nn.ModuleList([
            CrossModalGCNLayer(HS, n_heads) for _ in range(depth)
        ])
        self.bn_pep = nn.ModuleList([BatchNorm(HS) for _ in range(depth)])
        self.bn_tcr = nn.ModuleList([BatchNorm(HS) for _ in range(depth)])

        # ── TopK pooling after all GCN layers ────────────────────────
        self.topk_pep = TopKPooling(HS, ratio=K)
        self.topk_tcr = TopKPooling(HS, ratio=K)

        # ── cross-molecule attention on top-k features ───────────────
        self.peptide_cdr3_att = MultiHeadAttention(HS, n_heads)
        self.dropout_atom = nn.Dropout(p=0.2)

    def forward(self, peptide_graphs, cdr3_graphs, peptide_chems, cdr3_chems):
        """Full cross-modal GCN forward pass.

        Parameters
        ----------
        peptide_graphs : PyG Batch  atom-level graph batch for peptides
        cdr3_graphs : PyG Batch      atom-level graph batch for TCR CDR3
        peptide_chems : list[rdkit.Chem.Mol]
        cdr3_chems : list[rdkit.Chem.Mol]

        Returns
        -------
        dict with keys:
          peptide_topk : [B, k, H]
          cdr3_topk : [B, k, H]
          p_perm : [B, k]  local atom indices per molecule (ghost pads = 0)
          c_perm : [B, k]
          p_scores : [n_atoms]  TopK scores of all atoms
          c_scores : [n_atoms]
          p_topk_scores : [B, k]  scores of selected top-k atoms (ghost pads = 0)
          c_topk_scores : [B, k]
          p_indexs : all-atom index list (for Stage 1 Pearson loss)
          c_indexs : all-atom index list
          p_valid : [B, k]  1=real atom, 0=ghost padding
          c_valid : [B, k]
          joint_mask : [B, k, k, 1]  1=real atom pair, 0=ghost padding
          interaction_map : [B, k, k, H]  ghost-masked (padded positions = 0)
        """
        # ── initial projection ───────────────────────────────────────
        pep_x = peptide_graphs.x
        pep_ei = peptide_graphs.edge_index
        pep_ea = peptide_graphs.edge_attr
        pep_batch = peptide_graphs.batch
        tcr_x = cdr3_graphs.x
        tcr_ei = cdr3_graphs.edge_index
        tcr_ea = cdr3_graphs.edge_attr
        tcr_batch = cdr3_graphs.batch

        # ── graph augmentation: 5% node dropout (deepAntigen step ①) ──
        if self.training:
            pep_x, pep_ei, pep_ea, pep_batch = self._node_dropout(
                pep_x, pep_ei, pep_ea, pep_batch, drop_prob=0.05
            )
            tcr_x, tcr_ei, tcr_ea, tcr_batch = self._node_dropout(
                tcr_x, tcr_ei, tcr_ea, tcr_batch, drop_prob=0.05
            )

        pep_x = F.leaky_relu(self.init_w_pep(pep_x), 0.1)
        tcr_x = F.leaky_relu(self.init_w_tcr(tcr_x), 0.1)

        # ── per-layer: TGCN → SuperNodeExchange → GRU ────────────────
        for i in range(self.depth):
            pep_x, tcr_x, s_pep, s_tcr = self.layers[i](
                pep_x, pep_ei, pep_ea, pep_batch,
                tcr_x, tcr_ei, tcr_ea, tcr_batch,
            )
            pep_x = self.bn_pep[i](pep_x)
            tcr_x = self.bn_tcr[i](tcr_x)

        # ── TopK pooling (all-atom scoring) ─────────────────────────
        pep_topk, p_perm, p_scores, p_indexs, p_topk_scores, p_valid = self.topk_pep(
            pep_x, peptide_chems, pep_batch
        )
        tcr_topk, c_perm, c_scores, c_indexs, c_topk_scores, c_valid = self.topk_tcr(
            tcr_x, cdr3_chems, tcr_batch
        )
        # p_perm, c_perm: [B, k] — already local per-molecule indices from TopKPooling

        # ── joint valid mask: 1=real atom pair, 0=ghost padding ─────
        # [B, k] ⊗ [B, k] → [B, k, k] → unsqueeze → [B, k, k, 1]
        joint_mask = (p_valid.unsqueeze(-1) * c_valid.unsqueeze(1)).unsqueeze(-1)

        # ── cross-molecule attention on top-k ────────────────────────
        interaction_map = self.peptide_cdr3_att(pep_topk, tcr_topk)
        # Kill ghost-atom contributions: padded positions → zero features
        interaction_map = interaction_map * joint_mask
        interaction_map = self.dropout_atom(interaction_map)

        # ── zero out padded perms for safety ─────────────────────────
        p_perm = p_perm * p_valid.long()
        c_perm = c_perm * c_valid.long()

        return {
            "peptide_topk": pep_topk,            # [B, k, H_gcn]
            "cdr3_topk": tcr_topk,               # [B, k, H_gcn]
            "p_perm": p_perm,                    # [B, k] local atom indices
            "c_perm": c_perm,                    # [B, k]
            "p_scores": p_scores,
            "c_scores": c_scores,
            "p_topk_scores": p_topk_scores,       # [B, k] scores of top-k atoms
            "c_topk_scores": c_topk_scores,       # [B, k]
            "p_indexs": p_indexs,                 # all-atom indices (all, not top-k)
            "c_indexs": c_indexs,
            "p_valid": p_valid,                   # [B, k] 1=real, 0=ghost padding
            "c_valid": c_valid,                   # [B, k]
            "joint_mask": joint_mask,             # [B, k, k, 1]
            "interaction_map": interaction_map,    # [B, k, k, H_gcn] (ghost-masked)
        }

    @staticmethod
    def _node_dropout(x, edge_index, edge_attr, batch, drop_prob=0.05):
        """Randomly drop nodes + incident edges during training (graph augmentation).

        Aligned with deepAntigen: "训练阶段以 5% 概率随机丢弃节点及相连边".
        No-op during eval.
        """
        if drop_prob <= 0:
            return x, edge_index, edge_attr, batch

        n_total = x.size(0)
        keep_mask = torch.rand(n_total, device=x.device) > drop_prob

        # If all nodes would be dropped, keep at least one
        if keep_mask.sum() == 0:
            keep_mask[0] = True

        # Filter edges: keep only edges where BOTH endpoints survive
        src, dst = edge_index[0], edge_index[1]
        edge_keep = keep_mask[src] & keep_mask[dst]

        # Remap surviving node indices
        old_to_new = torch.full((n_total,), -1, dtype=torch.long, device=x.device)
        old_to_new[keep_mask] = torch.arange(keep_mask.sum(), device=x.device)

        new_edge_index = torch.stack([
            old_to_new[src[edge_keep]],
            old_to_new[dst[edge_keep]],
        ], dim=0)

        return (
            x[keep_mask],
            new_edge_index,
            edge_attr[edge_keep],
            batch[keep_mask],
        )

    def freeze_encoder(self):
        """Freeze init projections + all GCN layers + BN.

        Only TopK scoring networks and cross-attention remain trainable.
        This is Stage 1 of the two-stage fine-tuning protocol.
        """
        for p in self.init_w_pep.parameters():
            p.requires_grad = False
        for p in self.init_w_tcr.parameters():
            p.requires_grad = False
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = False
        for bn in self.bn_pep:
            for p in bn.parameters():
                p.requires_grad = False
        for bn in self.bn_tcr:
            for p in bn.parameters():
                p.requires_grad = False
        # Unfreeze TopK scoring + cross-attention
        for p in self.topk_pep.parameters():
            p.requires_grad = True
        for p in self.topk_tcr.parameters():
            p.requires_grad = True
        for p in self.peptide_cdr3_att.parameters():
            p.requires_grad = True

    def freeze_topk(self):
        """Freeze all GCN parameters.

        This is Stage 2 of the two-stage fine-tuning protocol.
        Only the downstream classifier remains trainable.
        """
        for p in self.parameters():
            p.requires_grad = False


# ============================================================================
#  PaperAlignedDeepGCN  — paper's independent encoder architecture
# ============================================================================

class PaperAlignedDeepGCN(nn.Module):
    """Paper-aligned GCN: 2 independent encoders + final MHA only.

    Architecture (pTCR_seq.py, Nature Communications 2025):
      peptide -> PaperEncoder(depth=5, k=20) -> [B, 20, 128]
      cdr3    -> PaperEncoder(depth=5, k=20) -> [B, 20, 128]
      MHA(sum output) -> [B, 128]
      Projector(128->64) + ReLU + Dropout(0.2)
      Classifier(64->1) -> logits

    No per-layer cross-modal exchange. Cross-attention only at final MHA.
    """

    def __init__(self, args: dict):
        super().__init__()
        HS = args['hidden_size']
        depth = args['depth']
        k = args['k']
        heads = args.get('heads', 4)
        in_channels = args.get('in_channels', 25)

        self.peptide_encoder = PaperEncoder(in_channels, HS, depth, k)
        self.cdr3_encoder = PaperEncoder(in_channels, HS, depth, k)
        self.peptide_cdr3_att = MultiHeadAttention(HS, heads, output_mode='sum')
        self.dropout = nn.Dropout(p=0.2)
        self.projector = pyg_linear.Linear(
            HS, int(0.5 * HS), weight_initializer='kaiming_uniform'
        )
        self.classier = pyg_linear.Linear(
            int(0.5 * HS), 1, weight_initializer='kaiming_uniform'
        )

    def forward(self, peptide_graphs, cdr3_graphs):
        peptide_fs = self.peptide_encoder(peptide_graphs)
        cdr3_fs = self.cdr3_encoder(cdr3_graphs)
        peptide_cdr3_intermap = self.peptide_cdr3_att(peptide_fs, cdr3_fs)
        proj = F.relu(self.dropout(self.projector(peptide_cdr3_intermap)))
        logits = self.classier(proj)
        return logits.squeeze(-1)
