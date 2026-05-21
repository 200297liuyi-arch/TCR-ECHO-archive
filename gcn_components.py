"""GCN components for atom-level graph processing with cross-modal exchange.

Architecture per layer:
  ① TGCN (deepAntigen GRU message passing) — local chemical neighbourhood
  ② SuperNodeExchange — s_pep ↔ s_tcr gated cross-molecule dialogue
  ③ GRU update — combine (prev_state, TGCN_output + global_context)
  + BatchNorm after each layer

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
    """Create super nodes via global pooling, then gated cross-attention.

    1. Super node = scatter_mean(atom_features, batch) for each molecule
    2. Gate:  σ(W_g · [s_pep, s_tcr])   controls how much foreign info enters
    3. Cross-attention between s_pep and s_tcr
    4. Gated residual update: s_new = s + gate ⊙ attention_output
    5. Broadcast s_new back to each atom: s_new[batch]
    """

    def __init__(self, hidden_channels: int, n_heads: int = 4):
        super().__init__()
        self.gate_linear = pyg_linear.Linear(
            2 * hidden_channels, hidden_channels,
            weight_initializer="kaiming_uniform"
        )
        # Use Linear instead of MultiheadAttention — with 1 super-node per
        # molecule, softmax over 1 key is always 1.0, so MHA degenerates
        # to a linear projection. Linear is strictly more efficient.
        self.cross_proj = pyg_linear.Linear(
            hidden_channels, hidden_channels,
            weight_initializer="kaiming_uniform"
        )

    def forward(self, pep_x, pep_batch, tcr_x, tcr_batch):
        """
        Returns
        -------
        pep_global_ctx : [n_pep_atoms, H]  broadcast super-node context
        tcr_global_ctx : [n_tcr_atoms, H]
        s_pep_new : [B, H]
        s_tcr_new : [B, H]
        """
        B = pep_batch.max().item() + 1

        # 1. create super nodes
        s_pep = scatter_mean(pep_x, pep_batch, dim=0)  # [B, H]
        s_tcr = scatter_mean(tcr_x, tcr_batch, dim=0)  # [B, H]

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
    """One complete GCN layer: TGCN → SuperNodeExchange → GRU.

    Per-layer computation flow (for both peptide and TCR):

      ① TGCN (deepAntigen's GRU message passing)
         ┌─ message: MLP(concat(neighbor_j, edge_attr))
         ├─ aggregate: sum over neighbors
         └─ update:   GRUCell(old_state, aggregated)  ← temporal smoothing

      ② SuperNodeExchange (gated cross-molecule dialogue)
         ┌─ Super node = scatter_mean(all_atom_features) per molecule
         ├─ Gate  = σ(W_g · [s_pep, s_tcr])            ← learned filter
         ├─ Cross-attention: s_pep (query) ⇄ s_tcr (key/value)
         └─ Broadcast: updated super node → every atom

      ③ Final GRU update
         new_state = GRUCell(prev_state, TGCN_output + global_context)

      This means each atom simultaneously perceives:
        - Its local chemical neighbourhood  (via TGCN message passing)
        - The other molecule's global state  (via SuperNodeExchange broadcast)
    """

    def __init__(self, hidden_channels: int, n_heads: int = 4):
        super().__init__()
        # ① TGCN replaces LocalMessagePassing — adds GRU temporal smoothing
        self.tgcn_pep = TGCN(hidden_channels)
        self.tgcn_tcr = TGCN(hidden_channels)

        # ② super node creation + gated cross-molecule exchange
        self.super_exchange = SuperNodeExchange(hidden_channels, n_heads)

        # ③ final GRU: fuse prev_state with (TGCN_output + global_context)
        self.gru_pep = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru_tcr = nn.GRUCell(hidden_channels, hidden_channels)

    def forward(self, pep_x, pep_edge_index, pep_edge_attr, pep_batch,
                tcr_x, tcr_edge_index, tcr_edge_attr, tcr_batch):
        # ① TGCN message passing — local chemical neighbourhood
        pep_tgcn = self.tgcn_pep(pep_x, pep_edge_index, pep_edge_attr, pep_batch)
        tcr_tgcn = self.tgcn_tcr(tcr_x, tcr_edge_index, tcr_edge_attr, tcr_batch)

        # ② super node creation + gated cross-molecule exchange
        pep_global, tcr_global, s_pep, s_tcr = self.super_exchange(
            pep_tgcn, pep_batch, tcr_tgcn, tcr_batch
        )

        # ③ GRU update: combine prev_state + local(TGCN) + global(cross-mol)
        pep_x_new = self.gru_pep(pep_x, pep_tgcn + pep_global)
        tcr_x_new = self.gru_tcr(tcr_x, tcr_tgcn + tcr_global)

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
    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
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
        return intermap * att  # [B, k_pep, k_cdr3, H]


# ══════════════════════════════════════════════════════════════════════
#  DeepGCN  — joint cross-modal GCN with per-layer SuperNodeExchange
# ══════════════════════════════════════════════════════════════════════

class DeepGCN(nn.Module):
    """Cross-modal GCN for TCR–peptide atom-level interaction.

    Architecture — joint encoder with per-layer cross-molecule dialogue:

      For each of L layers:
        ① TGCN (deepAntigen GRU message passing, pep + tcr independently)
        ② SuperNodeExchange (s_pep ↔ s_tcr gated cross-attention)
        ③ GRU update (prev_state → new_state with TGCN_output + global_context)
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
        pep_x = F.leaky_relu(self.init_w_pep(peptide_graphs.x), 0.1)
        tcr_x = F.leaky_relu(self.init_w_tcr(cdr3_graphs.x), 0.1)

        pep_ei = peptide_graphs.edge_index
        pep_ea = peptide_graphs.edge_attr
        pep_batch = peptide_graphs.batch
        tcr_ei = cdr3_graphs.edge_index
        tcr_ea = cdr3_graphs.edge_attr
        tcr_batch = cdr3_graphs.batch

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
