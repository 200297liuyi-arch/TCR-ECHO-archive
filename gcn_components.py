"""GCN components for atom-level graph processing with cross-modal exchange.

Architecture per layer:
  ① TGCN (deepAntigen GRU message passing) — local chemical neighbourhood
  ② SuperNodeExchange — s_pep ↔ s_tcr gated cross-molecule dialogue
  ③ GRU update — combine (prev_state, TGCN_output + global_context)
  + BatchNorm after each layer

After all L layers:
  TopKPooling (N/O-biased) → MultiHeadAttention → interaction_map [B,k,k,H]

Key insight: unlike deepAntigen's TCR model (independent encoders, cross-attn
only at the end), here peptide and TCR atoms engage in gated dialogue at EVERY
layer. Each atom perceives both its local chemical neighbourhood AND the other
molecule's global state throughout the encoding process.
"""

import math
from typing import Callable
from itertools import accumulate

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
        self.cross_attn = nn.MultiheadAttention(
            hidden_channels, n_heads, batch_first=True
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

        # 3. cross-attention: s_pep queries s_tcr's context, and vice versa
        s_pep_ctx, _ = self.cross_attn(
            s_pep.unsqueeze(1),    # query: pep super node
            s_tcr.unsqueeze(1),    # key:   tcr super node
            s_tcr.unsqueeze(1),    # value: tcr super node
        )  # [B, 1, H]
        s_tcr_ctx, _ = self.cross_attn(
            s_tcr.unsqueeze(1),
            s_pep.unsqueeze(1),
            s_pep.unsqueeze(1),
        )  # [B, 1, H]

        s_pep_ctx = s_pep_ctx.squeeze(1)  # [B, H]
        s_tcr_ctx = s_tcr_ctx.squeeze(1)  # [B, H]

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
#  Top-K pooling helpers  (with N / O atom constraint)
# ══════════════════════════════════════════════════════════════════════

def generate_O_N(chems, max_num_nodes):
    index_parallel = []
    index = []
    num_nodes = []
    cum_atom_num = 0
    for i, chem in enumerate(chems):
        atom_index_parallel = [
            idx + i * max_num_nodes
            for idx, atom in enumerate(chem.GetAtoms())
            if atom.GetSymbol() in ["N", "O"]
        ]
        index_parallel.extend(atom_index_parallel)
        atom_index = [
            idx + cum_atom_num
            for idx, atom in enumerate(chem.GetAtoms())
            if atom.GetSymbol() in ["N", "O"]
        ]
        num_nodes.append(len(atom_index))
        index.extend(atom_index)
        cum_atom_num += len(chem.GetAtoms())
    return index_parallel, num_nodes, index


def topk(x, ratio, chems, batch):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )
    rich_num_nodes = torch.full((batch_size,), max_num_nodes, device=x.device)
    cum_rich_num_nodes = torch.cat(
        [rich_num_nodes.new_zeros(1), rich_num_nodes.cumsum(dim=0)[:-1]], dim=0
    )
    void_num_nodes = max_num_nodes - num_nodes
    cum_void_num_nodes = torch.cat(
        [void_num_nodes.new_zeros(1), void_num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_rich_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    on_index_parallel, on_num, on_index = generate_O_N(chems, max_num_nodes)
    on_index_parallel = torch.LongTensor(on_index_parallel).to(x.device)
    offset = list(accumulate(on_num))
    offset = [0] + offset[:-1]

    indices = torch.where(perm == on_index_parallel[:, None])[1]
    indices, _ = indices.sort()

    k = num_nodes.new_full((num_nodes.size(0),), ratio)
    on_num_t = torch.tensor(on_num, device=x.device)
    offset_t = torch.tensor(offset, device=x.device)
    k = torch.min(k, on_num_t)
    pre_mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + offset_t[i]
        for i in range(batch_size)
    ]
    pre_mask = torch.cat(pre_mask, dim=0)
    mask = indices[pre_mask.long()]

    perm = perm.view(batch_size, max_num_nodes)
    perm = perm - cum_void_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    perm = perm[mask]

    return perm.long(), on_index


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
#  TopKPooling  — with N/O constraint
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
        perm, on_index = topk(score, self.ratio, chems, batch)
        x_top = xx[perm] * score[perm].view(-1, 1)
        bz = batch.max().item() + 1
        x_top = x_top.view(bz, self.ratio, -1)
        return x_top, perm, score[on_index], on_index


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
        TopKPooling (N/O-biased)  → select k key atoms per molecule
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
          p_perm : [B*k]  local atom indices within peptide molecules
          c_perm : [B*k]  local atom indices within TCR molecules
          p_scores : [B*k]  TopK attention scores (N/O atoms only)
          c_scores : [B*k]
          p_indexs : N/O atom index list (for structure loss)
          c_indexs : N/O atom index list
          interaction_map : [B, k, k, H]
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

        # ── TopK pooling (N/O-biased) ────────────────────────────────
        pep_topk, p_perm, p_scores, p_indexs = self.topk_pep(
            pep_x, peptide_chems, pep_batch
        )
        tcr_topk, c_perm, c_scores, c_indexs = self.topk_tcr(
            tcr_x, cdr3_chems, tcr_batch
        )

        # ── local atom indices within each molecule ──────────────────
        num_nodes_pep = scatter_add(
            torch.ones_like(pep_batch), pep_batch, dim=0
        )
        num_nodes_tcr = scatter_add(
            torch.ones_like(tcr_batch), tcr_batch, dim=0
        )
        p_perm_local = torch.zeros_like(p_perm)
        c_perm_local = torch.zeros_like(c_perm)
        for i, idx in enumerate(p_perm):
            g = pep_batch[idx]
            p_perm_local[i] = idx - sum(num_nodes_pep[:g.item()])
        for i, idx in enumerate(c_perm):
            g = tcr_batch[idx]
            c_perm_local[i] = idx - sum(num_nodes_tcr[:g.item()])

        # ── cross-molecule attention on top-k ────────────────────────
        interaction_map = self.peptide_cdr3_att(pep_topk, tcr_topk)

        return {
            "peptide_topk": pep_topk,            # [B, k, H_gcn]
            "cdr3_topk": tcr_topk,               # [B, k, H_gcn]
            "p_perm": p_perm_local,              # local atom indices
            "c_perm": c_perm_local,
            "p_scores": p_scores,
            "c_scores": c_scores,
            "p_indexs": p_indexs,                # N/O atom indices
            "c_indexs": c_indexs,
            "interaction_map": interaction_map,   # [B, k, k, H_gcn]
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
