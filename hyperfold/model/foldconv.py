"""Fold-Conv operator of Hyper-Fold.

Fold-Conv is a rank-K matrix-gated convolution over two hyperedges:

- a *sequence hyperedge* connecting neighbors with |s_i - s_j| <= l/2, and
- a *contact hyperedge* connecting the remaining spatial neighbors within
  radius r.

Each edge carries a 7-dim geometry feature ``delta_ij = (pos_ij, ori_ij,
dis_ij)`` built by the geometric feature builder (local position, relative
orientation frame, and distance). The geometric kernel network (GK-Net) maps
``(delta_ij, bucketed Delta_s_ij)`` to K gate coefficients, modulated by a
smooth edge gate ``sigma_ij = 0.5 * (1 - tanh(16 * d_bar * s_bar - 14))``.
The output is ``h_i = sum_j P_c(sigma_ij * g(delta_ij, Delta_s_ij) kron
W_v x_j)``.

The module also contains the bottleneck pre-activation residual unit
(:class:`FoldConvBlock`) and the local orientation-frame builder used to
construct the edge geometry.
"""
import math

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# graph utils
# ---------------------------------------------------------------------------

def radius(pos: Tensor, r: float, batch: Tensor):
    """Ball query for batched point clouds. Returns (row, col) edge indices."""
    already_sorted = (batch[:-1] <= batch[1:]).all().item()
    if not already_sorted:
        order = batch.argsort(stable=True)
        pos = pos[order]
        batch = batch[order]
    else:
        order = None

    sizes = batch.bincount()
    B = sizes.numel()
    M = int(sizes.max().item())
    if M > 0 and M * B <= 500000:
        ptr = sizes.cumsum(0, dtype=torch.long).roll(1)
        ptr[0] = 0
        arange_m = torch.arange(M, device=pos.device, dtype=torch.long)
        valid = arange_m.unsqueeze(0) < sizes.unsqueeze(1)
        padded = torch.zeros(B, M, pos.size(1), dtype=pos.dtype, device=pos.device)
        padded[valid] = pos
        dist = torch.cdist(padded, padded, p=2)
        mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (dist <= r)
        b_idx, src, dst = mask.nonzero(as_tuple=True)
        row = ptr[b_idx] + src
        col = ptr[b_idx] + dst
        if order is not None:
            row = order[row]
            col = order[col]
        return row, col

    row_list, col_list = [], []
    for g in torch.unique(batch, sorted=True):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        d = torch.cdist(pos[idx], pos[idx], p=2)
        src, dst = (d <= r).nonzero(as_tuple=True)
        row_list.append(idx[src])
        col_list.append(idx[dst])
    if len(row_list) == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pos.device)
        return empty, empty
    return torch.cat(row_list, dim=0), torch.cat(col_list, dim=0)


def remove_self_loops(edge_index: Tensor):
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def add_self_loops(edge_index: Tensor, num_nodes: int):
    loop_index = torch.arange(num_nodes, dtype=edge_index.dtype,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop_index], dim=1)


# ---------------------------------------------------------------------------
# Fold-Conv layer
# ---------------------------------------------------------------------------

def _kaiming_uniform(tensor: Tensor, size):
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class GKNet(nn.Module):
    """Geometric kernel network (GK-Net): maps the 7-dim edge geometry and
    the bucketed sequence displacement to K kernel coefficients; the first
    layer is indexed by the discretized sequence bucket (bucketed weight
    network)."""

    def __init__(self, l: int, edge_dim: int = 7, hidden_dim: int = 32,
                 out_dim: int = 1, num_mlp_layers: int = 2,
                 negative_slope: float = 0.2):
        super().__init__()
        self.l = l

        self.W_offset = nn.Parameter(torch.empty(l, edge_dim, hidden_dim))
        self.b_offset = nn.Parameter(torch.empty(l, hidden_dim))

        layers = []
        dims = [hidden_dim] * (num_mlp_layers - 1) + [out_dim]
        for i, d in enumerate(dims):
            if i > 0:
                layers.append(nn.LeakyReLU(negative_slope))
            layers.append(nn.Linear(dims[i - 1] if i > 0 else hidden_dim, d))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        _kaiming_uniform(self.W_offset.data,
                         size=[self.l, 7, self.W_offset.shape[-1]])
        self.b_offset.data.fill_(0.0)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, delta: Tensor, seq_idx: Tensor) -> Tensor:
        W = self.W_offset[seq_idx]
        b = self.b_offset[seq_idx]
        delta = delta.to(W.dtype)
        h = torch.bmm(delta.unsqueeze(1), W).squeeze(1) + b
        h = torch.nn.functional.leaky_relu(h, 0.2)
        return self.mlp(h)


class FoldConv(nn.Module):
    """Rank-K matrix-gated Fold-Conv layer.

    h_i = sum_j P_c(sigma_ij * g(delta_ij, Delta_s_ij) kron W_v x_j),
    where g is the GK-Net, sigma_ij the smooth edge gate, and P_c a
    pointwise projection.
    """

    def __init__(self, r: float, l: int, in_channels: int, out_channels: int,
                 k_small: int = 8, self_loops: bool = True,
                 weightnet_hidden: int = 32):
        super().__init__()
        self.r = r
        self.l = l
        self.k_small = k_small
        self.self_loops = self_loops

        self.gate_net = GKNet(l, edge_dim=7, hidden_dim=weightnet_hidden,
                              out_dim=k_small)
        self.W_v = nn.Linear(in_channels, in_channels, bias=False)
        self.pointwise = nn.Linear(k_small * in_channels, out_channels,
                                   bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        self.gate_net.reset_parameters()

    def _build_edges(self, pos, batch):
        row, col = radius(pos, self.r, batch)
        edge_index = torch.stack([row, col], dim=0)
        if self.self_loops:
            edge_index = remove_self_loops(edge_index)
            edge_index = add_self_loops(edge_index, num_nodes=pos.size(0))
        return edge_index

    def _edge_features(self, edge_index, pos, seq, ori):
        src, dst = edge_index[0], edge_index[1]
        pos_diff = pos[src] - pos[dst]
        distance = torch.norm(pos_diff, p=2, dim=-1, keepdim=True)
        pos_diff = pos_diff / (distance + 1e-9)

        ori_dst = ori[dst].reshape(-1, 3, 3)
        pos_local = torch.matmul(ori_dst, pos_diff.unsqueeze(2)).squeeze(2)

        ori_src = ori[src].reshape(-1, 3, 3)
        ori_rel = torch.sum(ori_dst * ori_src, dim=2)

        normed_distance = distance / self.r
        seq_diff = seq[src] - seq[dst]
        s = self.l // 2
        seq_diff = torch.clamp(seq_diff, min=-s, max=s)
        seq_idx = (seq_diff + s).squeeze(1).to(torch.int64)
        normed_length = torch.abs(seq_diff) / s

        delta = torch.cat([pos_local, ori_rel, distance], dim=1)
        smooth = 0.5 - torch.tanh(normed_distance * normed_length * 16.0 - 14.0) * 0.5
        return src, dst, delta, seq_idx, smooth

    def forward(self, x: Tensor, pos: Tensor, seq: Tensor, ori: Tensor,
                batch: Tensor) -> Tensor:
        if seq.dim() == 1:
            seq = seq.unsqueeze(-1)
        edge_index = self._build_edges(pos, batch)
        src, dst, delta, seq_idx, smooth = self._edge_features(
            edge_index, pos, seq, ori)

        h = self.W_v(x)
        kw = (self.gate_net(delta, seq_idx) * smooth).to(h.dtype)
        msg_op = torch.matmul(kw.unsqueeze(2), h[src].unsqueeze(1))
        msg = msg_op.reshape(msg_op.size(0), -1)
        msg = self.pointwise(msg)

        out = torch.zeros(x.size(0), msg.size(1), dtype=msg.dtype,
                          device=msg.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(msg), msg,
                            reduce='sum', include_self=False)
        return out


# ---------------------------------------------------------------------------
# residual building blocks
# ---------------------------------------------------------------------------

class _SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that skips normalization for a single training sample."""

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.size(0) == 1:
            return x
        return super().forward(x)


class _Linear(nn.Module):
    """BN -> LeakyReLU -> Dropout -> Linear."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0, bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1, momentum: float = 0.2):
        super().__init__()
        self.module = nn.Sequential(
            _SafeBatchNorm1d(in_channels, momentum=momentum),
            nn.LeakyReLU(leakyrelu_negative_slope),
            nn.Dropout(dropout),
            nn.Linear(in_channels, out_channels, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class _MLP(nn.Module):
    """BN -> LeakyReLU -> Dropout -> Linear (-> BN -> LeakyReLU -> Linear)."""

    def __init__(self, in_channels: int, mid_channels, out_channels: int,
                 dropout: float = 0.0, bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2, momentum: float = 0.2):
        super().__init__()
        module = [
            _SafeBatchNorm1d(in_channels, momentum=momentum),
            nn.LeakyReLU(leakyrelu_negative_slope),
            nn.Dropout(dropout),
        ]
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias=bias))
        module.append(_SafeBatchNorm1d(
            out_channels if mid_channels is None else mid_channels,
            momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class FoldConvBlock(nn.Module):
    """Bottleneck pre-activation residual unit wrapping a geometry-conv layer.

    Supports conv_type='foldconv' (default) plus the three expressivity-ladder
    rungs from :mod:`.rungs` ('scalar', 'additive', 'channel').
    """

    CONV_CLASSES = {'foldconv': FoldConv}

    def __init__(self, in_channels: int, out_channels: int,
                 base_width: float = 32.0, r: float = 10.0, l: int = 21,
                 k_small: int = 8, weightnet_hidden: int = 32,
                 dropout: float = 0.0, momentum: float = 0.2,
                 conv_type: str = 'foldconv'):
        super().__init__()
        width = int(out_channels * (base_width / 64.0))
        if in_channels != out_channels:
            self.identity = _Linear(in_channels, out_channels,
                                    dropout=dropout, bias=False,
                                    leakyrelu_negative_slope=0.1,
                                    momentum=momentum)
        else:
            self.identity = nn.Identity()

        self.input = _MLP(in_channels, None, width, dropout=dropout,
                          bias=False, leakyrelu_negative_slope=0.1,
                          momentum=momentum)
        if conv_type == 'foldconv':
            self.conv = FoldConv(r=r, l=l, in_channels=width,
                                 out_channels=width, k_small=k_small,
                                 weightnet_hidden=weightnet_hidden)
        else:
            # lazy import to avoid a circular dependency with .rungs
            from .rungs import AdditiveConv, ChannelGatedConv, ScalarGatedConv
            conv_classes = {**self.CONV_CLASSES,
                            'scalar': ScalarGatedConv,
                            'channel': ChannelGatedConv,
                            'additive': AdditiveConv}
            cls = conv_classes[conv_type]
            self.conv = cls(r=r, l=l, in_channels=width, out_channels=width,
                            weightnet_hidden=weightnet_hidden)
        self.output = _Linear(width, out_channels, dropout=dropout, bias=False,
                              leakyrelu_negative_slope=0.1, momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        return self.output(x) + identity


# ---------------------------------------------------------------------------
# orientation frames
# ---------------------------------------------------------------------------

def _normalize_l2(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def orientation(pos: Tensor) -> Tensor:
    """Local orientation frames from CA coordinates. pos: [N, 3]."""
    u = _normalize_l2(pos[1:] - pos[:-1])
    u1, u2 = u[1:], u[:-1]
    b = _normalize_l2(u2 - u1)
    n = _normalize_l2(torch.cross(u2, u1, dim=-1))
    o = _normalize_l2(torch.cross(b, n, dim=-1))
    ori = torch.stack([b, n, o], dim=1)
    return torch.cat([ori[:1], ori, ori[-1:]], dim=0)
