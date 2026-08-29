"""Ladder-rung layers for the operator-only ablation (Table 4 of the paper).

These layers implement the additive / scalar-gated / channel-gated rungs of
the expressivity ladder under the identical architecture: identical graphs,
edge features, and training recipe as Fold-Conv — only the way geometry
modulates neighbor content differs (additive < scalar-gated < channel-gated
< matrix-gated).
"""
import math

import torch
from torch import nn, Tensor

from .foldconv import FoldConv, GKNet


def _segment_softmax(logit: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    """Softmax of per-edge logits over the incoming edges of each node."""
    max_per = torch.full((num_nodes,), float('-inf'), dtype=logit.dtype,
                         device=logit.device)
    max_per.scatter_reduce_(0, dst, logit, reduce='amax', include_self=True)
    e = (logit - max_per[dst]).exp()
    denom = torch.zeros(num_nodes, dtype=logit.dtype, device=logit.device)
    denom.scatter_add_(0, dst, e)
    return e / (denom[dst] + 1e-16)


class ScalarGatedConv(FoldConv):
    """Scalar-gated (attention-style) layer: sum_j alpha_ij * W_v x_j.

    Same GK-Net trunk and smooth gate as Fold-Conv, but the gate is a
    single scalar per edge, softmax-normalized over the neighborhood. This is
    the scalar-gated rung of the expressivity ladder (attention variants).
    """

    def __init__(self, r: float, l: int, in_channels: int, out_channels: int,
                 self_loops: bool = True, weightnet_hidden: int = 32):
        nn.Module.__init__(self)
        self.r = r
        self.l = l
        self.self_loops = self_loops
        self.gate_net = GKNet(l, edge_dim=7,
                              hidden_dim=weightnet_hidden, out_dim=1)
        self.W_v = nn.Linear(in_channels, in_channels, bias=False)
        self.out = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in (self.W_v, self.out):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        self.gate_net.reset_parameters()

    def forward(self, x: Tensor, pos: Tensor, seq: Tensor, ori: Tensor,
                batch: Tensor) -> Tensor:
        if seq.dim() == 1:
            seq = seq.unsqueeze(-1)
        edge_index = self._build_edges(pos, batch)
        src, dst, delta, seq_idx, smooth = self._edge_features(
            edge_index, pos, seq, ori)

        h = self.W_v(x)
        logit = (self.gate_net(delta, seq_idx) * smooth).squeeze(-1)
        alpha = _segment_softmax(logit, dst, x.size(0)).to(h.dtype)
        msg = alpha.unsqueeze(-1) * h[src]

        out = torch.zeros(x.size(0), self.out.in_features, dtype=msg.dtype,
                          device=msg.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(msg), msg,
                            reduce='sum', include_self=False)
        return self.out(out)


class AdditiveConv(FoldConv):
    """Additive message passing: sum_j (W_v x_j + U delta_ij).

    Geometry and content enter through separate linear maps and are summed —
    there is no content-geometry binding. This is the additive rung of the
    ladder under the identical architecture.
    """

    def __init__(self, r: float, l: int, in_channels: int, out_channels: int,
                 self_loops: bool = True, weightnet_hidden: int = 32):
        nn.Module.__init__(self)
        self.r = r
        self.l = l
        self.self_loops = self_loops
        self.W_v = nn.Linear(in_channels, in_channels, bias=False)
        self.U = nn.Linear(7, in_channels, bias=False)
        self.out = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in (self.W_v, self.U, self.out):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, pos: Tensor, seq: Tensor, ori: Tensor,
                batch: Tensor) -> Tensor:
        if seq.dim() == 1:
            seq = seq.unsqueeze(-1)
        edge_index = self._build_edges(pos, batch)
        src, dst, delta, seq_idx, smooth = self._edge_features(
            edge_index, pos, seq, ori)

        msg = self.W_v(x)[src] + self.U(delta.to(x.dtype))

        out = torch.zeros(x.size(0), self.out.in_features, dtype=msg.dtype,
                          device=msg.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(msg), msg,
                            reduce='sum', include_self=False)
        return self.out(out)


class ChannelGatedConv(FoldConv):
    """Channel-gated (continuous-filter) layer: w_ij hadamard x_j.

    Same GK-Net trunk and smooth gate as Fold-Conv, but the gate is a
    per-channel vector: geometry modulates each channel independently without
    cross-channel mixing. This is the channel-gated rung of the ladder.
    """

    def __init__(self, r: float, l: int, in_channels: int, out_channels: int,
                 self_loops: bool = True, weightnet_hidden: int = 32):
        nn.Module.__init__(self)
        self.r = r
        self.l = l
        self.self_loops = self_loops
        self.filter_net = GKNet(l, edge_dim=7,
                                hidden_dim=weightnet_hidden,
                                out_dim=in_channels)
        self.out = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.out.weight, a=math.sqrt(5))
        self.filter_net.reset_parameters()

    def forward(self, x: Tensor, pos: Tensor, seq: Tensor, ori: Tensor,
                batch: Tensor) -> Tensor:
        if seq.dim() == 1:
            seq = seq.unsqueeze(-1)
        edge_index = self._build_edges(pos, batch)
        src, dst, delta, seq_idx, smooth = self._edge_features(
            edge_index, pos, seq, ori)

        w = (self.filter_net(delta, seq_idx) * smooth).to(x.dtype)
        msg = w * x[src]

        out = torch.zeros(x.size(0), self.out.in_features, dtype=msg.dtype,
                          device=msg.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(msg), msg,
                            reduce='sum', include_self=False)
        return self.out(out)
