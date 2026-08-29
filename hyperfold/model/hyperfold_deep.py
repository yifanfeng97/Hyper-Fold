"""Hyper-Fold-Deep: hierarchical protein-level variant of Hyper-Fold.

Hyper-Fold-Deep is the 4-stage hierarchical Fold-Conv backbone used for
protein-level classification (EC-number prediction and fold classification):

- 4 stages of 2 Fold-Conv blocks each, channels 256 -> 512 -> 1024 -> 2048,
- per-stage geometric radii r = (8, 12, 16, 20) Angstrom (both blocks of a
  stage share the stage radius), sequence window l (paper: l = 21),
- stride-2 sequence-average pooling between stages, so the residue axis is
  pooled L -> L/2 -> L/4 -> L/8,
- global mean pooling after the last block, then an MLP classifier head
  (channels[-1] -> max(channels[-1], num_classes) -> num_classes).

The same head serves both tasks: EC prediction uses num_classes=538 with a
multi-label BCE-with-logits loss, fold classification uses num_classes=1195
with a cross-entropy loss (the three fold test scenarios share the head).

Pooling semantics: nodes are
grouped by ``floor(seq / 2)`` over consecutive nodes; ``x``/``pos`` are
mean-pooled, ``seq`` becomes the group index (so sequence distances shrink by
2x per stage while the sequence window ``l`` stays fixed), ``ori`` is
mean-pooled over the flattened 9-D frame and L2-renormalized, and ``batch``
is max-pooled (the graph id is preserved).

Note on ``l``: the paper uses l = 21 for EC and l = 5 for fold
classification (a wide sequence window overfits the fold task). The
constructor default is ``l = 5``; set ``l=21`` for EC.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .foldconv import FoldConvBlock, _MLP


def _scatter_reduce(x: torch.Tensor, idx: torch.Tensor, num_groups: int,
                    reduce: str) -> torch.Tensor:
    """Pure-torch scatter reduce along dim 0 (replaces torch_scatter)."""
    out = torch.zeros(num_groups, *x.shape[1:], dtype=x.dtype, device=x.device)
    index = idx.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    out.scatter_reduce_(0, index, x, reduce=reduce, include_self=False)
    return out


def _global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global mean pooling over batched graphs."""
    out = torch.zeros(batch.max() + 1, x.size(1), dtype=x.dtype,
                      device=x.device)
    idx = batch.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    out.scatter_reduce_(0, idx, x, reduce='mean', include_self=False)
    return out


class SeqAvgPooling(nn.Module):
    """Stride-2 average pooling along the protein sequence.

    This is the inter-stage residue pooling of Hyper-Fold-Deep
    (L -> L/2 -> L/4 -> L/8).
    """

    def forward(self, x, pos, seq, ori, batch):
        if seq.dim() == 1:
            seq = seq.unsqueeze(-1)
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        num_groups = int(idx[-1].item()) + 1

        x = _scatter_reduce(x, idx, num_groups, 'mean')
        pos = _scatter_reduce(pos, idx, num_groups, 'mean')
        seq = _scatter_reduce(
            torch.div(seq, 2, rounding_mode='floor'), idx, num_groups, 'amax'
        )
        ori = ori.reshape(ori.size(0), -1)
        ori = _scatter_reduce(ori, idx, num_groups, 'mean')
        ori = F.normalize(ori, p=2, dim=-1)
        batch = _scatter_reduce(batch, idx, num_groups, 'amax')

        return x, pos, seq, ori, batch


class HyperFoldDeep(nn.Module):
    """Hyper-Fold-Deep: 4-stage hierarchical Fold-Conv classifier.

    4 stages of 2 Fold-Conv blocks each, channels 256 -> 512 -> 1024 -> 2048,
    radii (8, 12, 16, 20) Angstrom, bottleneck base width 16, residue
    pooling L -> L/2 -> L/4 -> L/8 between stages, global mean pooling, then
    an MLP head. ``num_classes`` selects the task head: 538 for multi-label
    EC-number prediction, 1195 for fold classification. The sequence window
    ``l`` is 21 for EC and 5 for fold classification (the reported configs).
    """

    def __init__(self, num_classes: int = 1195,
                 channels: Optional[List[int]] = None,
                 radii: Optional[List[float]] = None,
                 l: int = 5,
                 base_width: float = 16.0,
                 k_small: int = 8,
                 weightnet_hidden: int = 32,
                 embedding_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        if channels is None:
            channels = [256, 512, 1024, 2048]
        if radii is None:
            # Per-stage radii (8, 12, 16, 20) Angstrom.
            radii = [8.0, 12.0, 16.0, 20.0]
        if len(channels) != len(radii):
            raise ValueError("channels and radii must have the same length")

        self.embedding = nn.Embedding(num_embeddings=21,
                                      embedding_dim=embedding_dim)
        self.local_mean_pool = SeqAvgPooling()

        blocks = []
        in_channels = embedding_dim
        for i, r in enumerate(radii):
            # Two Fold-Conv blocks per stage.
            blocks.append(FoldConvBlock(
                in_channels=in_channels, out_channels=channels[i],
                base_width=base_width, r=r, l=l, k_small=k_small,
                weightnet_hidden=weightnet_hidden, dropout=dropout,
                conv_type='foldconv'))
            blocks.append(FoldConvBlock(
                in_channels=channels[i], out_channels=channels[i],
                base_width=base_width, r=r, l=l, k_small=k_small,
                weightnet_hidden=weightnet_hidden, dropout=dropout,
                conv_type='foldconv'))
            in_channels = channels[i]
        self.blocks = nn.ModuleList(blocks)

        self.classifier = _MLP(in_channels=channels[-1],
                               mid_channels=max(channels[-1], num_classes),
                               out_channels=num_classes, dropout=dropout)

    def forward(self, data=None, *, x=None, pos=None, seq=None, ori=None,
                batch=None) -> torch.Tensor:
        """Run the backbone and classifier head.

        Accepts either a data container with ``x`` (residue-type ids),
        ``pos`` (C-alpha coordinates), ``seq`` (sequence indices), ``ori``
        (orientation frames) and optional ``batch`` attributes, or the same
        tensors as keyword arguments. When ``batch`` is missing, all residues
        are treated as a single protein.
        """
        if data is not None:
            x, pos, seq, ori = data.x, data.pos, data.seq, data.ori
            batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x = self.embedding(x)

        # Local pool after the 2nd block of each of the first three stages;
        # global mean pool after the last block.
        for i, block in enumerate(self.blocks):
            x = block(x, pos, seq, ori, batch)
            if i == len(self.blocks) - 1:
                x = _global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(
                    x, pos, seq, ori, batch)

        return self.classifier(x)
