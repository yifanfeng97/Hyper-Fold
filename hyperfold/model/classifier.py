"""Hyper-Fold: the flat 6-block residue-level backbone for EC classification.

This is the model the paper calls **Hyper-Fold** (as opposed to the
hierarchical Hyper-Fold-Deep in :mod:`.hyperfold_deep`):

- residue embedding (21 -> 16),
- 6 Fold-Conv blocks at width 512, no pooling, geometric radii
  r = (8, 8, 8, 10, 10, 10) Angstrom and sequence windows
  l = (17, 17, 17, 21, 21, 21),
- readout of blocks {3, 6} (1-indexed): each readout level is global
  mean-pooled and the two are concatenated -> R^1024,
- MLP classifier head (BN -> LeakyReLU -> Dropout -> Linear -> BN ->
  LeakyReLU -> Linear) mapping R^1024 -> R^num_classes.

The readout set is configurable via ``readout_blocks`` (1-indexed); the
default (3, 6) reproduces the reference configuration exactly.
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .foldconv import FoldConvBlock, _MLP


def _global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global mean pooling over batched graphs."""
    out = torch.zeros(batch.max() + 1, x.size(1), dtype=x.dtype,
                      device=x.device)
    idx = batch.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    out.scatter_reduce_(0, idx, x, reduce='mean', include_self=False)
    return out


class HyperFoldClassifier(nn.Module):
    """Flat 6-block Hyper-Fold classifier (the paper's EC backbone).

    6 Fold-Conv blocks at width 512 with radii (8, 8, 8, 10, 10, 10) Angstrom
    and sequence windows (17, 17, 17, 21, 21, 21); the outputs of the
    ``readout_blocks`` (1-indexed, default blocks 3 and 6) are global
    mean-pooled and concatenated (2 * 512 = 1024 dim) before the MLP head.
    ``num_classes`` selects the task head (538 for EC-number prediction).
    """

    def __init__(self, num_classes: int = 538,
                 rs: Optional[Sequence[float]] = None,
                 ls: Optional[Sequence[int]] = None,
                 channels: int = 512,
                 base_width: float = 32.0,
                 readout_blocks: Tuple[int, ...] = (3, 6),
                 embedding_dim: int = 16,
                 k_small: int = 8,
                 weightnet_hidden: int = 32,
                 dropout: float = 0.2,
                 momentum: float = 0.2):
        super().__init__()
        if rs is None:
            rs = [8.0, 8.0, 8.0, 10.0, 10.0, 10.0]
        if ls is None:
            ls = [17, 17, 17, 21, 21, 21]
        rs, ls = list(rs), list(ls)
        if len(rs) != 6 or len(ls) != 6:
            raise ValueError("rs and ls must both have length 6")
        if not readout_blocks:
            raise ValueError("readout_blocks must be non-empty")
        for b in readout_blocks:
            if not 1 <= b <= 6:
                raise ValueError(
                    f"readout block {b} out of range; use 1-indexed blocks 1-6"
                )
        self.readout_blocks = tuple(readout_blocks)

        self.embedding = nn.Embedding(num_embeddings=21,
                                      embedding_dim=embedding_dim)
        self.blocks = nn.ModuleList([
            FoldConvBlock(in_channels=embedding_dim if i == 0 else channels,
                          out_channels=channels,
                          base_width=base_width,
                          r=rs[i], l=ls[i], k_small=k_small,
                          weightnet_hidden=weightnet_hidden, dropout=dropout,
                          momentum=momentum, conv_type='foldconv')
            for i in range(6)
        ])
        readout_dim = len(self.readout_blocks) * channels
        self.classifier = _MLP(in_channels=readout_dim,
                               mid_channels=max(readout_dim, num_classes),
                               out_channels=num_classes, dropout=dropout,
                               momentum=momentum)

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

        hiddens = []
        for i, block in enumerate(self.blocks):
            x = block(x, pos, seq, ori, batch)
            if i + 1 in self.readout_blocks:
                hiddens.append(_global_mean_pool(x, batch))

        out = torch.cat(hiddens, dim=-1)
        return self.classifier(out)
