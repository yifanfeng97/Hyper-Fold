"""Positional encodings for Hyper-Fold."""
import torch
from torch import nn, Tensor


class FourierPositionalEncoding3D(nn.Module):
    """Three-dimensional Fourier positional encoding (3D-PE):
    gamma(a) = [sin(2^m a), cos(2^m a)] for m=0..7 per axis -> 48-dim, then
    a learned linear projection to d_model."""

    def __init__(self, d_model: int, num_freqs: int = 8, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(3 * 2 * num_freqs, d_model)

    def forward(self, xyz: Tensor) -> Tensor:
        # xyz: [..., 3]
        angles = xyz.unsqueeze(-1) * self.freqs * self.scale  # [..., 3, F]
        feat = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [..., 3, 2F]
        return self.proj(feat.flatten(-2))
