"""Unit tests for the Fold-Conv operator."""
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hyperfold.model.foldconv import (  # noqa: E402
    FoldConv, FoldConvBlock, GKNet, orientation)


def _make_protein(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g).cumsum(dim=0) * 2.0
    x = torch.randn(n, 16, generator=g)
    seq = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
    batch = torch.zeros(n, dtype=torch.long)
    return x, pos, seq, batch


def test_shapes():
    x, pos, seq, batch = _make_protein()
    ori = orientation(pos)
    conv = FoldConv(r=8.0, l=17, in_channels=16, out_channels=32, k_small=8)
    out = conv(x, pos, seq, ori, batch)
    assert out.shape == (60, 32)
    assert torch.isfinite(out).all()


def test_block_residual_shape():
    x, pos, seq, batch = _make_protein()
    ori = orientation(pos)
    block = FoldConvBlock(in_channels=16, out_channels=512, base_width=32.0,
                          r=8.0, l=17, k_small=8)
    out = block(x, pos, seq, ori, batch)
    assert out.shape == (60, 512)
    block2 = FoldConvBlock(in_channels=512, out_channels=512,
                           base_width=32.0, r=10.0, l=21, k_small=8)
    assert block2(out, pos, seq, ori, batch).shape == (60, 512)


def test_se3_invariance():
    """Fold-Conv is invariant to global rotation/translation of the input."""
    torch.manual_seed(0)
    x, pos, seq, batch = _make_protein()
    conv = FoldConv(r=8.0, l=17, in_channels=16, out_channels=32, k_small=8)
    conv.eval()
    out1 = conv(x, pos, seq, orientation(pos), batch)

    theta = torch.tensor(0.7)
    rot = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                        [torch.sin(theta), torch.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
    pos2 = pos @ rot.T + torch.tensor([10.0, -5.0, 3.0])
    out2 = conv(x, pos2, seq, orientation(pos2), batch)
    assert torch.allclose(out1, out2, atol=1e-4), \
        (out1 - out2).abs().max().item()


def test_gknet_bucketing():
    """GK-Net first layer is indexed by the discretized sequence bucket."""
    net = GKNet(l=21, edge_dim=7, hidden_dim=32, out_dim=8)
    delta = torch.randn(5, 7)
    seq_idx = torch.tensor([0, 10, 20, 5, 15])
    out = net(delta, seq_idx)
    assert out.shape == (5, 8)
    # clamped displacements that fall into the same bucket give identical gates
    out2 = net(delta, torch.tensor([0, 10, 20, 5, 15]))
    assert torch.equal(out, out2)


def test_batch_isolation():
    """Two proteins in one batch must not exchange messages."""
    x, pos, seq, batch = _make_protein(n=40, seed=1)
    conv = FoldConv(r=8.0, l=17, in_channels=16, out_channels=16, k_small=4)
    conv.eval()
    ori = orientation(pos)
    alone = conv(x, pos, seq, ori, batch)
    # same protein batched with a far-away second copy
    pos_far = pos + 1000.0
    x2 = torch.cat([x, x])
    pos2 = torch.cat([pos, pos_far])
    seq2 = torch.cat([seq, seq])
    ori2 = torch.cat([ori, ori])
    batch2 = torch.cat([batch, batch + 1])
    together = conv(x2, pos2, seq2, ori2, batch2)
    assert torch.allclose(alone, together[:40], atol=1e-5)


if __name__ == '__main__':
    test_shapes()
    test_block_residual_shape()
    test_se3_invariance()
    test_gknet_bucketing()
    test_batch_isolation()
    print('all foldconv tests passed')
