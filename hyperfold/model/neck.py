"""Bi-level neck of Hyper-Fold-Pocket.

The neck fuses the two backbone readout levels into residue features: two
pairs of linear projectors (width -> d_model) produce the fused residue
features X_res (query content path) and the mask keys K_emb (mask
dot-product path), and the bi-level memory M_emb is the concatenation of
the per-level projected features plus a learnable level embedding
(e_low / e_high). Structurally unmapped positions are filled with a
learnable null vector.
"""
import torch
from torch import nn, Tensor


class BiLevelNeck(nn.Module):
    """Project the backbone readout levels and build the bi-level memory.

    Args:
        width: backbone residue feature width (512).
        d_model: decoder width the levels are projected to (256).
        readout_blocks: 1-INDEXED block numbers of the readout levels
            (converted to 0-indexed internally).
        mask_readout_blocks: 1-indexed block numbers of the mask-key path;
            defaults to ``readout_blocks``.
    """

    def __init__(self, width: int = 512, d_model: int = 256,
                 readout_blocks=(3, 6), mask_readout_blocks=None):
        super().__init__()
        self.readout_blocks = tuple(b - 1 for b in readout_blocks)
        self.mask_readout_blocks = tuple(
            b - 1 for b in mask_readout_blocks) \
            if mask_readout_blocks is not None else self.readout_blocks
        self.query_projs = nn.ModuleList(
            [nn.Linear(width, d_model) for _ in self.readout_blocks])
        self.key_projs = nn.ModuleList(
            [nn.Linear(width, d_model) for _ in self.mask_readout_blocks])
        self.level_embed = nn.Parameter(
            torch.zeros(len(self.readout_blocks), d_model))
        self.null_feat = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.null_feat, std=0.02)

    def forward(self, hidden, xyz_full: Tensor, mapped_mask: Tensor,
                res_mask: Tensor, L: int):
        """Fuse the readout levels and assemble the bi-level memory.

        Args:
            hidden: dict {0-indexed block idx: dense [N, L, width] tensor}
                from the backbone.
            xyz_full: [N, L, 3] protein-centered CA coordinates.
            mapped_mask: [N, L] bool, True at structurally mapped positions.
            res_mask: [N, L] bool, True at valid (non-padded) sequence
                positions.
            L: padded UniProt sequence length.

        Returns:
            X_res: [N, L, d_model] fused residue features (query content);
                null vector at unmapped positions.
            K_emb: [N, L, d_model] fused mask keys; null vector at unmapped
                positions.
            memory: [N, n_levels * L, d_model] bi-level memory M_emb, the
                per-level projected features plus the level embedding.
            mem_pad: [N, n_levels * L] bool key padding mask for the memory
                (``(~res_mask).repeat(1, n_levels)``).
        """
        N = mapped_mask.shape[0]
        d_model = self.null_feat.shape[0]
        device = mapped_mask.device

        X_res = self.null_feat.expand(N, L, d_model).clone()
        K_emb = self.null_feat.expand(N, L, d_model).clone()
        level_dense = [self.null_feat.expand(N, L, d_model).clone()
                       for _ in self.readout_blocks]
        for i in range(N):
            m = mapped_mask[i]
            if not m.any():
                continue
            proj_sum = 0
            for lv, bi in enumerate(self.readout_blocks):
                p = self.query_projs[lv](hidden[bi][i, m])
                level_dense[lv][i, m] = p
                proj_sum = proj_sum + p
            X_res[i, m] = proj_sum
            mask_sum = 0
            for lv, bi in enumerate(self.mask_readout_blocks):
                mask_sum = mask_sum + self.key_projs[lv](hidden[bi][i, m])
            K_emb[i, m] = mask_sum

        n_lv = len(self.readout_blocks)
        memory = torch.cat(
            [level_dense[lv] + self.level_embed[lv]
             for lv in range(n_lv)], dim=1)  # [N, n_lv*L, d_model]
        mem_pad = (~res_mask).repeat(1, n_lv)
        return X_res, K_emb, memory, mem_pad
