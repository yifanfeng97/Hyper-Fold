"""Hyper-Fold residue-level backbone.

The paper's residue-level encoder: a stack of 6 Fold-Conv blocks of width
512 with neighborhood radii RS = (8, 8, 8, 10, 10, 10) Angstrom and
sequence windows LS = (17, 17, 17, 21, 21, 21), bottleneck base width 32,
low-rank branch rank K = k_small = 8, dropout 0.1 and BN momentum 0.1.
The node input is a 21-class amino-acid embedding of dimension 16.

The backbone runs on the residue graph of each protein and scatters the
raw per-block features back onto the (padded) UniProt sequence axis;
positions without a mapped structure residue are left at zero. Projection
of the two readout levels to the decoder width is position-wise and is
therefore deferred to :class:`hyperfold.model.neck.BiLevelNeck`.
"""
import torch
from torch import nn, Tensor

from .foldconv import FoldConvBlock, orientation


class HyperFold(nn.Module):
    """Fold-Conv backbone producing per-block residue features on the
    UniProt sequence axis.

    Args:
        width: residue feature width of every Fold-Conv block (512).
        readout_blocks: 1-INDEXED block numbers (as in the paper) whose
            features feed the query path of the neck; converted to
            0-indexed internally.
        mask_readout_blocks: 1-indexed block numbers feeding the mask
            dot-product path; defaults to ``readout_blocks``.
        dropout: dropout inside the Fold-Conv blocks (0.1).
        momentum: BatchNorm momentum inside the Fold-Conv blocks (0.1).
        conv_type: geometry convolution of the blocks, 'foldconv' or one of
            the expressivity-ladder rungs (passed through to FoldConvBlock).
        k_small: rank K of the low-rank branch (8).
    """

    RS = (8.0, 8.0, 8.0, 10.0, 10.0, 10.0)
    LS = (17, 17, 17, 21, 21, 21)

    def __init__(self, width: int = 512, readout_blocks=(3, 6),
                 mask_readout_blocks=None, dropout: float = 0.1,
                 momentum: float = 0.1, conv_type: str = 'foldconv',
                 k_small: int = 8):
        super().__init__()
        self.width = width
        # the paper numbers blocks 1..6; store 0-indexed internally
        self.readout_blocks = tuple(b - 1 for b in readout_blocks)
        self.mask_readout_blocks = tuple(
            b - 1 for b in mask_readout_blocks) \
            if mask_readout_blocks is not None else self.readout_blocks
        self.aa_embedding = nn.Embedding(num_embeddings=21, embedding_dim=16)
        self.blocks = nn.ModuleList([
            FoldConvBlock(in_channels=16 if i == 0 else width,
                          out_channels=width, base_width=32.0,
                          r=self.RS[i], l=self.LS[i], k_small=k_small,
                          weightnet_hidden=32, dropout=dropout,
                          momentum=momentum, conv_type=conv_type)
            for i in range(6)
        ])

    def forward(self, input_feats, L: int):
        """Run the backbone and scatter block features onto the UniProt axis.

        Args:
            input_feats: collated batch dict with keys ``graph``
                (ProteinGraph batch), ``aatype`` [N, L], ``map_dict`` (per
                protein: PDB residue number string -> 1-based UniProt
                position string) and ``name``.
            L: padded UniProt sequence length.

        Returns:
            hidden: dict {0-indexed block idx: dense [N, L, width] tensor,
                zeros at unmapped positions} for every readout level.
            xyz_full: [N, L, 3] protein-centered CA coordinates (zeros at
                unmapped positions).
            mapped_mask: [N, L] bool, True where a structure residue is
                mapped to the sequence position.
        """
        graph = input_feats["graph"]
        N = input_feats["aatype"].shape[0]
        device = input_feats["aatype"].device

        all_x, all_pos, all_seq, all_ori, all_batch = [], [], [], [], []
        per_protein = []  # None or (uniprot_pos, xyz)
        for i in range(N):
            map_dict = input_feats["map_dict"][i]
            residue_number = graph[i].residue_number
            keep_list = [str(r.item()) in map_dict for r in residue_number]
            pdb2uniprot = [int(map_dict[str(r.item())]) - 1
                           for r, k in zip(residue_number, keep_list) if k]
            if len(pdb2uniprot) < 3:
                per_protein.append(None)
                continue
            assert max(pdb2uniprot) < L, input_feats["name"][i]
            keep = torch.tensor(keep_list, dtype=torch.bool, device=device)
            pos = graph[i].node_position[keep]
            uniprot_pos = torch.tensor(pdb2uniprot, dtype=torch.long,
                                       device=device)
            all_x.append(self.aa_embedding(input_feats["aatype"][i][uniprot_pos]))
            all_pos.append(pos)
            all_seq.append(uniprot_pos.float().unsqueeze(-1))
            all_ori.append(orientation(pos))
            all_batch.append(torch.full((len(pdb2uniprot),), len(all_batch),
                                        dtype=torch.long, device=device))
            per_protein.append((uniprot_pos, pos))

        x = torch.cat(all_x, dim=0)
        pos = torch.cat(all_pos, dim=0)
        seq = torch.cat(all_seq, dim=0)
        ori = torch.cat(all_ori, dim=0)
        batch = torch.cat(all_batch, dim=0)

        need_blocks = set(self.readout_blocks) | set(self.mask_readout_blocks)
        hidden = {}
        for i, block in enumerate(self.blocks):
            x = block(x, pos, seq, ori, batch)
            if i in need_blocks:
                hidden[i] = x

        hidden_dense = {bi: torch.zeros(N, L, self.width, device=device)
                        for bi in need_blocks}
        xyz_full = torch.zeros(N, L, 3, device=device)
        mapped_mask = torch.zeros(N, L, dtype=torch.bool, device=device)
        offset = 0
        for i in range(N):
            entry = per_protein[i]
            if entry is None:
                continue
            uniprot_pos, xyz = entry
            n = uniprot_pos.shape[0]
            # protein-centered coordinates keep the Fourier features
            # translation invariant, matching the backbone
            xyz_full[i, uniprot_pos] = xyz - xyz.mean(dim=0, keepdim=True)
            mapped_mask[i, uniprot_pos] = True
            for bi in need_blocks:
                hidden_dense[bi][i, uniprot_pos] = hidden[bi][offset:offset + n]
            offset += n
        return hidden_dense, xyz_full, mapped_mask
