"""Lightweight protein graph container for Hyper-Fold (dependency-free).

`ProteinGraph` carries the residue-level structural data the FoldConv
encoder and the pocket head need: Cα coordinates, PDB residue numbers and
per-protein sizes for a (possibly batched) set of proteins. It replaces the
external graph library object previously used for this role with a minimal,
self-contained structure supporting the same access patterns used across
the codebase (`graph[i].node_position`, `graph[i].residue_number`,
`graph.num_residue`, `graph.to(device)`, `graph_collate(...)`).
"""
import types
import warnings

import torch
from rdkit import Chem

from . import residue_constants


class ProteinGraph:
    """Residue-level protein structure graph.

    Fields:
        node_position: [R, 3] float tensor, Cα coordinates of all residues.
        residue_number: [R] long tensor, PDB residue number per residue.
        sizes: [G] long tensor, number of residues of each protein in the
            batch; a single protein has ``sizes == [R]``.
        aatype: optional [R] long tensor, amino-acid type index
            (``residue_constants.restype_order_with_x``, X = 20).
    """

    def __init__(self, node_position, residue_number, sizes=None, aatype=None,
                 is_batch=False):
        self.node_position = torch.as_tensor(node_position).float()
        self.residue_number = torch.as_tensor(residue_number).long()
        if sizes is None:
            sizes = [self.node_position.shape[0]]
        self.sizes = torch.as_tensor(sizes).long()
        assert self.sizes.sum().item() == self.node_position.shape[0], \
            f'sizes {self.sizes.tolist()} do not add up to ' \
            f'{self.node_position.shape[0]} residues'
        if aatype is not None:
            self.aatype = torch.as_tensor(aatype).long()
        else:
            self.aatype = None
        # True for graphs produced by graph_collate: __getitem__ then indexes
        # proteins even when the batch holds a single protein (mirrors the
        # previous graph container's batch semantics)
        self.is_batch = is_batch

    # ------------------------------------------------------------------
    # device handling (in-place, like the previous graph container)
    # ------------------------------------------------------------------
    def to(self, device):
        self.node_position = self.node_position.to(device)
        self.residue_number = self.residue_number.to(device)
        self.sizes = self.sizes.to(device)
        if self.aatype is not None:
            self.aatype = self.aatype.to(device)
        return self

    def cuda(self):
        return self.to('cuda')

    @property
    def device(self):
        return self.node_position.device

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def __len__(self):
        """Number of proteins in the batch."""
        return self.sizes.numel()

    @property
    def num_residue(self):
        """Number of residues (of the single protein, or total if batched)."""
        return self.node_position.shape[0]

    def __getitem__(self, i):
        """Batched graph -> i-th protein; unbatched single protein -> i-th
        residue (single-residue view, so ``graph[i].node_position`` is
        [1, 3])."""
        if self.is_batch or len(self) > 1:
            offset = int(self.sizes[:i].sum().item())
            n = int(self.sizes[i].item())
            aatype = self.aatype[offset:offset + n] \
                if self.aatype is not None else None
            return ProteinGraph(
                self.node_position[offset:offset + n],
                self.residue_number[offset:offset + n],
                sizes=[n], aatype=aatype)
        aatype = self.aatype[i:i + 1] if self.aatype is not None else None
        return ProteinGraph(self.node_position[i:i + 1],
                            self.residue_number[i:i + 1],
                            sizes=[1], aatype=aatype)

    def __repr__(self):
        return (f'ProteinGraph(num_proteins={len(self)}, '
                f'num_residue={self.num_residue})')


def graph_collate(graphs):
    """Collate a list of ProteinGraph into one batched ProteinGraph."""
    node_position = torch.cat([g.node_position for g in graphs], dim=0)
    residue_number = torch.cat([g.residue_number for g in graphs], dim=0)
    sizes = torch.cat([g.sizes for g in graphs], dim=0)
    aatype = None
    if all(g.aatype is not None for g in graphs):
        aatype = torch.cat([g.aatype for g in graphs], dim=0)
    return ProteinGraph(node_position, residue_number, sizes=sizes,
                        aatype=aatype, is_batch=True)


def load_pdb(pdb_file, sanitize=False, removeHs=True):
    """Parse a PDB file into a single-protein ProteinGraph (Cα per residue).

    Residues are taken in order of appearance in the file; residues without
    a Cα atom are skipped. ``residue_number`` is the PDB residue number;
    ``aatype`` maps the 3-letter residue name to
    ``residue_constants.restype_order_with_x`` (unknown -> X = 20).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol = Chem.MolFromPDBFile(pdb_file, removeHs=removeHs,
                                      sanitize=sanitize)
            conf = mol.GetConformer()
            positions, resnums, aatypes = [], [], []
            seen = set()
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info is None:
                    continue
                key = (info.GetChainId(), info.GetResidueNumber(),
                       info.GetInsertionCode())
                if key in seen:
                    continue
                if info.GetName().strip() != 'CA':
                    continue
                seen.add(key)
                p = conf.GetAtomPosition(atom.GetIdx())
                positions.append([p.x, p.y, p.z])
                resnums.append(info.GetResidueNumber())
                resname = info.GetResidueName().strip()
                aa1 = residue_constants.restype_3to1.get(resname, 'X')
                aatypes.append(residue_constants.restype_order_with_x[aa1])
            assert len(positions) > 0, 'no CA atoms found'
        graph = ProteinGraph(
            torch.tensor(positions, dtype=torch.float),
            torch.tensor(resnums, dtype=torch.long),
            aatype=torch.tensor(aatypes, dtype=torch.long))
    except Exception as e:
        assert False, f"Error loading PDB file {pdb_file}: {e}"
    return graph


def to_sequence(graph):
    """Recover the one-letter amino-acid sequence from ``graph.aatype``."""
    assert graph.aatype is not None, 'graph has no aatype'
    restypes_with_x = residue_constants.restypes_with_x
    return ''.join(restypes_with_x[int(a)] for a in graph.aatype)


class AtomLevelProtein:
    """Per-residue all-atom coordinate view.

    Mirrors the indexing interface ``calc_center`` relies on
    (``protein[i].node_position`` -> [A_i, 3] array of residue i's heavy
    atoms). Pocket-center computation (ConvexHull) follows the original
    UniSite protocol, which uses all heavy atoms rather than Cα only.
    """

    def __init__(self, atom_positions, residue_keys=None):
        self.atom_positions = atom_positions  # list of [A_i, 3] np.ndarray
        # (chain id, residue number, insertion code) per residue, same order
        self.residue_keys = residue_keys

    def __len__(self):
        return len(self.atom_positions)

    @property
    def num_residue(self):
        return len(self.atom_positions)

    def __getitem__(self, i):
        return types.SimpleNamespace(node_position=self.atom_positions[i])

    def align_to_resnums(self, resnums):
        """Return a copy aligned to a Cα graph's residue_number sequence.

        ``load_pdb`` skips residues lacking a Cα atom while this parser keeps
        every keyed residue, so the atom-level view can hold extra entries.
        Both views list residues in order of first appearance, hence the Cα
        sequence is a subsequence of this view: greedy in-order matching on
        residue number recovers the exact alignment.
        """
        assert self.residue_keys is not None, 'no residue keys stored'
        idx, j = [], 0
        for rn in resnums:
            while j < len(self.residue_keys) and self.residue_keys[j][1] != rn:
                j += 1
            if j == len(self.residue_keys):
                raise ValueError(f'residue number {rn} not found in atom view')
            idx.append(j)
            j += 1
        return AtomLevelProtein([self.atom_positions[k] for k in idx],
                                [self.residue_keys[k] for k in idx])


def load_pdb_atomlevel(pdb_file, removeHs=True):
    """Parse a PDB file into an AtomLevelProtein (heavy atoms per residue).

    Residues are keyed by (chain id, residue number, insertion code) and kept
    in order of first appearance — the same order ``load_pdb`` produces, so
    residue indices align with the Cα ProteinGraph of the same file.
    """
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=removeHs,
                                  sanitize=False)
    conf = mol.GetConformer()
    order, atoms = [], {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetChainId(), info.GetResidueNumber(),
               info.GetInsertionCode())
        if key not in atoms:
            atoms[key] = []
            order.append(key)
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms[key].append([p.x, p.y, p.z])
    assert order, f'no atoms found in {pdb_file}'
    return AtomLevelProtein([np.asarray(atoms[k], dtype=np.float32)
                             for k in order], residue_keys=order)
