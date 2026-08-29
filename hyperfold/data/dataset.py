"""Pocket-detection datasets and the ``Pockets`` target container.

Pure-structure data pipeline (no ESM): each sample carries the amino-acid
type sequence (unknown residues -> X), the precomputed residue-level
ProteinGraph (plain-dict cache written by ``scripts/precompute_graphs.py``),
and the PDB->sequence ``map_dict``.

Batch collation lives in :mod:`hyperfold.data.collate` (``length_collate`` /
``map_to`` are re-exported here for convenience).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from . import residue_constants
from . import utils as du
from .collate import length_collate, map_to  # noqa: F401  (re-export)
from .graph import ProteinGraph


class Pockets:
    """Ground-truth pocket set for one protein.

    Attributes:
        labels: [P] long tensor, class label per pocket (all 0 = pocket).
        pocket_masks: [P, L] float tensor, binary mask per pocket on the
            sequence axis.
        res_mask: [L] bool tensor, valid (non-padding) sequence positions.
    """

    def __init__(self, labels, pocket_masks, res_mask):
        self._labels = labels
        self._pocket_masks = pocket_masks
        self._res_mask = res_mask

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        if key == "labels":
            return self._labels
        elif key == "pocket_masks":
            return self._pocket_masks
        elif key == "res_mask":
            return self._res_mask
        elif isinstance(key, int):
            idx = key
            return {
                "labels": self._labels[idx],
                "pocket_masks": self._pocket_masks[idx],
                "res_mask": self._res_mask,
            }
        else:
            raise KeyError(f"Invalid key: {key}")

    @property
    def device(self):
        return self._labels.device

    def cuda(self):
        return Pockets(self._labels.cuda(), self._pocket_masks.cuda(),
                       self._res_mask.cuda())

    def numpy(self):
        return {
            "labels": self._labels.cpu().numpy(),
            "pocket_masks": self._pocket_masks.cpu().numpy(),
            "res_mask": self._res_mask.cpu().numpy(),
        }

    def to(self, device):
        return Pockets(self._labels.to(device), self._pocket_masks.to(device),
                       self._res_mask.to(device))


class PocketDataset(data.Dataset):
    """UniSite-DS pkl + precomputed protein graph (training / evaluation).

    Each pkl file provides the UniProt sequence, the pocket targets and the
    PDB->UniProt ``map_dict``; the residue graph comes from the plain-dict
    cache at ``{graph_cache}/{name}.pt`` (keys ``node_position`` [R, 3]
    float32 / ``residue_number`` [R] int64, see scripts/precompute_graphs.py).
    """

    def __init__(self, names, pkl_dir, graph_cache, coord_noise=0.0):
        self._names = names
        self._pkl_dir = pkl_dir
        self._graph_cache = graph_cache
        self._coord_noise = coord_noise

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        name = self._names[idx]
        pkl = du.read_pkl(os.path.join(self._pkl_dir, f'{name}.pkl'),
                          verbose=False)
        seq = pkl['sequence']
        length = len(seq)

        seq_with_x = [r if r in residue_constants.restypes else 'X'
                      for r in seq]
        aatype = torch.tensor(
            [residue_constants.restype_order_with_x[r] for r in seq_with_x]
        ).long()

        t = pkl['target']
        target = Pockets(
            labels=torch.as_tensor(np.asarray(t['labels'])).long(),
            pocket_masks=torch.as_tensor(np.asarray(t['pocket_masks'])).float(),
            res_mask=torch.as_tensor(np.asarray(t['res_mask'])).bool(),
        )
        graph = ProteinGraph(**torch.load(
            os.path.join(self._graph_cache, f'{name}.pt'), weights_only=True))
        if self._coord_noise > 0:
            graph.node_position = graph.node_position + torch.randn_like(
                graph.node_position) * self._coord_noise

        return {
            'name': name,
            'length': length,
            'aatype': aatype,
            'seq_idx': torch.arange(length) + 1,
            'res_mask': torch.ones(length, dtype=torch.bool),
            'target': target,
            'graph': graph,
            'map_dict': pkl['map_dict'],
        }


class BenchmarkDataset(data.Dataset):
    """Serves benchmark proteins (holo4k-sc / coach420) for zero-shot inference.

    ``rows``: list of dicts with name/sequence/pdb_file_path/length;
    ``graphs``: name -> ProteinGraph (pre-loaded with ``load_pdb``).
    The PDB->sequence mapping falls back to a positional 1:1 mapping from
    the graph's residue numbers (no curated mapping exists for benchmarks).
    """

    def __init__(self, rows, graphs):
        self._rows = rows
        self._graphs = graphs

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        row = self._rows[idx]
        name, seq = row['name'], row['sequence']
        length = len(seq)
        graph = self._graphs[name]
        # positional PDB->sequence mapping
        map_dict = {str(r): str(i + 1)
                    for i, r in enumerate(graph.residue_number.tolist())}

        restypes = residue_constants.restypes
        seq_with_x = [r if r in restypes else 'X' for r in seq]
        aatype = torch.tensor(
            [residue_constants.restype_order_with_x[r] for r in seq_with_x]
        ).long()

        return {
            'name': name,
            'length': length,
            'aatype': aatype,
            'seq_idx': torch.arange(length) + 1,
            'res_mask': torch.ones(length, dtype=torch.bool),
            'pdb_file_path': row['pdb_file_path'],
            'graph': graph,
            'map_dict': map_dict,
        }


def load_names(split_csv, graph_cache, max_seq_len, lengths_csv=None):
    """Split CSV names filtered by sequence length and graph availability.

    ``lengths_csv`` defaults to ``{graph_cache}/lengths.csv`` (written by
    scripts/precompute_graphs.py); proteins without a length entry or
    without a cached graph are dropped.
    """
    if lengths_csv is None:
        lengths_csv = os.path.join(graph_cache, 'lengths.csv')
    ids = pd.read_csv(split_csv).iloc[:, 0].astype(str).tolist()
    lengths = pd.read_csv(lengths_csv).set_index('name')['length'].to_dict()
    names = [
        n for n in ids
        if lengths.get(n, max_seq_len + 1) <= max_seq_len
        and os.path.exists(os.path.join(graph_cache, f'{n}.pt'))
    ]
    return names
