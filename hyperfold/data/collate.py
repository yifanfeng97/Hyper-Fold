"""Batch collation and nested device transfer for Hyper-Fold."""
import torch

from . import utils as du
from .graph import ProteinGraph, graph_collate


def length_collate(batch):
    """Collate variable-length protein features, padding to the batch max.

    ``name`` / ``target`` / ``graph`` / ``map_dict`` / ``pdb_file_path``
    entries are popped from each example and collated separately (graphs via
    ``graph_collate``); everything else is zero-padded along the residue axis
    and stacked with ``torch.utils.data.default_collate``.
    """
    max_len = max([x['length'] for x in batch])
    names = [x.pop('name') for x in batch]
    targets = [x.pop('target') for x in batch] if "target" in batch[0] else None
    graphs = [x.pop('graph') for x in batch] if "graph" in batch[0] else None
    map_dicts = [x.pop('map_dict') for x in batch] if "map_dict" in batch[0] else None
    pdb_file_paths = [x.pop('pdb_file_path') for x in batch] if "pdb_file_path" in batch[0] else None
    pad_example = lambda x: du.pad_feats(x, max_len)
    padded_batch = [pad_example(x) for x in batch]
    batch = torch.utils.data.default_collate(padded_batch)
    batch["name"] = names
    if targets is not None:
        batch["target"] = targets 
    if graphs is not None:
        batch["graph"] = graph_collate(graphs)
    if map_dicts is not None:
        batch["map_dict"] = map_dicts
    if pdb_file_paths is not None:
        batch["pdb_file_path"] = pdb_file_paths
    return batch


def map_to(feats, device):
    """Recursively move tensors in a nested structure to ``device``.

    Handles dict / list / tuple containers; leaves of type ``torch.Tensor``
    or ``ProteinGraph`` are transferred, everything else is passed through.
    Target containers (e.g. ``dataset.Pockets``, imported lazily to avoid a
    circular import) expose the same in-place-style ``.to(device)`` interface
    and are transferred too.
    """
    if isinstance(feats, dict):
        return {k: map_to(v, device) for k, v in feats.items()}
    if isinstance(feats, list):
        return [map_to(v, device) for v in feats]
    if isinstance(feats, tuple):
        return tuple(map_to(v, device) for v in feats)
    if isinstance(feats, (torch.Tensor, ProteinGraph)):
        return feats.to(device)
    to = getattr(feats, 'to', None)
    if callable(to) and not isinstance(feats, type):
        return to(device)
    return feats
