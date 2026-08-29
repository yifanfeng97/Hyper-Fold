"""Hyper-Fold: rank-K matrix-gated sequence-geometry learning for proteins.

Public API:
- HyperFold: 6-block residue-level backbone (Fold-Conv).
- HyperFoldDeep: 4-stage hierarchical protein-level backbone.
- HyperFoldClassifier: residue-level backbone + EC/fold classification head.
- HyperFoldPocket / PocketSetCriterion: anchored set-prediction pocket detector.
"""
from .model.foldconv import FoldConv, FoldConvBlock, GKNet, orientation
from .model.backbone import HyperFold
from .model.hyperfold_deep import HyperFoldDeep
from .model.classifier import HyperFoldClassifier
from .model.neck import BiLevelNeck
from .model.pocket import (
    HyperFoldPocket, PocketProposalNetwork, PocketSetCriterion)
from .model.position import FourierPositionalEncoding3D

__all__ = [
    'FoldConv', 'FoldConvBlock', 'GKNet', 'orientation',
    'HyperFold', 'HyperFoldDeep', 'HyperFoldClassifier',
    'BiLevelNeck', 'HyperFoldPocket', 'PocketProposalNetwork',
    'PocketSetCriterion', 'FourierPositionalEncoding3D',
]

__version__ = '1.0.0'
