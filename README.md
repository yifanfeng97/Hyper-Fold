<h1 align="center">Hyper-Fold</h1>

<p align="center">
  <b>Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling</b><br>
  Yifan Feng¹, Guanjie Cheng², Shihui Ying³, Shaoyi Du⁴, Yue Gao¹<br>
  ¹ Tsinghua University · ² Zhejiang University · ³ Shanghai University · ⁴ Xi'an Jiaotong University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.29207">
    <img src="https://img.shields.io/badge/arXiv-2608.29207-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e" alt="arXiv">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=1a1a2e" alt="PyTorch">
  </a>
  <img src="https://img.shields.io/badge/PyG%20%2F%20torchdrug-not%20required-4daf4a?style=for-the-badge&labelColor=1a1a2e" alt="No PyG / torchdrug">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=for-the-badge&labelColor=1a1a2e" alt="License">
  </a>
</p>

<p align="center">
  <a href="#about">About</a> &#xa0; | &#xa0;
  <a href="#quick-start">Quick Start</a> &#xa0; | &#xa0;
  <a href="#data--training">Data & Training</a> &#xa0; | &#xa0;
  <a href="#results">Results</a> &#xa0; | &#xa0;
  <a href="DATASETS.md">Datasets</a> &#xa0; | &#xa0;
  <a href="examples/README.md">Examples</a>
</p>

<div align="center">
  <img src="assets/bg.png" alt="Expressivity ladder and accuracy-latency trade-off" width="100%" />
</div>

*Left: the expressivity ladder of message passing — additive MP (GCN, GearNet) → scalar-gated (GAT) → channel-gated (SchNet) → our matrix-gated Fold-Conv, which provably approaches the complete bilinear ceiling. Right: pocket detection accuracy vs. latency — Hyper-Fold (10M) is 4.8× faster than UniSite-3D (683M) with +21.9% AP, and 1.7× faster than GearNet (23M) with +127.2% AP.*

## About

Protein structure modeling rests on a single computational primitive: the interaction between what a residue *is* (sequence content) and *where it sits* (3D geometry). Hyper-Fold is a new framework for protein 3D-structure learning that pushes this primitive to its expressive limit. Its core operator, the **Fold-Conv layer**, splits each residue's neighborhood into a **sequence hyperedge** and a **contact hyperedge**, and generates per-edge kernels with a **geometric kernel network (GK-Net)** — a rank-K matrix gate that strictly dominates the additive, scalar-gated and channel-gated layers used by classic methods (GCN, GAT, SchNet, GearNet). The result: stronger expressivity, higher accuracy, and lower latency at a fraction of the parameters.

All models are **purely structural** — the input is the residue type and Cα coordinates only, no sequence language model (ESM) features anywhere.

<div align="center">
  <img src="assets/backbone.png" alt="Hyper-Fold backbone" width="100%" />
</div>

*The Fold-Conv operator and the two backbone stacks: residue-level Hyper-Fold and protein-level Hyper-Fold-Deep.*

| Model | Task | Recipe |
|---|---|---|
| Hyper-Fold | EC number | 6 Fold-Conv blocks, width 512 |
| Hyper-Fold-Deep | fold classification | 4 stages, 256→2048, residue pooling |
| Hyper-Fold-Pocket | pocket detection | + bi-level neck, 50 PPN-anchored queries, transformer decoder, CDN |

<div align="center">
  <img src="assets/pocket_pipeline.png" alt="Hyper-Fold-Pocket architecture" width="100%" />
</div>

*Hyper-Fold-Pocket: the residue-level backbone plus a set-prediction detection head — a pocket proposal network (PPN) anchoring 50 queries, a 4-layer transformer decoder with 3D Fourier positional encoding, and contrastive denoising (CDN) training.*

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python scripts/predict.py --pdb examples/1qbuA.pdb   # or any PDB of yours
```

<div align="center">
  <img src="examples/1qbuA_panel.png" alt="Hyper-Fold-Pocket prediction on 1qbuA" width="100%" />
</div>

*Left: surface colored TP green / FP blue / FN red. Right: predicted pocket (violet) wraps the co-crystallized ligand (orange). More in [examples/](examples/README.md).*

## Data & training

```bash
bash scripts/download_data.sh   # UniSite-DS + HOLO4K-sc / COACH420; see DATASETS.md

torchrun --nproc_per_node=2 scripts/train_pocket.py --config configs/pocket.yaml
python scripts/train_ec.py    --config configs/ec.yaml
python scripts/train_fold.py  --config configs/fold.yaml
```

Evaluate with `scripts/eval_ap.py` (AP@IoU 0.3/0.5) and `scripts/eval_dcc_dca.py` (DCC/DCA@4 Å).

## Results

Pocket detection (all methods trained on UniSite-DS only; HOLO4K-sc is zero-shot; DCC/DCA are top-n success rates at 4 Å):

| Method | UniSite-DS AP@0.3 | UniSite-DS AP@0.5 | HOLO4K-sc AP@0.3 | HOLO4K-sc DCC | HOLO4K-sc DCA |
|---|---|---|---|---|---|
| Fpocket | 0.184 | 0.102 | 0.271 | 0.308 | 0.438 |
| Fpocket-rescore | 0.508 | 0.235 | 0.590 | 0.518 | 0.765 |
| P2Rank | 0.506 | 0.216 | 0.601 | 0.530 | <u>0.819</u> |
| Deep-Pocket | 0.427 | 0.233 | 0.542 | 0.493 | 0.737 |
| GrASP | 0.447 | 0.285 | 0.667 | 0.513 | 0.742 |
| VN-EGNN | 0.162 | 0.071 | 0.261 | <u>0.586</u> | 0.700 |
| UniSite-1D | 0.512 | 0.303 | 0.687 | 0.554 | 0.769 |
| UniSite-3D | <u>0.560</u> | <u>0.384</u> | <u>0.709</u> | 0.572 | 0.788 |
| **Hyper-Fold-Pocket (ours)** | **0.617** | **0.467** | **0.735** | **0.681** | **0.827** |

EC (Fmax@50% / AUPR@95): **0.789 / 0.874** · Fold / superfamily / family: **0.578 / 0.794 / 0.995**

## Tests

```bash
python tests/test_foldconv.py   # shapes, SE(3) invariance, GK-Net bucketing
```

## Citation

```bibtex
@article{feng2026hyperfold,
  title  = {Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry
            Learning for Proteins via Hypergraph Modeling},
  author = {Feng, Yifan and Cheng, Guanjie and Ying, Shihui and Du, Shaoyi and Gao, Yue},
  year   = {2026},
}
```
