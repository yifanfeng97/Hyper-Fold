# Datasets

Hyper-Fold uses three task suites. All structure processing in this repo is
Cα-only and handled by `hyperfold/data/graph.py` (rdkit); no torchdrug or PyG
is needed anywhere.

## 1. Pocket detection: UniSite-DS + HOLO4K-sc / COACH420

UniSite-DS ([paper](https://arxiv.org/pdf/2506.03237)) is a UniProt-centric
ligand-binding-site dataset: 11,510 unique proteins, each with a
representative PDB structure, a PDB↔UniProt residue mapping, and per-site
binary masks over the UniProt sequence. HOLO4K-sc and COACH420 (single-chain
*mlig* subsets of the
[P2Rank benchmarks](https://github.com/rdk/p2rank-datasets)) are used as
zero-shot test sets.

Download everything (UniSite-DS + benchmarks + graph cache build) with:

```bash
bash scripts/download_data.sh
```

which fetches the archives from
[huggingface.co/datasets/quanlin-wu/unisite-ds_v1](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1)
and extracts them into `datasets/`:

```
datasets/
├── unisite-ds/
│   ├── pkl_files/           # per-protein pkl (sequence, pdb_file_path,
│   │                        #   map_dict, target{labels, pocket_masks, res_mask})
│   │   ├── train_0.9.csv    # 90/10 split used in the paper
│   │   └── test_0.9.csv
│   ├── <UNP_ID>/            # representative structure + per-site files
│   └── graph_cache_v2/      # built by scripts/precompute_graphs.py
│       ├── <name>.pt        #   {node_position [N,3] f32, residue_number [N] i64}
│       └── lengths.csv
└── benchmark_datasets/
    ├── holo4k-sc/           # {name}/{name}_protein.pdb, _ligandK.pdb,
    │   ├── holo4k-sc.csv    #   _pocketK.txt; holo4k-sc_seq.csv; target_pkl/
    └── coach420/            # same layout
```

The graph cache is derived data; rebuild it any time with

```bash
python scripts/precompute_graphs.py \
    --pkl_dir datasets/unisite-ds/pkl_files \
    --graph_cache datasets/unisite-ds/graph_cache_v2
```

Notes:
- Training uses sequences of at most 800 residues (`data.max_seq_len`).
- The pocket model is purely structural — **no ESM embeddings are needed or
  used** anywhere in this pipeline.

## 2. EC number prediction

The Enzyme Commission dataset with the GearNet splits (sequence-identity
cutoffs 30/40/50/70/95%). Download from the GearNet release
([github.com/DeepGraphLearning/GearNet](https://github.com/DeepGraphLearning/GearNet))
or the CDConv mirror, and place it at:

```
datasets/EnzymeCommission/     # per-split subfolders with structure pkls
```

Point `configs/ec.yaml → data.root` at the directory. Each sample provides
Cα coordinates, residue types and a 538-dim multi-hot EC target.

## 3. Fold classification

The fold-classification benchmark with the standard three test scenarios
(fold / superfamily / family), from the GearNet/CDConv releases
([FoldClassification](https://github.com/DeepGraphLearning/GearNet) data
section). Place it at:

```
datasets/fold/               # train/valid/test_fold/test_superfamily/
                             # test_family splits
```

and set `configs/fold.yaml → data.root`. If your copy is in the original
`.hdf5` layout, convert once with
`FoldDataset.convert_hdf5_to_npz(root)` (needs `h5py`, conversion-time only).

## Checkpoints

`checkpoints/hyperfold_pocket_unisite_ds.pth` — the paper's main pocket
model (AP@0.3 0.617 / AP@0.5 0.467 on the UniSite-DS test split). Checkpoints
are self-contained (they embed their model config as a plain dict).

Classification checkpoints are not shipped; the configs in `configs/`
reproduce the reported models.
