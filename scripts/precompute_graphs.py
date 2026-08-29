"""Build the residue-graph cache for UniSite-DS (pure rdkit, no torchdrug).

For every protein named by the split CSVs, reads the preprocessed pkl
(sequence + pdb_file_path), parses the PDB file with rdkit
(``hyperfold.data.graph.load_pdb``: one C-alpha per residue) and writes
``{graph_cache}/{name}.pt`` as the plain tensor dict::

    {'node_position': [R, 3] float32, 'residue_number': [R] int64}

which loads with ``torch.load(..., weights_only=True)`` and feeds
``ProteinGraph(**...)`` (see hyperfold.data.dataset.PocketDataset).
Also writes ``{graph_cache}/lengths.csv`` (name,length from the pkl
sequence), consumed by ``hyperfold.data.dataset.load_names``.

Usage:
    python scripts/precompute_graphs.py [--limit N] [--workers 8]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from hyperfold.data import utils as du  # noqa: E402
from hyperfold.data.graph import load_pdb  # noqa: E402


def _resolve_pdb_path(pkl_dir, pdb_file_path):
    """pkl pdb_file_path is relative to the pkl directory (e.g.
    ``../A0A003/A0A003.pdb``)."""
    if os.path.isabs(pdb_file_path):
        return pdb_file_path
    return os.path.normpath(os.path.join(pkl_dir, pdb_file_path))


def _process_one(job):
    """-> (name, seq_len, record) or (name, None, error-string)."""
    name, pkl_dir, graph_cache = job
    try:
        pkl = du.read_pkl(os.path.join(pkl_dir, f'{name}.pkl'), verbose=False)
        seq_len = len(pkl['sequence'])
        out_path = os.path.join(graph_cache, f'{name}.pt')
        if not os.path.exists(out_path):
            pdb_path = _resolve_pdb_path(pkl_dir, pkl['pdb_file_path'])
            graph = load_pdb(pdb_path)
            record = {
                'node_position': torch.as_tensor(
                    graph.node_position).cpu().float(),
                'residue_number': torch.as_tensor(
                    graph.residue_number).cpu().long(),
            }
            assert record['node_position'].shape[0] == \
                record['residue_number'].shape[0]
            torch.save(record, out_path)
        return name, seq_len, None
    except Exception as e:
        return name, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', default='datasets/unisite-ds/pkl_files')
    parser.add_argument('--graph_cache',
                        default='datasets/unisite-ds/graph_cache_v2')
    parser.add_argument('--split_csvs', nargs='*', default=None,
                        help='split CSVs listing the proteins to cache; '
                             'default: {pkl_dir}/train_0.9.csv and test_0.9.csv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0,
                        help='only process the first N proteins (smoke test)')
    args = parser.parse_args()

    if args.split_csvs:
        csvs = args.split_csvs
    else:
        csvs = [os.path.join(args.pkl_dir, 'train_0.9.csv'),
                os.path.join(args.pkl_dir, 'test_0.9.csv')]
    names = []
    for csv_path in csvs:
        names.extend(pd.read_csv(csv_path).iloc[:, 0].astype(str).tolist())
    names = sorted(set(names))
    if args.limit > 0:
        names = names[:args.limit]
    print(f'{len(names)} proteins in scope -> {args.graph_cache}')
    os.makedirs(args.graph_cache, exist_ok=True)

    jobs = [(n, args.pkl_dir, args.graph_cache) for n in names]
    lengths, n_fail = {}, 0
    with Pool(args.workers) as pool:
        for i, (name, seq_len, err) in enumerate(
                pool.imap_unordered(_process_one, jobs, chunksize=32)):
            if err is not None:
                print(f'[WARN] skip {name}: {err}', flush=True)
                n_fail += 1
            else:
                lengths[name] = seq_len
            if (i + 1) % 1000 == 0:
                print(f'{i + 1}/{len(jobs)} done', flush=True)

    pd.DataFrame({'name': list(lengths), 'length': list(lengths.values())}
                 ).to_csv(os.path.join(args.graph_cache, 'lengths.csv'),
                          index=False)
    print(f'cached {len(lengths)}, failed {n_fail}; lengths.csv written')
    print('PRECOMPUTE_DONE')


if __name__ == '__main__':
    main()
