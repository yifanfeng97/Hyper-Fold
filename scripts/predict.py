"""Inference for Hyper-Fold-Pocket: benchmark zero-shot runs and single PDBs.

Benchmark mode (holo4k-sc / coach420) writes per-protein result pkl files
compatible with:
- scripts/eval_ap.py      (labels / scores / pocket_masks)
- scripts/eval_dcc_dca.py (scores / centers == hull_centers)

Single-PDB mode (``--pdb``) is the quick-start entry: run one structure and
print the predicted pockets (optionally saving the same pkl record).

No ESM embeddings are computed or consumed: the model input is the
amino-acid type sequence plus the CA graph parsed from the PDB file.

Pocket centers follow the original UniSite protocol: ConvexHull over all
heavy atoms of the masked residues (load_pdb_atomlevel +
align_to_resnums + calc_center).
"""
import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils import data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from hyperfold.data.collate import length_collate, map_to  # noqa: E402
from hyperfold.data.dataset import BenchmarkDataset  # noqa: E402
from hyperfold.data.graph import (  # noqa: E402
    load_pdb, load_pdb_atomlevel, to_sequence)
from hyperfold.model.losses import calc_center  # noqa: E402
from hyperfold.model.pocket import HyperFoldPocket  # noqa: E402


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    return d


def load_model(ckpt_path, device):
    """Load a HyperFoldPocket checkpoint (plain-dict conf)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_conf = dict_to_ns(ckpt['conf'])
    model = HyperFoldPocket(model_conf)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device).eval()
    print(f'model loaded from {ckpt_path} (epoch {ckpt.get("epoch")}, '
          f'metrics {ckpt.get("metrics")})')
    return model


def seq_axis_mask_to_graph(pred_masks, graph, map_dict):
    """Map (Q, L_seq) masks onto the graph residue axis (Q, num_residue)."""
    inv = {}  # seq pos (0-based) -> graph residue index
    for g_idx, res_num in enumerate(graph.residue_number.tolist()):
        seq_pos = map_dict.get(str(res_num))
        if seq_pos is not None:
            inv[int(seq_pos) - 1] = g_idx
    num_res = graph.num_residue
    out = np.zeros((pred_masks.shape[0], num_res), dtype=np.float32)
    for j in range(pred_masks.shape[1]):
        g = inv.get(j)
        if g is not None:
            out[:, g] = pred_masks[:, j]
    return out


def build_record(pred, graph, map_dict, pdb_file_path):
    """Model output + structure -> per-protein result record.

    Pocket centers: ConvexHull over all heavy atoms of the masked residues
    (the C-alpha parser may skip residues lacking a CA atom; the atom-level
    view is aligned back to the CA graph residue numbers when they differ).
    """
    mask_graph = seq_axis_mask_to_graph(pred['pocket_masks'], graph, map_dict)
    atom_view = load_pdb_atomlevel(pdb_file_path)
    if atom_view.num_residue != graph.num_residue:
        atom_view = atom_view.align_to_resnums(graph.residue_number.tolist())
    assert atom_view.num_residue == graph.num_residue
    centers = calc_center(atom_view, mask_graph, methods=['hull'])[
        'hull_centers']
    return {
        'labels': pred['labels'],
        'scores': pred['scores'],
        'pocket_masks': pred['pocket_masks'],
        'centers': centers,
        'hull_centers': centers,
        'pdb_file_path': pdb_file_path,
    }


def run_benchmark(args, device):
    data_root = args.data_root
    ds_dir = os.path.join(data_root, args.dataset)
    out_dir = os.path.join(args.out_root, f'{args.dataset}_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ds_dir, f'{args.dataset}.csv'))
    seq_df = pd.read_csv(os.path.join(ds_dir, f'{args.dataset}_seq.csv'))
    seq_map = dict(zip(seq_df['name'].astype(str), seq_df['sequence']))
    rows = []
    for _, r in df.iterrows():
        name = str(r['name'])
        seq = seq_map.get(name)
        if not isinstance(seq, str) or len(seq) == 0:
            print(f'[WARN] {name}: no sequence, skip')
            continue
        pdb_path = r['pdb_file_path']
        if not os.path.isabs(pdb_path):
            pdb_path = os.path.join(data_root, pdb_path)
        if not os.path.exists(pdb_path):
            # some benchmark csvs carry a stale top-level dir name
            # (e.g. coach420.csv points at coach/...); re-anchor on the
            # dataset directory
            parts = os.path.normpath(r['pdb_file_path']).split(os.sep)
            alt = os.path.join(ds_dir, *parts[1:])
            if os.path.exists(alt):
                pdb_path = alt
        rows.append({'name': name, 'sequence': seq, 'pdb_file_path': pdb_path,
                     'length': len(seq)})
    # length-sorted batching to bound padded batch size (OOM guard)
    rows.sort(key=lambda r: r['length'])
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f'{args.dataset}: {len(rows)} proteins to run')

    # ---- phase A: graph preload/validation (resume: skip existing pkls) ----
    graphs = {}
    valid_rows = []
    t0 = time.time()
    for i, row in enumerate(rows):
        name = row['name']
        out_pkl = os.path.join(out_dir, f'{name}.pkl')
        if os.path.exists(out_pkl):
            continue  # resume: already inferred
        try:
            graph = load_pdb(row['pdb_file_path'])
            if graph.num_residue > row['length']:
                print(f'[WARN] {name}: graph residues {graph.num_residue} '
                      f'> seq len {row["length"]}, skip')
                continue
            graphs[name] = graph
            valid_rows.append(row)
        except Exception as e:
            print(f'[WARN] {name}: preload failed: {e}')
        if (i + 1) % 100 == 0:
            print(f'  preload {i + 1}/{len(rows)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    print(f'preload done: {len(valid_rows)} valid '
          f'({time.time() - t0:.0f}s), loading model ...')

    # ---- phase B: model inference ----
    model = load_model(args.ckpt, device)
    dataset = BenchmarkDataset(valid_rows, graphs)
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=length_collate)

    n_done, t0 = 0, time.time()
    with torch.no_grad():
        for feats in loader:
            names = feats['name']
            try:
                feats = map_to(feats, device)
                outputs = model(feats)
                preds = model.post_process(outputs, feats['length'],
                                           return_numpy=True)
            except Exception as e:
                print(f'[WARN] batch failed ({names}): {e}')
                continue
            for i, pred in enumerate(preds):
                name = names[i]
                try:
                    rec = build_record(pred, graphs[name],
                                       feats['map_dict'][i],
                                       feats['pdb_file_path'][i])
                    with open(os.path.join(out_dir, f'{name}.pkl'), 'wb') as f:
                        pickle.dump(rec, f)
                    n_done += 1
                except Exception as e:
                    print(f'[WARN] {name}: post failed: {e}')
            if n_done and n_done % 200 < args.batch_size:
                print(f'  inferred {n_done}/{len(valid_rows)} '
                      f'({time.time() - t0:.0f}s)', flush=True)

    print(f'INFER_DONE {args.dataset}: {n_done} pkls -> {out_dir} '
          f'({time.time() - t0:.0f}s)')


def run_single_pdb(args, device):
    """Quick-start: predict pockets of one PDB file and print them."""
    pdb_path = args.pdb
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    graph = load_pdb(pdb_path)
    seq = to_sequence(graph)
    row = {'name': name, 'sequence': seq, 'pdb_file_path': pdb_path,
           'length': len(seq)}
    model = load_model(args.ckpt, device)
    dataset = BenchmarkDataset([row], {name: graph})
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=length_collate)
    with torch.no_grad():
        feats = next(iter(loader))
        feats = map_to(feats, device)
        outputs = model(feats)
        pred = model.post_process(outputs, feats['length'],
                                  return_numpy=True)[0]
    rec = build_record(pred, graph, feats['map_dict'][0], pdb_path)

    keep = rec['scores'] > args.score_thresh
    n_show = int(keep.sum())
    print(f'{name}: {len(seq)} residues, '
          f'{n_show}/{len(rec["scores"])} predicted pockets '
          f'(score > {args.score_thresh})')
    resnums = graph.residue_number.tolist()
    for q in np.where(keep)[0]:
        members = [resnums[i] for i, m in
                   enumerate(seq_axis_mask_to_graph(
                       rec['pocket_masks'][q:q + 1], graph,
                       feats['map_dict'][0])[0]) if m > 0]
        center = rec['hull_centers'][q]
        print(f'  pocket {q}: score={rec["scores"][q]:.3f} '
              f'center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) '
              f'residues={members}')
    if args.out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                    exist_ok=True)
        with open(args.out, 'wb') as f:
            pickle.dump(rec, f)
        print(f'saved -> {args.out}')


def main():
    parser = argparse.ArgumentParser(
        description='Hyper-Fold-Pocket inference (benchmark / single PDB)')
    parser.add_argument('--ckpt',
                        default='checkpoints/hyperfold_pocket_unisite_ds.pth',
                        help='HyperFoldPocket checkpoint (.pth); default: '
                             'the paper checkpoint in checkpoints/')
    parser.add_argument('--pdb', default=None,
                        help='single-PDB mode: path to one structure file')
    parser.add_argument('--dataset', choices=['holo4k-sc', 'coach420'],
                        default=None, help='benchmark mode dataset')
    parser.add_argument('--data_root', default='datasets/benchmark_datasets')
    parser.add_argument('--out_root', default='results/benchmark')
    parser.add_argument('--tag', default='hyperfold')
    parser.add_argument('--out', default=None,
                        help='single-PDB mode: optional output pkl path')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                        help='single-PDB mode: print pockets above this score')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0,
                        help='benchmark mode: only run the first N proteins '
                             '(smoke test)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.pdb is not None:
        run_single_pdb(args, device)
    elif args.dataset is not None:
        run_benchmark(args, device)
    else:
        parser.error('give either --pdb PATH (single structure) or '
                     '--dataset {holo4k-sc,coach420} (benchmark)')


if __name__ == '__main__':
    main()
