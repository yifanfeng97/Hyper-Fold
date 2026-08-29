"""Training script for Hyper-Fold-Pocket (pure-structure pocket detector).

Trains HyperFoldPocket (FoldConv
backbone + anchored-query set-prediction head) on UniSite-DS with
the precomputed ProteinGraph cache (plain dict of tensors, see
scripts/precompute_graphs.py). No ESM / 1D features are used anywhere.

Configuration is driven by ``configs/pocket.yaml`` (pyyaml); a handful of
CLI flags override the YAML values. DDP via torchrun with single-GPU
fallback; AdamW (weight decay only on >=2-dim params), linear warmup +
cosine decay, EMA of the weights for eval / best-checkpoint selection,
and AP@IoU 0.3 / 0.5 evaluation.

Launch:
    torchrun --nproc_per_node=2 scripts/train_pocket.py --config configs/pocket.yaml
    python scripts/train_pocket.py --config configs/pocket.yaml  # single GPU
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils import data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from hyperfold.data.collate import length_collate, map_to  # noqa: E402
from hyperfold.data.dataset import PocketDataset, load_names  # noqa: E402
from hyperfold.evaluation import mean_ap  # noqa: E402
from hyperfold.model.pocket import HyperFoldPocket, PocketSetCriterion  # noqa: E402
from hyperfold.utils.checkpoint import write_checkpoint  # noqa: E402
from hyperfold.utils.misc import ModelEMA  # noqa: E402


def dict_to_ns(d):
    """Nested dict -> nested SimpleNamespace (attribute access)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def ns_to_dict(ns):
    """Nested SimpleNamespace -> plain dict (for checkpoint storage)."""
    if isinstance(ns, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, (list, tuple)):
        return [ns_to_dict(v) for v in ns]
    return ns


def build_model_conf(model_cfg):
    """YAML ``model:`` section -> nested SimpleNamespace for HyperFoldPocket.

    The nested ``cdn:`` sub-section is flattened into the top level
    (dn_per_gt / dn_noise_feat / ...); ``matcher`` stays a namespace with
    cost_class / cost_mask / cost_dice.
    """
    cfg = dict(model_cfg)
    cdn = cfg.pop('cdn', {}) or {}
    cfg.update(cdn)
    return dict_to_ns(cfg)


def total_loss(loss_dict, loss_weights):
    """Weighted sum over loss terms: a term is weighted by the first weight
    name that appears in its key (e.g. ``loss_mask_dn`` -> ``mask``)."""
    total = 0.0
    for k, v in loss_dict.items():
        if not k.startswith('loss_'):
            continue
        for name, w in loss_weights.items():
            if name in k:
                total = total + w * v
                break
    return total


@torch.no_grad()
def evaluate(model, loader, device, iou_thresholds=(0.3, 0.5)):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        targets.extend([t.numpy() for t in batch['target']])
        feats = map_to(batch, device)
        outputs = model(feats)
        predictions.extend(
            model.post_process(outputs, feats['length'], return_numpy=True))
    metrics = {}
    for thr in iou_thresholds:
        _, eval_results, _ = mean_ap.eval_map(predictions, targets, 1,
                                              iou_thr=thr)
        metrics[f'AP@IoU_{thr}'] = float(eval_results[0]['ap'])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyper-Fold-Pocket trainer')
    parser.add_argument('--config', default='configs/pocket.yaml',
                        help='path to the YAML config')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    train_cfg = cfg['train']
    loss_weights = {k: float(v) for k, v in cfg['loss_weights'].items()}
    # CLI overrides
    output_dir = args.output_dir or cfg.get(
        'output_dir', 'train_outputs/hyperfold_pocket')
    epochs = args.epochs or int(train_cfg['epochs'])
    batch_size = args.batch_size or int(train_cfg['batch_size'])
    lr = args.lr or float(train_cfg['lr'])
    seed = args.seed if args.seed is not None else int(train_cfg['seed'])

    model_conf = build_model_conf(cfg['model'])
    model_conf_dict = ns_to_dict(model_conf)

    # DDP setup (launch with torchrun) with single-GPU fallback
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = rank == 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(os.path.join(output_dir, 'train_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    max_seq_len = int(data_cfg['max_seq_len'])
    lengths_csv = data_cfg.get('lengths_csv')  # default: {graph_cache}/lengths.csv
    train_names = load_names(data_cfg['train_csv'], data_cfg['graph_cache'],
                             max_seq_len, lengths_csv=lengths_csv)
    test_names = load_names(data_cfg['test_csv'], data_cfg['graph_cache'],
                            max_seq_len, lengths_csv=lengths_csv)
    print(f'train: {len(train_names)} proteins, '
          f'test: {len(test_names)} proteins')

    train_dataset = PocketDataset(train_names, data_cfg['pkl_dir'],
                                  data_cfg['graph_cache'],
                                  coord_noise=float(train_cfg['coord_noise']))
    train_sampler = data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if world_size > 1 else None
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(train_cfg['num_workers']), collate_fn=length_collate,
        drop_last=True)
    eval_loader = data.DataLoader(
        PocketDataset(test_names, data_cfg['pkl_dir'], data_cfg['graph_cache']),
        batch_size=int(train_cfg['eval_batch_size']), shuffle=False,
        num_workers=int(train_cfg['num_workers']), collate_fn=length_collate)

    model = HyperFoldPocket(model_conf).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    raw_model = model.module if hasattr(model, 'module') else model
    criterion = PocketSetCriterion(model_conf).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f'Number of model parameters: {num_params}')

    # weight decay only on >=2-dim params (weights); BN scale/bias, biases and
    # embeddings are exempt — uniform wd hurts BN-heavy encoders
    weight_decay = float(train_cfg['weight_decay'])
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    other_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': weight_decay},
         {'params': other_params, 'weight_decay': 0.0}],
        lr=lr, weight_decay=weight_decay)

    warmup_epochs = int(train_cfg['warmup_epochs'])
    if warmup_epochs > 0:
        def lr_lambda(epoch, warmup=warmup_epochs, total=epochs):
            if epoch < warmup:
                return (epoch + 1) / warmup
            t = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

    ema_decay = float(train_cfg['ema_decay'])
    ema = ModelEMA(raw_model, decay=ema_decay) if ema_decay > 0 else None

    metrics_path = os.path.join(output_dir, 'metrics.csv')
    best_ap = -1.0
    eval_every = int(train_cfg['eval_every'])
    clip_grad = float(train_cfg['clip_grad'])
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, num_steps, start = 0.0, 0, time.time()
        for batch in train_loader:
            feats = map_to(batch, device)
            targets = feats['target']
            outputs = model(feats, targets)
            loss_dict = criterion(outputs, targets)
            loss = total_loss(loss_dict, loss_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if ema is not None:
                ema.update(raw_model)
            epoch_loss += loss.item()
            num_steps += 1
        scheduler.step()
        avg_loss = epoch_loss / max(num_steps, 1)
        row = {'epoch': epoch, 'train_loss': avg_loss,
               'lr': scheduler.get_last_lr()[0],
               'epoch_time_sec': round(time.time() - start, 1)}

        if is_main and (epoch % eval_every == 0 or epoch == epochs):
            eval_model = ema.module if ema is not None else raw_model
            eval_metrics = evaluate(eval_model, eval_loader, device)
            row.update(eval_metrics)
            if eval_metrics['AP@IoU_0.5'] > best_ap:
                best_ap = eval_metrics['AP@IoU_0.5']
                write_checkpoint(
                    os.path.join(output_dir, 'best.pth'),
                    eval_model.state_dict(), model_conf_dict,
                    optimizer.state_dict(),
                    epoch, epoch * num_steps, metrics=eval_metrics)

        if is_main:
            write_checkpoint(
                os.path.join(output_dir, 'last.pth'),
                raw_model.state_dict(), model_conf_dict,
                optimizer.state_dict(), epoch, epoch * num_steps)
            pd.DataFrame([row]).to_csv(
                metrics_path, mode='a', header=not os.path.exists(metrics_path),
                index=False)
            print(f'[epoch {epoch}/{epochs}] ' +
                  ' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()
    print('TRAIN_DONE')


if __name__ == '__main__':
    main()
