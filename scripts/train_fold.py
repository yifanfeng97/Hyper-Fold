"""Fold classification trainer for Hyper-Fold-Deep.

Implements the paper's fold-classification recipe: single-label
1,195-class training on SCOPe 1.75 / Fold3D with cross-entropy loss (label
smoothing 0.1), structure-forcing augmentation (Gaussian coordinate noise
sigma = 0.15 Angstrom, residue-type masking p = 0.15) and accuracy metrics.

Recipe: SGD lr 1e-3,
momentum 0.9, wd 5e-4, batch 16, 400 epochs, 10-epoch linear warmup + step LR
decay (milestones 100/300, gamma 0.1), EMA 0.999, AMP, grad clip 10. After
training, the best (EMA) checkpoint is evaluated on all three test scenarios:
test_fold / test_superfamily / test_family. Configuration is driven by a YAML
file; CLI arguments override the YAML values. Single-GPU by default; DDP is
enabled automatically under ``torchrun``.
"""
import argparse
import json
import logging
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from hyperfold.data.classification_datasets import FoldDataset, collate_fn
from hyperfold.evaluation.classification import compute_accuracy
from hyperfold.utils.checkpoint import write_checkpoint
from hyperfold.utils.misc import ModelEMA, is_dist_avail_and_initialized
from hyperfold.data.utils import read_pkl
from train_ec import (
    MODEL_REGISTRY,
    _isolate_target_gpu,
    _is_main_process,
    _setup_distributed,
    build_model,
    build_optimizer,
    build_scheduler,
    model_name_and_kwargs,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyper-Fold-Deep fold classification trainer"
    )
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--gpu", type=int, help="GPU device id")
    parser.add_argument("--num-epochs", type=int, help="number of epochs")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--workers", type=int, help="data loading workers")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--ckpt-path", dest="ckpt_path", help="checkpoint path")
    parser.add_argument("--log-path", dest="log_path", help="log file path")
    parser.add_argument("--resume", help="checkpoint to resume from")
    parser.add_argument("--ema-decay", type=float, dest="ema_decay",
                        help="EMA decay (0.0 disables)")
    parser.add_argument("--eval-interval", type=int, dest="eval_interval",
                        help="evaluate every N epochs")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="enable AMP")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp",
                        default=None, help="disable AMP")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="compile the model with torch.compile")
    parser.add_argument("--no-compile", action="store_true", dest="no_compile",
                        default=None, help="disable torch.compile")
    return parser.parse_args()


def train_epoch(model, dataloader, loss_fn, optimizer, device, scaler,
                ema=None, clip_grad_norm=None):
    """Run one training epoch, returning mean CE loss."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(dataloader, desc="train")
    for data in pbar:
        data = data.to(device)
        y = data.y
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            out = model(data)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model.module if hasattr(model, "module") else model)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        pbar.set_postfix(loss=total_loss / n)
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def evaluate_acc(model, dataloader, device, use_amp):
    """Top-1 accuracy."""
    model.eval()
    preds, labels = [], []
    for data in tqdm(dataloader, desc="eval"):
        data = data.to(device)
        with torch.autocast(device.type, enabled=use_amp):
            out = model(data)
        preds.append(out.detach().cpu().numpy())
        labels.append(data.y.detach().cpu().numpy())
    return compute_accuracy(np.concatenate(labels), np.concatenate(preds))


def main() -> None:
    cli_args = parse_args()
    if cli_args.gpu is not None:
        # GPU isolation runs before any CUDA use.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    physical_gpu = int(cli_args.gpu) if cli_args.gpu is not None else 0

    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in vars(cli_args).items():
        if key == "config" or value is None:
            continue
        if key in ("amp", "no_amp", "compile", "no_compile"):
            continue
        cfg[key] = value
    if cli_args.amp is not None and cli_args.amp:
        cfg["amp"] = True
    if cli_args.no_amp:
        cfg["amp"] = False
    if cli_args.compile is not None and cli_args.compile:
        cfg["compile"] = True
    if cli_args.no_compile:
        cfg["compile"] = False

    cfg.setdefault("model", {"name": "hyperfold_deep"})
    cfg.setdefault("data_dir", "data/fold3d/new_fold3d")
    cfg.setdefault("num_classes", 1195)
    cfg.setdefault("num_epochs", 400)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("workers", 4)
    cfg.setdefault("seed", 0)
    cfg.setdefault("gpu", 0)
    model_name, _ = model_name_and_kwargs(cfg)
    cfg.setdefault(
        "ckpt_path",
        f"checkpoints/fold_{model_name}/best.pth"
    )
    cfg.setdefault("eval_interval", 1)
    cfg.setdefault("ema_decay", 0.0)
    cfg.setdefault("clip_grad_norm", 10.0)
    cfg.setdefault("compile", False)

    local_rank = _setup_distributed()
    distributed = is_dist_avail_and_initialized()
    is_main = _is_main_process()

    use_amp = cfg.get("amp", False)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", local_rank or None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger = setup_logging(cfg.get("log_path") if is_main else None)
    logger.info("--- Starting new fold run ---")
    logger.info("Config:\n%s", yaml.dump(cfg, default_flow_style=False))

    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Physical GPU: %d, device: %s", physical_gpu, device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(cfg["seed"]))

    seed = int(cfg["seed"])
    train_dataset = FoldDataset(root=cfg["data_dir"], split="train",
                                random_seed=seed,
                                noise_std=cfg.get("noise_std", 0.15),
                                mask_prob=cfg.get("mask_prob", 0.15))
    valid_dataset = FoldDataset(root=cfg["data_dir"], split="valid",
                                random_seed=seed)
    logger.info("Train: %d, Valid: %d, Classes: %d",
                len(train_dataset), len(valid_dataset),
                train_dataset.num_classes)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True,
                                           drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg["batch_size"]),
            sampler=train_sampler, drop_last=True,
            num_workers=int(cfg["workers"]), collate_fn=collate_fn
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
            drop_last=True, num_workers=int(cfg["workers"]),
            collate_fn=collate_fn
        )
    valid_loader = DataLoader(
        valid_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["workers"]), collate_fn=collate_fn
    )

    model_name, model_kwargs = model_name_and_kwargs(cfg)
    model = build_model(
        model_name, train_dataset.num_classes, model_kwargs
    ).to(device)
    if cfg.get("compile", False):
        logger.info("Compiling model with torch.compile(dynamic=True)")
        model = torch.compile(model, dynamic=True)
    num_params = sum(p.numel() for p in model.state_dict().values())
    logger.info("Model parameters: %d", num_params)
    raw_model = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )

    resume_path = cfg.get("resume")
    resume_state = None
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        # See train_ec.py: write_checkpoint uses pickle.HIGHEST_PROTOCOL, so
        # load through read_pkl rather than torch.load(weights_only=True).
        resume_state = read_pkl(resume_path, use_torch=True,
                                map_location=device, verbose=False)
        if isinstance(resume_state, dict) and "model" in resume_state:
            raw_model.load_state_dict(resume_state["model"])
        else:
            raw_model.load_state_dict(resume_state)

    optimizer = build_optimizer(raw_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    if resume_path and isinstance(resume_state, dict):
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            logger.info("Restored optimizer state")

    num_epochs = int(cfg["num_epochs"])
    start_epoch = 0
    if resume_path and isinstance(resume_state, dict):
        resume_epoch = resume_state.get("epoch")
        if resume_epoch is not None:
            start_epoch = int(resume_epoch) + 1
            logger.info("Resuming at epoch %d", start_epoch + 1)
    # Lambda/MultiStep schedulers are deterministic functions of the epoch;
    # replay the schedule up to the resume point.
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        for _ in range(start_epoch):
            scheduler.step()

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.1))
    )

    ema = None
    ema_decay = float(cfg["ema_decay"])
    if ema_decay > 0.0:
        ema = ModelEMA(raw_model, decay=ema_decay)
        logger.info("Using EMA with decay %.5f", ema_decay)

    best_valid_acc = -1.0
    best_epoch = 0
    if (resume_path and isinstance(resume_state, dict)
            and "metrics" in resume_state):
        best_valid_acc = float(
            resume_state["metrics"].get("best_val_acc", best_valid_acc)
        )

    num_steps = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            ema=ema, clip_grad_norm=cfg.get("clip_grad_norm")
        )

        do_eval = ((epoch + 1) % int(cfg["eval_interval"]) == 0) or \
                  (epoch == num_epochs - 1)
        if do_eval:
            # Evaluation (and checkpointing) run on the main process only.
            if is_main:
                eval_model = ema.module if ema is not None else raw_model
                valid_acc = evaluate_acc(eval_model, valid_loader, device,
                                         use_amp)

                logger.info(
                    "Epoch: %03d, TrainLoss: %.4f, Valid acc: %.4f",
                    epoch + 1, train_loss, valid_acc
                )

                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    best_epoch = epoch
                    best_weights = (
                        ema.module.state_dict() if ema is not None
                        else raw_model.state_dict()
                    )
                    write_checkpoint(
                        cfg["ckpt_path"], best_weights, cfg,
                        optimizer.state_dict(), epoch, epoch * num_steps,
                        logger=logger,
                        metrics={"best_val_acc": best_valid_acc,
                                 "val_acc": valid_acc},
                    )
                    logger.info("Saved best checkpoint to %s", cfg["ckpt_path"])
            else:
                logger.info("Epoch: %03d, TrainLoss: %.4f",
                            epoch + 1, train_loss)
            if distributed:
                torch.distributed.barrier()
        else:
            logger.info("Epoch: %03d, TrainLoss: %.4f", epoch + 1, train_loss)

        scheduler.step()

    logger.info("Best: Epoch %03d, Validation acc: %.4f",
                best_epoch + 1, best_valid_acc)

    # Final evaluation: best checkpoint on all three test scenarios.
    results = {"best_epoch": best_epoch + 1,
               "best_valid_acc": best_valid_acc}
    if is_main:
        ckpt = read_pkl(cfg["ckpt_path"], use_torch=True,
                        map_location=device, verbose=False)
        eval_model = ema.module if ema is not None else raw_model
        eval_model.load_state_dict(ckpt["model"])
        for test_split in ("test_fold", "test_superfamily", "test_family"):
            test_dataset = FoldDataset(root=cfg["data_dir"], split=test_split,
                                       random_seed=seed)
            test_loader = DataLoader(
                test_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
                num_workers=int(cfg["workers"]), collate_fn=collate_fn
            )
            acc = evaluate_acc(eval_model, test_loader, device, use_amp)
            results[test_split] = acc
            logger.info("Test %s acc: %.4f", test_split, acc)

        result_path = osp.splitext(cfg["ckpt_path"])[0] + "_results.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved test results to %s", result_path)

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _isolate_target_gpu()
    main()
