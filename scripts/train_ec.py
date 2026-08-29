"""EC-number prediction trainer for Hyper-Fold.

Implements the paper's EC recipe, restricted to the Hyper-Fold model
family: the flat 6-block ``HyperFoldClassifier`` (the paper's "Hyper-Fold"
EC backbone) and the hierarchical ``HyperFoldDeep`` ("Hyper-Fold-Deep").

Recipe: multi-label
BCE-with-logits with per-class weights, AdamW lr 1e-4 wd 0, batch 16, 500
epochs, 10-epoch linear warmup + cosine decay, EMA 0.999, AMP, grad clip 10.
Metrics: Fmax and micro-AUPR (GearNet protocol). Configuration is driven by a
YAML file; CLI arguments override the YAML values. Single-GPU by default;
DDP is enabled automatically under ``torchrun``.
"""
import argparse
import logging
import math
import os
import os.path as osp
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from hyperfold.data.classification_datasets import ECDataset, collate_fn
from hyperfold.evaluation.classification import (
    compute_fmax,
    compute_micro_aupr,
)
from hyperfold.model.classifier import HyperFoldClassifier
from hyperfold.model.hyperfold_deep import HyperFoldDeep
from hyperfold.utils.checkpoint import write_checkpoint
from hyperfold.utils.misc import ModelEMA, is_dist_avail_and_initialized
from hyperfold.data.utils import read_pkl


MODEL_REGISTRY = {
    "hyperfold": HyperFoldClassifier,
    "hyperfold_deep": HyperFoldDeep,
}


def _isolate_target_gpu():
    """Expose only the requested GPU to this process before importing torch.

    This prevents PyTorch from allocating a CUDA context on the default
    cuda:0 (which may be a different physical GPU) at import time. After
    isolation the target card is seen as cuda:0 inside the process.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def _load_subset(path: Optional[str]) -> Optional[set]:
    """Load a newline-separated list of protein names, returning a set."""
    if path is None:
        return None
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def model_name_and_kwargs(cfg: Dict[str, Any]) -> tuple:
    """Support both ``model: name`` and ``model: {name: ..., ...}`` configs."""
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        name = model_cfg.get("name", "hyperfold")
        kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
        return name, kwargs
    return model_cfg, cfg.get("model_kwargs", {})


def build_model(model_name: str, num_classes: int,
                model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Construct a model from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}"
        )
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("num_classes", num_classes)
    return MODEL_REGISTRY[model_name](**kwargs)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build AdamW or SGD optimizer."""
    optim = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if optim == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if optim == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if optim == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {optim}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    """Build cosine, step, or plateau learning-rate scheduler."""
    schedule = cfg.get("lr_schedule", "cosine").lower()
    num_epochs = int(cfg["num_epochs"])
    if schedule == "cosine":
        warmup = max(int(cfg.get("warmup_epochs", 0)), 0)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            p = (epoch - warmup) / max(1, num_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if schedule == "step":
        milestones = cfg.get("lr_milestones", [300, 400])
        gamma = float(cfg.get("lr_gamma", 0.1))
        warmup = max(int(cfg.get("warmup_epochs", 0)), 0)
        if warmup > 0:
            # Linear warmup, then multi-step decay at absolute milestones.
            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                factor = 1.0
                for m in milestones:
                    if epoch >= m:
                        factor *= gamma
                return factor

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

    if schedule == "plateau":
        factor = float(cfg.get("lr_factor", 0.6))
        patience = int(cfg.get("lr_patience", 5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        )

    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def train_epoch(model: nn.Module, dataloader: DataLoader,
                loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device, scaler: "torch.cuda.amp.GradScaler",
                ema: Optional[ModelEMA] = None,
                clip_grad_norm: Optional[float] = None) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n = 0
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
def evaluate(model: nn.Module, dataloader: DataLoader,
             device: torch.device, use_amp: bool) -> Dict[str, float]:
    """Evaluate the model and return GearNet-protocol EC metrics."""
    model.eval()
    probs, labels = [], []
    for data in tqdm(dataloader, desc="eval"):
        data = data.to(device)
        with torch.autocast(device.type, enabled=use_amp):
            out = model(data)
        prob = out.sigmoid().detach().cpu().numpy()
        y = data.y.detach().cpu().numpy()
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return {
        "fmax": compute_fmax(labels, probs),
        "aupr_micro": compute_micro_aupr(labels, probs),
    }


def merge_config(base_cfg: Dict[str, Any],
                 cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Return a copy of ``base_cfg`` overridden with explicitly provided CLI args."""
    cfg = {**base_cfg}
    overrides = vars(cli_args)

    # Handle AMP and compile flags explicitly so omitted flags don't clobber
    # the YAML value.
    if overrides.get("amp") is not None:
        cfg["amp"] = overrides["amp"]
    if overrides.get("no_amp") is not None:
        cfg["amp"] = not overrides["no_amp"]
    if overrides.get("compile") is not None:
        cfg["compile"] = overrides["compile"]
    if overrides.get("no_compile") is not None:
        cfg["compile"] = not overrides["no_compile"]

    for key, value in overrides.items():
        if key in ("config", "amp", "no_amp", "compile", "no_compile") or value is None:
            continue
        cfg[key] = value
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; omitted values do not override the YAML config."""
    parser = argparse.ArgumentParser(
        description="Hyper-Fold EC-number prediction trainer"
    )
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY),
                        help="model architecture")
    parser.add_argument("--gpu", type=int, help="GPU device id")
    parser.add_argument("--num-epochs", type=int, help="number of epochs")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight-decay", "--wd", type=float, dest="weight_decay",
                        help="weight decay")
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"],
                        help="optimizer")
    parser.add_argument("--lr-schedule", choices=["cosine", "step"],
                        dest="lr_schedule", help="LR schedule")
    parser.add_argument("--lr-milestones", nargs="+", type=int,
                        help="multi-step LR milestones")
    parser.add_argument("--lr-gamma", type=float, help="multi-step LR decay")
    parser.add_argument("--warmup-epochs", type=int, dest="warmup_epochs",
                        help="linear warmup epochs (cosine only)")
    parser.add_argument("--workers", type=int, help="data loading workers")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--ckpt-path", dest="ckpt_path", help="checkpoint path")
    parser.add_argument("--log-path", dest="log_path", help="log file path")
    parser.add_argument("--resume", help="checkpoint to resume from")
    parser.add_argument("--start-epoch", type=int, dest="start_epoch",
                        help="epoch to resume at")
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


def setup_logging(log_path: Optional[str]) -> logging.Logger:
    """Configure console + file logging."""
    logger = logging.getLogger("hyperfold")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path:
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def _setup_distributed() -> int:
    """Initialize DDP when launched via torchrun; return the local rank."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def _is_main_process() -> bool:
    return (not is_dist_avail_and_initialized()
            or torch.distributed.get_rank() == 0)


def main() -> None:
    cli_args = parse_args()
    physical_gpu = int(cli_args.gpu) if cli_args.gpu is not None else 0
    with open(cli_args.config, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    cfg = merge_config(base_cfg, cli_args)

    # Defaults for any values not supplied by config or CLI.
    cfg.setdefault("model", "hyperfold")
    cfg.setdefault("data_dir", "data/EnzymeCommission_cdconv")
    cfg.setdefault("num_classes", 538)
    cfg.setdefault("num_epochs", 500)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("optimizer", "adamw")
    cfg.setdefault("lr_schedule", "cosine")
    cfg.setdefault("lr_milestones", [300, 400])
    cfg.setdefault("lr_gamma", 0.1)
    cfg.setdefault("warmup_epochs", 10)
    cfg.setdefault("workers", 4)
    cfg.setdefault("seed", 0)
    cfg.setdefault("gpu", 0)
    model_name, _ = model_name_and_kwargs(cfg)
    cfg.setdefault(
        "ckpt_path",
        f"checkpoints/ec_{model_name}/best.pth"
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

    log_path = cfg.get("log_path") if is_main else None
    logger = setup_logging(log_path)
    logger.info("--- Starting new run ---")
    logger.info("Config:\n%s", yaml.dump(cfg, default_flow_style=False))

    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Physical GPU: %d, device: %s", physical_gpu, device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(cfg["seed"]))

    train_subset = _load_subset(cfg.get("train_subset"))
    train_dataset = ECDataset(
        root=cfg["data_dir"], split="train", random_seed=int(cfg["seed"]),
        protein_subset=train_subset
    )
    valid_dataset = ECDataset(
        root=cfg["data_dir"], split="valid", random_seed=int(cfg["seed"])
    )

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
        model_name, int(cfg["num_classes"]), model_kwargs
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
    resume_state: Optional[Dict[str, Any]] = None
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        # Checkpoints are written by utils.checkpoint.write_checkpoint, which
        # serializes with pickle.HIGHEST_PROTOCOL; torch.load(weights_only=True)
        # only supports protocol 2, so load through read_pkl instead.
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
    start_epoch = int(cfg.get("start_epoch", 0))
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

    if cfg.get("use_class_weights", True):
        weights = torch.as_tensor(train_dataset.weights, dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(weight=weights)
        logger.info("Using per-class weights for BCE loss")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        logger.info("Using unweighted BCE loss")

    ema = None
    ema_decay = float(cfg["ema_decay"])
    if ema_decay > 0.0:
        ema = ModelEMA(raw_model, decay=ema_decay)
        logger.info("Using EMA with decay %.5f", ema_decay)

    best_valid_fmax = -1.0
    best_epoch = 0
    if (
        resume_path
        and isinstance(resume_state, dict)
        and "metrics" in resume_state
    ):
        best_valid_fmax = float(
            resume_state["metrics"].get("best_val_fmax", best_valid_fmax)
        )
        logger.info("Resuming with best validation fmax: %.4f", best_valid_fmax)

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
        valid_fmax: Optional[float] = None
        if do_eval:
            # Evaluation (and checkpointing) run on the main process only.
            if is_main:
                eval_model = ema.module if ema is not None else raw_model
                valid_metrics = evaluate(eval_model, valid_loader, device,
                                         use_amp)

                valid_fmax = valid_metrics["fmax"]
                valid_aupr = valid_metrics["aupr_micro"]
                logger.info(
                    "Epoch: %03d, TrainLoss: %.4f, Valid fmax: %.4f, "
                    "Valid aupr_micro: %.4f",
                    epoch + 1, train_loss, valid_fmax, valid_aupr
                )

                if valid_fmax >= best_valid_fmax:
                    best_valid_fmax = valid_fmax
                    best_epoch = epoch
                    best_weights = (
                        ema.module.state_dict() if ema is not None
                        else raw_model.state_dict()
                    )
                    write_checkpoint(
                        cfg["ckpt_path"], best_weights, cfg,
                        optimizer.state_dict(), epoch, epoch * num_steps,
                        logger=logger,
                        metrics={
                            "best_val_fmax": best_valid_fmax,
                            **valid_metrics,
                        },
                    )
                    logger.info("Saved best checkpoint to %s", cfg["ckpt_path"])
            else:
                logger.info("Epoch: %03d, TrainLoss: %.4f",
                            epoch + 1, train_loss)
            if distributed:
                torch.distributed.barrier()
        else:
            logger.info("Epoch: %03d, TrainLoss: %.4f", epoch + 1, train_loss)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau requires a validation metric; under DDP only
            # the main process has one (the released recipes use cosine/step
            # schedules instead).
            if valid_fmax is not None:
                scheduler.step(valid_fmax)
        else:
            scheduler.step()

    logger.info(
        "Best: Epoch %03d, Validation fmax: %.4f",
        best_epoch + 1, best_valid_fmax
    )

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _isolate_target_gpu()
    main()
