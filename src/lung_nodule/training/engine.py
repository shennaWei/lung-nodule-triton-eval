"""Shared training engine: AMP loop, schedulers, checkpointing, metric tracking."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lung_nodule.utils.logging import get_logger


class MetricTracker:
    """Batch-size weighted running average of scalar metrics."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._weights: dict[str, float] = defaultdict(float)

    def update(self, metrics: Mapping[str, float], weight: int = 1) -> None:
        for key, value in metrics.items():
            self._sums[key] += float(value) * weight
            self._weights[key] += weight

    def compute(self) -> dict[str, float]:
        return {key: self._sums[key] / self._weights[key] for key in self._sums if self._weights[key] > 0}


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 1e-5) -> Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


def build_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.01,
):
    """Linear warmup followed by cosine decay, stepped once per batch."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    base_lr = optimizer.param_groups[0]["lr"]

    schedulers = []
    milestones = []
    if warmup_steps > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        )
        milestones.append(warmup_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    schedulers.append(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=base_lr * float(min_lr_ratio),
        )
    )
    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
    log_every: int = 50,
    epoch: int = 0,
    logger=None,
) -> dict[str, float]:
    logger = logger or get_logger()
    model.train()
    tracker = MetricTracker()
    started = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        images: torch.Tensor = batch["image"]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss, metrics = loss_fn(outputs, batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        tracker.update(metrics, weight=int(images.shape[0]))
        if logger is not None and log_every > 0 and step % log_every == 0:
            running = tracker.compute()
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch %d step %d | loss %.4f | %s | lr %.2e",
                epoch,
                step,
                running.get("loss", float("nan")),
                " ".join(f"{k}={v:.4f}" for k, v in running.items() if k != "loss"),
                lr,
            )
        if max_steps is not None and step >= max_steps:
            break

    summary = tracker.compute()
    summary["seconds"] = time.time() - started
    summary["lr"] = optimizer.param_groups[0]["lr"]
    return summary


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return averaged metrics plus per-sample predictions (score, label) if present."""
    model.eval()
    tracker = MetricTracker()
    predictions: list[dict[str, Any]] = []

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(batch["image"])
            _, metrics = loss_fn(outputs, batch)
        tracker.update(metrics, weight=int(batch["image"].shape[0]))

        if "label" in batch:
            scores = torch.sigmoid(outputs["logit"].squeeze(1)).detach().float().cpu().tolist()
            labels = batch["label"].detach().float().cpu().tolist()
            predictions.extend(
                {"score": float(score), "label": float(label)}
                for score, label in zip(scores, labels, strict=True)
            )
        if max_steps is not None and step >= max_steps:
            break

    return tracker.compute(), predictions


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": best_metric,
        "config": dict(config or {}),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def steps_per_epoch(loader: DataLoader, max_steps: int | None = None) -> int:
    steps = len(loader)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(1, steps)


def is_better(current: float, best: float | None, mode: str = "max", min_delta: float = 1e-4) -> bool:
    if best is None or math.isnan(best):
        return True
    return current > best + min_delta if mode == "max" else current < best - min_delta


__all__ = [
    "MetricTracker",
    "build_optimizer",
    "build_scheduler",
    "evaluate",
    "is_better",
    "load_checkpoint",
    "move_batch",
    "resolve_device",
    "save_checkpoint",
    "steps_per_epoch",
    "train_one_epoch",
]
