"""Stage-1 training entry point.

Usage::

    python -m lung_nodule.training.train_stage1 --config configs/stage1_swinunetr.yaml

The loop reports patch-level Dice and centre loss. End-to-end sensitivity lives in
the FROC report produced after inference (``lndet-predict`` + ``lndet-eval-froc``),
which is what the regression gate watches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from monai.data import Dataset, list_data_collate
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from lung_nodule.data.dataset import build_stage1_data_list, missing_scans
from lung_nodule.data.luna16 import load_split
from lung_nodule.data.transforms import build_stage1_train_transforms, build_stage1_val_transforms
from lung_nodule.models.factory import parameter_count
from lung_nodule.models.stage1_detector import build_stage1
from lung_nodule.training.engine import (
    build_optimizer,
    build_scheduler,
    evaluate,
    is_better,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    steps_per_epoch,
    train_one_epoch,
)
from lung_nodule.training.losses import build_stage1_loss
from lung_nodule.utils.config import load_config
from lung_nodule.utils.io import append_jsonl, ensure_dir, git_sha, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Stage-1 YAML config")
    parser.add_argument("--override", dest="overrides", action="append", default=[], help="a.b=c override")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop each epoch after N steps (debug)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)

    log_dir = ensure_dir(cfg.runtime.log_dir)
    logger = setup_logging(
        args.log_level, log_file=log_dir / "train.log", name="lung_nodule.stage1", force=True
    )
    if cfg.deterministic:
        set_determinism(int(cfg.seed))
    device = resolve_device(cfg.device)
    logger.info("device=%s config=%s", device, cfg.fingerprint())

    split = load_split(cfg.data.split_file)
    train_uids = split["train"]
    val_uids = split["val"]
    train_data = build_stage1_data_list(cfg.data.preprocessed_dir, train_uids)
    val_data = build_stage1_data_list(cfg.data.preprocessed_dir, val_uids)

    missing = missing_scans(cfg.data.preprocessed_dir, train_uids)
    if missing:
        logger.warning("%d training scans are not preprocessed yet (run `lndet-preprocess`)", len(missing))
    if not train_data:
        logger.error("no preprocessed training volumes found under %s", cfg.data.preprocessed_dir)
        return 1

    patch_size = tuple(int(v) for v in cfg.data.patch_size)
    train_dataset = Dataset(
        train_data,
        transform=build_stage1_train_transforms(
            patch_size=patch_size,
            pos=int(cfg.data.pos_samples),
            neg=int(cfg.data.neg_samples),
            num_samples=int(cfg.data.num_samples),
        ),
    )
    val_dataset = (
        Dataset(val_data, transform=build_stage1_val_transforms(patch_size=patch_size)) if val_data else None
    )

    num_workers = int(cfg.data.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=list_data_collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(cfg.data.val_batch_size),
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=device.type == "cuda",
            collate_fn=list_data_collate,
        )
        if val_dataset is not None
        else None
    )

    model = build_stage1(cfg.model).to(device)
    logger.info("stage-1 parameters: %.2fM", parameter_count(model) / 1e6)

    loss_fn = build_stage1_loss(cfg.loss.to_dict()).to(device)
    optimizer = build_optimizer(model, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    epochs = int(args.epochs or cfg.optim.epochs)
    steps = steps_per_epoch(train_loader, args.max_steps)
    scheduler = build_scheduler(
        optimizer,
        total_steps=steps * epochs,
        warmup_steps=int(float(cfg.optim.warmup_epochs) * steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg.optim.amp) and device.type == "cuda")

    start_epoch = 0
    best_dice: float | None = None
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_dice = payload.get("best_metric")
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    ckpt_dir = ensure_dir(cfg.runtime.checkpoint_dir)
    best_path = ckpt_dir / str(cfg.runtime.checkpoint_name)
    metrics_path = Path(cfg.runtime.metrics_file)
    val_every = int(cfg.runtime.val_every)

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler=scaler if scaler.is_enabled() else None,
            scheduler=scheduler,
            grad_clip=float(cfg.optim.grad_clip),
            max_steps=args.max_steps,
            log_every=int(cfg.runtime.log_every),
            epoch=epoch,
            logger=logger,
        )

        val_metrics: dict[str, float] = {}
        should_validate = val_loader is not None and (epoch + 1) % val_every == 0
        if should_validate:
            val_metrics, _ = evaluate(model, val_loader, loss_fn, device, max_steps=args.max_steps)

        record = {
            "epoch": epoch,
            "timestamp": utc_now_iso(),
            "git_sha": git_sha(),
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(metrics_path, record)
        logger.info(
            "epoch %d | train loss %.4f dice %.4f%s",
            epoch,
            train_metrics.get("loss", float("nan")),
            train_metrics.get("dice", float("nan")),
            f" | val dice {val_metrics['dice']:.4f}" if "dice" in val_metrics else "",
        )

        monitor = val_metrics.get("dice", train_metrics.get("dice"))
        if monitor is not None and is_better(float(monitor), best_dice, mode="max"):
            best_dice = float(monitor)
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_dice,
                config=cfg.to_dict(),
                extra={"monitor": "val_dice", "patch_size": list(patch_size)},
            )
            logger.info("new best patch Dice %.4f -> %s", best_dice, best_path)

        if (epoch + 1) % int(cfg.runtime.save_every) == 0:
            save_checkpoint(
                ckpt_dir / f"{Path(str(cfg.runtime.checkpoint_name)).stem}_last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_dice,
                config=cfg.to_dict(),
            )

    summary = {
        "stage": 1,
        "run_name": cfg.run_name,
        "config_fingerprint": cfg.fingerprint(),
        "git_sha": git_sha(),
        "epochs": epochs,
        "best_patch_dice": best_dice,
        "checkpoint": str(best_path),
        "finished_at": utc_now_iso(),
    }
    write_json(log_dir / "summary.json", summary)
    logger.info("training finished: best patch Dice %s", best_dice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
