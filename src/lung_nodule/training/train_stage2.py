"""Stage-2 training entry point (false-positive reduction).

Negative samples come from mined stage-1 false positives when
``data.hard_negatives`` exists, otherwise from LUNA16 ``candidates.csv``::

    python -m lung_nodule.training.train_stage2 --config configs/stage2_fp_reducer.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lung_nodule.data.candidates import load_training_samples
from lung_nodule.data.dataset import Stage2CandidateDataset, Stage2CropBatch
from lung_nodule.data.luna16 import load_split
from lung_nodule.data.transforms import build_stage2_transforms
from lung_nodule.eval.metrics import binary_auroc
from lung_nodule.models.factory import parameter_count
from lung_nodule.models.stage2_classifier import build_stage2
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
from lung_nodule.training.losses import build_stage2_loss
from lung_nodule.utils.config import load_config
from lung_nodule.utils.io import append_jsonl, ensure_dir, git_sha, utc_now_iso, write_json
from lung_nodule.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Stage-2 YAML config")
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--source", choices=("auto", "candidates", "hard_negatives"), default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    log_dir = ensure_dir(cfg.runtime.log_dir)
    logger = setup_logging(
        args.log_level, log_file=log_dir / "train.log", name="lung_nodule.stage2", force=True
    )
    device = resolve_device(cfg.device)
    logger.info("device=%s config=%s", device, cfg.fingerprint())

    split = load_split(cfg.data.split_file)
    hard_negatives_path = Path(str(cfg.data.hard_negatives))
    if args.source == "auto":
        source = "hard_negatives" if hard_negatives_path.exists() else "candidates"
    else:
        source = args.source
    logger.info("training samples from %s", source)

    common = {
        "candidates_csv": cfg.data.candidates,
        "hard_negatives_json": hard_negatives_path,
        "annotations_csv": cfg.data.annotations,
        "neg_pos_ratio": float(cfg.data.get("neg_pos_ratio", 2.0)),
        "seed": int(cfg.seed),
    }
    train_samples = load_training_samples(source, series_uids=split["train"], **common)
    val_samples = load_training_samples(source, series_uids=split["val"], **common)
    if not train_samples:
        logger.error("no stage-2 samples found for the training split")
        return 1
    if not val_samples:
        logger.warning("no stage-2 samples for the validation split; skipping validation")

    crop_size = tuple(int(v) for v in cfg.data.crop_size)
    use_mask_channel = bool(cfg.data.use_mask_channel)
    if use_mask_channel != (int(cfg.model.in_channels) == 2):
        logger.warning(
            "data.use_mask_channel=%s but model.in_channels=%s; make them consistent",
            use_mask_channel,
            cfg.model.in_channels,
        )

    train_dataset = Stage2CandidateDataset(
        train_samples,
        preprocessed_dir=cfg.data.preprocessed_dir,
        crop_size=crop_size,
        use_mask_channel=use_mask_channel,
        transform=build_stage2_transforms(augment=bool(cfg.data.augment)),
    )
    val_dataset = (
        Stage2CandidateDataset(
            val_samples,
            preprocessed_dir=cfg.data.preprocessed_dir,
            crop_size=crop_size,
            use_mask_channel=use_mask_channel,
            transform=build_stage2_transforms(augment=False),
        )
        if val_samples
        else None
    )

    collate = Stage2CropBatch(use_mask_channel=use_mask_channel)
    num_workers = int(cfg.data.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) > int(cfg.data.batch_size),
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(cfg.data.val_batch_size),
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        if val_dataset is not None
        else None
    )

    logger.info(
        "stage-2 samples: %d train / %d val (positives %d)",
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
        int(train_dataset.labels().sum()),
    )

    model = build_stage2(cfg.model).to(device)
    logger.info("stage-2 parameters: %.2fM", parameter_count(model) / 1e6)

    loss_fn = build_stage2_loss(cfg.loss.to_dict()).to(device)
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
    best_auroc: float | None = None
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_auroc = payload.get("best_metric")
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    ckpt_dir = ensure_dir(cfg.runtime.checkpoint_dir)
    best_path = ckpt_dir / str(cfg.runtime.checkpoint_name)
    metrics_path = Path(str(cfg.runtime.metrics_file))

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
        if val_loader is not None:
            val_metrics, predictions = evaluate(model, val_loader, loss_fn, device, max_steps=args.max_steps)
            if predictions:
                auroc = binary_auroc(
                    [row["score"] for row in predictions],
                    [row["label"] for row in predictions],
                )
                if auroc == auroc:  # not NaN
                    val_metrics["auroc"] = auroc

        record = {
            "epoch": epoch,
            "timestamp": utc_now_iso(),
            "git_sha": git_sha(),
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(metrics_path, record)
        logger.info(
            "epoch %d | train loss %.4f acc %.3f | val %s",
            epoch,
            train_metrics.get("loss", float("nan")),
            train_metrics.get("accuracy", float("nan")),
            " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()),
        )

        monitor = val_metrics.get("auroc", -val_metrics.get("loss", float("inf")))
        if monitor is not None and is_better(float(monitor), best_auroc, mode="max"):
            best_auroc = float(monitor)
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_auroc,
                config=cfg.to_dict(),
                extra={"monitor": "val_auroc", "source": source, "crop_size": list(crop_size)},
            )
            logger.info("new best val AUROC %.4f -> %s", best_auroc, best_path)

        if (epoch + 1) % int(cfg.runtime.save_every) == 0:
            save_checkpoint(
                ckpt_dir / f"{Path(str(cfg.runtime.checkpoint_name)).stem}_last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_auroc,
                config=cfg.to_dict(),
            )

    write_json(
        log_dir / "summary.json",
        {
            "stage": 2,
            "run_name": cfg.run_name,
            "config_fingerprint": cfg.fingerprint(),
            "git_sha": git_sha(),
            "source": source,
            "epochs": epochs,
            "best_val_auroc": best_auroc,
            "checkpoint": str(best_path),
            "n_train_samples": len(train_dataset),
            "finished_at": utc_now_iso(),
        },
    )
    logger.info("training finished: best val AUROC %s", best_auroc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
