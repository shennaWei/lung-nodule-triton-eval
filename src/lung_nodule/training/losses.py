"""Loss functions for both stages."""

from __future__ import annotations

import torch
from torch import nn


def soft_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Hard Dice on binarised probabilities, reported for monitoring only."""
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return float(((2.0 * intersection + eps) / (union + eps)).item())


class Stage1Loss(nn.Module):
    """Dice+CE on the nodule mask, class-weighted BCE on the centre heatmap."""

    def __init__(
        self,
        mask_weight: float = 1.0,
        center_weight: float = 0.5,
        dice_weight: float = 0.5,
        ce_weight: float = 1.0,
        center_pos_weight: float = 5.0,
    ) -> None:
        super().__init__()
        from monai.losses import DiceCELoss

        self.mask_weight = float(mask_weight)
        self.center_weight = float(center_weight)
        self.mask_loss = DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            include_background=True,
            batch=False,
            lambda_dice=float(dice_weight),
            lambda_ce=float(ce_weight),
        )
        # Soft Gaussian targets: the positive weight rescales the (rare) foreground.
        self.register_buffer("center_pos_weight", torch.tensor([float(center_pos_weight)]))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask_logits = outputs["mask_logits"]
        center_logits = outputs["center_logits"]
        mask_target = targets["mask"]
        center_target = targets["heatmap"]

        mask_loss = self.mask_loss(mask_logits, mask_target)
        center_loss = nn.functional.binary_cross_entropy_with_logits(
            center_logits,
            center_target,
            pos_weight=self.center_pos_weight.to(center_logits.device),
        )

        total = self.mask_weight * mask_loss + self.center_weight * center_loss
        with torch.no_grad():
            dice = soft_dice(torch.sigmoid(mask_logits), mask_target)
        metrics = {
            "loss": float(total.detach().item()),
            "loss_mask": float(mask_loss.detach().item()),
            "loss_center": float(center_loss.detach().item()),
            "dice": dice,
        }
        return total, metrics


class Stage2Loss(nn.Module):
    """BCE on the nodule logit plus a small SmoothL1 penalty on log-diameter."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        diameter_weight: float = 0.1,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.diameter_weight = float(diameter_weight)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.diameter_loss = nn.SmoothL1Loss(beta=0.1)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logit = outputs["logit"].squeeze(1)
        label = targets["label"].to(logit.device).float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            pos_weight=self.pos_weight.to(logit.device).view(1),
        )

        metrics = {
            "loss_bce": float(bce.detach().item()),
            "accuracy": float(((torch.sigmoid(logit) > 0.5).float() == label).float().mean().item()),
        }

        total = self.bce_weight * bce
        if self.diameter_weight > 0 and "diameter" in outputs:
            target_diameter = targets["diameter"].to(logit.device).float().clamp(min=1.0)
            # Only regress diameter for positives: negatives have no meaningful size.
            mask = label > 0.5
            if mask.any():
                predicted = outputs["diameter"].squeeze(1)[mask]
                diameter_loss = self.diameter_loss(predicted, torch.log(target_diameter[mask]))
                total = total + self.diameter_weight * diameter_loss
                metrics["loss_diameter"] = float(diameter_loss.detach().item())

        metrics["loss"] = float(total.detach().item())
        return total, metrics


def build_stage1_loss(config: dict) -> Stage1Loss:
    return Stage1Loss(
        mask_weight=float(config.get("mask_weight", 1.0)),
        center_weight=float(config.get("center_weight", 0.5)),
        dice_weight=float(config.get("dice_weight", 0.5)),
        ce_weight=float(config.get("ce_weight", 1.0)),
        center_pos_weight=float(config.get("center_pos_weight", 5.0)),
    )


def build_stage2_loss(config: dict) -> Stage2Loss:
    return Stage2Loss(
        bce_weight=float(config.get("bce_weight", 1.0)),
        diameter_weight=float(config.get("diameter_weight", 0.1)),
        pos_weight=float(config.get("pos_weight", 1.0)),
    )


__all__ = ["Stage1Loss", "Stage2Loss", "build_stage1_loss", "build_stage2_loss", "soft_dice"]
