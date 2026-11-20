"""Local PyTorch implementations of the runner protocols."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TorchStage1Runner:
    """Sliding-window inference over a full volume with Gaussian blending.

    Structurally satisfies :class:`lung_nodule.inference.base.Stage1Runner`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        device: str | torch.device = "cuda",
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        amp: bool = True,
    ) -> None:
        self.model = model.eval()
        self.patch_size = tuple(int(v) for v in patch_size)
        self.device = torch.device(device)
        self.sw_batch_size = int(sw_batch_size)
        self.overlap = float(overlap)
        self.amp = bool(amp) and self.device.type == "cuda"

    @torch.no_grad()
    def predict_volume(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        """``volume`` is ``(D, H, W)``; returns probabilities of the same shape."""
        from monai.inferers import sliding_window_inference

        tensor = torch.as_tensor(np.ascontiguousarray(volume), dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        def predictor(window: torch.Tensor) -> torch.Tensor:
            outputs = self.model(window)
            return torch.cat([outputs["mask_logits"], outputs["center_logits"]], dim=1)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=self.patch_size,
                sw_batch_size=self.sw_batch_size,
                predictor=predictor,
                overlap=self.overlap,
                mode="gaussian",
                padding_mode="constant",
                sw_device=self.device,
                device=self.device,
                progress=False,
            )
            probs = torch.sigmoid(logits).float().cpu().numpy()[0]

        return {"mask_prob": probs[0], "center_prob": probs[1]}


class TorchStage2Runner:
    """Batched scoring of candidate crops."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cuda",
        batch_size: int = 64,
        amp: bool = True,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.amp = bool(amp) and self.device.type == "cuda"

    @torch.no_grad()
    def score_crops(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crops = np.asarray(crops, dtype=np.float32)
        if crops.ndim != 5:
            raise ValueError(f"expected (N, C, D, H, W) crops, got {crops.shape}")
        if crops.shape[0] == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        probabilities: list[np.ndarray] = []
        diameters: list[np.ndarray] = []
        for start in range(0, crops.shape[0], self.batch_size):
            batch = torch.as_tensor(crops[start : start + self.batch_size]).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                outputs: dict[str, torch.Tensor] = self.model(batch)
            probabilities.append(torch.sigmoid(outputs["logit"]).squeeze(1).float().cpu().numpy())
            if "diameter" in outputs:
                diameters.append(torch.exp(outputs["diameter"].squeeze(1)).float().cpu().numpy())
            else:
                diameters.append(np.zeros(len(batch), dtype=np.float32))

        return np.concatenate(probabilities), np.concatenate(diameters)


def describe_runner(runner: Any) -> str:
    """Small helper used in logs / report metadata."""
    name = type(runner).__name__
    device = getattr(runner, "device", "n/a")
    return f"{name}(device={device})"


__all__ = ["TorchStage1Runner", "TorchStage2Runner", "describe_runner"]
