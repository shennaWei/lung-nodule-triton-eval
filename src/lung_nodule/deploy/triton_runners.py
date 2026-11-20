"""Triton-backed implementations of the stage-1 / stage-2 runner protocols.

Only window scheduling and crop building happen client-side; all network
computation runs inside Triton. The sliding-window grid is delegated to MONAI so
the Triton path and the local path share exactly the same tiling and Gaussian
blending.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lung_nodule.utils.logging import get_logger

LOGGER = get_logger("lung_nodule.triton")


class TritonError(RuntimeError):
    """Raised when the server is unreachable or a model is not ready."""


class TritonClientFactory:
    """Creates one gRPC client per thread (safe for ``workers > 1``)."""

    def __init__(
        self,
        url: str = "localhost:8001",
        client_timeout_s: float = 180.0,
        verbose: bool = False,
    ) -> None:
        self.url = url
        self.client_timeout_s = float(client_timeout_s)
        self.verbose = verbose
        self._local = threading.local()

    def client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            from tritonclient.grpc import InferenceServerClient

            client = InferenceServerClient(
                url=self.url, verbose=self.verbose, client_timeout=self.client_timeout_s
            )
            self._local.client = client
        return client

    def ensure_ready(self, model_names: Sequence[str]) -> dict[str, Any]:
        client = self.client()
        if not client.is_server_ready():
            raise TritonError(
                f"Triton server at {self.url} is not ready - start it with `make triton-up` and check the logs"
            )
        status: dict[str, Any] = {"url": self.url, "server": "ready", "models": {}}
        for name in model_names:
            if not client.is_model_ready(name):
                raise TritonError(f"model {name!r} is not ready on {self.url}")
            metadata = client.get_model_metadata(name)
            status["models"][name] = {
                "versions": list(metadata.versions),
                "platform": metadata.platform,
                "inputs": [
                    {"name": item.name, "shape": list(item.shape), "dtype": item.datatype}
                    for item in metadata.inputs
                ],
                "outputs": [
                    {"name": item.name, "shape": list(item.shape), "dtype": item.datatype}
                    for item in metadata.outputs
                ],
            }
        return status

    def close(self) -> None:
        client = getattr(self._local, "client", None)
        if client is not None:
            client.close()
            self._local.client = None


class _TritonModel:
    """Thin wrapper around one Triton model: input packing + output unpacking."""

    def __init__(
        self,
        factory: TritonClientFactory,
        model_name: str,
        input_name: str,
        output_names: Sequence[str],
        max_batch: int = 8,
    ) -> None:
        self.factory = factory
        self.model_name = model_name
        self.input_name = input_name
        self.output_names = tuple(output_names)
        self.max_batch = max(1, int(max_batch))

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        from tritonclient import grpc as grpcclient
        from tritonclient.utils import InferenceServerException

        client = self.factory.client()
        payload = np.ascontiguousarray(array, dtype=np.float32)
        infer_input = grpcclient.InferInput(self.input_name, list(payload.shape), "FP32")
        infer_input.set_data_from_numpy(payload)
        requested = [grpcclient.InferRequestedOutput(name) for name in self.output_names]

        try:
            result = client.infer(
                self.model_name,
                [infer_input],
                outputs=requested,
                client_timeout=self.factory.client_timeout_s,
            )
        except InferenceServerException as exc:  # pragma: no cover - needs a live server
            raise TritonError(f"inference failed for {self.model_name}: {exc}") from exc

        return [result.as_numpy(name) for name in self.output_names]


class TritonStage1Runner:
    """Sliding-window stage-1 inference where each window is scored by Triton."""

    def __init__(
        self,
        factory: TritonClientFactory,
        model_name: str = "stage1_swinunetr",
        input_name: str = "image",
        output_names: Sequence[str] = ("mask_logits", "center_logits"),
        patch_size: tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        max_batch: int = 4,
        device: str = "cpu",
    ) -> None:
        self.model = _TritonModel(factory, model_name, input_name, output_names, max_batch=max_batch)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.overlap = float(overlap)
        self.sw_batch_size = max(1, int(max_batch))
        self.device = device

    def predict_volume(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        import torch
        from monai.inferers import sliding_window_inference

        tensor = torch.as_tensor(np.ascontiguousarray(volume, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

        def predictor(window: torch.Tensor) -> torch.Tensor:
            logits = self.model.infer(window.detach().cpu().numpy())
            stacked = np.concatenate([np.asarray(item, dtype=np.float32) for item in logits], axis=1)
            return torch.from_numpy(stacked)

        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=predictor,
            overlap=self.overlap,
            mode="gaussian",
            padding_mode="constant",
            sw_device="cpu",
            device="cpu",
            progress=False,
        )
        probabilities = torch.sigmoid(logits).numpy()[0]
        return {"mask_prob": probabilities[0], "center_prob": probabilities[1]}


class TritonStage2Runner:
    """Batched candidate scoring through the stage-2 Triton model."""

    def __init__(
        self,
        factory: TritonClientFactory,
        model_name: str = "stage2_fp_reducer",
        input_name: str = "crop",
        output_names: Sequence[str] = ("logit", "diameter"),
        max_batch: int = 32,
    ) -> None:
        self.model = _TritonModel(factory, model_name, input_name, output_names, max_batch=max_batch)

    def score_crops(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crops = np.asarray(crops, dtype=np.float32)
        if crops.ndim != 5:
            raise ValueError(f"expected (N, C, D, H, W) crops, got {crops.shape}")
        if crops.shape[0] == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        probabilities: list[np.ndarray] = []
        diameters: list[np.ndarray] = []
        for start in range(0, crops.shape[0], self.model.max_batch):
            batch = crops[start : start + self.model.max_batch]
            outputs = self.model.infer(batch)
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            probabilities.append(1.0 / (1.0 + np.exp(-logits)))
            if len(outputs) > 1:
                diameters.append(np.exp(np.asarray(outputs[1], dtype=np.float32).reshape(-1)))
            else:
                diameters.append(np.zeros_like(logits))

        return np.concatenate(probabilities), np.concatenate(diameters)


def build_triton_runners(
    triton_cfg: Mapping[str, Any],
    url: str | None = None,
) -> tuple[TritonStage1Runner, TritonStage2Runner, TritonClientFactory, dict[str, Any]]:
    """Create both runners and validate that the server/models are ready."""
    stage1_cfg = triton_cfg.get("stage1", {})
    stage2_cfg = triton_cfg.get("stage2", {})
    factory = TritonClientFactory(
        url=url or str(triton_cfg.get("url", "localhost:8001")),
        client_timeout_s=float(triton_cfg.get("client_timeout_s", 180)),
        verbose=bool(triton_cfg.get("verbose", False)),
    )

    status = factory.ensure_ready(
        [
            str(stage1_cfg.get("model_name", "stage1_swinunetr")),
            str(stage2_cfg.get("model_name", "stage2_fp_reducer")),
        ]
    )
    LOGGER.info("connected to Triton at %s (%d models ready)", factory.url, len(status["models"]))

    stage1 = TritonStage1Runner(
        factory,
        model_name=str(stage1_cfg.get("model_name", "stage1_swinunetr")),
        input_name=str(stage1_cfg.get("input_name", "image")),
        output_names=tuple(stage1_cfg.get("output_names", ("mask_logits", "center_logits"))),
        patch_size=tuple(int(v) for v in stage1_cfg.get("patch_size", (96, 96, 96))),
        overlap=float(stage1_cfg.get("overlap", 0.5)),
        max_batch=int(stage1_cfg.get("max_batch", 4)),
    )
    stage2 = TritonStage2Runner(
        factory,
        model_name=str(stage2_cfg.get("model_name", "stage2_fp_reducer")),
        input_name=str(stage2_cfg.get("input_name", "crop")),
        output_names=tuple(stage2_cfg.get("output_names", ("logit", "diameter"))),
        max_batch=int(stage2_cfg.get("max_batch", 32)),
    )
    return stage1, stage2, factory, status


__all__ = [
    "TritonClientFactory",
    "TritonError",
    "TritonStage1Runner",
    "TritonStage2Runner",
    "build_triton_runners",
]
