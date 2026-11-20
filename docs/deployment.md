# Deployment on Triton Inference Server

## Prerequisites

| component | version used here |
| --- | --- |
| GPU driver | ≥ 550 (CUDA 12.4 runtime) |
| Docker + nvidia-container-toolkit | any recent |
| `nvcr.io/nvidia/tritonserver` | `24.08-py3` |
| client | `tritonclient[grpc] >= 2.45` |
| export | `torch >= 2.5`, `onnx >= 1.16`, `onnxruntime >= 1.18` |

Client-side requirements are in `pyproject.toml`; the server is a stock Triton
image, so there is nothing to compile.

## 1. Export

```bash
python -m lung_nodule.deploy.export_onnx \
    --config configs/pipeline.yaml \
    --stage1-checkpoint checkpoints/stage1_best.pt \
    --stage2-checkpoint checkpoints/stage2_best.pt \
    --model-repository deploy/triton/model_repository
```

The exporter writes `deploy/triton/model_repository/<model>/1/model.onnx` and
`model_metadata.json` next to the config, containing the SHA-256, opset and the
PyTorch↔ONNX parity check:

```json
{
  "stage": "stage1",
  "sha256": "…",
  "input_shape": [1, 1, 96, 96, 96],
  "verification": {"max_abs_diff_0": 1.2e-05, "cosine_0": 0.99999,
                   "max_abs_diff_1": 9.8e-06, "cosine_1": 0.99999,
                   "min_cosine": 0.99999}
}
```

`--fp16` converts weights to FP16 while keeping FP32 I/O; re-run the FROC gate
afterwards, because FP16 changes scores at the third or fourth decimal.

Why fixed spatial dims? Swin's window partition uses static reshapes; dynamic
batch is safe, dynamic spatial axes are not (short of exporting a graph per window
size). Triton is therefore fed 96³ windows and the client owns the tiling.

## 2. Start the server

```bash
make triton-up        # docker compose -f deploy/triton/docker-compose.yaml up -d
make triton-logs
```

Readiness endpoints:

```bash
curl -s localhost:8000/v2/health/ready     # server ready
curl -s localhost:8000/v2/models/stage1_swinunetr/config   # effective config
curl -s localhost:8001/v2/models/stage2_fp_reducer/stats   # queue stats
```

If a model fails to load, Triton keeps serving the others. Check the log line
`failed to load 'stage1_swinunetr'` — the usual cause is a `dims` mismatch between
`config.pbtxt` and the ONNX graph, which `--strict-model-config=true` surfaces
immediately instead of silently reshaping.

## 3. Verify with synthetic data

```bash
python scripts/triton_smoke_test.py --url localhost:8001 --repeat 3
```

This sends a random 96³ volume and a batch of 4 crops, checks output shapes and
prints latencies. It needs no patient data and is the fast feedback loop when you
change `config.pbtxt`.

## 4. Run a cohort

```bash
python -m lung_nodule.deploy.infer_triton \
    --config configs/triton_client.yaml \
    --pipeline-config configs/pipeline.yaml \
    --images-dir data/raw/luna16/subset0 \
    --split configs/splits/luna16_fold0.json --split-key val \
    --output artifacts/predictions/val.csv \
    --workers 2 --metrics
```

`--workers N` runs N scans concurrently. Each thread gets its own gRPC client, so
scale `instance_group.count` (or `--workers`) until the stage-1 queue depth in
`/v2/models/stage1_swinunetr/stats` stays near zero. `--metrics` snapshots the
Prometheus endpoint to `<output>.triton_metrics.txt`.

## 5. Tuning

| goal | knob | notes |
| --- | --- | --- |
| higher GPU utilisation | `max_batch_size` + `preferred_batch_size` | stage 1 is bandwidth bound; 4 windows per request is a good default |
| lower first-token latency | `max_queue_delay_microseconds` | 2000 µs favours batching, 0 favours latency |
| more concurrency | `instance_group.count` | two instances of stage 1 need ~2× VRAM (96³ Swin is ~4 GB with cudnn workspace) |
| higher throughput | TensorRT EP block in `config.pbtxt` | verify numerics; keep `precision_mode: FP16` only if the FROC gate stays green |
| CPU fallback | `instance_group { kind: KIND_CPU }` | works for stage 2, painfully slow for stage 1 |

Measured with `stage1` on an A10G (96³, batch 4): ~90 ms per batched request,
~26 windows per 512×512×300 volume → ~0.7 s/scan. Stage 2 adds ~15 ms for 40
candidates. Numbers vary with `cudnn_conv_algo_search` and clock throttling; treat
them as order-of-magnitude only.

## 6. Troubleshooting

| error | cause | fix |
| --- | --- | --- |
| `unexpected shape for input 'image'` | batch carried in `dims` | `dims` excludes the batch axis when `max_batch_size > 0` |
| `model not ready` at startup | ONNX opset newer than the ORT version in the image | export with `--opset 17` |
| CUDA OOM after a few requests | `max_batch_size` × instance count too high | lower batch or instance count; enable `cuda_memory_pool_byte_size` limits |
| client hangs | `client_timeout` shorter than stage-1 latency | raise `client_timeout_s` in `configs/triton_client.yaml` |
| outputs differ from PyTorch | FP16 export or TensorRT EP | rerun `--fp16` parity check; compare `min_cosine` |
| `model_metadata.json` stale | export ran into a different repository | rerun `make export` and check the SHA in the file |

## 7. Security notes

Triton's HTTP/gRPC endpoints are unauthenticated by default. In production, put it
behind a reverse proxy with mTLS, keep the model repository read-only (the compose
file mounts it `:ro`), and never expose port 8002 publicly — metrics reveal model
names and traffic patterns.
