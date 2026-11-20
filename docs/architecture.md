# Architecture

## Coordinate conventions

Everything user-facing is in **world millimetres (LPS)**, matching LUNA16
annotations. Voxel indices only appear inside cropping helpers
(`VolumeGeometry.world_to_voxel`, `crop_3d`) and are carried through the candidate
table as `voxel_z/y/x` so stage 2 can crop without recomputing the transform.

`VolumeGeometry` stores `origin`, `spacing`, `direction` and `shape` from
SimpleITK; `affine` is the 4×4 matrix mapping continuous voxel indices to world
coordinates. This is the only place where geometry is interpreted, which keeps the
rest of the pipeline free of axis-order ambiguity (SimpleITK is `(x, y, z)`, NumPy
arrays are `(z, y, x)` — the conversion happens once, in `load_volume`).

## Preprocessing

`lung_nodule.data.preprocess` writes one compressed `npz` per scan:

| key | dtype | content |
| --- | --- | --- |
| `image` | float16 | HU clipped to [-1000, 400], scaled to [0, 1] |
| `mask` | uint8 | union of nodule spheres (radius = diameter / 2) |
| `heatmap` | float16 | max of Gaussian blobs, σ = 0.35 × diameter (min 1 mm) |
| `meta` | str | JSON `origin`/`spacing`/`direction`/`shape` |

Resampling is linear to 1 mm isotropic. Because the spacing is fixed, physical
distances become voxel counts, diameters can be estimated from voxel volume, and
patch sizes have a consistent meaning across scans.

## Stage 1 — detector

- **Network**: `SwinUNETR(feature_size=48, out_channels=2)` from MONAI.
- **Targets**: channel 0 is the binary mask (Dice + CE), channel 1 is the Gaussian
  centre heatmap (BCE with `pos_weight=5`). The heatmap gives a clean, well-separated
  peak per nodule, which is what makes local-maximum extraction work.
- **Sampling**: `RandCropByPosNegLabeld(pos=2, neg=1)` → two thirds of the patches
  contain nodule voxels. Augmentations are flips, 90° rotations and small affine
  warps; all three keys (`image`, `mask`, `heatmap`) are transformed together.
- **Inference**: `sliding_window_inference` with 96³ windows, overlap 0.5 and
  Gaussian blending, batched 4 windows per forward pass.

## Candidate extraction

1. `maximum_filter(center_prob, window=5)` + threshold (`0.35`) → plateau
   components, each reduced to its centre of mass (`scipy.ndimage.label`).
2. Score = heatmap peak × mask support (configurable via `candidates.score_mode`).
3. Diameter = equivalent sphere of `mask_prob ≥ 0.5` voxels in a 17³ window,
   clipped to [3, 40] mm.
4. Distance NMS in world space (5 mm) removes duplicate peaks on the same nodule.

All of this lives in `lung_nodule.inference.postprocess` and is pure NumPy, so the
PyTorch path, the ONNX path and the Triton path share it.

## Stage 2 — false-positive reducer

- Input: `(2, 48, 48, 48)` — CT crop and the stage-1 mask probability.
- Trunk: 3D ResNet-18 (basic blocks, InstanceNorm, 32-channel stem, four stages).
- Heads: nodule logit and log-diameter (auxiliary task, improves the embedding and
  gives a size estimate for reporting).
- Training: positives from annotations, negatives from mined stage-1 false
  positives, balanced 1:2 by default.
- Fusion: `score = p₁ · (ε + (1 − ε) · p₂)` with `ε = 1e-3`, then a second NMS on
  the fused score. `combine.mode` switches to `stage2_only` or `mean` without
  touching the models.

## Backend symmetry

```
NodulePipeline (candidate extraction, crops, fusion, NMS)
        ├── TorchStage1Runner / TorchStage2Runner   (in-process)
        └── TritonStage1Runner / TritonStage2Runner (gRPC)
```

Both runner pairs implement the structural protocols in `inference/base.py`. The
sliding-window grid is MONAI's in both cases; only the per-window forward pass
differs, so a difference in FROC numbers can only come from the exported graph or
the serving configuration — which is exactly what the regression gate is for.

## Failure modes worth knowing

| symptom | likely cause |
| --- | --- |
| CPM drops only on Triton | ONNX parity < 0.9999, or a Triton `dims` / batch-size mismatch |
| candidates clustered on vessels | heatmap `pos_weight` too low; increase `center_weight` |
| no candidates at all | HU window changed, or heatmap threshold above the trained peak range |
| stage-2 collapses to all-negative | hard negatives dominate; lower `neg_pos_ratio` |
| diameter estimates biased small | `mask_radius_vox` too small for large nodules (> 20 mm) |
