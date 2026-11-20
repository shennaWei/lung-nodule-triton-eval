# Triton model repository

```
model_repository/
├── stage1_swinunetr/
│   ├── config.pbtxt          committed
│   ├── model_metadata.json   written by `lndet-export-onnx`
│   └── 1/
│       └── model.onnx        generated (git-ignored)
└── stage2_fp_reducer/
    ├── config.pbtxt
    ├── model_metadata.json
    └── 1/
        └── model.onnx
```

Populate the graphs with:

```bash
make export   # wraps python -m lung_nodule.deploy.export_onnx
```

The exporter verifies each graph against PyTorch with ONNX Runtime and writes the
max absolute difference and cosine similarity into `model_metadata.json`. CI
refuses to publish an image when `min_cosine < 0.9999`.

## Contract between the client and the servers

| model | input | shape (per batch item) | outputs |
| --- | --- | --- | --- |
| `stage1_swinunetr` | `image` | `1 x 96 x 96 x 96` float32, CT in `[0, 1]` | `mask_logits`, `center_logits` (both `1 x 96 x 96 x 96`) |
| `stage2_fp_reducer` | `crop` | `2 x 48 x 48 x 48` float32: image + stage-1 mask prob | `logit`, `diameter` (log-mm) |

Logits are returned unnormalised on purpose: sigmoid happens client-side so the
fusion weights can change without re-exporting the model.
