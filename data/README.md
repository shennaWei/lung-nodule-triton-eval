# Data layout

Nothing in this directory is committed except this README. The pipeline expects:

```
data/
├── raw/
│   └── luna16/
│       ├── subset0/ ... subset9/        # .mhd + .raw scans (LUNA16)
│       └── annotations/
│           ├── annotations.csv          # seriesuid,coordX,coordY,coordZ,diameter_mm
│           └── candidates.csv           # seriesuid,coordX,coordY,coordZ,class
└── preprocessed/
    └── luna16_1mm/
        ├── <series_uid>.npz             # image / mask / heatmap / geometry
        └── manifest.json
```

Build the cache with:

```bash
python -m lung_nodule.data.preprocess \
    --images-dir data/raw/luna16/subset0 \
    --annotations data/raw/luna16/annotations/annotations.csv \
    --output data/preprocessed/luna16_1mm
```

LUNA16 is distributed as a set of password-protected 7z archives at
<https://luna16.grand-challenge.org/>. The annotations and candidates CSVs are
part of the same release. `scripts/luna16_to_nifti.py` converts `.mhd` volumes to
`.nii.gz` if you prefer a single file format.
