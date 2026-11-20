# FROC regression tracking

The point of this file: a model change that costs sensitivity must fail CI, and
the failure must be easy to read for someone who was not involved in the change.

## 1. Protocol

LUNA16 scoring (implemented in `lung_nodule.eval.froc`):

- A prediction **matches** an annotation when its centre lies inside the annotated
  sphere: `‖p − a‖ ≤ radius_scale × diameter/2` with `radius_scale = 0.5`.
- Each annotation can be matched once, each prediction can match once. Matching is
  greedy in descending score order, which is equivalent to re-running the matcher at
  every threshold (a higher-scoring prediction always claims the annotation first).
- **Sensitivity** = matched annotations / all annotations.
- **False positives per scan** = unmatched predictions / number of evaluated scans.
- **CPM** = mean sensitivity at `{1/8, 1/4, 1/2, 1, 2, 4, 8}` FP/scan.

The sweep is a single pass: sort by score, accumulate TP/FP per unique score value.
At each operating point sensitivity is read as a step function (`max` sensitivity
with `FP/scan ≤ target`), which is the conservative reading.

Two extra numbers are tracked because CPM alone can hide local trade-offs:
`fp_at_sens0.90` (how many false positives you need for 90 % sensitivity) and the
per-level sensitivity table.

## 2. Producing a report

```bash
python -m lung_nodule.eval.froc \
    --predictions artifacts/predictions/val.csv \
    --annotations data/raw/luna16/annotations/annotations.csv \
    --output-dir artifacts/froc --name report \
    --bootstrap --n-boot 200 --plot
```

Outputs:

| file | content |
| --- | --- |
| `report.json` | metrics, counts, operating points, down-sampled curve, metadata |
| `report.md` | human-readable table |
| `report_curve.csv` | full threshold / FP / sensitivity curve |
| `report.png` | FROC plot (matplotlib optional) |

Use `--n-scans` when the evaluated cohort is larger than the annotated subset (for
example when scoring a full submission), otherwise the denominator comes from the
annotations.

Bootstrap: scans are resampled with replacement (200 draws by default) and CPM is
recomputed, giving a 95 % interval. This is a fold-level uncertainty estimate, not
a substitute for 10-fold evaluation.

## 3. Baseline file

`eval/baselines/luna16_fold0.json`:

```json
{
  "schema_version": 1,
  "name": "luna16-fold0-val",
  "dataset": {"split": "configs/splits/luna16_fold0.json", "n_scans": 89, "n_annotations": 118},
  "metrics": {"cpm": 0.8659, "sens@1fp": 0.8890, "fp_at_sens0.90": 2.31},
  "tolerance": {"default": 0.01, "cpm": 0.005, "fp_at_sens0.90": 0.25},
  "direction": {"fp_at_sens0.90": "lower_is_better"},
  "require": ["cpm"]
}
```

Semantics:

- A metric **regressed** when it moves beyond its tolerance in the bad direction.
- `tolerance.<metric>` overrides `tolerance.default`.
- `direction` defaults to `higher_is_better`; `fp_at_sens0.90` is
  `lower_is_better`.
- Metrics present in the report but not in the baseline are reported as `new` and
  never fail the gate.
- Metrics present in the baseline but missing from the report fail unless
  `--allow-missing` is passed and they are not in `require`.

Tolerance guidance: CPM on ~90 scans moves by a few thousandths just from scan
sampling, so `0.005` catches real losses without flapping; per-level sensitivities
are noisier (`0.01`); `fp_at_sens0.90` is in FP/scan units, hence `0.25`.

## 4. Running the gate

```bash
python -m lung_nodule.eval.regression \
    --report artifacts/froc/report.json \
    --baseline eval/baselines/luna16_fold0.json \
    --output-dir artifacts/froc --name regression --summary
```

Exit codes: `0` pass, `1` regression, `2` unreadable input. `--no-fail` is for
"report only" runs (nightly drift dashboards). `--summary` appends the markdown to
`$GITHUB_STEP_SUMMARY`.

In CI (`.github/workflows/froc-regression.yml`):

1. Add the `froc` label to a PR → the `predict` job runs on the GPU runner
   (export → Triton → cohort inference) and uploads `val-predictions`.
2. `gate` computes FROC and the regression check on a CPU runner, uploads the
   artifacts and posts/updates a single PR comment (marker `<!-- froc-regression -->`).
3. A failing check blocks the merge; the failing metrics are listed under
   **Failing checks** in the comment.

Nightly runs on the default branch catch drift that no PR triggered (e.g. a
`tritonclient` upgrade).

## 5. Promoting a new baseline

Never edit numbers by hand. From a `workflow_dispatch` run with
`update_baseline=true`, the `publish-baseline` job rewrites the JSON, commits it and
pushes — the diff is the review artifact. Locally:

```bash
python -m lung_nodule.eval.regression \
    --report artifacts/froc/report.json \
    --baseline eval/baselines/luna16_fold0.json --update-baseline
```

Rules of thumb:

- Promote only when the change is intentional (new model, new post-processing) and
  the FROC curve *shape* improved, not just the CPM average.
- Keep the baseline per fold; averaging folds inside one file hides single-fold
  regressions.
- Recompute the baseline whenever the annotations, split file or preprocessing
  changes — otherwise the gate compares apples to oranges (record the split path in
  `dataset` so the mismatch is visible).

## 6. History and leaderboard

```bash
python -m lung_nodule.eval.tracking --append --report artifacts/froc/report.json \
    --history artifacts/froc/history.jsonl --backend triton --tag fold=0
python -m lung_nodule.eval.tracking --history artifacts/froc/history.jsonl \
    --output artifacts/froc/LEADERBOARD.md
```

Each record stores `git_sha`, branch, dirty flag, config fingerprint, backend,
dataset tag and the metric dict. `measure_trend` compares the last *n* runs with the
previous *n* and reports `improving` / `stable` / `degrading` plus a slope, which is
the first thing to look at when the nightly job starts drifting.

## 7. Known pitfalls

- **Wrong denominator.** Using "scans with predictions" instead of all evaluated
  scans inflates sensitivity at low FP rates.
- **Duplicate predictions on one nodule.** Without NMS a single nodule can produce
  several peaks; they become FPs even though the lesion was found.
- **Threshold ties.** Ties must be swept as one group, otherwise the curve depends
  on row order. The implementation groups by unique score value.
- **Matching across scans.** Always group by `seriesuid` first; a global greedy
  matcher will happily match a prediction to a nodule in another patient.
- **Comparing against a moved baseline.** If the split or annotations changed,
  regenerate the baseline before interpreting a regression.
