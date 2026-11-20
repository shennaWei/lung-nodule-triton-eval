#!/usr/bin/env bash
# Full fold: preprocess -> train stage 1 -> infer -> mine negatives -> train stage 2
# -> export ONNX -> Triton inference -> FROC -> regression gate -> history.
#
# Every step is idempotent enough to be re-run; long steps are skipped when their
# outputs already exist (pass FORCE=1 to redo everything).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
FOLD=${FOLD:-0}
SPLIT=${SPLIT:-configs/splits/luna16_fold${FOLD}.json}
IMAGES_DIR=${IMAGES_DIR:-data/raw/luna16/subset0}
ANN=${ANN:-data/raw/luna16/annotations/annotations.csv}
CAND=${CAND:-data/raw/luna16/annotations/candidates.csv}
PRE=${PRE:-data/preprocessed/luna16_1mm}
PRED_TRAIN=${PRED_TRAIN:-artifacts/predictions/fold${FOLD}_train.csv}
PRED_VAL=${PRED_VAL:-artifacts/predictions/fold${FOLD}_val.csv}
HISTORY=${HISTORY:-artifacts/froc/history.jsonl}
BASELINE=${BASELINE:-eval/baselines/luna16_fold0.json}
FORCE=${FORCE:-0}

echo "== fold ${FOLD} =="

if [[ ! -f "${PRE}/manifest.json" || "${FORCE}" == "1" ]]; then
  echo "-- preprocess"
  ${PYTHON} -m lung_nodule.data.preprocess \
    --images-dir "${IMAGES_DIR}" --annotations "${ANN}" --output "${PRE}"
fi

if [[ ! -f checkpoints/stage1_best.pt || "${FORCE}" == "1" ]]; then
  echo "-- train stage 1"
  ${PYTHON} -m lung_nodule.training.train_stage1 --config configs/stage1_swinunetr.yaml \
    --override data.split_file="${SPLIT}"
fi

echo "-- stage-1 predictions on the training split (for hard-negative mining)"
${PYTHON} -m lung_nodule.inference.pipeline --config configs/pipeline.yaml \
  --images-dir "${IMAGES_DIR}" --split "${SPLIT}" --split-key train \
  --output "${PRED_TRAIN}" --no-stage2 --resume

echo "-- mine hard negatives"
${PYTHON} scripts/mine_hard_negatives.py --predictions "${PRED_TRAIN}" --annotations "${ANN}" \
  --split "${SPLIT}" --split-key train --per-scan 20 \
  --output artifacts/stage1/hard_negatives.json

if [[ ! -f checkpoints/stage2_best.pt || "${FORCE}" == "1" ]]; then
  echo "-- train stage 2"
  ${PYTHON} -m lung_nodule.training.train_stage2 --config configs/stage2_fp_reducer.yaml \
    --override data.split_file="${SPLIT}" --override data.candidates="${CAND}"
fi

echo "-- export ONNX"
${PYTHON} -m lung_nodule.deploy.export_onnx --config configs/pipeline.yaml \
  --stage1-checkpoint checkpoints/stage1_best.pt \
  --stage2-checkpoint checkpoints/stage2_best.pt \
  --model-repository deploy/triton/model_repository

if docker compose -f deploy/triton/docker-compose.yaml ps --status running --quiet triton | grep -q .; then
  echo "-- Triton inference on the validation split"
  ${PYTHON} -m lung_nodule.deploy.infer_triton --config configs/triton_client.yaml \
    --pipeline-config configs/pipeline.yaml --images-dir "${IMAGES_DIR}" \
    --split "${SPLIT}" --split-key val --output "${PRED_VAL}" --metrics
else
  echo "-- Triton is not running, using the local PyTorch pipeline instead"
  ${PYTHON} -m lung_nodule.inference.pipeline --config configs/pipeline.yaml \
    --images-dir "${IMAGES_DIR}" --split "${SPLIT}" --split-key val \
    --output "${PRED_VAL}" --resume
fi

echo "-- FROC"
${PYTHON} -m lung_nodule.eval.froc --predictions "${PRED_VAL}" --annotations "${ANN}" \
  --output-dir artifacts/froc --name "fold${FOLD}" --bootstrap --plot

echo "-- regression gate"
${PYTHON} -m lung_nodule.eval.regression --report artifacts/froc/fold${FOLD}.json \
  --baseline "${BASELINE}" --output-dir artifacts/froc --name "fold${FOLD}_regression"

echo "-- history"
${PYTHON} -m lung_nodule.eval.tracking --append --report artifacts/froc/fold${FOLD}.json \
  --history "${HISTORY}" --backend triton --tag fold="${FOLD}"
${PYTHON} -m lung_nodule.eval.tracking --history "${HISTORY}" \
  --output artifacts/froc/LEADERBOARD.md

echo "done: artifacts/froc/fold${FOLD}.md"
