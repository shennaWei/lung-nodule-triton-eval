# Convenience targets for the two-stage lung nodule pipeline.
# Every target is a thin wrapper around the `lndet-*` console scripts so the
# same commands work locally, in CI and inside containers.

PYTHON ?= python
CONFIG_STAGE1 ?= configs/stage1_swinunetr.yaml
CONFIG_STAGE2 ?= configs/stage2_fp_reducer.yaml
CONFIG_PIPELINE ?= configs/pipeline.yaml
CONFIG_TRITON ?= configs/triton_client.yaml

IMAGES_DIR ?= data/raw/luna16/subset0
ANN_CSV ?= data/raw/luna16/annotations/annotations.csv
CAND_CSV ?= data/raw/luna16/annotations/candidates.csv
PREPROCESSED ?= data/preprocessed/luna16_1mm
CKPT_DIR ?= checkpoints
ONNX_DIR ?= deploy/triton/model_repository
PRED_CSV ?= artifacts/predictions/luna16_val.csv
BASELINE ?= eval/baselines/luna16_fold0.json

.PHONY: help install preprocess train-stage1 train-stage2 export triton-up triton-down \
        triton-logs predict predict-triton froc froc-regression leaderboard lint format

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

preprocess:  ## Resample LUNA16 scans to 1 mm and cache image/mask/heatmap arrays
	$(PYTHON) -m lung_nodule.data.preprocess \
		--images-dir $(IMAGES_DIR) --annotations $(ANN_CSV) --output $(PREPROCESSED)

train-stage1:  ## Train the SwinUNETR detector
	$(PYTHON) -m lung_nodule.training.train_stage1 --config $(CONFIG_STAGE1)

train-stage2:  ## Train the 3D ResNet false-positive reducer
	$(PYTHON) -m lung_nodule.training.train_stage2 --config $(CONFIG_STAGE2)

export:  ## Export both stages to ONNX into the Triton model repository
	$(PYTHON) -m lung_nodule.deploy.export_onnx --config $(CONFIG_PIPELINE) \
		--stage1-checkpoint $(CKPT_DIR)/stage1_best.pt \
		--stage2-checkpoint $(CKPT_DIR)/stage2_best.pt \
		--model-repository $(ONNX_DIR)

triton-up:  ## Start the Triton Inference Server via docker compose
	docker compose -f deploy/triton/docker-compose.yaml up -d

triton-down:  ## Stop the Triton server
	docker compose -f deploy/triton/docker-compose.yaml down

triton-logs:  ## Follow Triton server logs
	docker compose -f deploy/triton/docker-compose.yaml logs -f triton

predict:  ## Run the local PyTorch pipeline over a cohort
	$(PYTHON) -m lung_nodule.inference.pipeline --config $(CONFIG_PIPELINE) \
		--images-dir $(IMAGES_DIR) --annotations $(ANN_CSV) --output $(PRED_CSV)

predict-triton:  ## Run the same cohort through Triton
	$(PYTHON) -m lung_nodule.deploy.infer_triton --config $(CONFIG_TRITON) \
		--pipeline-config $(CONFIG_PIPELINE) \
		--images-dir $(IMAGES_DIR) --annotations $(ANN_CSV) --output $(PRED_CSV)

froc:  ## Compute FROC / CPM for an existing prediction file
	$(PYTHON) -m lung_nodule.eval.froc --predictions $(PRED_CSV) --annotations $(ANN_CSV) \
		--output-dir artifacts/froc --plot

froc-regression:  ## Compare a FROC report against the committed baseline
	$(PYTHON) -m lung_nodule.eval.regression --report artifacts/froc/report.json \
		--baseline $(BASELINE) --output-dir artifacts/froc

leaderboard:  ## Render the CPM leaderboard from the run history
	$(PYTHON) -m lung_nodule.eval.tracking --history artifacts/froc/history.jsonl \
		--output artifacts/froc/LEADERBOARD.md

lint:  ## Static checks
	ruff check src scripts
	ruff format --check src scripts

format:  ## Auto-format the source tree
	ruff check --fix src scripts
	ruff format src scripts
