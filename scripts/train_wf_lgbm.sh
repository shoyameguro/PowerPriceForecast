#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_wf.sh – Train model with 7‑fold walk‑forward CV (train_model_wf.py)
# ---------------------------------------------------------------------------
# Usage:
#     bash scripts/train_wf.sh <yaml_cfg>
#
# Example:
#     bash scripts/train_wf.sh src/config/lgbm_baseline.yaml
# ---------------------------------------------------------------------------
set -euo pipefail

CFG_PATH=${1:-src/config/lgbm_baseline.yaml}
RAW_DIR="data"
PROCESSED_DIR="${RAW_DIR}/train"
MODEL_DIR="output/models_wf"

python -m src.training.train_model_wf \
    --cfg "${CFG_PATH}" \
    --input "${PROCESSED_DIR}/train.pkl" \
    --model_dir "${MODEL_DIR}"
