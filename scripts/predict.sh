#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# predict.sh – Generate predictions with Hydra-managed configuration
# ---------------------------------------------------------------------------
# Usage:
#     bash scripts/predict.sh <model_dir> <out_csv>
#
# Example:
#     bash scripts/predict.sh outputs/2023-10-05/15-00-00/models \
#         outputs/submissions/submission.csv
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR=${1:-models}
OUT_PATH=${2:-outputs/submissions/submission.csv}

python -m src.interface.predict \
    models="${MODEL_DIR}" \
    input=data/test/test.pkl \
    out="${OUT_PATH}"
