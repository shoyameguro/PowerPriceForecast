#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# preprocess_v2.sh – feature‑generation helper
# ---------------------------------------------------------------------------
# Runs the improved build_features_v2 pipeline for both train and test splits.
# Assumes raw CSV files are located at data/raw/ and processed outputs will be
# saved by save_data() into data/<split>/<split>.pkl .
# ---------------------------------------------------------------------------
# Usage:
#     bash scripts/preprocess_v2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

RAW_DIR="data/raw"

python -m src.features.build_features_v2 \
    --input "${RAW_DIR}/train.csv" \
    --split train

python -m src.features.build_features_v2 \
    --input "${RAW_DIR}/test.csv"  \
    --split test \
    --train "${RAW_DIR}/train.csv"
