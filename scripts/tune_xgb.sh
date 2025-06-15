#!/bin/bash
# Optuna tuning for XGBoost model
python -m src.training.tune_hyperparams \
  --cfg src/config/xgb_baseline.yaml \
  --input data/train/train.pkl \
  --trials 50 \
  --out tuning/xgb
