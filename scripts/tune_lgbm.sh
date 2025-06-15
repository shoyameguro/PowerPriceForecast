#!/bin/bash
# Optuna tuning for LightGBM model
python -m src.training.tune_hyperparams \
  --cfg src/config/lgbm_baseline.yaml \
  --input data/train/train.pkl \
  --trials 50 \
  --out tuning/lgbm
