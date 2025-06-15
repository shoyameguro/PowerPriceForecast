#!/bin/bash
# Optuna tuning for Darts NLinear model
python -m src.training.tune_hyperparams \
  --cfg src/config/nlinear.yaml \
  --input data/train/train.pkl \
  --trials 30 \
  --out tuning/nlinear
