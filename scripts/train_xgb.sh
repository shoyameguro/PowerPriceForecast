#!/bin/bash
# train.sh
python -m src.training.train_model \
  cfg=src/config/xgb_baseline.yaml \
  input=data/train/train.pkl
