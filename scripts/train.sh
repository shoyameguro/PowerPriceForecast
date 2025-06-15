#!/bin/bash
# train.sh
python -m src.training.train_model \
  --cfg src/config/lgbm_baseline.yaml \
  --input data/train/train.feather \
  --model_dir output/models