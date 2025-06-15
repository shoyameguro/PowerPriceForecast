#!/bin/bash
# train_lgbm.sh
python -m src.training.train_model \
  --cfg src/config/lgbm_baseline.yaml \
  --input data/train/train.pkl \
  --model_dir output/models
