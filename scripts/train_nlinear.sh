#!/bin/bash
# train_nlinear.sh
python -m src.training.train_model \
  --cfg src/config/nlinear.yaml \
  --input data/train/train.pkl \
  --model_dir output/models
