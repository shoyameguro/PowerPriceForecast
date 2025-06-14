#!/bin/bash
# preprocess.sh
python -m src.features.build_features --input data/raw/train.csv --split train
python -m src.features.build_features --input data/raw/test.csv  --split test