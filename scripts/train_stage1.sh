#!/usr/bin/env bash
python -m src.training.tune_hyperparams cv_stage=stage1 trials=100
