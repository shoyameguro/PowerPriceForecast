# PowerPriceForecast

Utility scripts for training power price forecasting models.

## Training Stages

Two CV stages can be executed via Hydra:

```
# Stage 1
python -m src.training.train_model cv_stage=stage1

# Stage 2
python -m src.training.train_model cv_stage=stage2
```

Stage 1 uses many short purged walk-forward folds for tuning and feature
selection. Stage 2 performs year-ahead evaluation.
