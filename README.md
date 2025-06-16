# PowerPriceForecast

Utility scripts for training power price forecasting models.

## Training Stages

Two CV stages can be executed via Hydra. Convenience scripts are provided:

```
# Stage 1 (hyperparameter tuning)
scripts/train_stage1.sh

# Stage 2 (year-ahead evaluation)
scripts/train_stage2.sh
```

Stage 1 uses many short purged walk-forward folds for tuning and feature
selection. Stage 2 performs year-ahead evaluation.

### Feature selection

The Optuna-based selector can be enabled by overriding the `feature_select` group:

```bash
python -m src.training.tune_hyperparams feature_select=optuna
```
The selected feature names are saved to `selected_features.txt` inside the run directory.
