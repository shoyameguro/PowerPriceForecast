#!/usr/bin/env python
"""Optuna-based hyperparameter search with Hydra integration.

Run **inside a Hydra run dir** so that all artifacts
(`best_params.yaml`, `study.pkl`, `optuna.log`) are grouped per experiment.

Typical usage:

```bash
python -m src.tuning.tune_hyperparams \
    cfg=src/config/lgbm_baseline.yaml \
    input=data/train/train.pkl \
    trials=50
```

or rely on defaults in `conf/tune.yaml` and simply execute

```bash
python -m src.tuning.tune_hyperparams
```
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import hydra
import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.models import MODEL_REGISTRY
from src.utils.io import read_pickle
from src.training.train_model import drop_by_patterns

# ---------------------------------------------------------------------------
# Logging setup -------------------------------------------------------------
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyper‑param suggestion spaces ---------------------------------------------
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, model_name: str, base: dict) -> dict:  # noqa: C901 (complex)
    """Return parameter dict sampled for *model_name*."""
    params = base.copy()
    if model_name == "lgbm":
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 512),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
            }
        )
    elif model_name == "xgb":
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
            }
        )
    elif model_name == "nlinear":
        params.update(
            {
                "input_chunk_length": trial.suggest_int("input_chunk_length", 48, 168),
                "output_chunk_length": trial.suggest_int("output_chunk_length", 24, 72),
                "n_epochs": trial.suggest_int("n_epochs", 30, 100),
            }
        )
    return params


# ---------------------------------------------------------------------------
# Objective builder ---------------------------------------------------------
# ---------------------------------------------------------------------------

def make_objective(X, y, tss, model_name: str, base_params: dict):
    model_cls = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:  # noqa: WPS430 (nested func OK here)
        params = suggest_params(trial, model_name, base_params)
        params["early_stopping_rounds"] = base_params.get("early_stopping_rounds", 50)
        rmses = []
        for tr_idx, val_idx in tss.split(X):
            model = model_cls(params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], valid=(X.iloc[val_idx], y.iloc[val_idx]))
            preds = model.predict(X.iloc[val_idx])
            rmse = math.sqrt(mean_squared_error(y.iloc[val_idx], preds))
            rmses.append(rmse)
        score = float(np.mean(rmses))
        trial.set_user_attr("rmse", score)
        return score

    return objective


# ---------------------------------------------------------------------------
# Main worker ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def run(cfg_path: str, input_path: str, n_trials: int, out_dir: Path):
    logger.info("Config: %s", cfg_path)
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = read_pickle(input_path)

    logger.info("Data shape: %s", df.shape)

    target_col = cfg.get("target_col", "price_actual")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = drop_by_patterns(X, cfg.get("features_exclude", []))

    cv_conf = cfg.get("cv", {})
    tss_kwargs = {"n_splits": cv_conf.get("n_splits", 3)}
    if (test_hours := cv_conf.get("test_hours")):
        tss_kwargs["test_size"] = test_hours
    tss = TimeSeriesSplit(**tss_kwargs)

    model_name = cfg.get("model_name", "lgbm")
    base_params = cfg.get("params", {}).copy()

    objective = make_objective(X, y, tss, model_name, base_params)

    # Optuna study ---------------------------------------------------------
    study = optuna.create_study(direction="minimize", study_name=f"tune_{model_name}")
    logger.info("Starting Optuna: %d trials", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Optuna finished. Best RMSE = %.4f", study.best_value)

    # Save artifacts -------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    best_params = base_params.copy()
    best_params.update(study.best_params)
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.safe_dump(best_params, f)
    joblib.dump(study, out_dir / "study.pkl")

    logger.info("Artifacts saved to %s", out_dir.resolve())


# ---------------------------------------------------------------------------
# Hydra entry‑point ---------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../conf", config_name="tune", version_base=None)
def main(cfg: DictConfig):  # noqa: D401
    """Hydra injects *cfg*."""

    # *cfg.out* is already `${hydra:runtime.output_dir}/tuning` by default
    run(
        to_absolute_path(cfg.cfg),          # model YAML
        to_absolute_path(cfg.input),        # training data
        cfg.trials,                         # number of trials
        Path(cfg.out).resolve(),            # output dir inside Hydra run
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    main()
