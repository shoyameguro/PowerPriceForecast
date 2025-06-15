"""Optuna-based hyperparameter search for ML models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .train_model import drop_by_patterns
from ..models import MODEL_REGISTRY
from ..utils.io import read_pickle


# ---------------------------------------------------------------------------
# Hyperparameter suggestion spaces
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, model_name: str, base: dict) -> dict:
    """Return parameter dict sampled for *model_name*."""
    params = base.copy()
    if model_name == "lgbm":
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        params["num_leaves"] = trial.suggest_int("num_leaves", 31, 512)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 1.0)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 1.0)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 20, 100)
        params["lambda_l1"] = trial.suggest_float("lambda_l1", 0.0, 1.0)
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 0.0, 1.0)
    elif model_name == "xgb":
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 4, 12)
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["n_estimators"] = trial.suggest_int("n_estimators", 200, 1500)
        params["lambda_l1"] = trial.suggest_float("lambda_l1", 0.0, 1.0)
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 0.0, 1.0)
    elif model_name == "nlinear":
        params["input_chunk_length"] = trial.suggest_int("input_chunk_length", 48, 168)
        params["output_chunk_length"] = trial.suggest_int("output_chunk_length", 24, 72)
        params["n_epochs"] = trial.suggest_int("n_epochs", 30, 100)
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(X, y, tss, model_name: str, base_params: dict):
    model_cls = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, base_params)
        params["early_stopping_rounds"] = base_params.get("early_stopping_rounds", 50)
        rmses = []
        for tr_idx, val_idx in tss.split(X):
            model = model_cls(params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], valid=(X.iloc[val_idx], y.iloc[val_idx]))
            preds = model.predict(X.iloc[val_idx])
            rmse = math.sqrt(mean_squared_error(y.iloc[val_idx], preds))
            rmses.append(rmse)
        return float(np.mean(rmses))

    return objective


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def run(cfg_path: str, input_path: str, n_trials: int, out_dir: Path) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = read_pickle(input_path)

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
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_params = base_params.copy()
    best_params.update(study.best_params)
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.safe_dump(best_params, f)
    joblib.dump(study, out_dir / "study.pkl")
    print("Best RMSE", study.best_value)
    print("Best params saved to", out_dir / "best_params.yaml")


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    ap.add_argument("--cfg", required=True, help="Path to model config YAML")
    ap.add_argument("--input", required=True, help="Training features pickle")
    ap.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    ap.add_argument("--out", type=Path, default=Path("tuning_results"))
    args = ap.parse_args()

    run(args.cfg, args.input, args.trials, args.out)


if __name__ == "__main__":
    main()
