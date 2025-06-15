#!/usr/bin/env python
"""Train ML model (LightGBM or XGBoost) with time-series CV.

This script now uses ``hydra`` for configuration. Each run is executed in a
timestamped directory under ``outputs/`` and logs are stored automatically.

Example::

  python -m src.training.train_model cfg=src/config/lgbm_baseline.yaml
"""

import yaml, joblib, math, re, logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.utils.io import read_pickle
from src.models import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def drop_by_patterns(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    """Drop columns whose names match any literal or regex pattern."""
    cols_to_drop: list[str] = []
    for pat in patterns:
        if pat.startswith("regex:"):
            regex = pat.split("regex:")[1]
            cols_to_drop.extend([c for c in df.columns if re.match(regex, c)])
        else:
            cols_to_drop.append(pat)
    return df.drop(columns=list(set(cols_to_drop)), errors="ignore")

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def train_fold(X_tr, y_tr, X_val, y_val, model_cls, params):
    """Train one fold and return fitted model plus validation preds."""
    model = model_cls(params)
    model.fit(X_tr, y_tr, valid=(X_val, y_val))
    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return model, preds, rmse


def run(cfg_path: str, input_path: str, model_dir: str):
    logger = logging.getLogger(__name__)

    # 1. config & data ------------------------------------------------------
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df  = read_pickle(input_path)

    # 2. target & feature separation ---------------------------------------
    target_col = cfg.get("target_col", "price_actual")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. column exclusions --------------------------------------------------
    X = drop_by_patterns(X, cfg.get("features_exclude", []))

    # 4. CV setup -----------------------------------------------------------
    n_splits = cfg.get("cv", {}).get("n_splits", 3)
    test_hours = cfg.get("cv", {}).get("test_hours", None)
    tss_kwargs = {"n_splits": n_splits}
    if test_hours:
        # translate hours to sample count (df is hourly)
        tss_kwargs["test_size"] = test_hours
    tss = TimeSeriesSplit(**tss_kwargs)

    # 5. train folds --------------------------------------------------------
    oof = np.zeros(len(df))
    models_path = Path(model_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_name = cfg.get("model_name", "lgbm")
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_params = cfg["params"].copy()
    model_params["early_stopping_rounds"] = cfg.get("cv", {}).get("early_stopping_rounds", 100)

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        model, preds, rmse_fold = train_fold(
            X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[val_idx], y.iloc[val_idx], model_cls, model_params
        )
        oof[val_idx] = preds
        model.save(models_path / f"{model_name}_fold{fold}.pkl")
        logger.info("Fold %d: RMSE = %.4f", fold, rmse_fold)

    # 6. overall CV score ---------------------------------------------------
    rmse_oof = math.sqrt(mean_squared_error(y, oof))
    logger.info("OOF RMSE: %.4f", rmse_oof)

    # 7. retrain on full data ----------------------------------------------
    final_model = model_cls(model_params)
    final_model.fit(X, y, valid=(X, y))
    final_model.save(models_path / "model.pkl")
    joblib.dump(cfg, models_path / "train_config.pkl")
    logger.info("Training complete. Models saved to %s", models_path)

# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for Hydra."""
    run(
        to_absolute_path(cfg.cfg),
        to_absolute_path(cfg.input),
        cfg.model_dir,
    )

if __name__ == "__main__":
    main()

