#!/usr/bin/env python
"""Train LightGBM/XGBoost with **recommended walk-forward CV** (7 folds).

Runs under ``hydra`` so each execution is stored in ``outputs/`` with a
timestamped directory and associated log file.

Fold design (hourly data)  ────────────────────────────────────────────────
F1 : 2015‑01‑02  → 2015‑09‑30  (train) | 2015‑10‑01 → 2015‑12‑31 (val)
F2 : 2015‑01‑02  → 2016‑03‑31  | 2016‑04‑01 → 2016‑06‑30
F3 : 2015‑01‑02  → 2016‑06‑30  | 2016‑07‑01 → 2016‑09‑30
F4 : 2015‑01‑02  → 2016‑09‑30  | 2016‑10‑01 → 2016‑12‑31
F5 : 2015‑01‑02  → 2017‑03‑31  | 2017‑04‑01 → 2017‑06‑30
F6 : 2015‑01‑02  → 2017‑06‑30  | 2017‑07‑01 → 2017‑09‑30
F7 : 2015‑01‑02  → 2017‑09‑30  | 2017‑10‑01 → 2017‑12‑31

A purge gap of **7 days** (168 h) is applied before each validation window.
"""
from __future__ import annotations

import yaml, joblib, math, re, logging
from pathlib import Path
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# mlfinlab provides PurgedWalkForwardCV
# try:
#     from mlfinlab.cross_validation import PurgedWalkForwardCV
# except ImportError:  # graceful fallback
#     raise ImportError("Please install mlfinlab: pip install mlfinlab")

from src.utils.io import read_pickle
from src.models import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_by_patterns(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    cols_to_drop: list[str] = []
    for pat in patterns:
        if pat.startswith("regex:"):
            regex = pat.split("regex:")[1]
            cols_to_drop.extend([c for c in df.columns if re.match(regex, c)])
        else:
            cols_to_drop.append(pat)
    return df.drop(columns=list(set(cols_to_drop)), errors="ignore")

# ---------------------------------------------------------------------------
# Walk‑forward split specification
# ---------------------------------------------------------------------------

def build_folds(df: pd.DataFrame, ts_col: str = "timestamp"):
    """Return list of (train_idx, val_idx) tuples following recommended scheme."""
    date_series = pd.to_datetime(df[ts_col])
    # boundaries
    cuts = [
        ("2015-10-01", "2015-12-31"),
        ("2016-04-01", "2016-06-30"),
        ("2016-07-01", "2016-09-30"),
        ("2016-10-01", "2016-12-31"),
        ("2017-04-01", "2017-06-30"),
        ("2017-07-01", "2017-09-30"),
        ("2017-10-01", "2017-12-31"),
    ]
    purge_gap = timedelta(days=7)

    folds = []
    for val_start_str, val_end_str in cuts:
        val_start = pd.Timestamp(val_start_str, tz=None)
        val_end   = pd.Timestamp(val_end_str, tz=None) + timedelta(hours=23)  # inclusive end day

        train_mask = date_series < (val_start - purge_gap)
        val_mask   = (date_series >= val_start) & (date_series <= val_end)

        tr_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((tr_idx, val_idx))
    return folds

# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

def train_fold(X_tr, y_tr, X_val, y_val, model_cls, params):
    model = model_cls(params)
    model.fit(X_tr, y_tr, valid=(X_val, y_val))
    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return model, preds, rmse


def run(cfg_path: str, input_path: str, model_dir: str):
    logger = logging.getLogger(__name__)

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = read_pickle(input_path)

    ts_col = "timestamp" if "timestamp" in df.columns else "time"

    target_col = cfg.get("target_col", "price_actual")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # column exclusions
    X = drop_by_patterns(X, cfg.get("features_exclude", []))

    # assemble folds
    folds = build_folds(df, ts_col)

    models_path = Path(model_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_name = cfg.get("model_name", "lgbm")
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_params = cfg["params"].copy()
    model_params["early_stopping_rounds"] = cfg.get("cv", {}).get("early_stopping_rounds", 100)

    oof = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
        model, preds, rmse_fold = train_fold(
            X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[val_idx], y.iloc[val_idx], model_cls, model_params
        )
        oof[val_idx] = preds
        model.save(models_path / f"{model_name}_fold{fold}.pkl")
        logger.info(
            "Fold %d: %6d train | %5d val rows -> RMSE %.4f",
            fold,
            len(tr_idx),
            len(val_idx),
            rmse_fold,
        )

    rmse_oof = math.sqrt(mean_squared_error(y, oof))
    logger.info("Overall OOF RMSE: %.4f", rmse_oof)

    # full‑data model
    final_model = model_cls(model_params)
    final_model.fit(X, y, valid=(X, y))
    final_model.save(models_path / "model_full.pkl")
    joblib.dump(cfg, models_path / "train_config.pkl")
    logger.info("Training complete. Models saved to %s", models_path)

# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(
        to_absolute_path(cfg.cfg),
        to_absolute_path(cfg.input),
        cfg.model_dir,
    )

if __name__ == "__main__":
    main()
