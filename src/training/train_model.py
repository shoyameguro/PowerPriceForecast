#!/usr/bin/env python
"""Train ML model (LightGBM or XGBoost) with time-series CV.

This script uses **Hydra** for configuration. Each run is executed in a
timestamped directory under ``outputs/`` (see ``hydra.run.dir`` in your
YAML), so all artifacts remain neatly grouped per experiment.

Example::

    python -m src.training.train_model

The default configuration is ``conf/config.yaml``. Override any parameter
on the CLI, e.g. ``model_dir=my_models``.
"""

import logging
import math
import re
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error

from src.utils.cv import build_cv

from src.models import MODEL_REGISTRY
from src.utils.io import read_pickle

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
# Core
# ---------------------------------------------------------------------------

def train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_cls,
    params: dict,
):
    """Train one fold and return fitted model plus validation preds."""
    model = model_cls(params)
    model.fit(X_tr, y_tr, valid=(X_val, y_val))
    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return model, preds, rmse


def run(cfg_path: str, input_path: str, model_dir: Path, cv_stage: str, cv_override: dict | None = None):
    logger = logging.getLogger(__name__)

    # 1. Load config & data -------------------------------------------------
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if cv_override:
        cfg["cv"] = {**cfg.get("cv", {}), **cv_override}
    cfg["cv_stage"] = cv_stage
    df = read_pickle(input_path)

    # 2. Target & feature separation --------------------------------------
    target_col = cfg.get("target_col", "price_actual")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Column exclusions --------------------------------------------------
    X = drop_by_patterns(X, cfg.get("features_exclude", []))

    # 4. CV setup -----------------------------------------------------------
    tss = build_cv(cfg["cv"])

    # 5. Train folds --------------------------------------------------------
    oof = np.zeros(len(df))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.get("model_name", "lgbm")
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_params = cfg["params"].copy()
    model_params["early_stopping_rounds"] = cfg["cv"].get("early_stopping_rounds", 100)

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        if len(tr_idx) == 0:
            logger.warning("Skipping fold %d because training set is empty", fold)
            continue

        model, preds, rmse_fold = train_fold(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
            model_cls,
            model_params,
        )
        oof[val_idx] = preds
        model.save(model_dir / f"{model_name}_fold{fold}.pkl")
        logger.info("fold_%d_rmse = %.4f", fold, rmse_fold)

    # 6. Overall CV score ---------------------------------------------------
    rmse_oof = math.sqrt(mean_squared_error(y, oof))
    logger.info("OOF RMSE: %.4f", rmse_oof)

    # 7. Retrain on full data ----------------------------------------------
    final_model = model_cls(model_params)
    final_model.fit(X, y, valid=(X, y))
    final_model.save(model_dir / "model.pkl")
    joblib.dump(cfg, model_dir / "train_config.pkl")
    logger.info("Training complete. Artifacts saved to %s", model_dir)


# ---------------------------------------------------------------------------
# Hydra entry‑point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Hydra wraps this function, injecting the parsed config."""
    run(
        to_absolute_path(cfg.cfg),          # original YAML
        to_absolute_path(cfg.input),        # training data
        Path(cfg.model_dir).resolve(),      # already mapped inside outputs/
        cfg.get("cv_stage", "stage1"),
        cfg.get("cv", {}),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
