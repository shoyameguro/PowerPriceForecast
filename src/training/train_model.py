#!/usr/bin/env python
"""Train LightGBM model with time‑series CV.
Usage:
  python -m src.training.train_model \
      --cfg   src/config/lgbm_baseline.yaml \
      --input data/train/train.pkl \
      --model_dir output/models
"""
import argparse, yaml, joblib, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.utils.io import read_pickle
from src.models.lgbm_model import LGBMWrapper

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

def train_fold(X_tr, y_tr, X_val, y_val, lgb_params):
    """Train one fold and return fitted model plus validation preds."""
    model = LGBMWrapper(lgb_params)
    model.fit(X_tr, y_tr, valid=(X_val, y_val))
    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return model, preds, rmse


def main(cfg_path: str, input_path: str, model_dir: str):
    # 1. config & data ------------------------------------------------------
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df  = read_pickle(input_path)

    # 2. column exclusions --------------------------------------------------
    df = drop_by_patterns(df, cfg.get("features_exclude", []))
    target_col = cfg.get("target_col", "price_actual")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. CV setup -----------------------------------------------------------
    n_splits = cfg.get("cv", {}).get("n_splits", 3)
    test_hours = cfg.get("cv", {}).get("test_hours", None)
    tss_kwargs = {"n_splits": n_splits}
    if test_hours:
        # translate hours to sample count (df is hourly)
        tss_kwargs["test_size"] = test_hours
    tss = TimeSeriesSplit(**tss_kwargs)

    # 4. train folds --------------------------------------------------------
    oof = np.zeros(len(df))
    models_path = Path(model_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    lgb_params = cfg["params"].copy()
    lgb_params["early_stopping_rounds"] = cfg.get("cv", {}).get("early_stopping_rounds", 100)

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        model, preds, rmse_fold = train_fold(
            X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[val_idx], y.iloc[val_idx], lgb_params
        )
        oof[val_idx] = preds
        model.save(models_path / f"lgbm_fold{fold}.pkl")
        print(f"Fold {fold}: RMSE = {rmse_fold:.4f}")

    # 5. overall CV score ---------------------------------------------------
    rmse_oof = math.sqrt(mean_squared_error(y, oof))
    print(f"OOF RMSE: {rmse_oof:.4f}")

    # 6. retrain on full data ----------------------------------------------
    final_model = LGBMWrapper(lgb_params)
    final_model.fit(X, y, valid=(X, y))
    final_model.save(models_path / "model.pkl")
    joblib.dump(cfg, models_path / "train_config.pkl")
    print("Training complete. Models saved to", models_path)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    args = parser.parse_args()

    main(str(args.cfg), str(args.input), str(args.model_dir))