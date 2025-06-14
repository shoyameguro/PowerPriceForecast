#!/usr/bin/env python
"""Train pipeline.
Example:
  python -m src.training.train_model --config src/config/lgbm_baseline.yaml
"""
import argparse, yaml, joblib, math
from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
from src.utils.io import read_data
from src.models.lgbm_model import LGBMWrapper


def main(cfg):
    df = read_data("train")  # load feature-engineered training set
    y = df[cfg["target_col"]]
    X = df.drop(columns=cfg["features_exclude"] + [cfg["target_col"]])

    tss = TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"])
    oof = np.zeros(len(df))
    models_dir = Path("output/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for i, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = LGBMWrapper(cfg["params"] | {"early_stopping_rounds": cfg["cv"]["early_stopping_rounds"]})
        model.fit(X_tr, y_tr, valid=(X_val, y_val))
        oof[val_idx] = model.predict(X_val)
        model.save(models_dir / f"lgbm_fold{i}.pkl")
        print(f"Fold {i}: RMSE =", math.sqrt(mean_squared_error(y_val, oof[val_idx])))

    print("OOF RMSE:", math.sqrt(mean_squared_error(y, oof)))
    joblib.dump(cfg, models_dir / "train_config.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)