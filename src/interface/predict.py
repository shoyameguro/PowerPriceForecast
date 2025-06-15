#!/usr/bin/env python
"""Inference script.
Example:
  python -m src.interface.predict --models output/models --out output/submissions/submission.csv
"""
import argparse, joblib, glob
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import read_data


def load_models(path: Path):
    models = []
    for p in glob.glob(str(path / "lgbm_fold*.pkl")):
        models.append(joblib.load(p))
    return models


def main(args):
    df = read_data("test") if args.input is None else pd.read_feather(args.input)
    cfg = joblib.load(Path(args.models) / "train_config.pkl")
    X = df.drop(columns=cfg["features_exclude"] + [cfg["target_col"]])
    models = load_models(Path(args.models))
    preds = np.mean([m.predict(X) for m in models], axis=0)
    sub = pd.DataFrame({"time": df["time"], "price_actual": preds})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("Saved submission to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())