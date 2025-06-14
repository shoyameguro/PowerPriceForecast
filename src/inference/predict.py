#!/usr/bin/env python
"""Batch inference script."""
import argparse, joblib, glob
from pathlib import Path
import pandas as pd, numpy as np
from src.utils.io import read_data

def load_models(path: Path):
    return [joblib.load(p) for p in sorted(path.glob("lgbm_fold*.pkl"))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--split", choices=["test"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = read_data(args.split)
    X = df.drop(columns=["time", "timestamp", "price_actual"], errors="ignore")
    models = load_models(args.models)
    preds = np.mean([m.predict(X) for m in models], axis=0)

    sub = pd.DataFrame({"time": df["time"], "price_actual": preds})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("submission saved to", args.out)

if __name__ == "__main__":
    main()
