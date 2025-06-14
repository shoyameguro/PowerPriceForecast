#!/usr/bin/env python
"""Weighted average ensemble of multiple submission files."""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main(args):
    files = args.files
    weights = args.weights or [1/len(files)]*len(files)
    assert len(files) == len(weights)
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0][["time"]].copy()
    preds = np.zeros(len(base))
    for w, df in zip(weights, dfs):
        preds += w * df["price_actual"].values
    base["price_actual"] = preds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print("Ensemble saved to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args())