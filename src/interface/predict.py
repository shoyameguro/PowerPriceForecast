#!/usr/bin/env python
"""Inference script.
Example:
  python -m src.interface.predict --models output/models --out output/submissions/submission.csv
"""
import argparse, joblib, glob
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import read_data, read_pickle
from src.models.lgbm_model import LGBMWrapper


def load_models(path: Path):
    """Load all LightGBM wrapper models from the given directory."""
    models: list[LGBMWrapper] = []
    for p in glob.glob(str(path / "lgbm_fold*.pkl")):
        wrapper = LGBMWrapper(params={})
        wrapper.load(Path(p))
        models.append(wrapper)
    return models


def main(args):
    # 1) データ読み込み
    if args.input is None:
        df = read_data("test")
    else:
        # autodetect pickle vs feather
        if str(args.input).endswith(".pkl"):
            df = read_pickle(args.input)
        else:
            df = pd.read_feather(args.input)

    # 2) モデル側メタデータを先に読み出す
    model0 = load_models(Path(args.models))[0]      # 1個取れば列情報は分かる
    feature_names = model0.feature_names

    # 3) 列をモデル順に並べ替え・欠損列はゼロ埋め
    X = df.reindex(columns=feature_names, copy=True)
    missing = X.columns[X.isnull().all()]
    if len(missing):
        X[missing] = 0.0

    # 4) すべてのモデルで平均
    models = [model0] + load_models(Path(args.models))[1:]
    preds = np.mean([m.predict(X) for m in models], axis=0)

    # 5) 提出ファイル
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