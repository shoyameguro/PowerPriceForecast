#!/usr/bin/env python
"""Inference script with Hydra support.

This script loads trained models from a directory and generates
predictions for the test data. It now runs under :mod:`hydra` so each
execution is stored in ``outputs/`` with a timestamped directory and the
model directory can be specified via a config file or command line
override.

Example::

    python -m src.interface.predict \
        models=outputs/2023-10-05/15-00-00/models \
        out=outputs/submissions/submission.csv
"""
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


from src.utils.io import read_data, read_pickle
from src.models import MODEL_REGISTRY, LGBMWrapper


def load_models(path: Path):
    """Load all CV fold models from a directory regardless of type."""
    models = []
    for p in sorted(path.glob("*_fold*.pkl")):
        meta = joblib.load(p)
        model_type = meta.get("model_type", "lgbm")
        cls = MODEL_REGISTRY.get(model_type, LGBMWrapper)
        wrapper = cls(params={})
        wrapper.load(p)
        models.append(wrapper)
    return models


def run(models: str, input_path: str | None, out_path: str) -> None:
    """Generate predictions from trained models."""

    # 1) データ読み込み
    if input_path is None:
        df = read_data("test")
    else:
        if str(input_path).endswith(".pkl"):
            df = read_pickle(input_path)
        else:
            df = pd.read_feather(input_path)

    # 2) モデル側メタデータを先に読み出す
    model0 = load_models(Path(models))[0]  # 1個取れば列情報は分かる
    feature_names = model0.feature_names

    # 3) 列をモデル順に並べ替え・欠損列はゼロ埋め
    X = df.reindex(columns=feature_names, copy=True)
    missing = X.columns[X.isnull().all()]
    if len(missing):
        X[missing] = 0.0

    # 4) すべてのモデルで平均
    model_list = [model0] + load_models(Path(models))[1:]
    preds = np.mean([m.predict(X) for m in model_list], axis=0)

    # 5) 提出ファイル
    sub = pd.DataFrame({"time": df["time"], "price_actual": preds})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, header=False)
    print("Saved submission to", out_path)

@hydra.main(config_path="../../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    run(
        to_absolute_path(cfg.models),
        to_absolute_path(cfg.input) if cfg.input else None,
        to_absolute_path(cfg.out),
    )


if __name__ == "__main__":
    main()

