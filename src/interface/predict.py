#!/usr/bin/env python
"""Inference script with Hydra support.

This version automatically resolves the *models* directory inside a
Hydra training run. 使い方は 2 通りあります。

1. **明示的に指定**  
   `python -m src.interface.predict models=outputs/2025-06-15/23-40-12/models`

2. **最新 run を自動検出**  
   `python -m src.interface.predict models=latest`

`models=latest` とすると ``outputs/`` 配下でもっとも *mtime* が新しい
`<date>/<time>/models/` を探索して読み込みます。
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from src.models import LGBMWrapper, MODEL_REGISTRY
from src.utils.io import read_data, read_pickle

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_candidates(base: Path) -> list[Path]:
    """Return all `<date>/<time>/models` dirs under *base* (depth=2)."""
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    candidates: list[Path] = []
    for day_dir in base.iterdir():
        if not (day_dir.is_dir() and pattern.match(day_dir.name)):
            continue
        for time_dir in day_dir.iterdir():
            mdir = time_dir / "models"
            if mdir.is_dir():
                candidates.append(mdir)
    return candidates


def _find_latest_models(base_dir: str | Path = "outputs") -> Path | None:
    base = Path(base_dir)
    candidates = _list_candidates(base)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_models_path(arg: str | None) -> Path:
    """Convert *arg* to `Path`.

    * ``None`` or empty   -> try latest under ``outputs/``
    * ``latest``          -> same as above
    * other string        -> absolute via `to_absolute_path` (keeps absolute)
    """
    if not arg or arg == "latest":
        latest = _find_latest_models()
        if latest is None:
            _LOG.error("No models directory found under outputs/. Abort.")
            sys.exit(1)
        _LOG.info("Using latest models at %s", latest)
        return latest.resolve()

    path = Path(to_absolute_path(arg))
    if not path.exists():
        _LOG.error("models path %s does not exist", path)
        sys.exit(1)
    return path.resolve()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_models(path: Path):
    """Load all CV‑fold models from *path*."""
    models = []
    for p in sorted(path.glob("*_fold*.pkl")):
        meta = joblib.load(p)
        model_type = meta.get("model_type", "lgbm")
        cls = MODEL_REGISTRY.get(model_type, LGBMWrapper)
        wrapper = cls(params={})
        wrapper.load(p)
        models.append(wrapper)
    if not models:
        raise FileNotFoundError(f"No *_fold*.pkl found in {path}")
    return models


def run(models: Path, input_path: str | None, out_path: Path) -> None:
    """Generate predictions from trained models."""

    # 1) Load test data ----------------------------------------------------
    if input_path is None:
        df = read_data("test")
    else:
        if str(input_path).endswith(".pkl"):
            df = read_pickle(input_path)
        else:
            df = pd.read_feather(input_path)

    # 2) Model metadata ----------------------------------------------------
    model_list = load_models(models)
    feature_names = model_list[0].feature_names

    # 3) Align columns -----------------------------------------------------
    X = df.reindex(columns=feature_names, copy=True)
    missing = X.columns[X.isnull().all()]
    if len(missing):
        X[missing] = 0.0

    # 4) Predict -----------------------------------------------------------
    preds = np.mean([m.predict(X) for m in model_list], axis=0)

    # 5) Save submission ---------------------------------------------------
    sub = pd.DataFrame({"time": df["time"], "price_actual": preds})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, header=False)
    _LOG.info("Saved submission to %s", out_path)


# ---------------------------------------------------------------------------
# Hydra entry‑point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Hydra injects *cfg* here."""

    models_path = _resolve_models_path(cfg.models)

    input_abs = (
        to_absolute_path(cfg.input) if cfg.input else None
    )  # test data may be inside repo

    # *out* は Hydra run dir 内の相対パスで十分なので絶対化しない
    out_path = Path(cfg.out)

    run(models_path, input_abs, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
