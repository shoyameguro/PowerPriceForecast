"""Cross-validation utilities."""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple
from sklearn.model_selection import TimeSeriesSplit


def build_cv(cfg: dict):
    """Return a CV splitter instance from a config dictionary."""
    cv_name = cfg.get("name", "timeseries")
    if cv_name == "purged_walk":
        return PurgedWalkForwardCV(
            n_splits=cfg.get("n_splits", 3),
            test_size=cfg.get("test_hours", cfg.get("test_size", 0)),
            gap=cfg.get("gap_hours", cfg.get("gap", 0)),
        )
    if cv_name == "timeseries":
        kwargs = {"n_splits": cfg.get("n_splits", 3)}
        if (ts := cfg.get("test_hours")) is not None:
            kwargs["test_size"] = ts
        return TimeSeriesSplit(**kwargs)
    raise ValueError(f"Unknown cv.name: {cv_name}")


class PurgedWalkForwardCV:
    """Generator for walk-forward splits with a purge gap."""

    def __init__(self, n_splits: int, test_size: int, gap: int) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D401
        """Return number of CV splits."""
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs."""
        n_samples = len(X)
        step = self.test_size + self.gap
        for i in range(self.n_splits):
            train_end = i * step
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            if test_end > n_samples:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
