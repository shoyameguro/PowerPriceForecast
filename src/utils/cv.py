"""Cross-validation utilities."""

from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit

from .purged_walk import PurgedWalkForwardCV


def build_cv(conf: dict):
    """Create a CV splitter according to configuration."""
    name = conf.get("name", "timeseries").lower()
    n_splits = conf.get("n_splits", 5)

    if name == "purged_walk":
        return PurgedWalkForwardCV(
            n_splits=n_splits,
            test_size=conf.get("test_hours", conf.get("test_size", 0)),
            gap=conf.get("gap_hours", conf.get("gap", 0)),
        )
    elif name == "timeseries":
        kwargs = {"n_splits": n_splits}
        if conf.get("test_hours") is not None:
            kwargs["test_size"] = conf["test_hours"]
        return TimeSeriesSplit(**kwargs)
    else:
        raise ValueError(f"Unknown CV name: {name}")

__all__ = ["build_cv", "PurgedWalkForwardCV"]
