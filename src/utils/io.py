from pathlib import Path
import pandas as pd
import pickle


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV with automatic datetime parse for column 'time'."""
    return pd.read_csv(path, parse_dates=["time"])


# -------------------- pickle convenience -----------------------------------

def save_pickle(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------- project‑specific data API -----------------------------

def save_data(df: pd.DataFrame, split: str):
    """Save processed DataFrame to project location as **pickle**.

    Args:
        df    : pandas DataFrame to persist.
        split : "train" or "test" – used to build the path
                 data/<split>/<split>.pkl
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = Path(f"data/{split}/{split}.pkl")
    save_pickle(df, path)


def read_data(split: str) -> pd.DataFrame:
    """Load processed features for the given split (train / test)."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = Path(f"data/{split}/{split}.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return read_pickle(path)
