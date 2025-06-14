from pathlib import Path
import pandas as pd

def read_data(split: str, root: Path = Path("data")) -> pd.DataFrame:
    """split ∈ {raw, interim, processed}"""
    return pd.read_feather(root / split / f"{split}.feather")

def save_data(df: pd.DataFrame, split: str, root: Path = Path("data")) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    df.to_feather(root / split / f"{split}.feather")