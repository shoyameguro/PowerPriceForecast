#!/usr/bin/env python
"""Feature engineering pipeline.
Usage:  python -m src.features.build_features --input data/raw/train.csv --split train
Results are saved to ``data/<split>/<split>.pkl`` via :func:`save_data`.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

ES_HOL = holidays.country_holidays("ES")

DEF_LAGS = [1, 2, 24, 168]  # hours

# lag/rolling を取る候補列は run-time に決定 (train は price_actual を含むが test には無い)
NUM_COLS_BASE = ["total_load_actual"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar‑based features. Handles CET/CEST with mixed offsets safely."""
    # --- 1. robust timestamp parsing ---
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    # Convert to Spain local time (CET/CEST)
    ts = ts.dt.tz_convert("Europe/Madrid")

    # --- 2. calendar fields ---
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = ts.dt.date.map(lambda d: int(d in ES_HOL))

    # cyclic encoding (hour)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- 3. store cleaned timestamp (naive) for potential downstream joins ---
    df["timestamp"] = ts.dt.tz_localize(None)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    cols = [c for c in NUM_COLS_BASE if c in df.columns]
    for col in cols:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
    return df


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUM_COLS_BASE if c in df.columns]
    for col in cols:
        df[f"{col}_roll_mean24"] = df[col].rolling(24).mean()
        df[f"{col}_roll_std24"] = df[col].rolling(24).std()
    return df


def main(args):
    df = pd.read_csv(args.input)

    # If we are preparing test features and a training file is provided,
    # prepend the last week of training data so that lag/rolling features
    # can be computed without losing the first test rows.
    if args.split == "test" and args.train is not None:
        df_tr = pd.read_csv(args.train)
        window = max(DEF_LAGS)
        df_tr = df_tr.tail(window)
        df_tr["__split"] = "train"
        df["__split"] = "test"
        df = pd.concat([df_tr, df], ignore_index=True)
        # remove target so NaNs in test portion don't wipe rows
        if "price_actual" in df.columns:
            df = df.drop(columns=["price_actual"])
    else:
        df["__split"] = args.split

    df = add_time_features(df)
    df = add_lag_features(df, DEF_LAGS)
    df = add_rolling(df)

    # drop rows with NaNs created by lagging/rolling
    df = df.dropna().reset_index(drop=True)

    # if extra training rows were prepended, keep only test part
    if "__split" in df.columns and args.split == "test":
        df = df[df["__split"] == "test"].reset_index(drop=True)
        df = df.drop(columns=["__split"])
    elif "__split" in df.columns:
        df = df.drop(columns=["__split"])

    from src.utils.io import save_data
    save_data(df, args.split)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--split", type=str, choices=["train", "test"], required=True)
    p.add_argument("--train", type=Path, help="Path to raw training CSV used for computing initial lags in test mode")
    args = p.parse_args()
    main(args)