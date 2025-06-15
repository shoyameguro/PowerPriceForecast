#!/usr/bin/env python
"""Improved feature‑engineering pipeline for Spain power‑price competition (v2).
Adds richer calendar, demand/generation, weather, and ratio features while
retaining CLI compatibility with previous version.

Usage examples
--------------
Train split::
    python -m src.features.build_features_v2 \
        --input data/raw/train.csv --split train

Test split::
    python -m src.features.build_features_v2 \
        --input data/raw/test.csv --split test --train data/raw/train.csv

Results are saved via ``save_data`` into ``data/<split>/<split>.pkl``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import pandas as pd
import holidays

from src.utils.io import save_data  # unchanged helper

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIG ---------------------------------------------------------
# ---------------------------------------------------------------------------
ES_HOL = holidays.country_holidays("ES")
TARGET = "price_actual"  # present only in train CSV

# lags (hours)
DEF_LAGS_SHORT = [1, 2]
DEF_LAGS_DAILY = [24]
DEF_LAGS_WEEKLY = [168]

ROLL_WINDOWS: dict[int, list[str]] = {
    24: ["total_load_actual"],    # daily
    168: ["total_load_actual"],   # weekly
}

NUM_DIFF_COLS = ["total_load_actual"]
GEN_PREFIX = "generation_"

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _to_local(ts: pd.Series) -> pd.Series:
    """Convert UTC timestamps to Europe/Madrid tz (returns Timestamp[ns, tz])."""
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert("Europe/Madrid")


def _cyclic_encode(series: pd.Series, period: int, prefix: str, df: pd.DataFrame):
    angle = 2 * np.pi * series / period
    df[f"{prefix}_sin"] = np.sin(angle)
    df[f"{prefix}_cos"] = np.cos(angle)

# ---------------------------------------------------------------------------
# FEATURE BUILDERS -----------------------------------------------------------
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts_local = _to_local(df["time"])

    df["hour"] = ts_local.dt.hour.astype(np.int8)
    df["dow"] = ts_local.dt.dayofweek.astype(np.int8)  # 0=Mon
    df["month"] = ts_local.dt.month.astype(np.int8)

    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["is_holiday"] = ts_local.dt.date.map(lambda d: int(d in ES_HOL)).astype(np.int8)

    _cyclic_encode(df["hour"], 24, "hour", df)
    _cyclic_encode(df["dow"], 7, "dow", df)
    _cyclic_encode(df["month"], 12, "month", df)

    # long‑term index (seconds since epoch)
    df["t_idx"] = (ts_local.view("int64") // 10 ** 9).astype(np.int64)

    # naive timestamp (no tz) for optional merges
    df["timestamp"] = ts_local.dt.tz_localize(None)
    return df


def select_numeric_candidates(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in (TARGET, "time"):
            continue
        if df[col].dtype.kind in "biufc" and not col.endswith("icon"):
            out.append(col)
    return out


def add_lag_features(df: pd.DataFrame, cols: Iterable[str], lags: Sequence[int]):
    for col in cols:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
    return df


def add_rolling_features(df: pd.DataFrame, windows: dict[int, Sequence[str]]):
    for win, cols in windows.items():
        for col in cols:
            s = df[col]
            df[f"{col}_roll_mean{win}"] = s.rolling(win).mean()
            df[f"{col}_roll_std{win}"] = s.rolling(win).std()
    return df


def add_diff_pct(df: pd.DataFrame, cols: Sequence[str]):
    for col in cols:
        df[f"{col}_diff1"] = df[col] - df[col].shift(1)
        df[f"{col}_pct1"] = df[col].pct_change(1)
        df[f"{col}_diff24"] = df[col] - df[col].shift(24)
        df[f"{col}_pct24"] = df[col].pct_change(24)
    return df


def add_generation_shares(df: pd.DataFrame):
    gen_cols = [c for c in df.columns if c.startswith(GEN_PREFIX)]
    if not gen_cols:
        return df

    df["generation_total"] = df[gen_cols].sum(axis=1)

    fossil_cols = [c for c in gen_cols if "fossil" in c]
    renewable_cols = [c for c in gen_cols if any(k in c for k in ["solar", "wind", "hydro", "biomass", "other_renewable"])]

    df["share_fossil"] = df[fossil_cols].sum(axis=1) / df["generation_total"]
    df["share_renewable"] = df[renewable_cols].sum(axis=1) / df["generation_total"]
    return df

# ---------------------------------------------------------------------------
# MAIN PIPELINE --------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, split: str, df_tr_tail: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construct features for *df* (raw). If *split=='test'*, prepend *df_tr_tail* to
    provide history for lag/rolling calculation, then discard before return.
    """
    if split == "test" and df_tr_tail is not None:
        df_tr_tail = df_tr_tail.copy()
        df_tr_tail["__origin"] = "train_tail"
        df["__origin"] = "test"
        df = pd.concat([df_tr_tail, df], ignore_index=True, sort=False)

    # Core transforms
    df = add_calendar_features(df)

    numeric_cols = select_numeric_candidates(df)

    lags = DEF_LAGS_SHORT + DEF_LAGS_DAILY + DEF_LAGS_WEEKLY
    df = add_lag_features(df, numeric_cols, lags)
    df = add_rolling_features(df, ROLL_WINDOWS)
    df = add_diff_pct(df, NUM_DIFF_COLS)
    df = add_generation_shares(df)

    # Remove rows with NaNs induced by shifting/rolling
    df = df.dropna().reset_index(drop=True)

    # Keep only test rows if we prepended history
    if "__origin" in df.columns:
        if split == "test":
            df = df[df["__origin"] == "test"].reset_index(drop=True)
        df = df.drop(columns=["__origin"], errors="ignore")

    # Ensure target is never present in test features to prevent accidental leakage
    if split == "test" and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df

# ---------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Spain power price feature builder v2 (no leakage)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--train", type=Path, help="Raw train CSV (provides tail rows for test lags)")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input)

    df_tr_tail = None
    if args.split == "test" and args.train is not None:
        df_train_raw = pd.read_csv(args.train)
        need = max(DEF_LAGS_WEEKLY) + max(ROLL_WINDOWS.keys())
        df_tr_tail = df_train_raw.tail(need)

    df_feat = build_features(df_raw, args.split, df_tr_tail)
    save_data(df_feat, args.split)


if __name__ == "__main__":
    cli()
