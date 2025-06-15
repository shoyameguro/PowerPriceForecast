#!/usr/bin/env python
"""Feature‑engineering pipeline v3 – Spain power price.

Key improvements over v2
------------------------
* Outlier handling for pressure / wind_speed (clip + NaN)
* Forward/backward fill of sporadic NaN (<1 %)
* Constant‑column removal
* Added physical interaction features:
  - East‑West pressure gradient (Valencia − Madrid)
  - Beaufort category for each wind‑speed column
  - Net load (total_load − total renewable generation)
  - Renewable ramp (Δ share_renewable over 2 h)
* Same CLI interface and leakage‑safe test handling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import pandas as pd
import holidays

from src.utils.io import save_data  # project helper unchanged

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIG ---------------------------------------------------------
# ---------------------------------------------------------------------------
ES_HOL = holidays.country_holidays("ES")
TARGET = "price_actual"

# lag config (hours)
DEF_LAGS_SHORT = [1, 2]
DEF_LAGS_DAILY = [24]
DEF_LAGS_WEEKLY = [168]

ROLL_WINDOWS: dict[int, list[str]] = {
    24: ["total_load_actual"],
    168: ["total_load_actual"],
}

NUM_DIFF_COLS = ["total_load_actual"]
GEN_PREFIX = "generation_"

# Wind speed Beaufort bins (m/s)
_BEAUFORT_BINS = [0, 2, 5, 8, 11, 14, 30]
_BEAUFORT_LABELS = list(range(len(_BEAUFORT_BINS) - 1))

# ---------------------------------------------------------------------------
# BASIC HELPERS --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _to_local(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert("Europe/Madrid")


def _cyclic_encode(series: pd.Series, period: int, prefix: str, df: pd.DataFrame):
    angle = 2 * np.pi * series / period
    df[f"{prefix}_sin"] = np.sin(angle)
    df[f"{prefix}_cos"] = np.cos(angle)

# ---------------------------------------------------------------------------
# OUTLIER & NAN HANDLING -----------------------------------------------------
# ---------------------------------------------------------------------------

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Pressure: plausible 960‑1040 hPa
    for c in [col for col in df.columns if col.endswith("_pressure")]:
        df[c] = df[c].clip(lower=960, upper=1040)

    # Wind speed: >30 m/s set to NaN (hurricane‑class)
    for c in [col for col in df.columns if col.endswith("_wind_speed")]:
        df.loc[df[c] > 30, c] = np.nan
    return df


def fill_sparse_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by time to preserve chronology
    df = df.sort_values("time")
    return df.fillna(method="ffill").fillna(method="bfill")

# ---------------------------------------------------------------------------
# CALENDAR FEATURES ----------------------------------------------------------
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts_local = _to_local(df["time"])

    df["hour"] = ts_local.dt.hour.astype(np.int8)
    df["dow"] = ts_local.dt.dayofweek.astype(np.int8)
    df["month"] = ts_local.dt.month.astype(np.int8)

    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["is_holiday"] = ts_local.dt.date.map(lambda d: int(d in ES_HOL)).astype(np.int8)

    _cyclic_encode(df["hour"], 24, "hour", df)
    _cyclic_encode(df["dow"], 7, "dow", df)
    _cyclic_encode(df["month"], 12, "month", df)

    df["t_idx"] = (ts_local.view("int64") // 10 ** 9).astype(np.int64)
    df["timestamp"] = ts_local.dt.tz_localize(None)
    return df

# ---------------------------------------------------------------------------
# DOMAIN‑SPECIFIC FEATURES ---------------------------------------------------
# ---------------------------------------------------------------------------

def add_generation_shares(df: pd.DataFrame) -> pd.DataFrame:
    gen_cols = [c for c in df.columns if c.startswith(GEN_PREFIX)]
    if not gen_cols:
        return df

    df["generation_total"] = df[gen_cols].sum(axis=1)

    fossil_cols = [c for c in gen_cols if "fossil" in c]
    renewable_cols = [
        c for c in gen_cols if any(k in c for k in [
            "solar", "wind", "hydro", "biomass", "other_renewable"])]

    df["share_fossil"] = df[fossil_cols].sum(axis=1) / df["generation_total"]
    df["share_renewable"] = df[renewable_cols].sum(axis=1) / df["generation_total"]

    # generation_renewable absolute (必要: net load)
    df["generation_renewable"] = df[renewable_cols].sum(axis=1)
    return df


def add_pressure_gradient(df: pd.DataFrame) -> pd.DataFrame:
    if {"valencia_pressure", "madrid_pressure"}.issubset(df.columns):
        df["pressure_grad_valencia_madrid"] = df["valencia_pressure"] - df["madrid_pressure"]
    return df


def add_wind_beaufort(df: pd.DataFrame) -> pd.DataFrame:
    for c in [col for col in df.columns if col.endswith("_wind_speed")]:
        cat = pd.cut(df[c], bins=_BEAUFORT_BINS, labels=_BEAUFORT_LABELS, right=False)
        df[f"{c}_beaufort"] = cat.astype(pd.Int8Dtype())
    return df


def add_load_net(df: pd.DataFrame) -> pd.DataFrame:
    if {"total_load_actual", "generation_renewable"}.issubset(df.columns):
        df["load_net"] = df["total_load_actual"] - df["generation_renewable"]
    return df

# ---------------------------------------------------------------------------
# LAG / ROLLING / DIFFERENCE -------------------------------------------------
# ---------------------------------------------------------------------------

def select_numeric_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TARGET, "time") and df[c].dtype.kind in "biufc" and not c.endswith("icon")]


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


def add_renewable_ramp(df: pd.DataFrame):
    if "share_renewable" in df.columns and "share_renewable_lag2" in df.columns:
        df["renewable_ramp"] = df["share_renewable"] - df["share_renewable_lag2"]
    return df

# ---------------------------------------------------------------------------
# MAIN PIPELINE --------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, split: str, df_tr_tail: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate features for *df* (raw). If *split == 'test'*, *df_tr_tail* provides
    history rows for lag/rolling calculation and is discarded before return.
    """
    if split == "test" and df_tr_tail is not None:
        df_tr_tail = df_tr_tail.copy()
        df_tr_tail["__origin"] = "train_tail"
        df["__origin"] = "test"
        df = pd.concat([df_tr_tail, df], ignore_index=True, sort=False)

    # ---------------- Outlier & NaN handling ----------------
    df = clip_outliers(df)
    df = fill_sparse_nan(df)  # keeps chronology

    # ---------------- Core feature blocks -------------------
    df = add_calendar_features(df)
    df = add_generation_shares(df)
    df = add_pressure_gradient(df)
    df = add_wind_beaufort(df)
    df = add_load_net(df)

    # ---------------- Lag / rolling / diff ------------------
    numeric_cols = select_numeric_candidates(df)
    lags = DEF_LAGS_SHORT + DEF_LAGS_DAILY + DEF_LAGS_WEEKLY
    df = add_lag_features(df, numeric_cols, lags)
    df = add_rolling_features(df, ROLL_WINDOWS)
    df = add_diff_pct(df, NUM_DIFF_COLS)
    df = add_renewable_ramp(df)

    # ---------------- Cleanup -------------------------------
    # Drop constant columns (once per split)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const_cols, errors="ignore")

    # For training: remove rows with NaNs introduced by lag/rolling
    if split == "train":
        df = df.dropna().reset_index(drop=True)

    # Keep only test rows if we prepended history
    if "__origin" in df.columns:
        if split == "test":
            df = df[df["__origin"] == "test"].reset_index(drop=True)
        df = df.drop(columns=["__origin"], errors="ignore")

    # Ensure target absent in test
    if split == "test" and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df

# ---------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Spain power price feature builder v3 (enhanced)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--train", type=Path, help="Raw train CSV (for history rows when split=test)")
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
