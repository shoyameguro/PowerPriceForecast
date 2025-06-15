#!/usr/bin/env python
"""
Usage:
    python check_submission.py path/to/submission.csv
"""
import sys
from pathlib import Path
import pandas as pd

def check_submission(csv_path: Path) -> None:
    # ---- 1. 読み込み ----
    # ヘッダ行が無い想定なので names=[] を与える
    df = pd.read_csv(csv_path, header=None, names=["ts", "value"])

    # ---- 2. タイムスタンプを datetime64[ns, UTC] に変換 ----
    # もともと "+01:00" が付いた ISO 形式なので tz_localize は不要
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # ---- 3. 期待レンジを作成 ----
    start   = pd.Timestamp("2018-01-01 00:00:00+01:00").tz_convert("UTC")
    end     = pd.Timestamp("2018-12-31 23:00:00+01:00").tz_convert("UTC")
    expected = pd.date_range(start, end, freq="H", tz="UTC")

    # ---- 4. 検査 ----
    missing     = expected.difference(df["ts"])
    duplicates  = df[df.duplicated("ts", keep=False)]
    extra       = df[~df["ts"].isin(expected)]

    # ---- 5. レポート ----
    print(f"Expected rows : {len(expected):>5}")   # 8 760
    print(f"Actual rows   : {len(df):>5}")
    print(f"Missing hours : {len(missing):>5}")
    print(f"Duplicates    : {len(duplicates):>5}")
    print(f"Extra hours   : {len(extra):>5}")

    if not missing.empty:
        print("\n--- Missing timestamps (first 20) ---")
        print(missing[:20].to_series().dt.tz_convert("Europe/Madrid"))

    if not duplicates.empty:
        print("\n--- Duplicate rows (showing up to 10) ---")
        print(duplicates.head(10))

    if not extra.empty:
        print("\n--- Extra timestamps (showing up to 20) ---")
        print(extra.head(20))

    # ---- 6. 自動チェック（0 件ならパス） ----
    assert len(missing) == 0, "❌ 時系列に欠損があります"
    assert len(duplicates) == 0, "❌ タイムスタンプが重複しています"
    assert len(extra) == 0, "❌ 想定外のタイムスタンプが含まれています"
    print("\n✅ すべて問題ありません！")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python check_submission.py path/to/submission.csv")
    check_submission(Path(sys.argv[1]))