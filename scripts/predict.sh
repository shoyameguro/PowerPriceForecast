#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# predict.sh – Generate predictions with the *latest* trained models.
# ---------------------------------------------------------------------------
# USAGE:
#     bash scripts/predict.sh            # ← 最新の models/ を自動検出し、その run 直下へ submission.csv
#     bash scripts/predict.sh <path>     # ← <path> にある models/ を使い、同じ run 直下へ submission.csv
#     bash scripts/predict.sh <path> <csv>
#
# 例:
#     # 1) 最新 run で学習したモデルを使って提出ファイル作成
#     bash scripts/predict.sh
#
#     # 2) 任意の run を指定
#     bash scripts/predict.sh outputs/2025-06-15/23-40-12/models
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------- 引数 -----------------------------------------------------------
MODEL_DIR=${1:-latest}   # 第一引数: models ディレクトリ or "latest"
OUT_ARG=${2-}            # 第二引数: 提出 CSV (省略可)

# ---------- models=latest を解決 -----------------------------------------
if [[ "$MODEL_DIR" == "latest" ]]; then
  # outputs/**/models で最終更新が新しいものを取得
  MODEL_DIR=$(find outputs -type d -path 'outputs/*/*/models' -printf '%T@ %p\n' \
               | sort -n                    \
               | tail -n1                   \
               | cut -d' ' -f2-)
  if [[ -z "$MODEL_DIR" ]]; then
    echo "[ERROR] outputs/**/models が見つかりません" >&2
    exit 1
  fi
  echo "[INFO] 最新のモデル: $MODEL_DIR"
fi

# ---------- 出力ファイル ---------------------------------------------------
if [[ -n "$OUT_ARG" ]]; then
  OUT_PATH="$OUT_ARG"                  # ユーザ指定
else
  OUT_PATH="$(dirname "$MODEL_DIR")/submission.csv"  # run 直下に置く
fi

# ---------- 実行 -----------------------------------------------------------
python -m src.interface.predict \
    models="${MODEL_DIR}" \
    input=data/test/test.pkl \
    out="${OUT_PATH}"

echo "[INFO] prediction → ${OUT_PATH}"
