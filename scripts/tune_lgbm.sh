#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tune.sh – Run Optuna tuning inside a Hydra run directory
# ---------------------------------------------------------------------------
# USAGE:
#     bash scripts/tune.sh                     # → trials は tune.yaml の値
#     bash scripts/tune.sh <cfg_yaml> <data>   # → trials は tune.yaml
#     bash scripts/tune.sh <cfg_yaml> <data> <trials>
# ---------------------------------------------------------------------------
set -euo pipefail

CFG=${1:-src/config/lgbm_baseline.yaml}  # モデル設定 YAML
INPUT=${2:-data/train/train.pkl}         # 学習データ pkl
TRIALS=${3-}                             # 第 3 引数があれば上書き

CMD=(python -m src.training.tune_hyperparams cfg="${CFG}" input="${INPUT}")

# trials を指定したときだけ CLI で上書き
if [[ -n "$TRIALS" ]]; then
  CMD+=(trials="${TRIALS}")
fi

# 実行
"${CMD[@]}"

# 出力は Hydra の run dir:
#   outputs/YYYY-MM-DD/HH-MM-SS/tuning/{best_params.yaml, study.pkl}