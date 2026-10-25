#!/usr/bin/env bash
# Cross-domain GEPA v7: Pareto-bug 修复版 + 只用 seed1-4（去掉强势的 seed0）
# - 修复：run_iteration 的 build_pareto 现在用 val instance_ids（之前误用 train ids，
#   导致 Pareto 退化成"永远选 agg 最高的 seed0"）。
# - 复用：train 分数取自 v5（无 val split，subscores 即 train），
#         val 分数取自 v6（held-out val，2 trials）。按 persona 文本匹配。
# Agent: gpt-5.4 high reasoning | User: user_simulator gpt-5.2
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="${REPO}/experiments/persona_search/scripts/crossdomain_gepa.py"
SPLIT="${REPO}/experiments/persona_search/configs/crossdomain_variance_split_v3.json"
RUN_DIR="${REPO}/data/persona_opt/pgepa_runs/xdom_v7_3domain"

# 只用 seed1-4（去掉 brinks_00 / seed0）
SEEDS=(
  "${REPO}/data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md"
  "${REPO}/data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
  "${REPO}/data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
  "${REPO}/data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
)

"${REPO}/.venv/bin/python3" "${SCRIPT}" \
  --seed-personas "${SEEDS[@]}" \
  --split-file "${SPLIT}" \
  --split train \
  --val-split val \
  --val-cap 25 \
  --val-min-domain-size 5 \
  --run-dir "${RUN_DIR}" \
  --iterations 40 \
  --minibatch 10 \
  --valset-cap 50 \
  --mutation-edit-pct "60-80%" \
  --seed 42 \
  --persona-id-base xdom7 \
  --resume-seeds-from \
    "${REPO}/data/persona_opt/pgepa_runs/xdom_v5_3domain/pgepa_state.json" \
    "${REPO}/data/persona_opt/pgepa_runs/xdom_v6_3domain/pgepa_state.json" \
  --val-num-trials 2 \
  --print-pareto \
  "$@"
