#!/usr/bin/env bash
# Cross-domain GEPA v6: train split minibatch + val split held-out Pareto
# Agent: gpt-5.4 high reasoning | User: user_simulator gpt-5.2
# Train minibatch: 50 tasks (20T+6A+24R, 按比例抽 10 tasks/iter)
# Val held-out: 25 tasks (10T+4A+11R, airline 全保留)
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="${REPO}/experiments/persona_search/scripts/crossdomain_gepa.py"
SPLIT="${REPO}/experiments/persona_search/configs/crossdomain_variance_split_v3.json"
RUN_DIR="${REPO}/data/persona_opt/pgepa_runs/xdom_v6_3domain"

SEEDS=(
  "${REPO}/data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
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
  --persona-id-base xdom6 \
  --resume-seeds-from "${REPO}/data/persona_opt/pgepa_runs/xdom_v5_3domain/pgepa_state.json" \
  --val-num-trials 2 \
  --print-pareto \
  "$@"
