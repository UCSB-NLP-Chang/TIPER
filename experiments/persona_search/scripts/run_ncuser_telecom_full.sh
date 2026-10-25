#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Run NCUser non-collaborative behavior presets on the telecom TEST set,
# matching the existing telecom fullset test setup:
#   agent gpt-5.4 (high) / user gpt-5.4 (high)
#   --num-trials 3 --seed 300 --max-concurrency 25 --max-steps 120 --timeout 600
#   --auto-resume, 5 presets (none/emotional/tangential/fragment/unavailable)
#   54 test tasks × 3 trials × 5 presets = 810 rollouts
#
# Usage:
#   bash experiments/persona_search/scripts/run_ncuser_telecom_full.sh          # full (background)
#   bash experiments/persona_search/scripts/run_ncuser_telecom_full.sh pilot    # 10×1×5 smoke
set -uo pipefail
cd "$REPO"

MODE=${1:-full}
LOG=/tmp/ncuser_telecom_${MODE}.log
echo "===== START $(date) mode=$MODE =====" >>"$LOG"
.venv/bin/python experiments/persona_search/scripts/run_ncuser_telecom.py \
  --mode "$MODE" \
  --agent-llm gpt-5.4 \
  --user-llm gpt-5.4 \
  --agent-llm-args '{"reasoning_effort":"high"}' \
  --user-llm-args '{"reasoning_effort":"high"}' \
  --num-trials 3 \
  --seed 300 \
  --max-concurrency 20 \
  --max-steps 120 \
  --sim-timeout 600 \
  --run-tag ncuser_telecom \
  >>"$LOG" 2>&1
rc=$?
echo "===== END $(date) rc=$rc =====" >>"$LOG"
