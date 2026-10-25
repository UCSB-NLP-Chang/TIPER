#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Run NCUser behavior presets on VitaBench delivery domain,
# gpt-5.4 high as agent + user (reasoning_effort via vitabench models.yaml).
# Settings aligned to the pgepa delivery test eval (scripts/vita_pgepa.py):
#   agent/user gpt-5.4 high, evaluator gpt-4.1, max_steps 100, seed 300,
#   enable_think, max_concurrency 16, language chinese (matches your pgepa
#   chinese delivery runs -> none preset reproduces pgepa no_persona, behavior
#   Delta directly comparable to your chinese persona Delta).
#   full = pgepa 40-task test split (ids[60:100], seed 300) x 3 trials x 5 presets
#        = 600 rollouts
#   pilot = 10 tasks x 1 trial x 5 presets = 50 rollouts
#
# Usage:
#   bash experiments/persona_search/scripts/run_ncuser_vitabench.sh pilot
#   bash experiments/persona_search/scripts/run_ncuser_vitabench.sh full
set -uo pipefail
cd "$REPO/vendor/vitabench"

MODE=${1:-pilot}
LOG=/tmp/ncuser_vitabench_${MODE}.log
echo "===== START $(date) mode=$MODE =====" >>"$LOG"
.venv/bin/python run_ncuser_delivery.py \
  --mode "$MODE" \
  --agent-llm gpt-5.4 \
  --user-llm gpt-5.4 \
  --evaluator-llm gpt-4.1 \
  --language chinese \
  --num-trials 3 \
  --seed 300 \
  --max-concurrency 16 \
  --max-steps 100 \
  --log-level WARNING \
  >>"$LOG" 2>&1
rc=$?
echo "===== END $(date) rc=$rc =====" >>"$LOG"
