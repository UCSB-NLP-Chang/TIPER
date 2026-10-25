#!/bin/bash
# DeepSeek v4 Pro fullset: max reasoning.
# 8 cells = 4 domains × {seed, optim}.
# Routes through fireworks shim (:4115) → Cresta /proxy/fireworks.
# Token: .env CRESTA_VERTEX_TOKEN (shared with vertex shim).
# wrapper loop + auto-resume until dry-run gap=0.
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_ds_max.log"
CELLS="airline:alaska_01:seed airline:alaska_01:opt retail:gc_02:seed retail:gc_02:opt banking:cw_03:seed banking:cw_03:opt telecom:brinks_02:seed telecom:brinks_02:opt"
AGENT_LLM="anthropic/accounts/fireworks/models/deepseek-v4-pro"

# Route through local fireworks shim (:4115) → Cresta /proxy/fireworks.
# The shim injects reasoning_effort=max and carries the Bearer token.
export ANTHROPIC_API_BASE="http://127.0.0.1:4115"
export ANTHROPIC_API_KEY="dummy"

run() {
  .venv/bin/python3 experiments/persona_search/scripts/fullset_build.py \
    --cells $CELLS \
    --agent-llm "$AGENT_LLM" \
    --agent-effort max --user-effort xhigh \
    --save-prefix fullset_ds_max --concurrency 20 \
    --num-trials 4 \
    "$@"
}

iter=0
while true; do
  iter=$((iter+1))
  echo "[$(date)] === iteration $iter ===" >> "$LOG"
  run >> "$LOG" 2>&1
  rc=$?
  gap=$(run --dry-run 2>&1 | grep "合计补跑" | tail -1)
  echo "[$(date)] rc=$rc  $gap" >> "$LOG"
  if echo "$gap" | grep -q "trial-slots: 0"; then
    echo "[$(date)] ===== ALL 8 CELLS DONE =====" >> "$LOG"
    break
  fi
  echo "[$(date)] not done (or crashed) — restarting in 30s" >> "$LOG"
  sleep 30
done
