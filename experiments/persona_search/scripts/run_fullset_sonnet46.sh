#!/bin/bash
# Claude Sonnet 4.6 fullset: high reasoning (thinking budget 4096).
# 8 cells = 4 domains × {seed, optim}.
# Routes through vertex proxy shim (:4112) → Cresta /proxy/vertex.
# Token: reuse VERTEX_PROXY_TOKEN (same as fireworks shim).
# wrapper 循环重启 + auto-resume, 直到 dry-run gap=0.
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_sonnet46.log"
CELLS="telecom:brinks_02:seed telecom:brinks_02:opt"
AGENT_LLM="claude-sonnet-4-6"

# Route through local vertex proxy shim (:4112) → Cresta /proxy/vertex.
# The shim carries the real Bearer token; ANTHROPIC_API_KEY is a dummy.
export ANTHROPIC_API_BASE="http://127.0.0.1:4112"
export ANTHROPIC_API_KEY="dummy"

run() {
  .venv/bin/python3 experiments/persona_search/scripts/fullset_build.py \
    --cells $CELLS \
    --agent-llm "$AGENT_LLM" \
    --agent-effort high --user-effort xhigh \
    --save-prefix fullset_sonnet46 --concurrency 20 \
    --num-trials 3 \
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
