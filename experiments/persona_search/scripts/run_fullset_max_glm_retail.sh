#!/bin/bash
# 步骤③ retail 额外 island: gc_03 + gc_04 各 seed+opt = 4 cells (gc_02 已在主 wrapper 跑完).
# 验证 GLM max 下 retail 反向是否系统性 (gc_02 Δ=+0.009; gpt-5.4 full set: gc_03 -0.029, gc_04 -0.041).
# 走 shim :4114 (fireworks 独立 quota), agent max + user xhigh, 3 trials, concurrency 20.
# wrapper 循环重启 + auto-resume, 直到 dry-run gap=0.
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_max_glm_retail.log"
CELLS="retail:gc_03:seed retail:gc_03:opt retail:gc_04:seed retail:gc_04:opt"
AGENT_LLM="anthropic/accounts/fireworks/models/glm-5p2"

export ANTHROPIC_API_BASE="http://127.0.0.1:4114"
export ANTHROPIC_API_KEY="dummy"

run() {
  .venv/bin/python3 experiments/persona_search/scripts/fullset_build.py \
    --cells $CELLS \
    --agent-llm "$AGENT_LLM" \
    --agent-effort max --user-effort xhigh \
    --save-prefix fullset_max_glm --concurrency 20 \
    "$@"
}

iter=0
while true; do
  iter=$((iter+1))
  echo "[$(date)] === retail extra iteration $iter ===" >> "$LOG"
  run >> "$LOG" 2>&1
  rc=$?
  gap=$(run --dry-run 2>&1 | grep "合计补跑" | tail -1)
  echo "[$(date)] rc=$rc  $gap" >> "$LOG"
  if echo "$gap" | grep -q "trial-slots: 0"; then
    echo "[$(date)] ===== RETAIL EXTRA 4 CELLS DONE =====" >> "$LOG"
    break
  fi
  echo "[$(date)] not done (or crashed) — restarting in 30s" >> "$LOG"
  sleep 30
done
