#!/bin/bash
# 步骤③: GLM-5.2 fullset, agent reasoning=max + user gpt-5.4 xhigh, 3 trials.
# 8 cells = 4 domains × {seed, optim}:
#   airline  alaska_01 (v5 gen12 optim / v5 gen0 seed, cert Δ=−0.065)
#   retail   gc_02     (full-set Δ=−0.044, retail 最强)
#   banking  cw_03     (bm25, full-set Δ=−0.069)
#   telecom  brinks_02 (full-set Δ=−0.266, full set 上反超 brinks_00)
# total concurrency=20 (串行 cell, 每 cell 内 tau2 --max-concurrency 20).
# 写到 fullset_max_glm/，不碰 canonical fullset/。auto-resume 断点续跑。
# wrapper 循环重启: 进程崩了自动续, 直到 dry-run gap=0.
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_max_glm.log"
CELLS="airline:alaska_01:seed airline:alaska_01:opt retail:gc_02:seed retail:gc_02:opt banking:cw_03:seed banking:cw_03:opt telecom:brinks_02:seed telecom:brinks_02:opt"
AGENT_LLM="anthropic/accounts/fireworks/models/glm-5p2"

# Route GLM through the local fireworks shim (:4114) → Cresta /proxy/fireworks path,
# which has its own quota (the direct anthropic/ path hit the $50/day Cresta vertex
# budget mid-run and infra-errorred all telecom/banking sims). The shim carries the
# real Bearer token itself; ANTHROPIC_API_KEY is a dummy. shim env already sets
# GLM_REASONING_EFFORT=max so effort is injected even if litellm drops the kwarg.
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
