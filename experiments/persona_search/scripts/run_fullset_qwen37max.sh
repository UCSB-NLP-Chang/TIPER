#!/bin/bash
# Qwen 3.7 Max fullset via 阿里云百炼.
# Agent: qwen3.7-max (high reasoning) via aliyun
# User:  gpt-5.4 (high reasoning) via default OpenAI endpoint
# 12 cells = 4 domains × {seed, opt, no_persona}:
#   telecom  brinks_02  (Δ=−0.266, telecom 最强)
#   banking  cw_03      (bm25, Δ=−0.069)
#   airline  alaska_01  (v5, Δ=−0.065)
#   retail   gc_02      (Δ=−0.044)
# wrapper 循环 auto-resume, 直到 dry-run gap=0.
#
# Usage:
#   bash experiments/persona_search/scripts/run_fullset_qwen37max.sh
#   nohup bash experiments/persona_search/scripts/run_fullset_qwen37max.sh > /tmp/fullset_qwen37max.log 2>&1 &
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_qwen37max.log"

# --- Force IPv4: aliyun returns AAAA records but IPv6 is unreachable from here ---
mkdir -p /tmp/ipv4_site
cp experiments/persona_search/scripts/_force_ipv4.py /tmp/ipv4_site/sitecustomize.py
export PYTHONPATH="/tmp/ipv4_site${PYTHONPATH:+:$PYTHONPATH}"

# Agent endpoint (阿里云百炼) — passed per-call, NOT as global env var,
# so the user-sim (gpt-5.4) still routes to the default OpenAI endpoint.
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALIYUN_KEY="REDACTED_ALIYUN_KEY"

CELLS="airline:alaska_01:seed airline:alaska_01:opt airline:shared:no_persona retail:gc_02:seed retail:gc_02:opt retail:shared:no_persona telecom:brinks_02:seed telecom:brinks_02:opt telecom:shared:no_persona banking:cw_03:seed banking:cw_03:opt banking:shared:no_persona"
AGENT_LLM="openai/qwen3.7-max"
SAVE_PREFIX="fullset_qwen37max"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
  .venv/bin/python3 experiments/persona_search/scripts/fullset_build.py \
    --cells $CELLS \
    --agent-llm "$AGENT_LLM" \
    --agent-effort high \
    --agent-api-base "$ALIYUN_BASE" \
    --agent-api-key "$ALIYUN_KEY" \
    --save-prefix "$SAVE_PREFIX" --concurrency 10 \
    --num-trials 3 \
    "$@"
}

log "=== Qwen 3.7 Max fullset (agent=qwen37max@aliyun high, user=gpt-5.4 high) ==="
log "Agent: $AGENT_LLM via $ALIYUN_BASE"
log "User: gpt-5.4 high (default OpenAI)"
log "Cells: 4 domains × {seed, opt, no_persona} = 12"
log "Save prefix: $SAVE_PREFIX"

iter=0
while true; do
  iter=$((iter+1))
  log "=== iteration $iter ==="
  run >> "$LOG" 2>&1
  rc=$?
  gap=$(run --dry-run 2>&1 | grep "合计补跑" | tail -1)
  log "rc=$rc  $gap"
  if echo "$gap" | grep -q "trial-slots: 0"; then
    log "===== ALL 12 CELLS DONE ====="
    break
  fi
  log "not done (or crashed) — restarting in 30s"
  sleep 30
done

log "=== ALL DONE ==="
