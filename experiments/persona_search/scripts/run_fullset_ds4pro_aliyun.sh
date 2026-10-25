#!/bin/bash
# DeepSeek v4 Pro fullset via 阿里云百炼.
# Agent: deepseek-v4-pro (high reasoning) via aliyun
# User:  gpt-5.4 (high reasoning) via default OpenAI endpoint
# 12 cells = 4 domains × {seed, opt, no_persona}:
#   telecom  brinks_02  (Δ=−0.266, telecom 最强)
#   banking  cw_03      (bm25, Δ=−0.069)
#   airline  alaska_01  (v5, Δ=−0.065)
#   retail   gc_02      (Δ=−0.044)
# wrapper 循环 auto-resume, 直到 dry-run gap=0.
#
# Usage:
#   bash experiments/persona_search/scripts/run_fullset_ds4pro_aliyun.sh
#   nohup bash experiments/persona_search/scripts/run_fullset_ds4pro_aliyun.sh > /tmp/fullset_ds4pro_aliyun.log 2>&1 &
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_ds4pro_aliyun.log"

# --- Force IPv4: aliyun returns AAAA records but IPv6 is unreachable from here ---
mkdir -p /tmp/ipv4_site
cp experiments/persona_search/scripts/_force_ipv4.py /tmp/ipv4_site/sitecustomize.py
export PYTHONPATH="/tmp/ipv4_site${PYTHONPATH:+:$PYTHONPATH}"

# Agent endpoint (阿里云百炼) — passed per-call, NOT as global env var,
# so the user-sim (gpt-5.4) still routes to the default OpenAI endpoint.
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALIYUN_KEY="REDACTED_ALIYUN_KEY"

# Cells: airline/retail (done) → telecom → banking (slowest, last)
CELLS="airline:alaska_01:seed airline:alaska_01:opt airline:shared:no_persona retail:gc_02:seed retail:gc_02:opt retail:shared:no_persona telecom:brinks_02:seed telecom:brinks_02:opt telecom:shared:no_persona banking:cw_03:seed banking:cw_03:opt banking:shared:no_persona"
AGENT_LLM="openai/deepseek-v4-pro"
SAVE_PREFIX="fullset_ds4pro_aliyun"

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

log "=== DeepSeek v4 Pro fullset (agent=ds4pro@aliyun high, user=gpt-5.4 high) ==="
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

# --- Summary ---
log ""
log "=== Pick top/last trials ==="
.venv/bin/python3 experiments/persona_search/scripts/pick_top_last_trials.py \
  --prefix "$SAVE_PREFIX" --keep 3 | tee -a "$LOG"
log "=== ALL DONE ==="
