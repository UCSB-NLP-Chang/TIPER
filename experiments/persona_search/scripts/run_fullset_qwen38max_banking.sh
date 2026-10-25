#!/bin/bash
# Qwen 3.8 Max Preview — banking domain only (bm25 retrieval).
# Agent: qwen3.8-max-preview (high reasoning) via aliyun
# User:  gpt-5.4 (high reasoning) via default OpenAI endpoint
# 3 cells = banking × {seed, opt, no_persona}
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/fullset_qwen38max_banking.log"

# --- Force IPv4 ---
mkdir -p /tmp/ipv4_site
cp experiments/persona_search/scripts/_force_ipv4.py /tmp/ipv4_site/sitecustomize.py
export PYTHONPATH="/tmp/ipv4_site${PYTHONPATH:+:$PYTHONPATH}"

ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALIYUN_KEY="REDACTED_ALIYUN_KEY"

CELLS="banking:cw_03:seed banking:cw_03:opt banking:shared:no_persona"
AGENT_LLM="openai/qwen3.8-max-preview"
SAVE_PREFIX="fullset_qwen38max"

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

log "=== Qwen 3.8 Max Preview banking (agent=qwen38max@aliyun high, user=gpt-5.4 high) ==="
log "Agent: $AGENT_LLM via $ALIYUN_BASE"
log "Cells: banking × {seed, opt, no_persona} = 3"
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
    log "===== ALL 3 CELLS DONE ====="
    break
  fi
  log "not done (or crashed) — restarting in 30s"
  sleep 30
done

log "=== ALL DONE ==="
