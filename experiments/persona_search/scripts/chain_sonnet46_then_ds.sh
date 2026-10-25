#!/bin/bash
# Chain: wait for sonnet46 fullset → pick data → start fireworks shim → run deepseek fullset → pick data.
set -uo pipefail
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
LOG="/tmp/chain_sonnet46_ds.log"

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

# ─── Phase 1: wait for sonnet46 fullset to finish ───
log "Phase 1: waiting for sonnet46 fullset (pid 7905)..."
while kill -0 7905 2>/dev/null; do
  sleep 300  # check every 5 min
done
log "sonnet46 fullset process ended."

# Verify completion via dry-run
export ANTHROPIC_API_BASE="http://127.0.0.1:4112"
export ANTHROPIC_API_KEY="dummy"
gap=$(.venv/bin/python3 experiments/persona_search/scripts/fullset_build.py \
  --cells airline:alaska_01:seed airline:alaska_01:opt retail:gc_02:seed retail:gc_02:opt \
         banking:cw_03:seed banking:cw_03:opt telecom:brinks_02:seed telecom:brinks_02:opt \
  --agent-llm claude-sonnet-4-6 --agent-effort high --user-effort xhigh \
  --save-prefix fullset_sonnet46 --concurrency 20 --num-trials 4 \
  --dry-run 2>&1 | grep "合计补跑" | tail -1)
log "sonnet46 dry-run gap: $gap"

# ─── Phase 2: pick sonnet46 data ───
log "Phase 2: picking sonnet46 top3/last3 data..."
.venv/bin/python3 experiments/persona_search/scripts/pick_top_last_trials.py \
  --prefix fullset_sonnet46 --keep 3 >> "$LOG" 2>&1
log "sonnet46 data picked."

# ─── Phase 3: start fireworks shim for deepseek ───
log "Phase 3: starting fireworks shim on :4115..."
# Load CRESTA_VERTEX_TOKEN from .env
CRESTA_VERTEX_TOKEN=$(grep "^CRESTA_VERTEX_TOKEN=" "$REPO/.env" | cut -d= -f2-)
export VERTEX_PROXY_TOKEN="$CRESTA_VERTEX_TOKEN"
export GLM_REASONING_EFFORT="max"

# Start shim on port 4115 (separate from any existing :4113/:4114)
.venv/bin/python3 anthropic_to_fireworks_proxy.py --port 4115 &
SHIM_PID=$!
log "fireworks shim started: pid=$SHIM_PID, port=4115"
sleep 3

# Smoke test
smoke=$(curl -s http://127.0.0.1:4115/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"accounts/fireworks/models/deepseek-v4-pro","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
log "shim smoke test: ${smoke:0:100}"

# ─── Phase 4: run deepseek fullset ───
log "Phase 4: starting deepseek v4 pro fullset (max reasoning, 4 trials)..."
export ANTHROPIC_API_BASE="http://127.0.0.1:4115"
export ANTHROPIC_API_KEY="dummy"

bash experiments/persona_search/scripts/run_fullset_ds_max.sh >> "$LOG" 2>&1
log "deepseek fullset finished."

# ─── Phase 5: pick deepseek data ───
log "Phase 5: picking deepseek top3/last3 data..."
.venv/bin/python3 experiments/persona_search/scripts/pick_top_last_trials.py \
  --prefix fullset_ds_max --keep 3 >> "$LOG" 2>&1
log "deepseek data picked."

# Clean up shim
kill $SHIM_PID 2>/dev/null
log "=== ALL DONE ==="
