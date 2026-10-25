#!/usr/bin/env bash
# Continue the brinks_04 search from its current best persona for a few more
# generations, to test whether difficulty can still climb past 0.457.
#
# parallel_gepa.py has no resume, and the per-parent reflective traces (eval_batch)
# are not persisted in pgepa_state.json -- so we seed a FRESH short run from the
# best persona (gen3 idx4). gen 0 re-confirms it (rebuilding traces), then 2 more
# generations mutate from there. Same models / Vertex routing as the pair run.
#
# Usage:
#   bash experiments/persona_search/scripts/run_brinks_04_continue.sh
#   G=1 bash experiments/persona_search/scripts/run_brinks_04_continue.sh   # only 1 more gen
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

ANTHROPIC_API_KEY="REDACTED_ANTHROPIC_VERTEX_TOKEN"
PY="${PY:-.venv/bin/python}"
G="${G:-2}"
K="${K:-5}"
B="${B:-2}"
MB="${MB:-10}"
TOP_T="${TOP_T:-2}"
PHASE_PER_RUN="${PHASE_PER_RUN:-10}"
MAX_TASKS="${MAX_TASKS:-35}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
PARENT_TEMP="${PARENT_TEMP:-0.15}"
MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-2.0}"
LOG_DIR="${LOG_DIR:-data/persona_opt/pgepa_runs/logs}"
DRY_RUN="${DRY_RUN:-0}"

SEED="${SEED:-data/persona_opt/seeds_brinks/brinks_04_cont/gen_00.md}"
BASE="${BASE:-pgepa_brinks_04_cont}"
RUN_DIR="${RUN_DIR:-data/persona_opt/pgepa_runs/${BASE}}"

# Model/API routing (identical to the pair run).
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-cc_user_con}"
USER_LLM="${USER_LLM:-claude-opus-4-7}"
EFFORT="${EFFORT:-high}"
USE_BEDROCK="${USE_BEDROCK:-0}"
MUTATION_MODEL="${MUTATION_MODEL:-vertex_ai/claude-opus-4-7}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export CRESTA_VERTEX_TOKEN="${CRESTA_VERTEX_TOKEN:-${ANTHROPIC_API_KEY}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"

VERTEX_SHIM_PORT="${VERTEX_SHIM_PORT:-4112}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${VERTEX_SHIM_PORT}}"
SHIM_LOG="${LOG_DIR}/vertex_shim.log"

mkdir -p "$LOG_DIR"

if [[ "$USE_BEDROCK" != "1" && "$DRY_RUN" != "1" ]]; then
  if ! (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null; then
    echo "Starting Anthropic->Vertex shim on :${VERTEX_SHIM_PORT}"
    VERTEX_PROXY_TOKEN="$CRESTA_VERTEX_TOKEN" \
      nohup "$PY" anthropic_to_vertex_proxy.py --port "$VERTEX_SHIM_PORT" \
      >"$SHIM_LOG" 2>&1 &
    echo "  shim pid $!  log $SHIM_LOG"
    for _ in $(seq 1 30); do
      (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null && break
      sleep 0.5
    done
  else
    echo "Reusing existing shim on :${VERTEX_SHIM_PORT}"
  fi
  exec 3>&- 2>/dev/null || true
fi

if [[ ! -f "$SEED" ]]; then
  echo "Missing seed: $SEED" >&2
  exit 1
fi

extra=()
if [[ "$DRY_RUN" == "1" ]]; then extra+=(--dry-run); fi
if [[ "$USE_BEDROCK" == "1" ]]; then
  extra+=(--use-bedrock)
else
  extra+=(--no-bedrock --anthropic-base-url "$ANTHROPIC_BASE_URL" --anthropic-api-key dummy)
fi

log="${LOG_DIR}/${BASE}.log"
per_run_rollouts=$((MAX_TASKS + G * (K * MB + TOP_T * MAX_TASKS)))

echo "Continue brinks_04 from best persona"
echo "  seed    $SEED  (= gen3 idx4 best, agg 0.457)"
echo "  run_dir $RUN_DIR"
echo "  log     $log"
echo "  G=$G K=$K B=$B MB=$MB TOP_T=$TOP_T MAX_TASKS=$MAX_TASKS phase_conc=$PHASE_PER_RUN"
echo "  rollout estimate=$per_run_rollouts  (MAX_TASKS + G*(K*MB + TOP_T*MAX_TASKS))"
echo

nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
  --seed-persona "$SEED" \
  --persona-id-base "$BASE" \
  --run-dir "$RUN_DIR" \
  --domains telecom \
  --generations "$G" \
  --children-per-gen "$K" \
  --parents-per-gen "$B" \
  --minibatch "$MB" \
  --top-t "$TOP_T" \
  --max-tasks "$MAX_TASKS" \
  --parent-softmax-temperature "$PARENT_TEMP" \
  --max-length-ratio "$MAX_LENGTH_RATIO" \
  --phase-max-concurrency "$PHASE_PER_RUN" \
  --severe-gate-rate "$SEVERE_GATE_RATE" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-type "$USER_TYPE" \
  --user-llm "$USER_LLM" \
  --effort "$EFFORT" \
  --mutation-model "$MUTATION_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  "${extra[@]}" \
  >"$log" 2>&1 &

pid=$!
echo "Run launched. pid $pid"
echo "Monitor:  tail -f $log"
