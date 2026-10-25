#!/usr/bin/env bash
# Launch two Brinks-seed parallel_gepa optimizations side by side.
#
# Serial Pareto search: ONE parent -> ONE child per iteration, parents drawn from
# GEPA's per-instance Pareto frontier, each child forced >=MIN_DISSIM novel vs all
# prior personas and free of task-specific content. No length control, no
# cross-child parallelism (MAX_CONCURRENCY is only tau2's within-eval sim fan-out).
#
# Usage:
#   bash experiments/persona_search/scripts/run_brinks_pgepa_pair.sh
#
# Useful overrides:
#   ITERATIONS=24 MB=10 MIN_DISSIM=0.30 MAX_CONCURRENCY=10 bash experiments/persona_search/scripts/run_brinks_pgepa_pair.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_brinks_pgepa_pair.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

ANTHROPIC_API_KEY="REDACTED_ANTHROPIC_VERTEX_TOKEN"
PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-${G:-24}}"   # G kept as a legacy alias
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-${PHASE_PER_RUN:-10}}"
MAX_TASKS="${MAX_TASKS:-35}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
LOG_DIR="${LOG_DIR:-data/persona_opt/pgepa_runs/logs}"
DRY_RUN="${DRY_RUN:-0}"

# Make model/API routing explicit instead of relying on parallel_gepa.py defaults.
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-cc_user_con}"
USER_LLM="${USER_LLM:-claude-opus-4-7}"
EFFORT="${EFFORT:-high}"
USE_BEDROCK="${USE_BEDROCK:-0}"
BEDROCK_PROFILE="${BEDROCK_PROFILE:-}"
MUTATION_MODEL="${MUTATION_MODEL:-vertex_ai/claude-opus-4-7}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

# Vertex proxy env for in-process litellm (mutation model).
# CRESTA_VERTEX_TOKEN uses the same uuid:secret as ANTHROPIC_API_KEY.
export CRESTA_VERTEX_TOKEN="${CRESTA_VERTEX_TOKEN:-${ANTHROPIC_API_KEY}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"

# cc_user_con backbone -> Vertex: the `claude` CLI's native Vertex mode only does
# Google ADC (can't use a uuid:secret), so we run the local Anthropic-Messages ->
# Vertex shim and point the simulator's anthropic_base_url branch at it. Both runs
# share one shim. Only needed when NOT on Bedrock and NOT a dry run.
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

SEEDS=(
  "brinks_03_verbose_polite:data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
  "brinks_04_confused_drip_feeding:data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
)

echo "Launching Brinks serial-Pareto parallel_gepa pair"
echo "  ITERATIONS=$ITERATIONS MB=$MB MIN_DISSIM=$MIN_DISSIM MAX_TASKS=$MAX_TASKS"
echo "  max_concurrency per run=$MAX_CONCURRENCY (within-eval sim fan-out only)"
echo "  severe_gate_rate=$SEVERE_GATE_RATE max_mutation_attempts=$MAX_MUT_ATTEMPTS"
echo "  agent:   --agent-llm $AGENT_LLM --agent-llm-args '$AGENT_LLM_ARGS'"
  echo "  user:    --user-type $USER_TYPE --user-llm $USER_LLM --effort $EFFORT use_bedrock=$USE_BEDROCK"
  if [[ "$USE_BEDROCK" != "1" ]]; then
    echo "           via Vertex shim $ANTHROPIC_BASE_URL (loc=$VERTEXAI_LOCATION proj=$VERTEXAI_PROJECT)"
  fi
  echo "  mutate:  --mutation-model $MUTATION_MODEL (litellm vertex_ai via CRESTA_VERTEX_TOKEN)"
echo "  judge:   --judge-model $JUDGE_MODEL (failed-only)"
per_run_rollouts=$((MAX_TASKS + ITERATIONS * (MB + 2 * MAX_TASKS)))
pair_rollouts=$((per_run_rollouts * 2))
echo "  rollout estimate per run~=$per_run_rollouts  pair~=$pair_rollouts"
echo "    formula: MAX_TASKS + ITERATIONS * (MB + 2*MAX_TASKS)  (minibatch + 2-stage full confirm; upper bound when every child passes the minibatch screen)"
echo

for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="pgepa_${name}"
  run_dir="data/persona_opt/pgepa_runs/${base}"
  log="${LOG_DIR}/${base}.log"

  if [[ ! -f "$seed" ]]; then
    echo "Missing seed: $seed" >&2
    exit 1
  fi

  extra=()
  if [[ "$DRY_RUN" == "1" ]]; then
    extra+=(--dry-run)
  fi
  if [[ "$USE_BEDROCK" == "1" ]]; then
    extra+=(--use-bedrock)
  else
    extra+=(--no-bedrock --anthropic-base-url "$ANTHROPIC_BASE_URL" --anthropic-api-key dummy)
  fi
  if [[ -n "$BEDROCK_PROFILE" ]]; then
    extra+=(--bedrock-profile "$BEDROCK_PROFILE")
  fi

  echo "[$name]"
  echo "  seed    $seed"
  echo "  run_dir $run_dir"
  echo "  log     $log"

  nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$seed" \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains telecom \
    --iterations "$ITERATIONS" \
    --minibatch "$MB" \
    --min-dissimilarity "$MIN_DISSIM" \
    --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
    --max-tasks "$MAX_TASKS" \
    --max-concurrency "$MAX_CONCURRENCY" \
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
  echo "  pid     $pid"
  echo
done

echo "Both runs launched."
echo "Monitor:"
echo "  $PY experiments/persona_search/scripts/monitor_brinks_pgepa_pair.py --watch 20"
