#!/usr/bin/env bash
# Island-model persona search for GLM-5.2 on telecom.
# 4 islands (brinks_00, brinks_02, brinks_03, p0003), agent via fireworks proxy (:4114).
# User-sim = user_simulator + gpt-5.4 (standard, not cc_user).
#
# Usage:
#   bash experiments/persona_search/scripts/run_islands_telecom_glm52.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_islands_telecom_glm52.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
# Vertex path: 15 req/min for non-Anthropic models. 4 islands × concurrency 1 = 4 concurrent sims.
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split.json}"
RUN_TAG="${RUN_TAG:-islands_telecom_glm52_v1}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

# Agent: GLM-5.2 via vertex proxy (dev-proxy routes claude-glm-5.2 to fireworks)
# Vertex path has no rate limit; fireworks path is capped at 5 req/min.
AGENT_LLM="${AGENT_LLM:-anthropic/claude-glm-5.2}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{}}"
VERTEX_SHIM_PORT="${VERTEX_SHIM_PORT:-4112}"

# User-sim: standard user_simulator + gpt-5.4
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"

# Mutation / judge (use OpenAI)
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

# Route agent calls through vertex proxy (supports GLM/DeepSeek without rate limit)
export ANTHROPIC_API_BASE="http://127.0.0.1:${VERTEX_SHIM_PORT}"
export ANTHROPIC_API_KEY="dummy"

mkdir -p "$LOG_DIR"

# Verify vertex shim is running
if ! (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null; then
  echo "ERROR: vertex proxy not running on :${VERTEX_SHIM_PORT}" >&2
  echo "Start it first:  VERTEX_PROXY_TOKEN=<token> nohup .venv/bin/python anthropic_to_vertex_proxy.py --port ${VERTEX_SHIM_PORT} &" >&2
  exit 1
fi
exec 3>&- 2>/dev/null || true
echo "Vertex proxy OK on :${VERTEX_SHIM_PORT}"

# 4 islands in 2 waves (vertex proxy 15 req/min limit for non-Anthropic models)
WAVE1=(
  "brinks_00:data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
  "brinks_02:data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
)
WAVE2=(
  "brinks_03:data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
  "p0003:data/persona_opt/lineages/p0003/gen_00.md"
)
SEEDS=("${WAVE1[@]}" "${WAVE2[@]}")

echo "Launching ${#SEEDS[@]}-island telecom persona search (GLM-5.2)"
echo "  ITERATIONS=$ITERATIONS MB=$MB MIN_DISSIM=$MIN_DISSIM"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY  total cap=$((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]}))"
echo "  split=$SPLIT_FILE  shared pool=NONE (independent islands)"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL  judge=$JUDGE_MODEL"
echo

launch_wave() {
  local wave_name="$1"; shift
  local specs=("$@")
  local pids=()

  echo
  echo "===== Wave $wave_name: ${#specs[@]} islands ($(date)) ====="

  for spec in "${specs[@]}"; do
    name="${spec%%:*}"
    seed="${spec#*:}"
    base="isl_${name}"
    run_dir="${RUN_ROOT}/${base}"
    log="${LOG_DIR}/${base}.log"

    if [[ ! -f "$seed" ]]; then
      echo "Missing seed: $seed" >&2
      exit 1
    fi

    extra=(--resume --no-bedrock)
    extra+=(--anthropic-base-url "http://127.0.0.1:${VERTEX_SHIM_PORT}" --anthropic-api-key dummy)

    if [[ "$DRY_RUN" == "1" ]]; then
      extra+=(--dry-run)
    fi

    echo "[$name] seed=$seed run_dir=$run_dir log=$log"

    nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
      --seed-persona "$seed" \
      --persona-id-base "$base" \
      --run-dir "$run_dir" \
      --domains telecom \
      --split-file "$SPLIT_FILE" \
      --iterations "$ITERATIONS" \
      --minibatch "$MB" \
      --min-dissimilarity "$MIN_DISSIM" \
      --max-semantic-similarity "$MAX_SEM_SIM" \
      --embedding-model "$EMBEDDING_MODEL" \
      --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
      --max-concurrency "$PER_ISLAND_CONCURRENCY" \
      --severe-gate-rate "$SEVERE_GATE_RATE" \
      --agent-llm "$AGENT_LLM" \
      --agent-llm-args "$AGENT_LLM_ARGS" \
      --user-type "$USER_TYPE" \
      --user-llm "$USER_LLM" \
      --effort "$EFFORT" \
      --mutation-model "$MUTATION_MODEL" \
      --mutation-effort "$MUTATION_EFFORT" \
      --judge-model "$JUDGE_MODEL" \
      "${extra[@]}" \
      >"$log" 2>&1 &

    pids+=("$!")
    echo "  pid $!"
    sleep 2
  done

  echo "Wave $wave_name pids: ${pids[*]}"
  echo "Monitor: tail -f ${LOG_DIR}/isl_*.log"
  wait "${pids[@]}"
  echo "===== Wave $wave_name done ($(date)) ====="
}

launch_wave 1 "${WAVE1[@]}"
launch_wave 2 "${WAVE2[@]}"

echo
echo "All 4 islands finished."
