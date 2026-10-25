#!/usr/bin/env bash
# Island-model persona search for DeepSeek v4-pro on telecom — ENRICHED mode (v2).
# 2 islands (brinks_00, p0003), agent via fireworks proxy (:4114).
# Key changes vs v1 (basic):
#   - --mutation-context enriched (direction+score summaries + failure traces)
#   - --mutation-edit-pct 50-60% (larger edits to escape local optima)
#   - --summary-inject-limit 40 (richer context for mutator)
#   - --early-stop-consecutive-gated 3 (stop early if 3 consecutive gated)
#   - --scoring-mode mean, --val-num-trials 2
#   - --print-pareto for monitoring
#   - Fresh run dir (v2), does NOT --resume from v1 state
#   - Reuses v1 seed evals via --seed-reuse-glob
#
# Usage:
#   bash experiments/persona_search/scripts/run_islands_telecom_dsv4pro_v2.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_islands_telecom_dsv4pro_v2.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-5}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split.json}"
RUN_TAG="${RUN_TAG:-islands_telecom_dsv4pro_v2}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

# Enriched mutation settings
MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-50-60%}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"
EARLY_STOP="${EARLY_STOP:-3}"

# Agent: DeepSeek v4-pro via fireworks proxy (no 15 req/min rate limit)
AGENT_LLM="${AGENT_LLM:-anthropic/accounts/fireworks/models/deepseek-v4-pro}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{}}"
FIREWORKS_SHIM_PORT="${FIREWORKS_SHIM_PORT:-4114}"

# User-sim: standard user_simulator + gpt-5.4
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"

# Mutation / judge (use OpenAI)
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.5}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

# Route agent calls through fireworks proxy
export ANTHROPIC_API_BASE="http://127.0.0.1:${FIREWORKS_SHIM_PORT}"
export ANTHROPIC_API_KEY="dummy"

mkdir -p "$LOG_DIR"

# Verify fireworks shim is running
if ! (exec 3<>"/dev/tcp/127.0.0.1/${FIREWORKS_SHIM_PORT}") 2>/dev/null; then
  echo "ERROR: fireworks proxy not running on :${FIREWORKS_SHIM_PORT}" >&2
  echo "Start it first:  FIREWORKS_PROXY_TOKEN=<token> nohup .venv/bin/python anthropic_to_fireworks_proxy.py --port ${FIREWORKS_SHIM_PORT} &" >&2
  exit 1
fi
exec 3>&- 2>/dev/null || true
echo "Fireworks proxy OK on :${FIREWORKS_SHIM_PORT}"

# 2 islands — brinks_00 + p0003, concurrency=20
SEEDS=(
  "brinks_00:data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
  "p0003:data/persona_opt/lineages/p0003/gen_00.md"
)

echo "Launching ${#SEEDS[@]}-island telecom persona search (DeepSeek v4-pro, ENRICHED v2)"
echo "  ITERATIONS=$ITERATIONS NUM_TRIALS=$NUM_TRIALS MB=$MB MIN_DISSIM=$MIN_DISSIM"
echo "  mutation_context=$MUTATION_CONTEXT  edit_pct=$MUTATION_EDIT_PCT  summary_limit=$SUMMARY_INJECT_LIMIT"
echo "  early_stop_gated=$EARLY_STOP"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY"
echo "  split=$SPLIT_FILE"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo

pids=()
for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="isl_ds_${name}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}.log"

  if [[ ! -f "$seed" ]]; then
    echo "Missing seed: $seed" >&2
    exit 1
  fi

  extra=(--no-bedrock)
  extra+=(--anthropic-base-url "http://127.0.0.1:${FIREWORKS_SHIM_PORT}" --anthropic-api-key dummy)
  extra+=(--seed-reuse-glob "data/simulations/gepa_opt/isl_ds_${name}_g00_seed*/eval_*/telecom/results.json")

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
    --num-trials "$NUM_TRIALS" \
    --val-num-trials "$NUM_TRIALS" \
    --scoring-mode mean \
    --minibatch "$MB" \
    --mutation-context "$MUTATION_CONTEXT" \
    --mutation-edit-pct "$MUTATION_EDIT_PCT" \
    --summary-inject-limit "$SUMMARY_INJECT_LIMIT" \
    --early-stop-consecutive-gated "$EARLY_STOP" \
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
    --print-pareto \
    "${extra[@]}" \
    >"$log" 2>&1 &

  pids+=("$!")
  echo "  pid $!"
  sleep 2
done

echo
echo "All ${#SEEDS[@]} islands launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/isl_ds_*.log"
echo "  grep 'ACCEPTED\|GATED\|full_diff' ${LOG_DIR}/isl_ds_*.log"
