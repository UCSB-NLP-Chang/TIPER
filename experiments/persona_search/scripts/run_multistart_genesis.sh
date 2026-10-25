#!/usr/bin/env bash
# Multi-start genesis: generate N diverse initial personas via LLM, then run
# each as a separate GEPA island. This gives the search multiple starting
# points in persona space, avoiding the single-start local-optimum trap.
#
# Usage:
#   DOMAIN=airline bash experiments/persona_search/scripts/run_multistart_genesis.sh
#   DOMAIN=retail  bash experiments/persona_search/scripts/run_multistart_genesis.sh
#   DRY_RUN=1 DOMAIN=airline bash experiments/persona_search/scripts/run_multistart_genesis.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

DOMAIN="${DOMAIN:?Set DOMAIN=airline or DOMAIN=retail}"
PY="${PY:-.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

# --- Shared settings (match mut55 genesis experiments) ---
NUM_SEEDS="${NUM_SEEDS:-4}"
ITERATIONS="${ITERATIONS:-15}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-5}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.5}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"
SEED_GEN_MODEL="${SEED_GEN_MODEL:-openai/gpt-5.5}"

# --- Domain-specific settings ---
case "$DOMAIN" in
  airline)
    SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/airline_alaska_split_v5.json}"
    MB="${MB:-8}"
    EDIT_PCT="${EDIT_PCT:-25-35%}"
    MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"
    ;;
  retail)
    SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/retail_guitar_center_split_v4.json}"
    MB="${MB:-10}"
    EDIT_PCT="${EDIT_PCT:-40-50%}"
    MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"
    ;;
  *)
    echo "Unsupported domain: $DOMAIN" >&2; exit 1 ;;
esac

RUN_TAG="${RUN_TAG:-multistart_genesis_${DOMAIN}}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
SHARED_POOL="${SHARED_POOL:-${RUN_ROOT}/shared_seen_pool.jsonl}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
SEED_DIR="${RUN_ROOT}/seeds"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

mkdir -p "$LOG_DIR" "$SEED_DIR" "$(dirname "$SHARED_POOL")"

# --- Step 1: Generate diverse initial seeds (skip if already exist) ---
SEEDS_META="${SEED_DIR}/seeds_meta.json"
if [[ -f "$SEEDS_META" ]]; then
  echo "Seeds already exist at $SEED_DIR — reusing"
else
  echo "=== Step 1: Generating $NUM_SEEDS diverse seeds for $DOMAIN ==="
  "$PY" experiments/persona_search/scripts/generate_genesis_seeds.py \
    --domain "$DOMAIN" \
    --num-seeds "$NUM_SEEDS" \
    --model "$SEED_GEN_MODEL" \
    --output-dir "$SEED_DIR"
fi

# Build SEEDS array from generated files
SEEDS=()
for i in $(seq 0 $((NUM_SEEDS - 1))); do
  name=$(printf "gs_%02d" "$i")
  seed_file="${SEED_DIR}/genesis_seed_$(printf '%02d' "$i").md"
  if [[ ! -f "$seed_file" ]]; then
    echo "Missing seed: $seed_file" >&2; exit 1
  fi
  SEEDS+=("${name}:${seed_file}")
done

# Also add a no-seed (pure genesis) island for comparison
SEEDS+=("gs_pure:")  # empty seed = genesis mode

echo
echo "=== Step 2: Launching $((${#SEEDS[@]})) islands for $DOMAIN ==="
echo "  ITERATIONS=$ITERATIONS MB=$MB EDIT=$EDIT_PCT"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY  total cap=$((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]}))"
echo "  split=$SPLIT_FILE  shared pool=$SHARED_POOL"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL"
echo

pids=()
for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="isl_${name}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}.log"

  seed_args=()
  if [[ -n "$seed" && -f "$seed" ]]; then
    seed_args+=(--seed-persona "$seed")
  fi
  # no --seed-persona = genesis mode for gs_pure

  extra=(--resume)
  if [[ "$DRY_RUN" == "1" ]]; then
    extra+=(--dry-run)
  fi

  echo "[$name] seed=${seed:-<genesis>} run_dir=$run_dir"

  ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}" nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
    "${seed_args[@]}" \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains "$DOMAIN" \
    --split-file "$SPLIT_FILE" \
    --shared-seen-pool "$SHARED_POOL" \
    --island-name "$name" \
    --summary-inject-limit "$SUMMARY_INJECT_LIMIT" \
    --iterations "$ITERATIONS" \
    --minibatch "$MB" \
    --mutation-edit-pct "$EDIT_PCT" \
    --mutation-context "$MUTATION_CONTEXT" \
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

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 2
done

echo
echo "All ${#SEEDS[@]} islands launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/isl_*.log"
echo "  grep -E '\\[iter|best=' ${LOG_DIR}/isl_*.log | tail -20"
echo "  wc -l $SHARED_POOL"
