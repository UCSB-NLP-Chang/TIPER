#!/usr/bin/env bash
# Resume GEPA optimization for brinks_02 and brinks_04 ONLY (gpt-5.4 user).
# Same config as run_islands_telecom_gpt54.sh but only 2 islands.
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-30}"
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-10}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split.json}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"
RUN_TAG="${RUN_TAG:-islands_telecom_gpt54user}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
SHARED_POOL="${SHARED_POOL:-${RUN_ROOT}/shared_seen_pool.jsonl}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

# models
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

mkdir -p "$LOG_DIR"

SEEDS=(
  "brinks_02:data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
  "brinks_04:data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
)

echo "Resuming 2-island GEPA optimization (brinks_02 + brinks_04)"
echo "  ITERATIONS=$ITERATIONS MB=$MB"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo

pids=()
for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="g54_${name}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}_resume.log"

  echo "[$name] resuming from $run_dir -> log=$log"

  nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$seed" \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains telecom \
    --split-file "$SPLIT_FILE" \
    --shared-seen-pool "$SHARED_POOL" \
    --island-name "$name" \
    --summary-inject-limit "$SUMMARY_INJECT_LIMIT" \
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
    --no-bedrock \
    --resume \
    >"$log" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 2
done

echo
echo "Launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/g54_brinks_0{2,4}_resume.log"
echo "  RUN_TAG=$RUN_TAG .venv/bin/python experiments/persona_search/scripts/monitor_islands.py"
