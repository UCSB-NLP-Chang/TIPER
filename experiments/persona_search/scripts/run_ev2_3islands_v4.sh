#!/usr/bin/env bash
# Enriched-v2 3-island search: brinks_00, 01, 04 (val=20×2trials, concurrency=10)
# Continues from ev2_5islands_v2 best candidates as seeds
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-5}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-10}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split_v20.json}"

RUN_TAG="${RUN_TAG:-ev2_3islands_v4}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
SHARED_POOL="${SHARED_POOL:-${RUN_ROOT}/shared_seen_pool.jsonl}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-xhigh}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-high}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-40-50%}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

mkdir -p "$LOG_DIR" "$(dirname "$SHARED_POOL")"

# Seeds: original brinks personas
SEEDS=(
  "brinks_00:${RUN_ROOT}/seeds/brinks_00.md"
  "brinks_01:${RUN_ROOT}/seeds/brinks_01.md"
  "brinks_02:${RUN_ROOT}/seeds/brinks_02.md"
  "brinks_03:${RUN_ROOT}/seeds/brinks_03.md"
  "brinks_04:${RUN_ROOT}/seeds/brinks_04.md"
)

echo "Enriched-v2 3-island search (val=20): ${#SEEDS[@]} islands (brinks_00, 01, 04)"
echo "  Seeds from ev2_5islands_v2 best candidates"
echo "  ITERATIONS=$ITERATIONS MB=$MB NUM_TRIALS=2 SCORING=mean(avg@2)"
echo "  mutation_context=enriched  split=$SPLIT_FILE (val=20)"
echo "  concurrency: ${PER_ISLAND_CONCURRENCY}/island × ${#SEEDS[@]} = $((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]})) total"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo

pids=()
for entry in "${SEEDS[@]}"; do
  SEED_NAME="${entry%%:*}"
  SEED_FILE="${entry#*:}"
  BASE="ev2v4_${SEED_NAME}"
  RUN_DIR="${RUN_ROOT}/${BASE}"
  LOG="${LOG_DIR}/${BASE}.log"

  if [[ ! -f "$SEED_FILE" ]]; then
    echo "SKIP $SEED_NAME: missing seed $SEED_FILE" >&2
    continue
  fi

  echo "[$SEED_NAME] seed=$SEED_FILE run_dir=$RUN_DIR log=$LOG"

  nohup "$PY" -u experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$SEED_FILE" \
    --persona-id-base "$BASE" \
    --run-dir "$RUN_DIR" \
    --domains telecom \
    --split-file "$SPLIT_FILE" \
    --shared-seen-pool "$SHARED_POOL" \
    --island-name "$SEED_NAME" \
    --summary-inject-limit 40 \
    --iterations "$ITERATIONS" \
    --minibatch "$MB" \
    --num-trials 2 \
    --val-num-trials 2 \
    --scoring-mode mean \
    --mutation-context enriched \
    --mutation-edit-pct "$MUTATION_EDIT_PCT" \
    --min-dissimilarity "$MIN_DISSIM" \
    --max-semantic-similarity "$MAX_SEM_SIM" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
    --max-concurrency "$PER_ISLAND_CONCURRENCY" \
    --severe-gate-rate "$SEVERE_GATE_RATE" \
    --early-stop-consecutive-gated 3 \
    --agent-llm "$AGENT_LLM" \
    --agent-llm-args "$AGENT_LLM_ARGS" \
    --user-type "$USER_TYPE" \
    --user-llm "$USER_LLM" \
    --effort "$EFFORT" \
    --mutation-model "$MUTATION_MODEL" \
    --mutation-effort "$MUTATION_EFFORT" \
    --judge-model "$JUDGE_MODEL" \
    --no-bedrock \
    --print-pareto \
    >"$LOG" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 2
done

echo
echo "All ${#SEEDS[@]} islands launched: pids ${pids[*]}"
echo "Monitor:"
echo "  .venv/bin/python experiments/persona_search/scripts/monitor_islands.py --launcher experiments/persona_search/scripts/run_ev2_3islands_v4.sh --full"
echo "  .venv/bin/python experiments/persona_search/scripts/monitor_islands.py --launcher experiments/persona_search/scripts/run_ev2_3islands_v4.sh --full --watch 15"
echo "  tail -f ${LOG_DIR}/ev2v4_brinks_0{0,1,4}.log"
