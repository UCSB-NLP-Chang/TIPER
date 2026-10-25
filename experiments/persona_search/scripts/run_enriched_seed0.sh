#!/usr/bin/env bash
# GEPA with enriched mutation context: hypothesis-driven + failure traces + outcome summaries.
# Same settings as fail2_seed0_gpt54user baseline, only --mutation-context enriched added.
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-5}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-15}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split_fail2.json}"
RUN_TAG="${RUN_TAG:-enriched_seed0_gpt54user}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

# models — identical to baseline
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-40-50%}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

SEED_NAME="brinks_00"
SEED_FILE="data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
BASE="enriched_${SEED_NAME}"
RUN_DIR="${RUN_ROOT}/${BASE}"
LOG="${LOG_DIR}/${BASE}.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$SEED_FILE" ]]; then
  echo "Missing seed: $SEED_FILE" >&2
  exit 1
fi

echo "GEPA enriched mutation context: ${SEED_NAME}"
echo "  ITERATIONS=$ITERATIONS MB=$MB NUM_TRIALS=2 SCORING=mean(avg@2)"
echo "  mutation_context=enriched (hypothesis-driven + failure traces + outcome summaries)"
echo "  split=$SPLIT_FILE"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)  mutate=$MUTATION_MODEL  judge=$JUDGE_MODEL"
echo

echo "[$SEED_NAME] seed=$SEED_FILE run_dir=$RUN_DIR log=$LOG"

nohup "$PY" -u experiments/persona_search/scripts/parallel_gepa.py \
  --seed-persona "$SEED_FILE" \
  --persona-id-base "$BASE" \
  --run-dir "$RUN_DIR" \
  --domains telecom \
  --split-file "$SPLIT_FILE" \
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
echo "  pid $pid"
echo
echo "Monitor:"
echo "  tail -f $LOG"
