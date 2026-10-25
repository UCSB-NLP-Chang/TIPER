#!/usr/bin/env bash
# Enriched mutation on brinks_02/03/04, parallel launch.
# Same eval settings as enriched_seed0/seed1 but with Plan C + timeout scoring.
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
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split_fail2.json}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-high}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-40-50%}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

mkdir -p "$LOG_DIR"

SEEDS=(
  "brinks_02:data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
  "brinks_03:data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
  "brinks_04:data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
)

echo "Enriched mutation: 3 seeds (brinks_02/03/04)"
echo "  ITERATIONS=$ITERATIONS MB=$MB NUM_TRIALS=2 SCORING=mean(avg@2)"
echo "  mutation_context=enriched (hypothesis-driven)"
echo "  early_stop=2 consecutive diversity-gated 8/8"
echo "  split=$SPLIT_FILE"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo

for entry in "${SEEDS[@]}"; do
  SEED_NAME="${entry%%:*}"
  SEED_FILE="${entry#*:}"
  RUN_TAG="enriched_${SEED_NAME}_gpt54user"
  RUN_DIR="${RUN_ROOT}/${RUN_TAG}/enriched_${SEED_NAME}"
  LOG="${LOG_DIR}/enriched_${SEED_NAME}.log"

  if [[ ! -f "$SEED_FILE" ]]; then
    echo "SKIP $SEED_NAME: missing seed $SEED_FILE" >&2
    continue
  fi

  echo "[$SEED_NAME] seed=$SEED_FILE run_dir=$RUN_DIR log=$LOG"

  # Capture mutation prompts + proposer responses for ONE run (brinks_04) so we
  # can inspect the new reasoning-first format on real data.
  CAP_DIR=""
  if [[ "$SEED_NAME" == "brinks_04" ]]; then
    CAP_DIR="data/persona_opt/mutation_captures_brinks_04"
    mkdir -p "$CAP_DIR"
    echo "  [capture] MUTATION_CAPTURE_DIR=$CAP_DIR"
  fi

  MUTATION_CAPTURE_DIR="$CAP_DIR" nohup "$PY" -u experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$SEED_FILE" \
    --persona-id-base "enriched_${SEED_NAME}" \
    --run-dir "$RUN_DIR" \
    --domains telecom \
    --split-file "$SPLIT_FILE" \
    --seed-reuse-glob "data/simulations/gepa_opt/enriched_${SEED_NAME}_g00_seed*/eval_*/telecom/results.json" \
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
  echo "  pid $pid"
done

echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/enriched_brinks_0{2,3,4}.log"
