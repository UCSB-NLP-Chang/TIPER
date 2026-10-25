#!/usr/bin/env bash
# Island-model persona search: 5 parallel parallel_gepa optimizations, one per
# Brinks seed, telecom only. Each island keeps its OWN candidate pool but they all
# SHARE one file-locked "seen" pool so the personas they generate stay UNIQUE
# across islands (embedding-cosine + lexical dedup; short 2-line summaries of each
# island's children are injected back into every other island's mutation prompt).
#
# Selection uses a held-out VAL split; minibatch is sampled from TRAIN; TEST is
# left untouched for a post-hoc generalization check (see telecom_island_split.json).
#
# Concurrency: 5 islands x --max-concurrency 3 = 15 total in-flight sims (cap).
#
# Usage:
#   bash experiments/persona_search/scripts/run_islands_telecom.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_islands_telecom.sh
#   ITERATIONS=20 PER_ISLAND_CONCURRENCY=3 bash experiments/persona_search/scripts/run_islands_telecom.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

ANTHROPIC_API_KEY="REDACTED_ANTHROPIC_VERTEX_TOKEN"
PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-5}"   # 5 islands x 5 = 25 total
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split.json}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"
RUN_TAG="${RUN_TAG:-islands_telecom_v1}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
SHARED_POOL="${SHARED_POOL:-${RUN_ROOT}/shared_seen_pool.jsonl}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

# models / routing (mirror run_brinks_pgepa_pair.sh)
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-cc_user_con}"
USER_LLM="${USER_LLM:-claude-opus-4-7}"
EFFORT="${EFFORT:-high}"
USE_BEDROCK="${USE_BEDROCK:-0}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

# Vertex proxy env for in-process litellm (mutation + embedding models).
export CRESTA_VERTEX_TOKEN="${CRESTA_VERTEX_TOKEN:-${ANTHROPIC_API_KEY}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

VERTEX_SHIM_PORT="${VERTEX_SHIM_PORT:-4112}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${VERTEX_SHIM_PORT}}"
SHIM_LOG="${LOG_DIR}/vertex_shim.log"

mkdir -p "$LOG_DIR" "$(dirname "$SHARED_POOL")"

# start the Anthropic->Vertex shim once (shared by all 5 islands) unless on Bedrock/dry-run
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

# one island per seed: <island-name>:<seed-file>
SEEDS=(
  "brinks_00:data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
  "brinks_01:data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md"
  "brinks_02:data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
  "brinks_03:data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
  "brinks_04:data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
)

echo "Launching ${#SEEDS[@]}-island telecom persona search"
echo "  ITERATIONS=$ITERATIONS MB=$MB MIN_DISSIM=$MIN_DISSIM MAX_SEM_SIM=$MAX_SEM_SIM"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY  total in-flight cap=$((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]}))"
echo "  split=$SPLIT_FILE  shared pool=$SHARED_POOL"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo

pids=()
for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="isl_${name}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}.log"

  if [[ ! -f "$seed" ]]; then
    echo "Missing seed: $seed" >&2
    exit 1
  fi

  extra=(--resume)
  if [[ "$DRY_RUN" == "1" ]]; then
    extra+=(--dry-run)
  fi
  # gen-0 cross-run warm-start: reuse the SEED's prior sims where they exist.
  # Verified sound: these old runs used today's gen_00.md seeds (seed mtime Jun 10
  # predates the runs Jun 14-16). brinks_00/02 never ran a g00_seed -> no reuse.
  case "$name" in
    brinks_01) extra+=(--seed-reuse-glob "data/simulations/gepa_opt/pgepa_brinks01_run1_g00_seed/eval_*/telecom/results.json") ;;
    brinks_03) extra+=(--seed-reuse-glob "data/simulations/gepa_opt/pgepa_brinks_03_verbose_polite_g00_seed/eval_*/telecom/results.json") ;;
    brinks_04) extra+=(--seed-reuse-glob "data/simulations/gepa_opt/pgepa_brinks_04_confused_drip_feeding_g00_seed/eval_*/telecom/results.json") ;;
  esac
  if [[ "$USE_BEDROCK" == "1" ]]; then
    extra+=(--use-bedrock)
  else
    extra+=(--no-bedrock --anthropic-base-url "$ANTHROPIC_BASE_URL" --anthropic-api-key dummy)
  fi

  echo "[$name] seed=$seed run_dir=$run_dir log=$log"

  ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
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
    "${extra[@]}" \
    >"$log" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 2  # small stagger so the gen-0 seed evals don't all hit the pool/shim at once
done

echo
echo "All ${#SEEDS[@]} islands launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/isl_*.log"
echo "  wc -l $SHARED_POOL   # total unique personas across islands"
