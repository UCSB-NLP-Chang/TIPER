#!/usr/bin/env bash
# Banking persona search — single-island parallel_gepa seeded from the CarWorks
# persona. Faithfully mirrors the earlier banking_v2/bk_00 config (see that run's
# pgepa_state.json "config"), with ONE deliberate change: the agent is gpt-5.5,
# not gpt-5.4.
#
# Why gpt-5.5: banking under gpt-5.4 is already floored (no_persona ASR ~27%), so
# a persona has almost no room left to push the agent down (measured: telecom
# personas transferred only +0.01~+0.05 on banking). gpt-5.5 has a higher
# baseline ASR, leaving real headroom for the search to find difficulty.
#
# User is the plain user_simulator (gpt-5.4) — no Claude, so no Anthropic shim is
# needed. gpt-5.x agent/user route through the Cresta Vertex litellm patch; the
# mutation/lint/embedding models use OPENAI_API_KEY. All creds come from .env.
#
# Retrieval: bm25 (offline, no API key), matching the transfer / difficulty runs.
#
# Usage:
#   bash experiments/persona_search/scripts/run_banking_carworks_gepa.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_banking_carworks_gepa.sh
#   SEED_FILE=path/to/carworks.md bash experiments/persona_search/scripts/run_banking_carworks_gepa.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Creds (OPENAI_API_KEY, CRESTA_VERTEX_TOKEN, VERTEXAI_*) are loaded automatically:
# parallel_gepa imports tau2.utils.utils, which calls load_dotenv() at import time.

PY="${PY:-.venv/bin/python}"
SEED_FILE="${SEED_FILE:-data/persona_opt/seeds_banking_k1/carworks_00/gen_00.md}"
PERSONA_BASE="${PERSONA_BASE:-carworks_00}"
RUN_TAG="${RUN_TAG:-banking_carworks}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/banking_knowledge_split_v2.json}"
DRY_RUN="${DRY_RUN:-0}"

# --- search hyperparams (mirror bk_00) ---
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-5}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
LINT_MODEL="${LINT_MODEL:-openai/gpt-5.4-mini}"
CONCURRENCY="${CONCURRENCY:-10}"

# --- models / routing (agent bumped to gpt-5.5) ---
AGENT_LLM="${AGENT_LLM:-gpt-5.5}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
RETRIEVAL="${RETRIEVAL:-bm25}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$SEED_FILE" ]]; then
  echo "ERROR: carworks seed not found: $SEED_FILE" >&2
  echo "  Create it (format like data/persona_opt/seeds_banking_k1/bk_00_cooperative/gen_00.md)" >&2
  echo "  or pass SEED_FILE=... to this script." >&2
  exit 1
fi

extra=(--resume --no-bedrock)
[[ "$DRY_RUN" == "1" ]] && extra+=(--dry-run)

echo "Launching banking CarWorks persona search"
echo "  seed=$SEED_FILE  base=$PERSONA_BASE  run_dir=$RUN_ROOT"
echo "  split=$SPLIT_FILE  retrieval=$RETRIEVAL"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL(effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo "  iterations=$ITERATIONS minibatch=$MB min_dissim=$MIN_DISSIM max_sem_sim=$MAX_SEM_SIM concurrency=$CONCURRENCY"
echo

log="${LOG_DIR}/${PERSONA_BASE}.log"
nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
  --seed-persona "$SEED_FILE" \
  --persona-id-base "$PERSONA_BASE" \
  --run-dir "$RUN_ROOT" \
  --domains banking \
  --retrieval-config "$RETRIEVAL" \
  --split-file "$SPLIT_FILE" \
  --iterations "$ITERATIONS" \
  --minibatch "$MB" \
  --min-dissimilarity "$MIN_DISSIM" \
  --max-semantic-similarity "$MAX_SEM_SIM" \
  --embedding-model "$EMBEDDING_MODEL" \
  --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
  --lint-model "$LINT_MODEL" \
  --max-concurrency "$CONCURRENCY" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-type "$USER_TYPE" \
  --user-llm "$USER_LLM" \
  --effort "$EFFORT" \
  --mutation-model "$MUTATION_MODEL" \
  --mutation-effort "$MUTATION_EFFORT" \
  --judge-model "$JUDGE_MODEL" \
  --num-trials 1 \
  "${extra[@]}" \
  >"$log" 2>&1 &

echo "launched pid $!  log $log"
echo "Monitor: tail -f $log"
