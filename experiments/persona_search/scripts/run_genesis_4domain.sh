#!/usr/bin/env bash
# Genesis mode GEPA: 4 domains, enriched mutation context, no seed persona.
# Starts from no-persona baseline and lets the proposer create the first persona
# from scratch based on agent success/failure traces.
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
ITERATIONS="${ITERATIONS:-20}"
MB="${MB:-10}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_DOMAIN_CONCURRENCY="${PER_DOMAIN_CONCURRENCY:-5}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
RUN_TAG="${RUN_TAG:-genesis_4domain_enriched}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

# models
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-40-50%}"
LINT_MODEL="${LINT_MODEL:-openai/gpt-5.4-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

# per-domain config: domain_name:split_file
DOMAINS=(
  "telecom:experiments/persona_search/configs/telecom_island_split.json"
  "airline:experiments/persona_search/configs/airline_alaska_split_v5.json"
  "retail:experiments/persona_search/configs/retail_guitar_center_split_v2.json"
  "banking:experiments/persona_search/configs/banking_knowledge_split_v2.json"
)

mkdir -p "$LOG_DIR"

echo "Genesis mode GEPA: 4 domains, enriched mutation context"
echo "  ITERATIONS=$ITERATIONS MB=$MB CONCURRENCY=$PER_DOMAIN_CONCURRENCY"
echo "  mutation_context=enriched  mutation_edit_pct=$MUTATION_EDIT_PCT"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
echo

pids=()
for spec in "${DOMAINS[@]}"; do
  domain="${spec%%:*}"
  split_file="${spec#*:}"
  base="genesis_${domain}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}.log"

  if [[ ! -f "$split_file" ]]; then
    echo "Missing split file: $split_file" >&2
    exit 1
  fi

  # per-domain overrides
  domain_extra=()
  domain_mb="$MB"
  if [[ "$domain" == "banking" ]]; then
    domain_mb=5
    domain_extra+=(--retrieval-config bm25)
  fi

  # reuse fullset no_persona sims for gen 0 (same user/agent config)
  reuse_file="data/simulations/fullset/${domain}/shared_no_persona/results.json"
  if [[ -f "$reuse_file" ]]; then
    domain_extra+=(--seed-reuse-glob "$reuse_file")
  fi

  echo "[$domain] split=$split_file mb=$domain_mb run_dir=$run_dir log=$log"

  nohup "$PY" -u experiments/persona_search/scripts/parallel_gepa.py \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains "$domain" \
    --split-file "$split_file" \
    --iterations "$ITERATIONS" \
    --minibatch "$domain_mb" \
    --num-trials 2 \
    --val-num-trials 2 \
    --scoring-mode mean \
    --mutation-context enriched \
    --mutation-edit-pct "$MUTATION_EDIT_PCT" \
    --min-dissimilarity "$MIN_DISSIM" \
    --max-semantic-similarity "$MAX_SEM_SIM" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
    --lint-model "$LINT_MODEL" \
    --max-concurrency "$PER_DOMAIN_CONCURRENCY" \
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
    --resume \
    ${domain_extra[@]+"${domain_extra[@]}"} \
    >"$log" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 3  # stagger gen-0 evals
done

echo
echo "All ${#DOMAINS[@]} domains launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/genesis_*.log"
