#!/usr/bin/env bash
# Component-level ablation v2 — telecom, enriched mutation context.
# Three new experiments to isolate the validity *gate* from the *judge*,
# across two agents (gpt-5.4 / deepseek-v4-flash):
#
#   1. no_judge_ds4flash : --no-validity-judge  + agent=deepseek-v4-flash
#                          (judge OFF + gate OFF; ds4flash version of existing
#                           no_judge which was gpt-5.4)
#   2. no_gate_gpt54     : --no-validity-gate   + agent=gpt-5.4
#                          (judge ON, gate OFF — isolates gate's filtering role)
#   3. no_gate_ds4flash  : --no-validity-gate   + agent=deepseek-v4-flash
#                          (judge ON, gate OFF, on the weaker agent)
#
# Shared with the enriched main run (enriched_seed0_gpt54user):
#   domain=telecom, seed=brinks_00, split=telecom_island_split_fail2.json,
#   mutation-context=enriched, 15 iters, minibatch=5, num_trials=2 (val),
#   scoring=mean, user=gpt-5.4 (high), mutation=gpt-5.5 (xhigh), judge=gpt-5.2.
# Compare on test cert@3 (run run_test_cert_enriched.py after).
set -uo pipefail

cd "$(dirname "$0")/../.."  # repo root
PY="${PY:-.venv/bin/python}"

# --- OpenAI key for user/mutation/judge/embeddings (parallel_gepa 不读 .env) ---
if [[ -z "${OPENAI_API_KEY:-}" && -f .env ]]; then
  export OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and not in .env}"

# --- shared search settings (mirror enriched_seed0) ---
ITERATIONS="${ITERATIONS:-15}"
MB="${MB:-5}"
NUM_TRIALS="${NUM_TRIALS:-2}"
SCORING="${SCORING:-mean}"
MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"
MUTATION_EDIT_PCT="${MUTATION_EDIT_PCT:-40-50%}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-8}"
SIM_TIMEOUT="${SIM_TIMEOUT:-7200}"
MAX_STEPS="${MAX_STEPS:-120}"
MAX_INFRA_RETRIES="${MAX_INFRA_RETRIES:-2}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"

USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.5}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"

SEED_NAME="brinks_00"
SEED_FILE="data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split_fail2.json}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/enriched_ablation_v2}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
mkdir -p "$LOG_DIR"

[[ -f "$SEED_FILE" ]] || { echo "Missing seed: $SEED_FILE" >&2; exit 1; }
[[ -f "$SPLIT_FILE" ]] || { echo "Missing split: $SPLIT_FILE" >&2; exit 1; }

# --- agent presets ---
GPT54_AGENT_LLM="gpt-5.4"
GPT54_AGENT_ARGS='{"reasoning_effort":"high"}'

FW_PORT="${FW_PORT:-4114}"
DSFLASH_AGENT_LLM="openai/accounts/fireworks/models/deepseek-v4-flash-0731"
DSFLASH_AGENT_ARGS='{"api_base":"http://127.0.0.1:'"${FW_PORT}"'/v1","api_key":"dummy"}'

if ! curl -s -m 10 "http://127.0.0.1:${FW_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer dummy" \
    -d '{"model":"accounts/fireworks/models/deepseek-v4-flash-0731","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
    | grep -q '"content"'; then
  echo "ERROR: fireworks proxy not answering on :${FW_PORT}" >&2
  echo "  start it: nohup python3 -u openai_to_fireworks_proxy.py --port ${FW_PORT} &" >&2
  exit 1
fi
echo "fireworks proxy OK on :${FW_PORT}"

# --- which experiments to launch (default: all 3) ---
EXPS="${EXPS:-no_judge_ds4flash no_gate_gpt54 no_gate_ds4flash}"

launch() {
  local name="$1" agent_llm="$2" agent_args="$3" gate_flag="$4"
  local run_dir="${RUN_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  local base="ablation_${name}"
  echo "[$name] agent=$agent_llm flag=$gate_flag"
  echo "  run_dir=$run_dir log=$log"
  nohup "$PY" -u experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$SEED_FILE" \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains telecom \
    --split-file "$SPLIT_FILE" \
    --iterations "$ITERATIONS" \
    --minibatch "$MB" \
    --num-trials "$NUM_TRIALS" \
    --val-num-trials "$NUM_TRIALS" \
    --scoring-mode "$SCORING" \
    --mutation-context "$MUTATION_CONTEXT" \
    --mutation-edit-pct "$MUTATION_EDIT_PCT" \
    --min-dissimilarity "$MIN_DISSIM" \
    --max-semantic-similarity "$MAX_SEM_SIM" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
    --max-concurrency "$PER_ISLAND_CONCURRENCY" \
    --sim-timeout "$SIM_TIMEOUT" \
    --max-steps "$MAX_STEPS" \
    --max-infra-retries "$MAX_INFRA_RETRIES" \
    --summary-inject-limit "$SUMMARY_INJECT_LIMIT" \
    --agent-llm "$agent_llm" \
    --agent-llm-args "$agent_args" \
    --user-type "$USER_TYPE" \
    --user-llm "$USER_LLM" \
    --effort "$EFFORT" \
    --mutation-model "$MUTATION_MODEL" \
    --mutation-effort "$MUTATION_EFFORT" \
    --judge-model "$JUDGE_MODEL" \
    --no-bedrock \
    --print-pareto \
    $gate_flag \
    --resume \
    >"$log" 2>&1 &
  echo "  pid $!"
}

for exp in $EXPS; do
  case "$exp" in
    no_judge_ds4flash)
      launch "$exp" "$DSFLASH_AGENT_LLM" "$DSFLASH_AGENT_ARGS" "--no-validity-judge" ;;
    no_gate_gpt54)
      launch "$exp" "$GPT54_AGENT_LLM" "$GPT54_AGENT_ARGS" "--no-validity-gate" ;;
    no_gate_ds4flash)
      launch "$exp" "$DSFLASH_AGENT_LLM" "$DSFLASH_AGENT_ARGS" "--no-validity-gate" ;;
    *)
      echo "unknown exp: $exp" >&2; exit 1 ;;
  esac
done

echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "After convergence, cert@3 with run_test_cert_enriched.py (set RUN_DIR per exp)."
