#!/usr/bin/env bash
# Pure-genesis persona search from NO-PERSONA on 3 domains,
# targeting MiniMax-M3 (via Fireworks) as the agent under test.
#
# Agent:  anthropic/accounts/fireworks/models/minimax-m3
#         via local anthropic->fireworks shim on :4113 (ANTHROPIC_API_BASE).
#         Shim injects reasoning_effort=high (EFFORT env, default high) when the
#         caller passes neither reasoning_effort nor thinking. We pass
#         agent-llm-args={} so the shim injects high — verified minimax-m3 accepts
#         it via the fireworks Anthropic endpoint (smoke 0 infra error).
# User:   gpt-5.4 (high)  | Mutation: gpt-5.5 xhigh | Judge: gpt-5.2
#
# Anthropic prompt caching is enabled in llm_utils.py (3 breakpoints).
# Genesis mode = no --seed-persona: gen-0 evaluates the no-persona baseline and
# the proposer authors the first persona from those failure traces.
# Per domain, NUM_ISLANDS (default 1) pure-genesis islands run in parallel.
# Total concurrency ~= 3 domains x PER_ISLAND_CONCURRENCY (default 8) ~= 24.
#
# Prereq: the anthropic->fireworks shim must be listening on :4113
#   start it:
#   FIREWORKS_PROXY_TOKEN=<uuid:secret> \
#     nohup python3 -u anthropic_to_fireworks_proxy.py --port 4113 &
#
# Usage:
#   bash experiments/persona_search/scripts/run_genesis_minimax.sh                  # all 3 domains
#   DOMAINS="telecom" bash .../run_genesis_minimax.sh                 # single domain
#   MUTATION_CONTEXT=weird DOMAINS=telecom bash ...                   # weird mode
#   RUN_TAG=minimax_smoke NUM_ISLANDS=1 ITERATIONS=2 bash ...         # smoke (fresh RUN_TAG)
set -uo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"

# --- load secrets from .env ---
load_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-; }
[[ -f .env ]] || { echo "ERROR: no .env in repo root" >&2; exit 1; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then export OPENAI_API_KEY="$(load_env OPENAI_API_KEY)"; fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and not in .env}"

# Fireworks token for the minimax agent (anthropic->fireworks shim reads this).
# The shim itself was started with FIREWORKS_PROXY_TOKEN; here we only need
# ANTHROPIC_API_BASE/KEY so litellm's anthropic provider points at the shim.
export ANTHROPIC_API_BASE="http://127.0.0.1:${FW_PORT:-4113}"
export ANTHROPIC_API_KEY="dummy"   # shim replaces auth with its own token

# --- Agent: minimax-m3 via local anthropic->fireworks shim on :4113 ---
FW_PORT="${FW_PORT:-4113}"
AGENT_LLM="${AGENT_LLM:-anthropic/accounts/fireworks/models/minimax-m3}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{}}"
# EFFORT controls shim-side reasoning_effort injection; "" disables (minimax may reject).
EFFORT="${EFFORT:-high}"

# Shim liveness check.
if ! pgrep -f "anthropic_to_fireworks_proxy.py" >/dev/null 2>&1; then
  echo "ERROR: anthropic_to_fireworks_proxy.py not running" >&2
  echo "  start: FIREWORKS_PROXY_TOKEN=<tok> nohup python3 -u anthropic_to_fireworks_proxy.py --port ${FW_PORT} &" >&2
  exit 1
fi
if ! curl -s -o /dev/null -m 5 "http://127.0.0.1:${FW_PORT}/" ; then
  echo "ERROR: nothing listening on :${FW_PORT}" >&2; exit 1
fi
echo "anthropic->fireworks shim OK on :${FW_PORT} (agent=$AGENT_LLM args=$AGENT_LLM_ARGS)"

# --- Shared settings ---
NUM_ISLANDS="${NUM_ISLANDS:-1}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-8}"
SIM_TIMEOUT="${SIM_TIMEOUT:-7200}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
USER_MB="${MB:-}"
USER_ITERS="${ITERATIONS:-}"
USER_ESTOP="${EARLY_STOP:-}"
MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"
SUMMARY_INJECT_LIMIT="${SUMMARY_INJECT_LIMIT:-40}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.5}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"
RUN_TAG="${RUN_TAG:-genesis_minimax_3domain}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
DOMAINS="${DOMAINS:-telecom retail airline}"

# --- Per-domain settings (split / minibatch / edit% / iters / early-stop) ---
dom_cfg() {
  case "$1" in
    telecom)  SPLIT=experiments/persona_search/configs/telecom_island_split.json;          MB="${USER_MB:-10}"; EDIT="50-60%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    airline)  SPLIT=experiments/persona_search/configs/airline_alaska_split_v5.json;       MB="${USER_MB:-8}";  EDIT="25-35%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    retail)   SPLIT=experiments/persona_search/configs/retail_guitar_center_split_v4.json; MB="${USER_MB:-10}"; EDIT="40-50%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    *) echo "unknown domain $1" >&2; return 1 ;;
  esac
}

echo "=== Genesis-from-no-persona search: agent=minimax-m3 (via fireworks) ==="
echo "  domains: $DOMAINS   islands/domain: $NUM_ISLANDS   iters: 15   early-stop: 3   context: $MUTATION_CONTEXT"
echo "  cache: 3-breakpoint prompt caching ON (llm_utils.py)"
echo "  run root: $RUN_ROOT"
echo

pids=()
for dom in $DOMAINS; do
  dom_cfg "$dom" || exit 1
  [[ -f "$SPLIT" ]] || { echo "missing split $SPLIT" >&2; exit 1; }
  POOL="${RUN_ROOT}/${dom}/shared_seen_pool.jsonl"
  LOG_DIR="${RUN_ROOT}/${dom}/logs"
  mkdir -p "$LOG_DIR"

  for i in $(seq 0 $((NUM_ISLANDS - 1))); do
    base="isl_${dom}_g${i}"
    run_dir="${RUN_ROOT}/${dom}/${base}"
    log="${LOG_DIR}/${base}.log"
    extra=(--resume --seed "$i" --shared-seen-pool "$POOL" --island-name "${dom}_g${i}")

    echo "[$dom/g$i] genesis (no seed persona)  split=$SPLIT mb=$MB edit=$EDIT iters=$ITERS early_stop=$ESTOP context=$MUTATION_CONTEXT"
    nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
      --persona-id-base "$base" \
      --run-dir "$run_dir" \
      --domains "$dom" \
      --split-file "$SPLIT" \
      --iterations "$ITERS" \
      --num-trials "$NUM_TRIALS" \
      --val-num-trials "$NUM_TRIALS" \
      --scoring-mode mean \
      --minibatch "$MB" \
      --mutation-context "$MUTATION_CONTEXT" \
      --mutation-edit-pct "$EDIT" \
      --summary-inject-limit "$SUMMARY_INJECT_LIMIT" \
      --early-stop-consecutive-gated "$ESTOP" \
      --min-dissimilarity "$MIN_DISSIM" \
      --max-semantic-similarity "$MAX_SEM_SIM" \
      --embedding-model "$EMBEDDING_MODEL" \
      --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
      --max-concurrency "$PER_ISLAND_CONCURRENCY" \
      --sim-timeout "$SIM_TIMEOUT" \
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
      "${extra[@]}" \
      >"$log" 2>&1 &
    pids+=("$!")
    echo "  pid $!"
    sleep 2
  done
done

echo
echo "Launched ${#pids[@]} islands: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${RUN_ROOT}/*/logs/isl_*.log"
echo "  grep -hE '\\[iter [0-9]+\\].*(-> idx|rejected)' ${RUN_ROOT}/*/logs/isl_*.log | tail -20"
