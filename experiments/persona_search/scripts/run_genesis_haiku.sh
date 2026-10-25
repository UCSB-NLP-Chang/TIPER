#!/usr/bin/env bash
# Pure-genesis persona search from NO-PERSONA on 3 domains,
# targeting Claude Haiku 4.5 as the agent under test.
#
# Agent:  vertex_ai/claude-haiku-4-5  (reasoning_effort=high -> litellm native
#         thinking budget 4096; tau2's claude- prefix patch does not apply to
#         the vertex_ai/ prefix, but litellm 1.86.2 maps reasoning_effort
#         natively for vertex_ai Claude models).
#         via local vertex RR+backoff shim on :4115 (CRESTA_VERTEX_PROXY_BASE).
# User:   gpt-5.4 (high)  | Mutation: gpt-5.5 xhigh | Judge: gpt-5.2
#
# Anthropic prompt caching is enabled in llm_utils.py (3 breakpoints:
# system content-block + last tool + last conversation message), so repeated
# agent calls hit cache_read at ~10% input price. Verified ~60% agent-cost
# reduction on retail; telecom (more steps) expected 65-70%.
#
# Genesis mode = no --seed-persona: gen-0 evaluates the no-persona baseline and
# the proposer authors the first persona from those failure traces.
# Per domain, NUM_ISLANDS (default 1) pure-genesis islands run in parallel.
# Total concurrency ~= 3 domains x PER_ISLAND_CONCURRENCY (default 6) ~= 18.
#
# Prereq: the vertex RR+backoff shim must be listening on :4115
#   (pgrep -f vertex_rr_shim).  start it:
#   nohup python3 -u vertex_rr_shim.py --port 4115 &
#
# Usage:
#   bash experiments/persona_search/scripts/run_genesis_haiku.sh             # all 3 domains, 1 island
#   DOMAINS="telecom" bash .../run_genesis_haiku.sh            # single domain
#   PER_ISLAND_CONCURRENCY=8 DOMAINS=telecom bash ...          # single-domain, more parallel
#   RUN_TAG=haiku_smoke NUM_ISLANDS=1 ITERATIONS=2 bash ...    # smoke (use a fresh RUN_TAG)
#   (注意: 不要用 --dry-run 探测真实 RUN_DIR —— dry-run 会写 mock state
#    污染 run_dir, 之后 --resume 会秒退. 冒烟请换独立 RUN_TAG.)
set -uo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"

# --- load secrets from .env (parallel_gepa does not read .env itself) ---
load_env() {  # <key>
  grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-
}
[[ -f .env ]] || { echo "ERROR: no .env in repo root" >&2; exit 1; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then export OPENAI_API_KEY="$(load_env OPENAI_API_KEY)"; fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and not in .env}"

# Vertex credentials for the haiku agent. Allow command-line override (e.g. if
# the user exports a different CRESTA_VERTEX_TOKEN that also authorizes
# /proxy/vertex); otherwise take them from .env.
if [[ -z "${CRESTA_VERTEX_TOKEN:-}" ]];        then export CRESTA_VERTEX_TOKEN="$(load_env CRESTA_VERTEX_TOKEN)"; fi
if [[ -z "${VERTEXAI_PROJECT:-}" ]];           then export VERTEXAI_PROJECT="$(load_env VERTEXAI_PROJECT)"; fi
if [[ -z "${VERTEXAI_LOCATION:-}" ]];          then export VERTEXAI_LOCATION="$(load_env VERTEXAI_LOCATION)"; fi
: "${CRESTA_VERTEX_TOKEN:?CRESTA_VERTEX_TOKEN not set and not in .env}"
: "${VERTEXAI_PROJECT:?VERTEXAI_PROJECT not set and not in .env}"
: "${VERTEXAI_LOCATION:=us-east5}"

# --- Agent: claude-haiku-4-5 via local vertex shim on :4115 ---
VTX_PORT="${VTX_PORT:-4115}"
export CRESTA_VERTEX_PROXY_BASE="http://127.0.0.1:${VTX_PORT}"
AGENT_LLM="${AGENT_LLM:-vertex_ai/claude-haiku-4-5}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"

# Shim liveness check: the transparent proxy answers 405 on GET / (no do_GET
# route at root), 200/4xx on a real upstream path. pgrep is the reliable check.
if ! pgrep -f "vertex_rr_shim.py" >/dev/null 2>&1; then
  echo "ERROR: vertex_rr_shim.py not running" >&2
  echo "  start it: nohup python3 -u vertex_rr_shim.py --port ${VTX_PORT} &" >&2
  exit 1
fi
# Confirm the port is actually accepting connections.
if ! curl -s -o /dev/null -m 5 "http://127.0.0.1:${VTX_PORT}/" ; then
  echo "ERROR: nothing listening on :${VTX_PORT} (shim process exists but port closed)" >&2
  exit 1
fi
echo "vertex RR shim OK on :${VTX_PORT} (agent=$AGENT_LLM proxy=$CRESTA_VERTEX_PROXY_BASE)"

# --- Shared settings ---
NUM_ISLANDS="${NUM_ISLANDS:-1}"
# ITERATIONS / EARLY_STOP default per-domain in dom_cfg (all: 15 iters, early-stop 3)
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-8}"   # 3 域 x 6 = 18 总并发
# dev-proxy 20 req/min 是按 token 全局限; vertex shim RR+backoff 兜底, 但 haiku
# high reasoning 每 call 带 thinking budget 4096, 单 sim 慢, 留宽 sim timeout.
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
RUN_TAG="${RUN_TAG:-genesis_haiku_3domain}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"

DOMAINS="${DOMAINS:-telecom retail airline}"

# --- Per-domain settings (split / minibatch / edit% / iters / early-stop) ---
# 沿用 flash genesis 的 split 与超参; haiku 能力弱于 flash, 跑起来后按需调.
dom_cfg() {
  case "$1" in
    telecom)  SPLIT=experiments/persona_search/configs/telecom_island_split.json;          MB="${USER_MB:-10}"; EDIT="50-60%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    airline)  SPLIT=experiments/persona_search/configs/airline_alaska_split_v5.json;       MB="${USER_MB:-8}";  EDIT="25-35%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    retail)   SPLIT=experiments/persona_search/configs/retail_guitar_center_split_v4.json; MB="${USER_MB:-10}"; EDIT="40-50%"; ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    *) echo "unknown domain $1" >&2; return 1 ;;
  esac
}

echo "=== Genesis-from-no-persona search: agent=claude-haiku-4-5 (high) ==="
echo "  domains: $DOMAINS   islands/domain: $NUM_ISLANDS   iters: 15   early-stop: 3"
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

    echo "[$dom/g$i] genesis (no seed persona)  split=$SPLIT mb=$MB edit=$EDIT iters=$ITERS early_stop=$ESTOP"
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
