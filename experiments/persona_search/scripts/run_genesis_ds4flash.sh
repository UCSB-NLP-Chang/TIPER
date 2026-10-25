#!/usr/bin/env bash
# Pure-genesis persona search from NO-PERSONA on all 4 domains,
# targeting DeepSeek v4 Flash as the agent under test.
#
# Agent:  openai/accounts/fireworks/models/deepseek-v4-flash-0731
#         via local openai->fireworks proxy on :4114 (OpenAI protocol, so the
#         $defs schema bug of the fireworks *Anthropic* endpoint does not apply;
#         verified: tool calls + reasoning_effort accepted).
# User:   gpt-5.4 (high)  | Mutation: gpt-5.5 xhigh | Judge: gpt-5.2
#
# Genesis mode = no --seed-persona: gen-0 evaluates the no-persona baseline and
# the proposer authors the first persona from those failure traces.
# Per domain, NUM_ISLANDS (default 1) pure-genesis islands run in parallel.
# Total concurrency ~= 4 domains x PER_ISLAND_CONCURRENCY (default 6) ~= 24.
#
# Prereq: the openai->fireworks proxy must be listening on :4114
#   (ps aux | grep openai_to_fireworks_proxy). It carries the Cresta token in
#   its own env; nothing to export here.
#
# Usage:
#   bash experiments/persona_search/scripts/run_genesis_ds4flash.sh            # all 4 domains, 1 island each
#   DOMAINS="telecom retail" bash .../run_genesis_ds4flash.sh   # subset
#   PER_ISLAND_CONCURRENCY=15 DOMAINS=telecom bash ...          # 单域时加大并发
#   RUN_TAG=dsflash_smoke NUM_ISLANDS=1 ITERATIONS=2 bash ...   # smoke
#   (注意: 不要用 --dry-run 探测真实 RUN_DIR —— dry-run 会写 mock state
#    污染 run_dir, 之后 --resume 会秒退. 冒烟请换独立 RUN_TAG.)
set -uo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"

# --- OpenAI key for user/mutation/judge/embeddings (parallel_gepa 不读 .env) ---
if [[ -z "${OPENAI_API_KEY:-}" && -f .env ]]; then
  export OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and not in .env}"

# --- Agent: deepseek-v4-flash via local fireworks proxy (OpenAI protocol) ---
FW_PORT="${FW_PORT:-4114}"
AGENT_LLM="${AGENT_LLM:-openai/accounts/fireworks/models/deepseek-v4-flash-0731}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"api_base\":\"http://127.0.0.1:${FW_PORT}/v1\",\"api_key\":\"dummy\"}}"

if ! curl -s -m 10 "http://127.0.0.1:${FW_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer dummy" \
    -d "{\"model\":\"accounts/fireworks/models/deepseek-v4-flash-0731\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
    | grep -q '"content"'; then
  echo "ERROR: fireworks proxy not answering on :${FW_PORT}" >&2
  echo "  start it: nohup python3 -u openai_to_fireworks_proxy.py --port ${FW_PORT} &" >&2
  exit 1
fi
echo "fireworks proxy OK on :${FW_PORT} (agent=$AGENT_LLM)"

# --- Shared settings ---
NUM_ISLANDS="${NUM_ISLANDS:-1}"
# ITERATIONS / EARLY_STOP 不设全局默认 —— 默认值在 dom_cfg 里按域给
# (所有域: 15 iters; early-stop: telecom/airline/retail=3, banking=6)
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-6}"   # 4 域 x 6 = 24 总并发
# agent 调用全挤在 :4114 shim 的 19 req/min 全局限流上, 24 并发下每 turn 排队
# ~75s; 默认 600s sim timeout 会大面积 timeout (实测 banking 30/30 全超).
SIM_TIMEOUT="${SIM_TIMEOUT:-7200}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
# 用户在命令行给的全局覆盖（per-domain dom_cfg 会用这些，未给则用各域默认）
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
RUN_TAG="${RUN_TAG:-genesis_ds4flash_4domain}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"

DOMAINS="${DOMAINS:-telecom retail airline banking}"

# --- Per-domain settings (split / minibatch / edit% / retrieval / iters / early-stop) ---
dom_cfg() {
  case "$1" in
    telecom)  SPLIT=experiments/persona_search/configs/telecom_island_split.json;              MB="${USER_MB:-10}"; EDIT="50-60%"; RETR="";      ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    airline)  SPLIT=experiments/persona_search/configs/airline_alaska_split_v5.json;           MB="${USER_MB:-8}";  EDIT="25-35%"; RETR="";      ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    retail)   SPLIT=experiments/persona_search/configs/retail_guitar_center_split_v4.json;     MB="${USER_MB:-10}"; EDIT="40-50%"; RETR="";      ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-3}" ;;
    banking)  SPLIT=experiments/persona_search/configs/banking_knowledge_split_v2.json;        MB="${USER_MB:-5}";  EDIT="40-50%"; RETR="bm25";  ITERS="${USER_ITERS:-15}"; ESTOP="${USER_ESTOP:-6}" ;;
    *) echo "unknown domain $1" >&2; return 1 ;;
  esac
}

echo "=== Genesis-from-no-persona search: agent=deepseek-v4-flash-0731 ==="
echo "  domains: $DOMAINS   islands/domain: $NUM_ISLANDS   iters: 15 (all domains)"
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
    [[ -n "$RETR" ]] && extra+=(--retrieval-config "$RETR")

    echo "[$dom/g$i] genesis (no seed persona)  split=$SPLIT mb=$MB edit=$EDIT retr=${RETR:-none} iters=$ITERS early_stop=$ESTOP"
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
