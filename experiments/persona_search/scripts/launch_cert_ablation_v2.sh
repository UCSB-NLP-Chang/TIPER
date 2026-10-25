#!/usr/bin/env bash
# Cert@3 for ablation v2 experiments (telecom, enriched). Compare best persona test
# difficulty against the full judge+gate baseline (enriched_seed0 cert).
#
# Usage:
#   EXPS=no_gate_gpt54 bash experiments/persona_search/scripts/launch_cert_ablation_v2.sh
#   EXPS="no_judge_ds4flash no_gate_ds4flash" bash experiments/persona_search/scripts/launch_cert_ablation_v2.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"

if [[ -z "${OPENAI_API_KEY:-}" && -f .env ]]; then
  export OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"

RUN_ROOT="data/persona_opt/pgepa_runs/enriched_ablation_v2"
OUT_BASE="gepa_opt/test_ablation_v2"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

SPLIT="experiments/persona_search/configs/telecom_island_split_fail2.json"
FW_PORT="${FW_PORT:-4114}"

DSFLASH_AGENT_LLM="openai/accounts/fireworks/models/deepseek-v4-flash-0731"
DSFLASH_AGENT_ARGS='{"api_base":"http://127.0.0.1:'"${FW_PORT}"'/v1","api_key":"dummy"}'
GPT54_AGENT_LLM="gpt-5.4"
GPT54_AGENT_ARGS='{"reasoning_effort":"high"}'

EXPS="${EXPS:-no_gate_gpt54 no_judge_ds4flash no_gate_ds4flash}"

launch() {
  local name="$1" agent_llm="$2" agent_args="$3"
  local state="$RUN_ROOT/$name/pgepa_state.json"
  local log="$LOG_DIR/cert_${name}.log"
  [[ -f "$state" ]] || { echo "[$name] missing state $state" >&2; return 1; }
  echo "[$name] cert@3 agent=$agent_llm -> $OUT_BASE/$name"
  nohup "$PY" -u experiments/persona_search/scripts/run_test_cert_enriched.py \
    --state "$state" \
    --split "$SPLIT" \
    --out-base "$OUT_BASE/$name" \
    --top-k 3 --num-trials 3 \
    --agent-llm "$agent_llm" --agent-llm-args "$agent_args" \
    --user-type user_simulator --user-llm gpt-5.4 --effort high \
    --max-concurrency 8 --max-steps 120 --sim-timeout 7200 \
    --max-infra-retries 2 \
    >"$log" 2>&1 &
  echo "  pid $! log=$log"
}

for exp in $EXPS; do
  case "$exp" in
    no_judge_ds4flash) launch "$exp" "$DSFLASH_AGENT_LLM" "$DSFLASH_AGENT_ARGS" ;;
    no_gate_gpt54)     launch "$exp" "$GPT54_AGENT_LLM"   "$GPT54_AGENT_ARGS" ;;
    no_gate_ds4flash)  launch "$exp" "$DSFLASH_AGENT_LLM" "$DSFLASH_AGENT_ARGS" ;;
    *) echo "unknown exp: $exp" >&2; exit 1 ;;
  esac
done
echo
echo "Monitor: tail -f $LOG_DIR/cert_*.log"
