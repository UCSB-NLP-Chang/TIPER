#!/usr/bin/env bash
# Test-set cert for haiku-genesis best personas (airline + retail only).
# Uses the best persona from the haiku genesis search
# (data/persona_opt/pgepa_runs/genesis_haiku_3domain/<dom>/isl_<dom>_g0/
#  exploration_best_persona.md), written by parallel_gepa on search exit.
#
# nop baselines already exist (haiku_test/{airline,retail}_nop, 3 trials each)
# so we only run the opt (persona) arms here.
#
# airline: v5 val 30 tasks (no test split), 3 trials
# retail:  test 54 tasks, 3 trials
# Agent: vertex_ai/claude-haiku-4-5 (high) via :4115 vertex shim.
set -uo pipefail
cd "$(dirname "$0")/../.."

# --- env from .env ---
for k in OPENAI_API_KEY CRESTA_VERTEX_TOKEN CRESTA_VERTEX_PROXY_BASE VERTEXAI_PROJECT; do
  if [[ -z "${!k:-}" && -f .env ]]; then
    export "$k=$(grep -E "^${k}=" .env | head -1 | cut -d= -f2-)"
  fi
done
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"
# point at the local vertex RR shim
export CRESTA_VERTEX_PROXY_BASE="http://127.0.0.1:${VTX_PORT:-4115}"

AGENT="${AGENT_LLM:-vertex_ai/claude-haiku-4-5}"
AGENT_ARGS='{"reasoning_effort":"high"}'
USER_ARGS='{"reasoning_effort": "high"}'
TAU2=.venv/bin/tau2
RUN=data/persona_opt/pgepa_runs/genesis_haiku_3domain
OUT="${OUT:-haiku_genesis_test}"
CONC="${CONC:-6}"
TIMEOUT=7200

# shim liveness
pgrep -f vertex_rr_shim.py >/dev/null 2>&1 || { echo "ERROR: vertex_rr_shim not running" >&2; exit 1; }

ids() { python3 -c "import json;print(' '.join(map(str,json.load(open('/tmp/ds4flash_cert/task_ids.json'))['$1'])))"; }

launch() { # name domain persona_file taskset_key
  local name=$1 dom=$2 persona=$3 tset=$4
  local cmd=("$TAU2" run --domain "$dom" --user user_simulator
    --user-llm gpt-5.4 --user-llm-args "$USER_ARGS"
    --agent-llm "$AGENT" --agent-llm-args "$AGENT_ARGS"
    --num-trials 3 --max-concurrency "$CONC"
    --save-to "${OUT}/${name}" --auto-resume
    --max-steps 120 --timeout "$TIMEOUT")
  if [[ -n "$persona" ]]; then
    local pjson
    pjson=$(python3 - "$persona" <<'PYEOF'
import json, sys
pre = "CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.\n\n"
print(json.dumps({"custom_guidelines": pre + open(sys.argv[1]).read()}))
PYEOF
)
    cmd+=(--user-persona "$pjson")
  fi
  echo "[launch] $name  (persona=${persona:-nop}, tasks=$tset)"
  "${cmd[@]}" --task-ids $(ids "$tset") >"/tmp/haiku_genesis_test/${name}.log" 2>&1 &
}

mkdir -p /tmp/haiku_genesis_test

# opt arms only (haiku genesis best persona); nop baselines reused from haiku_test/
# DOMAINS env selects which domains to certify (default all 3).
DOMAINS="${DOMAINS:-airline retail telecom}"
for d in $DOMAINS; do
  case $d in
    airline) launch airline_opt airline "$RUN/airline/isl_airline_g0/exploration_best_persona.md" airline ;;
    retail)  launch retail_opt  retail  "$RUN/retail/isl_retail_g0/exploration_best_persona.md"  retail ;;
    telecom) launch telecom_opt  telecom  "$RUN/telecom/isl_telecom_g0/exploration_best_persona.md"  telecom ;;
    *) echo "skip unknown domain $d" >&2 ;;
  esac
done

echo "launched opt runs for: $DOMAINS (3 trials each). nop baselines reused from haiku_test/"
wait
echo "ALL DONE"
