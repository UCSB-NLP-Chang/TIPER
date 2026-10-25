#!/usr/bin/env bash
# Test-set cert with claude-haiku-4-5 (high reasoning) as the agent under test,
# on airline/retail/telecom. Mirrors the ds4flash cert exactly:
#   - same task splits (airline = v5 val 30 tasks; retail/telecom = test 54)
#   - same personas (ds4flash genesis best) + no-persona baseline
#   - 3 trials, user gpt-5.4 high
# Agent route: vertex_ai/claude-haiku-4-5 via dev-proxy /proxy/vertex
#   (CRESTA_VERTEX_TOKEN from .env; cresta_vertex.py auto-patches litellm).
#   Verified 2026-08-12: reasoning_effort=high natively maps to thinking
#   budget in litellm 1.86.2 for vertex_ai/claude-*; thinking shows in response.
#   NOTE: same dev-proxy 20 req/min per-token global cap as the fireworks route.
set -uo pipefail
cd "$(dirname "$0")/../.."

# --- env from .env (tau2 CLI children inherit) ---
for k in OPENAI_API_KEY CRESTA_VERTEX_TOKEN CRESTA_VERTEX_PROXY_BASE VERTEXAI_PROJECT; do
  if [[ -z "${!k:-}" && -f .env ]]; then
    export "$k=$(grep -E "^${k}=" .env | head -1 | cut -d= -f2-)"
  fi
done
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"   # Claude lives in us-east5

AGENT="${AGENT_LLM:-vertex_ai/claude-haiku-4-5}"
AGENT_ARGS='{"reasoning_effort":"high"}'
USER_ARGS='{"reasoning_effort": "high"}'
TAU2=.venv/bin/tau2
RUN=data/persona_opt/pgepa_runs/genesis_ds4flash_4domain
OUT="${OUT:-haiku_test}"
CONC="${CONC:-6}"
TIMEOUT=7200

# sanity: haiku reachable via vertex route
if ! .venv/bin/python - <<'EOF' >/dev/null 2>&1
import litellm, src.tau2.utils.cresta_vertex
litellm.completion(model='vertex_ai/claude-haiku-4-5',
    messages=[{'role':'user','content':'hi'}], max_tokens=8192,
    reasoning_effort='high', num_retries=0)
EOF
then echo "ERROR: vertex haiku probe failed" >&2; exit 1; fi
echo "vertex haiku OK (agent=$AGENT, effort=high)"

ids() { python3 -c "import json;print(' '.join(map(str,json.load(open('/tmp/ds4flash_cert/task_ids.json'))['$1'])))"; }

launch() { # name domain persona_file
  local name=$1 dom=$2 persona=$3; shift 3
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
  echo "[launch] $name"
  "${cmd[@]}" --task-ids $(ids "$1") >"/tmp/ds4flash_cert/${OUT}_${name}.log" 2>&1 &
}

# airline has no test split -> use val 30 (fresh 3 trials; agent differs from flash era)
launch airline_opt airline "$RUN/airline/isl_airline_g0/exploration_best_persona.md" airline
launch airline_nop airline "" airline
launch retail_opt retail "$RUN/retail/isl_retail_g0/exploration_best_persona.md" retail
launch retail_nop retail "" retail
launch telecom_opt telecom "$RUN/telecom/isl_telecom_g0/exploration_best_persona.md" telecom
launch telecom_nop telecom "" telecom

echo "launched 6 runs"
wait
echo "ALL DONE"
