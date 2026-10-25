#!/usr/bin/env bash
# Test-set certification for the ds4flash 4-domain genesis search.
#   telecom/retail/banking: fresh 3-trial runs on the split's held-out TEST set
#   airline: v5 split has no test -> reuse the 2-trial val evals from the search
#            (copied out of gepa_opt so search artifacts stay untouched) and
#            top up with trial 3 (--num-trials 3 --auto-resume fills only trial 2)
# Configs per domain: best persona (opt) + no-persona baseline (nop).
# banking best == no-persona (search verdict), so only nop is run there.
set -uo pipefail
cd "$(dirname "$0")/../.."

if [[ -z "${OPENAI_API_KEY:-}" && -f .env ]]; then
  export OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi

AGENT="openai/accounts/fireworks/models/deepseek-v4-flash-0731"
AGENT_ARGS='{"api_base":"http://127.0.0.1:4114/v1","api_key":"dummy"}'
USER_ARGS='{"reasoning_effort": "high"}'
TAU2=.venv/bin/tau2
RUN=data/persona_opt/pgepa_runs/genesis_ds4flash_4domain
OUT=ds4flash_test
CONC="${CONC:-6}"
TIMEOUT=7200

# same preamble eval_persona.py uses for --user-persona
PREAMBLE='CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.\n\n'

ids() { python3 -c "import json;print(' '.join(map(str,json.load(open('/tmp/ds4flash_cert/task_ids.json'))['$1'])))"; }

pids=()
launch() { # name domain trials persona_file [extra...]
  local name=$1 dom=$2 trials=$3 persona=$4; shift 4
  local cmd=("$TAU2" run --domain "$dom" --user user_simulator
    --user-llm gpt-5.4 --user-llm-args "$USER_ARGS"
    --agent-llm "$AGENT" --agent-llm-args "$AGENT_ARGS"
    --num-trials "$trials" --max-concurrency "$CONC"
    --save-to "${OUT}/${name}" --auto-resume
    --max-steps 120 --timeout "$TIMEOUT" "$@")
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
  echo "[launch] $name (dom=$dom trials=$trials persona=${persona:-none})"
  "${cmd[@]}" >"/tmp/ds4flash_cert/${name}.log" 2>&1 &
  pids+=($!)
}

# --- airline top-up: copy existing 2-trial val evals, then add trial 3 ---
mkdir -p "data/simulations/${OUT}"
for cfg in opt nop; do
  if [[ $cfg == opt ]]; then src=data/simulations/gepa_opt/isl_airline_g0_g10_k00_full_diff/eval_0001/airline
  else src=data/simulations/gepa_opt/isl_airline_g0_g00_seed/eval_0001/airline; fi
  dst="data/simulations/${OUT}/airline_${cfg}"
  [[ -f "$dst/results.json" ]] || { mkdir -p "$dst"; cp "$src/results.json" "$dst/results.json"; }
done

# --- telecom ---
launch telecom_opt telecom 3 "$RUN/telecom/isl_telecom_g0/exploration_best_persona.md" --task-ids $(ids telecom)
launch telecom_nop telecom 3 "" --task-ids $(ids telecom)
# --- retail ---
launch retail_opt retail 3 "$RUN/retail/isl_retail_g0/exploration_best_persona.md" --task-ids $(ids retail)
launch retail_nop retail 3 "" --task-ids $(ids retail)
# --- banking (best == no-persona) ---
launch banking_nop banking_knowledge 3 "" --retrieval-config bm25 --task-ids $(ids banking)
# --- airline top-up (trial 3 only) ---
launch airline_opt airline 3 "$RUN/airline/isl_airline_g0/exploration_best_persona.md" --task-ids $(ids airline)
launch airline_nop airline 3 "" --task-ids $(ids airline)

echo "launched ${#pids[@]} runs: ${pids[*]}"
wait
echo "ALL DONE"
