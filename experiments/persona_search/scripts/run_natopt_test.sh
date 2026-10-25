#!/usr/bin/env bash
# Test eval for naturalness-optimized personas (correct agent reasoning_effort).
# Runs 5 optimized personas + p03 seed on 54 test tasks × 1 trial.
# Uses same settings as run_test_gpt54.sh for comparability with GEPA baselines.
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-120}"
SIM_TIMEOUT="${SIM_TIMEOUT:-600}"
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/telecom_island_split.json}"
LOG_DIR="${LOG_DIR:-data/persona_opt/naturalness_opt/test_logs}"

mkdir -p "$LOG_DIR"

TEST_TASKS=$("$PY" -c "
import json
tasks = json.load(open('$SPLIT_FILE'))['test']['telecom']
print(' '.join(tasks))
")
N_TASKS=$(echo $TEST_TASKS | wc -w | tr -d ' ')

ROLEPLAY_PREAMBLE="CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.

"

run_persona() {
  local pname="$1"
  local pfile="$2"
  local save_to="naturalness_opt_test/${pname}"
  local log="${LOG_DIR}/${pname}.log"

  if [[ ! -f "$pfile" ]]; then
    echo "SKIP $pname: missing $pfile" >&2
    return 1
  fi

  local persona_text
  persona_text=$(cat "$pfile")
  local user_persona
  user_persona=$("$PY" -c "
import json, sys
text = sys.stdin.read()
preamble = '''$ROLEPLAY_PREAMBLE'''
print(json.dumps({'custom_guidelines': preamble + text}))
" <<< "$persona_text")

  local user_args
  user_args=$("$PY" -c "import json; print(json.dumps({'reasoning_effort': '$EFFORT'}))")

  echo "[$(date +%H:%M:%S)] START $pname ($N_TASKS tasks, concurrency=$MAX_CONCURRENCY)"

  "$PY" -m tau2.cli run \
    --domain telecom \
    --user "$USER_TYPE" \
    --user-llm "$USER_LLM" \
    --user-llm-args "$user_args" \
    --user-persona "$user_persona" \
    --agent-llm "$AGENT_LLM" \
    --agent-llm-args "$AGENT_LLM_ARGS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --max-steps "$MAX_STEPS" \
    --timeout "$SIM_TIMEOUT" \
    --save-to "$save_to" \
    --auto-resume \
    --task-ids $TEST_TASKS \
    >"$log" 2>&1

  echo "[$(date +%H:%M:%S)] DONE  $pname"
}

echo "=== Naturalness-optimized test eval ($N_TASKS tasks × $NUM_TRIALS trial) ==="
echo "Agent: $AGENT_LLM (reasoning=$EFFORT), User: $USER_LLM (reasoning=$EFFORT)"
echo

# p03 seed (no GEPA baseline exists)
run_persona "p03_seed" "data/persona_opt/naturalness_seed_p03.json" &
P1=$!

# 5 optimized personas — extract persona_text from best_persona.json
for seed in b00 b02 b03 b04 p03; do
  best="data/persona_opt/naturalness_opt/${seed}/best_persona.json"
  tmp="/tmp/natopt_${seed}_text.txt"
  "$PY" -c "import json; print(json.load(open('$best'))['persona_text'])" > "$tmp"
  run_persona "natopt_${seed}" "$tmp" &
done

wait
echo
echo "=== ALL DONE ==="

# Summary
"$PY" -u -c "
import json, glob, os

print()
print('Persona              N    Diff    Dur(s)  Msgs')
print('-' * 55)

# GEPA seed baselines (already correct)
gepa_map = {'b00': 'seed_brinks_00', 'b02': 'seed_brinks_02',
            'b03': 'seed_brinks_03', 'b04': 'seed_brinks_04'}
for nk, gk in sorted(gepa_map.items()):
    p = f'data/simulations/gepa_opt/test_gpt54user/{gk}/results.json'
    if os.path.exists(p):
        d = json.load(open(p))
        sims = d['simulations']
        rr = [s['reward_info']['reward'] for s in sims]
        diff = 1 - sum(rr)/len(rr)
        dur = sum(s.get('duration',0) for s in sims)/len(sims)
        msgs = sum(len(s.get('messages',[])) for s in sims)/len(sims)
        print(f'{nk}_seed (GEPA)     {len(rr):>4}  {diff:>6.3f}  {dur:>7.0f}  {msgs:>4.0f}')

# New test results
for p in sorted(glob.glob('data/simulations/naturalness_opt_test/*/results.json')):
    name = p.split('/')[-2]
    d = json.load(open(p))
    sims = d['simulations']
    rr = [s['reward_info']['reward'] for s in sims if s.get('reward_info')]
    if not rr: continue
    diff = 1 - sum(rr)/len(rr)
    dur = sum(s.get('duration',0) for s in sims)/len(sims)
    msgs = sum(len(s.get('messages',[])) for s in sims)/len(sims)
    print(f'{name:<20} {len(rr):>4}  {diff:>6.3f}  {dur:>7.0f}  {msgs:>4.0f}')
"
