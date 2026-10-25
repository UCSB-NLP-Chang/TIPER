#!/usr/bin/env bash
# Test set evaluation for brinks_02 idx9 + brinks_04 idx1,idx8 (val-top per island).
# 3 personas × 54 test tasks × 1 trial = 162 rollouts
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
LOG_DIR="${LOG_DIR:-data/persona_opt/test_eval_gpt54user/logs}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

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
  local save_to="gepa_opt/test_gpt54user/${pname}"
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

echo "brinks_02/04 val-top test: $N_TASKS tasks × 3 personas = $((N_TASKS * 3)) rollouts"
echo "Agent: $AGENT_LLM  User: $USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo

run_persona "valtop_brinks_02_idx9" \
  "data/persona_opt/test_personas_gpt54/g54_brinks_02_valtop_idx9.md" &
pid1=$!

run_persona "valtop_brinks_04_idx1" \
  "data/persona_opt/test_personas_gpt54/g54_brinks_04_valtop_idx1.md" &
pid2=$!

run_persona "valtop_brinks_04_idx8" \
  "data/persona_opt/test_personas_gpt54/g54_brinks_04_valtop_idx8.md" &
pid3=$!

echo "Launched: pids $pid1 $pid2 $pid3"
echo "Monitor: tail -f ${LOG_DIR}/valtop_brinks_0{2,4}_*.log"

wait $pid1 $pid2 $pid3

echo
echo "========== ALL DONE =========="
"$PY" -c "
import json, glob
root = 'data/simulations/gepa_opt/test_gpt54user'
for name in ['valtop_brinks_02_idx9', 'valtop_brinks_04_idx1', 'valtop_brinks_04_idx8']:
    fs = sorted(glob.glob(f'{root}/{name}/eval_*/*/results.json'))
    if not fs:
        fs = sorted(glob.glob(f'{root}/{name}/results.json'))
    if not fs:
        print(f'  >> {name:30s}  NO RESULTS')
        continue
    sims = json.load(open(fs[-1])).get('simulations', [])
    rewards = [r for r in [(s.get('reward_info') or {}).get('reward') for s in sims] if r is not None]
    n = len(rewards)
    avg = sum(rewards)/n if n else 0
    print(f'  >> {name:30s}  n={n:3d}  success={avg:.3f}  difficulty={1-avg:.3f}')
"
