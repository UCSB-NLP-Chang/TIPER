#!/usr/bin/env bash
# No-persona baseline on val + test splits (gpt-5.4 agent + gpt-5.4 user).
# 2 splits × 1 trial = 30 (val) + 54 (test) = 84 rollouts
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
LOG_DIR="${LOG_DIR:-data/persona_opt/nopersona_gpt54/logs}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

mkdir -p "$LOG_DIR"

USER_ARGS=$("$PY" -c "import json; print(json.dumps({'reasoning_effort': '$EFFORT'}))")

run_split() {
  local split="$1"

  local tasks
  tasks=$("$PY" -c "
import json
tasks = json.load(open('$SPLIT_FILE'))['${split}']['telecom']
print(' '.join(tasks))
")
  local n_tasks
  n_tasks=$(echo $tasks | wc -w | tr -d ' ')

  local save_to="gepa_opt/nopersona_gpt54/${split}"
  local log="${LOG_DIR}/nopersona_${split}.log"

  echo "[$(date +%H:%M:%S)] START ${split} (${n_tasks} tasks, concurrency=${MAX_CONCURRENCY})"

  "$PY" -m tau2.cli run \
    --domain telecom \
    --user "$USER_TYPE" \
    --user-llm "$USER_LLM" \
    --user-llm-args "$USER_ARGS" \
    --agent-llm "$AGENT_LLM" \
    --agent-llm-args "$AGENT_LLM_ARGS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --max-steps "$MAX_STEPS" \
    --timeout "$SIM_TIMEOUT" \
    --save-to "$save_to" \
    --auto-resume \
    --task-ids $tasks \
    >"$log" 2>&1

  echo "[$(date +%H:%M:%S)] DONE  ${split}"

  "$PY" -c "
import json, glob
fs = sorted(glob.glob('data/simulations/${save_to}/eval_*/*/results.json'))
if not fs:
    fs = sorted(glob.glob('data/simulations/${save_to}/results.json'))
if not fs:
    print('  >> no results found'); exit()
sims = json.load(open(fs[-1])).get('simulations', [])
rewards = [r for r in [(s.get('reward_info') or {}).get('reward') for s in sims] if r is not None]
n = len(rewards)
avg = sum(rewards)/n if n else 0
print(f'  >> ${split}: n={n}  success={avg:.3f}  difficulty={1-avg:.3f}')
"
}

echo "No-persona baseline: val(30) + test(54) = 84 rollouts"
echo "Agent: $AGENT_LLM  User: $USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "Concurrency: $MAX_CONCURRENCY"
echo

# Run val and test in parallel
run_split "val" &
pid_val=$!

run_split "test" &
pid_test=$!

echo "Launched: val(pid=$pid_val) test(pid=$pid_test)"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/nopersona_val.log"
echo "  tail -f ${LOG_DIR}/nopersona_test.log"

wait $pid_val $pid_test

echo
echo "========== ALL DONE =========="
echo "Results: ls data/simulations/gepa_opt/nopersona_gpt54/"
