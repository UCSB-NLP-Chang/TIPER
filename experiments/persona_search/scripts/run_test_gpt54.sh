#!/usr/bin/env bash
# Test set evaluation for gpt-5.4 user island experiment.
# Runs seed+best pairs sequentially, 2 at a time (concurrency 20 each).
# brinks_00 last (no mutated best, seed only).
# 9 personas × 54 test tasks × 1 trial = 486 rollouts
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

# Extract test task IDs
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

run_pair() {
  local seed_name="$1" seed_file="$2" best_name="$3" best_file="$4"

  echo
  echo "========== PAIR: $seed_name + $best_name =========="

  run_persona "$seed_name" "$seed_file" &
  local pid1=$!

  if [[ -n "$best_name" && -n "$best_file" ]]; then
    run_persona "$best_name" "$best_file" &
    local pid2=$!
    echo "  pids: $pid1 $pid2"
    wait $pid1 $pid2
  else
    echo "  pid: $pid1 (seed only)"
    wait $pid1
  fi

  echo "========== PAIR DONE: $seed_name + $best_name =========="

  # Print results for this pair immediately
  "$PY" -c "
import json, glob
root = 'data/simulations/gepa_opt/test_gpt54user'
for name in ['$seed_name', '$best_name']:
    if not name: continue
    fs = sorted(glob.glob(f'{root}/{name}/eval_*/*/results.json'))
    if not fs: continue
    sims = json.load(open(fs[-1])).get('simulations', [])
    rewards = [(s.get('reward_info') or {}).get('reward') for s in sims]
    rewards = [r for r in rewards if r is not None]
    n = len(rewards)
    avg = sum(rewards)/n if n else 0
    infra = sum(1 for s in sims if not any(r is not None for r in [(s.get('reward_info') or {}).get('reward')]))
    print(f'  >> {name:30s}  n={n:3d}  success={avg:.3f}  difficulty={1-avg:.3f}  infra={infra}')
"
}

echo "Test evaluation: $N_TASKS tasks × $NUM_TRIALS trial × 9 personas = $((N_TASKS * NUM_TRIALS * 9)) rollouts"
echo "Agent: $AGENT_LLM  User: $USER_TYPE/$USER_LLM (effort=$EFFORT)"
echo "Concurrency: $MAX_CONCURRENCY per persona (2 parallel = $((MAX_CONCURRENCY * 2)) total)"
echo

# Group 1: brinks_03 (best=0.500, most interesting)
run_pair \
  "seed_brinks_03" "data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md" \
  "best_brinks_03_idx8" "data/persona_opt/test_personas_gpt54/g54_brinks_03_best_idx8.md"

# Group 2: brinks_02 (best=0.200, nice 3-level chain)
run_pair \
  "seed_brinks_02" "data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md" \
  "best_brinks_02_idx10" "data/persona_opt/test_personas_gpt54/g54_brinks_02_best_idx10.md"

# Group 3: brinks_01 (best=0.200)
run_pair \
  "seed_brinks_01" "data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md" \
  "best_brinks_01_idx2" "data/persona_opt/test_personas_gpt54/g54_brinks_01_best_idx2.md"

# Group 4: brinks_04 (best=0.167)
run_pair \
  "seed_brinks_04" "data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md" \
  "best_brinks_04_idx4" "data/persona_opt/test_personas_gpt54/g54_brinks_04_best_idx4.md"

# Group 5: brinks_00 (seed only, no mutated beat)
run_pair \
  "seed_brinks_00" "data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md" \
  "" ""

echo
echo "========== ALL DONE =========="
echo "Results: ls data/simulations/gepa_opt/test_gpt54user/"

# Aggregate
echo
echo "=== SUMMARY ==="
"$PY" -c "
import json, glob, os
root = 'data/simulations/gepa_opt/test_gpt54user'
for d in sorted(glob.glob(f'{root}/*/eval_*/*/results.json')):
    parts = d.split('/')
    pname = parts[-4]
    sims = json.load(open(d)).get('simulations', [])
    rewards = [(s.get('reward_info') or {}).get('reward') for s in sims]
    rewards = [r for r in rewards if r is not None]
    n = len(rewards)
    avg = sum(rewards)/n if n else 0
    difficulty = 1 - avg
    print(f'{pname:30s}  n={n:3d}  success={avg:.3f}  difficulty={difficulty:.3f}')
"
