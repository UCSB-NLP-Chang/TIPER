#!/usr/bin/env bash
# Test-set evaluation for DeepSeek v4-pro GEPA best personas.
# Runs best persona from each island on the test split (54 tasks) x 3 trials.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-.venv/bin/python}"
RUN_ROOT="data/persona_opt/pgepa_runs/islands_telecom_dsv4pro_v2"
SPLIT_FILE="experiments/persona_search/configs/telecom_island_split.json"
NUM_TRIALS="${NUM_TRIALS:-3}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"

AGENT_LLM="anthropic/accounts/fireworks/models/deepseek-v4-pro"
USER_TYPE="user_simulator"
USER_LLM="gpt-5.4"
EFFORT="high"
FIREWORKS_SHIM_PORT="${FIREWORKS_SHIM_PORT:-4114}"

export ANTHROPIC_API_BASE="http://127.0.0.1:${FIREWORKS_SHIM_PORT}"
export ANTHROPIC_API_KEY="dummy"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep OPENAI_API_KEY .env 2>/dev/null | cut -d= -f2)}"

# Extract test task IDs from split file
TEST_TASKS=($($PY -c "
import json
split = json.load(open('$SPLIT_FILE'))
for t in split['test']['telecom']:
    print(t)
"))
echo "Test set: ${#TEST_TASKS[@]} tasks x $NUM_TRIALS trials"

ROLEPLAY_PREAMBLE="CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.

"

run_eval() {
  local label="$1"
  local persona_json="$2"  # empty string = no persona
  local save_to="$3"

  echo ""
  echo "=========================================="
  echo "  $label -> $save_to"
  echo "=========================================="

  local cmd=(
    .venv/bin/tau2 run
    --domain telecom
    --user "$USER_TYPE"
    --user-llm "$USER_LLM"
    --user-llm-args "{\"reasoning_effort\": \"$EFFORT\"}"
    --agent-llm "$AGENT_LLM"
    --agent-llm-args "{}"
    --num-trials "$NUM_TRIALS"
    --max-concurrency "$MAX_CONCURRENCY"
    --save-to "$save_to"
    --auto-resume
    --task-ids "${TEST_TASKS[@]}"
  )

  if [[ -n "$persona_json" ]]; then
    cmd+=(--user-persona "$persona_json")
  fi

  "${cmd[@]}" 2>&1
}

# 1. No-persona baseline
run_eval "no_persona baseline" "" \
  "gepa_opt/dsv4pro_v2_test/no_persona"

# 2. Each island's best persona
for island in brinks_00 p0003; do
  TREE="${RUN_ROOT}/isl_ds_${island}/lineage_tree.json"
  STATE="${RUN_ROOT}/isl_ds_${island}/pgepa_state.json"

  if [[ ! -f "$TREE" ]]; then
    echo "SKIP $island: no lineage_tree.json"
    continue
  fi

  BEST_IDX=$($PY -c "import json; s=json.load(open('$STATE')); print(s.get('best_idx', 0))")
  PERSONA_TEXT=$($PY -c "
import json, sys
tree = json.load(open('$TREE'))
sys.stdout.write(tree['personas']['idx$BEST_IDX'])
")
  BEST_DIFF=$($PY -c "import json; s=json.load(open('$STATE')); print(f\"{s.get('best_agg', 0):.3f}\")")

  PERSONA_JSON=$(python3 -c "
import json, sys
text = sys.stdin.read()
print(json.dumps({'custom_guidelines': '''$ROLEPLAY_PREAMBLE''' + text}))
" <<< "$PERSONA_TEXT")

  run_eval "$island best (idx$BEST_IDX, diff=$BEST_DIFF)" "$PERSONA_JSON" \
    "gepa_opt/dsv4pro_v2_test/${island}_best"
done

# 3. Seed personas
for island in brinks_00 p0003; do
  TREE="${RUN_ROOT}/isl_ds_${island}/lineage_tree.json"
  if [[ ! -f "$TREE" ]]; then continue; fi

  SEED_TEXT=$($PY -c "
import json, sys
tree = json.load(open('$TREE'))
sys.stdout.write(tree['personas']['idx0'])
")

  SEED_JSON=$(python3 -c "
import json, sys
text = sys.stdin.read()
print(json.dumps({'custom_guidelines': '''$ROLEPLAY_PREAMBLE''' + text}))
" <<< "$SEED_TEXT")

  run_eval "$island seed (idx0)" "$SEED_JSON" \
    "gepa_opt/dsv4pro_v2_test/${island}_seed"
done

echo ""
echo "=== All test evals complete ==="
echo "Results in data/simulations/gepa_opt/dsv4pro_v2_test/"
