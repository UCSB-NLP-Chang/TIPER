#!/usr/bin/env bash
# Test-set certification for fail² GEPA: evaluate top2 + seed on 54 test tasks × 3 trials.
# Seed already has 1 trial → runs 2 more; top2 candidates run 3 fresh trials.
# Computes avg@3 and pass@3 per task.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-.venv/bin/python}"
STATE="data/persona_opt/pgepa_runs/fail2_seed0_gpt54user/f2_brinks_00/pgepa_state.json"
SPLIT="experiments/persona_search/configs/telecom_island_split_fail2.json"
SEED_FILE="data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
OUT_BASE="gepa_opt/test_f2_cert"

AGENT_LLM="gpt-5.4"
AGENT_LLM_ARGS='{"reasoning_effort":"high"}'
USER_TYPE="user_simulator"
USER_LLM="gpt-5.4"
EFFORT="high"
MAX_CONCURRENCY=15
SIM_TIMEOUT=600
MAX_STEPS=100
MAX_INFRA_RETRIES=2

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

# Extract test task IDs
TEST_TASKS=$("$PY" -c "
import json
split = json.load(open('$SPLIT'))
for t in split['test']['telecom']:
    print(t)
")

# Get top2 candidates by val agg
TOP2=$("$PY" -c "
import json
state = json.load(open('$STATE'))
cands = state.get('candidates', [])
cands.sort(key=lambda c: c.get('agg', 0), reverse=True)
for c in cands[:2]:
    print(c['idx'])
")

echo "=== Test-set certification ==="
echo "Test tasks: $(echo "$TEST_TASKS" | wc -l | tr -d ' ')"
echo "Top 2 candidates: $TOP2"
echo

run_eval() {
    local label="$1"
    local persona_file="$2"
    local num_trials="$3"
    local save_to="${OUT_BASE}/${label}"

    local persona_text
    persona_text=$(cat "$persona_file")

    echo "[$label] Running $num_trials trials on test set..."
    "$PY" -c "
import sys, json, subprocess
sys.path.insert(0, 'experiments/persona_search/scripts')
from types import SimpleNamespace
from eval_persona import _run_domain_with_retry

args = SimpleNamespace(
    user_type='$USER_TYPE',
    user_llm='$USER_LLM',
    effort='$EFFORT',
    agent_llm='$AGENT_LLM',
    agent_llm_args='$AGENT_LLM_ARGS',
    num_trials=$num_trials,
    max_concurrency=$MAX_CONCURRENCY,
    max_steps=$MAX_STEPS,
    sim_timeout=$SIM_TIMEOUT,
    max_infra_retries=$MAX_INFRA_RETRIES,
    use_bedrock=False,
    persona_text='''$(echo "$persona_text" | sed "s/'/'\\\\''/g")''',
    retrieval_config='bm25',
)

task_ids = '''$TEST_TASKS'''.strip().split('\n')
_run_domain_with_retry(args, 'telecom', task_ids, '$save_to')
print(f'[$label] Done. Results at data/simulations/$save_to/results.json')
"
}

# 1. Seed: already has 1 trial, run 2 more (auto-resume will skip existing)
echo "--- Seed (brinks_00) ---"
# Copy existing seed test results if not already in cert dir
SEED_SAVE="${OUT_BASE}/seed_brinks_00"
SEED_EXISTING="data/simulations/gepa_opt/test_gpt54user/seed_brinks_00"
SEED_CERT_DIR="data/simulations/${SEED_SAVE}"
if [[ -f "${SEED_EXISTING}/results.json" ]] && [[ ! -f "${SEED_CERT_DIR}/results.json" ]]; then
    mkdir -p "$SEED_CERT_DIR"
    cp "${SEED_EXISTING}/results.json" "${SEED_CERT_DIR}/results.json"
    echo "  Copied existing 1-trial seed results to cert dir"
fi
run_eval "seed_brinks_00" "$SEED_FILE" 3

# 2. Top 2 candidates
for idx in $TOP2; do
    echo
    echo "--- Candidate idx${idx} ---"
    # Extract persona text from state
    PERSONA_TMP=$(mktemp /tmp/persona_idx${idx}_XXXX.md)
    "$PY" -c "
import json
state = json.load(open('$STATE'))
for c in state['candidates']:
    if c['idx'] == $idx:
        print(c['persona_text'])
        break
" > "$PERSONA_TMP"
    run_eval "cand_idx${idx}" "$PERSONA_TMP" 3
    rm -f "$PERSONA_TMP"
done

echo
echo "=== All evaluations complete ==="
echo "Analyze with:"
echo "  $PY experiments/persona_search/scripts/analyze_test_cert_f2.py"
