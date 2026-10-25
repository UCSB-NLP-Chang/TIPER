#!/usr/bin/env bash
# Monitor natopt_v2 optimization (PID from $1), then run test eval for all 5 seeds.
# Usage: nohup bash experiments/persona_search/scripts/monitor_and_test_v2.sh 22777 > /tmp/natopt_v2_monitor.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."

OPTIM_PID="${1:?Usage: $0 <optim_pid>}"
PY=.venv/bin/python
SPLIT="experiments/persona_search/configs/telecom_island_split_fail2.json"
TEST_TASKS=$("$PY" -c "import json; print(' '.join(json.load(open('$SPLIT'))['test']['telecom']))")
N_TASKS=$(echo $TEST_TASKS | wc -w | tr -d ' ')
NUM_TRIALS=3
MAX_CONCURRENCY=5
AGENT_LLM="gpt-5.4"
USER_LLM="gpt-5.4"
AGENT_LLM_ARGS='{"reasoning_effort":"high"}'
USER_LLM_ARGS='{"reasoning_effort":"high"}'
LOG=/tmp/natopt_v2_monitor.log

echo "[monitor] Watching PID $OPTIM_PID, will test when done"
echo "[monitor] Test: $N_TASKS tasks × $NUM_TRIALS trials"

# Wait for optimization to finish
while kill -0 "$OPTIM_PID" 2>/dev/null; do
    echo "[monitor] $(date '+%H:%M') optim PID $OPTIM_PID still running"
    sleep 1800
done
echo "[monitor] $(date '+%H:%M') optim finished"
sleep 5

# Show optimization results
echo ""
echo "=== OPTIMIZATION RESULTS ==="
"$PY" -c "
import json, os
for seed in ['b00','b02','b03','b04','p03']:
    h = f'data/persona_opt/naturalness_opt_v2/{seed}/history.json'
    b = f'data/persona_opt/naturalness_opt_v2/{seed}/best_persona.json'
    if os.path.exists(h):
        hist = json.load(open(h))
        seed_jsd = hist[0].get('jsd', 0)
        best = json.load(open(b)) if os.path.exists(b) else {}
        best_jsd = best.get('jsd', seed_jsd)
        accepted = sum(1 for x in hist if x.get('type') == 'accepted')
        print(f'  {seed}: JSD {seed_jsd:.4f} -> {best_jsd:.4f} (d={best_jsd-seed_jsd:+.4f}) accepted={accepted}/{len(hist)-1}')
"

# Run test eval for nat-opt personas only (seeds already have ev2v4_cert data)
echo ""
echo "=== RUNNING TEST EVAL ==="
for seed in b00 b02 b03 b04 p03; do
    BEST="data/persona_opt/naturalness_opt_v2/${seed}/best_persona.json"
    SAVE_TO="naturalness_opt_v2/${seed}/test"

    if [ ! -f "$BEST" ]; then
        echo "[test] SKIP $seed: no best_persona.json"
        continue
    fi

    # Check if test already fully done
    if [ -f "data/simulations/${SAVE_TO}/results.json" ]; then
        DONE=$("$PY" -c "
import json
d=json.load(open('data/simulations/${SAVE_TO}/results.json'))
rr=[s for s in d['simulations'] if s.get('reward_info')]
expected=$N_TASKS*$NUM_TRIALS
print('yes' if len(rr)>=expected*0.9 else 'no')
")
        if [ "$DONE" = "yes" ]; then
            echo "[test] SKIP $seed: test results complete"
            continue
        fi
        echo "[test] $seed: partial results, will auto-resume"
    fi

    PERSONA_TEXT=$("$PY" -c "import json; print(json.load(open('$BEST'))['persona_text'])")
    USER_PERSONA=$("$PY" -c "
import json, sys
text = sys.stdin.read()
print(json.dumps({'custom_guidelines': text, 'reveal_all_info': False, 'preset': None, 'use_preset_from_task': False}))
" <<< "$PERSONA_TEXT")

    echo "[test] $(date '+%H:%M') START $seed ($N_TASKS tasks × $NUM_TRIALS trials)"

    "$PY" -m tau2.cli run \
        --domain telecom \
        --task-split base \
        --agent llm_agent \
        --agent-llm "$AGENT_LLM" \
        --agent-llm-args "$AGENT_LLM_ARGS" \
        --user user_simulator \
        --user-llm "$USER_LLM" \
        --user-llm-args "$USER_LLM_ARGS" \
        --user-persona "$USER_PERSONA" \
        --num-trials "$NUM_TRIALS" \
        --max-steps 120 \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-to "$SAVE_TO" \
        --auto-resume \
        --task-ids $TEST_TASKS \
        2>&1 | tail -5

    echo "[test] $(date '+%H:%M') DONE $seed"
done

# Summary
echo ""
echo "=== TEST RESULTS ==="
"$PY" -c "
import json, os, glob
print(f'{\"Seed\":<6} {\"N\":>4} {\"Diff\":>8} {\"Type\":<6}')
print('-' * 30)
for seed in ['b00','b02','b03','b04','p03']:
    p = f'data/simulations/naturalness_opt_v2/{seed}/test/results.json'
    if os.path.exists(p):
        d = json.load(open(p))
        sims = d['simulations']
        rr = [s['reward_info']['reward'] for s in sims if s.get('reward_info')]
        diff = 1 - sum(rr)/len(rr) if rr else 0
        print(f'{seed:<6} {len(rr):>4} {diff:>8.3f} opt')
    else:
        print(f'{seed:<6}  —      —    (missing)')
"

echo ""
echo "[monitor] ALL DONE @ $(date '+%H:%M')"
