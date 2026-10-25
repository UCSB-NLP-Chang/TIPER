#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Test-set eval for the TERMINAL-OPTIMIZED cw_01 persona ("Queued-Up Customer",
# best idx4 from the islands_banking_cardworks_v2_terminal/isl_cw_01 search).
# 57 test tasks x 3 trials, terminal_use, conc 30 — same config as the other
# testmatrix_cardworks conditions so it's directly comparable to:
#   no_persona_terminal 0.637 / cw_01_seed_terminal 0.632 / cw_01_opt_terminal 0.635
# Uses a bash array for args (avoids the inline-nohup task-ids quoting bug).
set -u
cd "$REPO"

TAU2=.venv/bin/tau2
PY=.venv/bin/python

IDS=$($PY -c "
import json
d=json.load(open('experiments/persona_search/configs/banking_knowledge_split_v2.json'))
t=d['test']; ids=t['banking'] if isinstance(t,dict) else t
print(' '.join(ids))
")

PREAMBLE='CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.\n\n'

PERSONA_FILE=data/persona_opt/pgepa_runs/islands_banking_cardworks_v2_terminal/isl_cw_01/exploration_best_persona.md
PERSONA=$($PY -c "
import json
txt=open('$PERSONA_FILE').read()
preamble='$PREAMBLE'.replace('\\n','\n')
print(json.dumps({'custom_guidelines': preamble+txt}, ensure_ascii=False))
")

SAVE_TO=gepa_opt/testmatrix_cardworks/cw_01_topt_terminal/banking

args=()
args+=( --domain banking_knowledge --user user_simulator --user-llm gpt-5.4 \
        --user-llm-args '{"reasoning_effort":"high"}' \
        --agent-llm gpt-5.5 --agent-llm-args '{"reasoning_effort":"high"}' \
        --num-trials 3 --max-concurrency 30 --save-to "$SAVE_TO" --auto-resume \
        --retrieval-config terminal_use --max-steps 120 --timeout 600 \
        --task-ids $IDS --user-persona "$PERSONA" )

echo "[cw01-topt] @ $(date) — launching terminal-opt cw_01 test eval"
"$TAU2" run "${args[@]}" 2>&1 | tail -30
echo "[cw01-topt] @ $(date) — done"
