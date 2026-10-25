#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Run terminal_use retrieval for the OPT personas of cw_01 / cw_03 / cw_04 on the
# same 57-task test set x 3 trials. cw_00 opt_terminal already done (0.659,
# +0.021 vs no_persona_terminal 0.637 — gain scaled UP vs bm25 +0.012). This
# checks whether the opt gain holds across the other islands under the stronger
# agent retrieval. no_persona_terminal baseline (0.637) already exists.
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

run_cond () {
  local name="$1" persona_file="$2"
  local save="gepa_opt/testmatrix_cardworks/${name}/banking"
  local persona=$($PY -c "
import json
txt=open('$persona_file').read()
preamble='$PREAMBLE'.replace('\\n','\n')
print(json.dumps({'custom_guidelines': preamble+txt}, ensure_ascii=False))
")
  echo "[cwterm2] @ $(date) — running $name (terminal_use, persona=$persona_file)"
  "$TAU2" run \
    --domain banking_knowledge --user user_simulator --user-llm gpt-5.4 \
    --user-llm-args '{"reasoning_effort":"high"}' \
    --agent-llm gpt-5.5 --agent-llm-args '{"reasoning_effort":"high"}' \
    --num-trials 3 --max-concurrency 30 --save-to "$save" --auto-resume \
    --retrieval-config terminal_use --max-steps 120 --timeout 600 \
    --task-ids $IDS --user-persona "$persona" 2>&1 | tail -20
  echo "[cwterm2] @ $(date) — $name done"
}

run_cond "cw_01_opt_terminal" /tmp/cw_terminal/cw_01_opt.md
run_cond "cw_03_opt_terminal" /tmp/cw_terminal/cw_03_opt.md
run_cond "cw_04_opt_terminal" /tmp/cw_terminal/cw_04_opt.md

echo "[cwterm2] @ $(date) — cw_01/03/04 opt terminal 全部完成"
