#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Wait for the current testmatrix chain to finish, then run TWO extra conditions
# on terminal_use retrieval: cw_00 seed + cw_00 opt, on the same 57-task test set
# x 3 trials. Lets us see whether the cw_00 personas still beat no_persona when the
# agent uses the stronger terminal_use retrieval (no_persona_terminal diff=0.63 vs
# bm25 0.73 → terminal gives the agent +0.10 ASR, less headroom for personas).
set -u
cd "$REPO"

TAU2=.venv/bin/tau2
PY=.venv/bin/python

echo "[cw00term] @ $(date) — 等当前 testmatrix chain 退出"
while pgrep -f 'run_testmatrix_chain' | grep -qv grep; do
  sleep 30
done
echo "[cw00term] chain 已退出 @ $(date) — 开始 cw_00 terminal_use 两条件"

# 57 test task ids
IDS=$($PY -c "
import json
d=json.load(open('experiments/persona_search/configs/banking_knowledge_split_v2.json'))
t=d['test']; ids=t['banking'] if isinstance(t,dict) else t
print(' '.join(ids))
")

# method-actor preamble (identical to eval_persona.py / run_testmatrix_chain.sh)
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
  echo "[cw00term] @ $(date) — running $name (terminal_use, persona=$persona_file)"
  "$TAU2" run \
    --domain banking_knowledge --user user_simulator --user-llm gpt-5.4 \
    --user-llm-args '{"reasoning_effort":"high"}' \
    --agent-llm gpt-5.5 --agent-llm-args '{"reasoning_effort":"high"}' \
    --num-trials 3 --max-concurrency 30 --save-to "$save" --auto-resume \
    --retrieval-config terminal_use --max-steps 120 --timeout 600 \
    --task-ids $IDS --user-persona "$persona" 2>&1 | tail -20
  echo "[cw00term] @ $(date) — $name done"
}

run_cond "cw_00_seed_terminal" /tmp/cw00_terminal/seed.md
run_cond "cw_00_opt_terminal"  /tmp/cw00_terminal/opt.md

echo "[cw00term] @ $(date) — cw_00 terminal 两条件完成"
