#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Wait for the cw_01/03/04 OPT terminal batch to finish, then run the SEED
# personas on terminal_use so we can compute opt-vs-seed under terminal for all
# islands (cw_00 seed/opt already done: seed 0.620 / opt 0.659 / Δ+0.039).
set -u
cd "$REPO"

TAU2=.venv/bin/tau2
PY=.venv/bin/python

echo "[cwseedterm] @ $(date) — 等当前 cw opt terminal 批次退出"
while pgrep -f 'run_testmatrix_cw_remaining' | grep -qv grep; do
  sleep 30
done
echo "[cwseedterm] opt 批次已退出 @ $(date) — 开始 cw_01/03/04 seed terminal"

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
  echo "[cwseedterm] @ $(date) — running $name (terminal_use, persona=$persona_file)"
  "$TAU2" run \
    --domain banking_knowledge --user user_simulator --user-llm gpt-5.4 \
    --user-llm-args '{"reasoning_effort":"high"}' \
    --agent-llm gpt-5.5 --agent-llm-args '{"reasoning_effort":"high"}' \
    --num-trials 3 --max-concurrency 30 --save-to "$save" --auto-resume \
    --retrieval-config terminal_use --max-steps 120 --timeout 600 \
    --task-ids $IDS --user-persona "$persona" 2>&1 | tail -20
  echo "[cwseedterm] @ $(date) — $name done"
}

run_cond "cw_01_seed_terminal" /tmp/cw_terminal/cw_01_seed.md
run_cond "cw_03_seed_terminal" /tmp/cw_terminal/cw_03_seed.md
run_cond "cw_04_seed_terminal" /tmp/cw_terminal/cw_04_seed.md

echo "[cwseedterm] @ $(date) — cw_01/03/04 seed terminal 全部完成"
