#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Chain that fires after the current bm25 test matrix driver exits:
#   1. banking_v2 seed  (Cooperative Banking Customer)        bm25 57x3
#   2. banking_v2 best  (Self-Diagnosing Banking Customer)     bm25 57x3
#   3. no_persona + terminal_use                               terminal 57x3
# All same agent/user config as the CardWorks matrix (gpt-5.5 agent high,
# user_simulator gpt-5.4 high, max-steps 120, timeout 600, num-trials 3, conc 25).
# Persona conditions use the same method-actor wrapper as eval_persona.
set -u
cd "$REPO"

TAU2=.venv/bin/tau2
PY=.venv/bin/python

echo "[chain] @ $(date) — 等当前 test matrix driver 退出"
while pgrep -f 'run_test_matrix_banking' | grep -qv grep; do
  sleep 60
done
echo "[chain] driver 已退出 @ $(date) — 开始链式 3 条件"

# test 57 ids
IDS=$($PY -c "
import json
d=json.load(open('experiments/persona_search/configs/banking_knowledge_split_v2.json'))
t=d['test']; ids=t['banking'] if isinstance(t,dict) else t
print(' '.join(ids))
")

# method-actor preamble (identical to eval_persona.py)
PREAMBLE='CRITICAL: You are a method actor. Your #1 priority is to stay in character as described below — every single turn, no exceptions. Never break character to be helpful, cooperative, or efficient beyond what your persona would do. If your persona is terse, give minimal information even if you know more. If your persona withholds details, do NOT volunteer them unprompted. The agent must EARN every piece of information through the right questions.\n\n'

run_cond () {
  local name="$1" retrieval="$2" persona_file="$3"
  local save="gepa_opt/testmatrix_cardworks/${name}/banking"
  local args=()
  args+=( --domain banking_knowledge --user user_simulator --user-llm gpt-5.4 \
          --user-llm-args '{"reasoning_effort":"high"}' \
          --agent-llm gpt-5.5 --agent-llm-args '{"reasoning_effort":"high"}' \
          --num-trials 3 --max-concurrency 30 --save-to "$save" --auto-resume \
          --retrieval-config "$retrieval" --max-steps 120 --timeout 600 \
          --task-ids $IDS )
  if [ -n "$persona_file" ]; then
    local persona=$($PY -c "
import json,sys
txt=open('$persona_file').read()
preamble='$PREAMBLE'.replace('\\n','\n')
print(json.dumps({'custom_guidelines': preamble+txt}, ensure_ascii=False))
")
    args+=( --user-persona "$persona" )
  fi
  echo "[chain] @ $(date) — running $name (retrieval=$retrieval, persona=${persona_file:-none})"
  "$TAU2" run "${args[@]}" 2>&1 | tail -20
  echo "[chain] @ $(date) — $name done"
}

# 1. banking_v2 seed
run_cond "banking_v2_seed"    bm25        "/tmp/banking_v2_personas/seed.md"
# 2. banking_v2 best (idx4)
run_cond "banking_v2_best"    bm25        "/tmp/banking_v2_personas/best_idx4.md"
# 3. no_persona + terminal_use
run_cond "no_persona_terminal" terminal_use ""

echo "[chain] @ $(date) — 全部 3 条件完成"
