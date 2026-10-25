#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Wait for the current banking test matrix (bm25) driver to finish, then run
# ONE extra condition: no_persona + terminal_use retrieval on the same test set
# (57 tasks x 3 trials), so we can compare bm25 vs terminal_use on the no_persona
# baseline under identical agent/user config.
set -u
cd "$REPO"

echo "[wait] @ $(date) — 等当前 test matrix driver 退出"
# block until no run_test_matrix_banking driver is alive
while pgrep -f 'run_test_matrix_banking' | grep -qv grep; do
  sleep 60
done
echo "[wait] driver 已退出 @ $(date) — 启动 no_persona + terminal_use"

# 57 test task ids
IDS=$(.venv/bin/python -c "
import json
d=json.load(open('experiments/persona_search/configs/banking_knowledge_split_v2.json'))
t=d['test']
ids=t['banking'] if isinstance(t,dict) else t
print(' '.join(ids))
")

TAU2=.venv/bin/tau2
SAVE_TO=gepa_opt/testmatrix_cardworks/no_persona_terminal/banking

nohup "$TAU2" run \
  --domain banking_knowledge \
  --user user_simulator \
  --user-llm gpt-5.4 \
  --user-llm-args '{"reasoning_effort":"high"}' \
  --agent-llm gpt-5.5 \
  --agent-llm-args '{"reasoning_effort":"high"}' \
  --num-trials 3 \
  --max-concurrency 25 \
  --save-to "$SAVE_TO" \
  --auto-resume \
  --retrieval-config terminal_use \
  --max-steps 120 \
  --timeout 600 \
  --task-ids $IDS \
  > /tmp/testmatrix_no_persona_terminal.log 2>&1 &

echo "launched pid $!  log /tmp/testmatrix_no_persona_terminal.log"
echo "save_to $SAVE_TO"
