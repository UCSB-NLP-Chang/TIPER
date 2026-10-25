#!/usr/bin/env bash
# Evaluate the two Brinks seed personas and their final pGEPA optimized personas
# on the full telecom base split.
#
# Defaults:
#   4 personas * 114 telecom-base tasks * 2 trials = 912 rollouts
#
# Useful overrides:
#   MAX_CONCURRENCY=30 bash experiments/persona_search/scripts/run_brinks_fullset_eval.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_brinks_fullset_eval.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MAX_TASKS="${MAX_TASKS:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-data/persona_opt/pgepa_runs/logs}"
OUT="${OUT:-data/persona_opt/pgepa_runs/brinks_fullset_eval.json}"

USER_TYPE="${USER_TYPE:-cc_user_con}"
USER_LLM="${USER_LLM:-claude-opus-4-7}"
EFFORT="${EFFORT:-high}"
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
mkdir -p "$LOG_DIR"

# Native Vertex env for cc_user_con (claude CLI inherits these).
export CLAUDE_CODE_USE_VERTEX="${CLAUDE_CODE_USE_VERTEX:-1}"
export CLAUDE_CODE_SKIP_VERTEX_AUTH="${CLAUDE_CODE_SKIP_VERTEX_AUTH:-1}"
export ANTHROPIC_VERTEX_BASE_URL="${ANTHROPIC_VERTEX_BASE_URL:-https://dev-proxy.ops2.internal.cresta.ai/proxy/vertex}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-REDACTED_ANTHROPIC_VERTEX_TOKEN}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"

extra=""
if [[ "$DRY_RUN" == "1" ]]; then
  extra="$extra --dry-run"
fi
if [[ "$MAX_TASKS" != "0" ]]; then
  extra="$extra --max-tasks $MAX_TASKS"
fi

echo "Launching Brinks fullset eval"
echo "  personas=4 split=telecom.base max_tasks=$MAX_TASKS trials=$NUM_TRIALS"
echo "  rollout estimate=$(( (MAX_TASKS == 0 ? 114 : MAX_TASKS) * 4 * NUM_TRIALS ))"
echo "  max_concurrency=$MAX_CONCURRENCY"
echo "  user=$USER_TYPE $USER_LLM via native Vertex ($ANTHROPIC_VERTEX_BASE_URL)"
echo "  agent=$AGENT_LLM $AGENT_LLM_ARGS"
echo "  out=$OUT"

log="${LOG_DIR}/brinks_fullset_eval.log"
nohup "$PY" experiments/persona_search/scripts/eval_brinks_final_fullset.py \
  --out "$OUT" \
  --num-trials "$NUM_TRIALS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --user-type "$USER_TYPE" \
  --user-llm "$USER_LLM" \
  --effort "$EFFORT" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --no-bedrock \
  $extra \
  >"$log" 2>&1 &

echo "  pid=$!"
echo "  log=$log"
echo
echo "Monitor:"
echo "  tail -f $log"
