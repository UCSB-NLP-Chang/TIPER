#!/usr/bin/env bash
# Extra test eval: 3 additional personas on ALL 3 domains test set
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${REPO}/.venv/bin"
SPLIT="${REPO}/experiments/persona_search/configs/crossdomain_variance_split_v2.json"
CAND_DIR="${REPO}/data/persona_opt/test_eval_candidates"
DOMAINS=(telecom airline retail)

run_domain() {
  local NAME="$1"
  local DOMAIN="$2"
  local PERSONA_ARG="$3"
  local SAVE_TO="gepa_opt/test_eval/${NAME}/${DOMAIN}"

  local TASK_IDS
  TASK_IDS=$("${VENV}/python3" -c "
import json
split = json.load(open('${SPLIT}'))
for tid in split['test'].get('${DOMAIN}', []):
    print(tid)
")

  local TIDS=()
  while IFS= read -r line; do
    [ -n "$line" ] && TIDS+=("$line")
  done <<< "$TASK_IDS"

  if [ ${#TIDS[@]} -eq 0 ]; then
    echo "  [${DOMAIN}] no test tasks, skipping"
    return
  fi

  echo "  [${DOMAIN}] ${#TIDS[@]} tasks -> ${SAVE_TO}"

  local CMD=("${VENV}/tau2" run \
    --domain "${DOMAIN}" \
    --user user_simulator \
    --user-llm gpt-5.2 \
    --agent-llm gpt-5.4 \
    --agent-llm-args '{"reasoning_effort":"high"}' \
    --num-trials 1 \
    --max-concurrency 10 \
    --max-steps 120 \
    --timeout 600 \
    --save-to "${SAVE_TO}" \
    --auto-resume \
    --task-ids "${TIDS[@]}")

  if [ -n "$PERSONA_ARG" ]; then
    CMD+=(--user-persona "$PERSONA_ARG")
  fi

  "${CMD[@]}" || echo "  WARNING: ${NAME}/${DOMAIN} exited with error $?"
}

for CAND in v2_seed6 v3_mut13 v3_mut11; do
  PERSONA_JSON=$("${VENV}/python3" -c "import json; print(json.dumps({'custom_guidelines': open('${CAND_DIR}/${CAND}.md').read()}))")

  echo ""
  echo "=============================="
  echo "  Evaluating: ${CAND}"
  echo "=============================="

  for DOMAIN in "${DOMAINS[@]}"; do
    run_domain "$CAND" "$DOMAIN" "$PERSONA_JSON"
  done
done

echo ""
echo "Extra evaluations complete."
