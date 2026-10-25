#!/bin/bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Relaunch p0001/p0002 GEPA optims WITH validity gating, resuming from gepa_state.bin.
# Inherits current env (cresta Vertex proxy + AWS_PROFILE + OpenAI keys).
set -u
cd "$REPO"

launch() {
  local sid="$1"  # p0001 / p0002
  local rd="data/persona_opt/gepa_runs/${sid}_aggressive50_20260606_151059"
  nohup .venv/bin/python experiments/persona_search/scripts/gepa_runner.py \
    --seed-persona "data/persona_opt/lineages/${sid}/gen_00.md" \
    --persona-id "gepa_${sid}_v2_50t" \
    --run-dir "$rd" \
    --max-metric-calls 1000 --minibatch-size 10 --num-trials 2 \
    --domains telecom --max-concurrency 10 \
    --mutation-edit-pct '~30%' \
    --patience 0 --selection-strategy epsilon_greedy \
    --agent-llm gpt-5.4 --agent-llm-args '{"reasoning_effort":"high"}' \
    --user-type cc_user_con --user-llm claude-opus-4-7 --effort high \
    --mutation-model bedrock/us.anthropic.claude-opus-4-7 \
    --sim-timeout 600 --max-steps 120 --max-infra-retries 2 \
    --retrieval-config terminal_use \
    --judge-model gpt-5.2 --judge-concurrency 16 --judge-maj 3 \
    --judge-valset --severe-gate-rate 0.05 --valset-judge-max-failures 15 \
    >> "$rd/run_log.txt" 2>> "$rd/run_log_stderr.txt" &
  echo "  launched $sid pid=$! (run-dir $rd)"
}

launch p0001
launch p0002
