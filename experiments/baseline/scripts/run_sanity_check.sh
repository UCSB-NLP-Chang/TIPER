#!/bin/bash
set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be set}"

PYTHON=/mnt/data3/yutao/conda_envs/tau2bench/bin/python
TASK_IDS="0 1 3 4 10"

echo "===== tau3 retail sanity check | agent=gemini-3-pro | user=gpt-5 ====="
$PYTHON -m tau2.cli run \
    --domain retail \
    --agent llm_agent \
    --agent-llm "gemini/gemini-3-pro-preview" \
    --user user_simulator \
    --user-llm "gpt-4o" \
    --task-ids $TASK_IDS \
    --num-trials 1 \
    --max-concurrency 3 \
    --save-to results/sanity_check_tau3_retail_$(date +%m%d%H%M%S)
echo "===== done ====="
