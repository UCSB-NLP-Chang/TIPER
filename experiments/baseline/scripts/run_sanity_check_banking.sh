#!/bin/bash
set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be set}"

PYTHON=/mnt/data3/yutao/conda_envs/tau2bench/bin/python

echo "===== tau3 banking_knowledge sanity check | agent=gemini-3-pro | user=gpt-4o | retrieval=bm25 ====="
$PYTHON -m tau2.cli run \
    --domain banking_knowledge \
    --agent llm_agent \
    --agent-llm "gemini/gemini-3-pro-preview" \
    --user user_simulator \
    --user-llm "gpt-4o" \
    --retrieval-config bm25 \
    --task-ids task_001 task_002 task_003 task_004 task_005 \
    --num-trials 1 \
    --max-concurrency 3 \
    --save-to results/sanity_check_tau3_banking_$(date +%m%d%H%M%S)
echo "===== done ====="
