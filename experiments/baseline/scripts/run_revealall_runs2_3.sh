#!/bin/bash
# Run reveal-all experiments run2 and run3
# Requires simulation_guidelines.md to have reveal-all modification
set -e

export AWS_PROFILE=us-west-2-staging_dev
export OPENAI_API_KEY=$(grep OPENAI_API_KEY "$(dirname "$0")/../.env" | head -1 | cut -d= -f2)

TASK_IDS_FILE="/tmp/all_114_task_ids.txt"
TASK_ARGS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && TASK_ARGS+=("$line")
done < "$TASK_IDS_FILE"

PREFIX="t1"
MAX_CONCURRENCY=5
MAX_STEPS=200

echo "============================================"
echo "  Reveal-all experiments - run2 and run3"
echo "  Temperature: 1.0 (default)"
echo "  $(date)"
echo "============================================"

# Run2 - Sonnet agent
echo ""
echo ">>> revealall_sonnet_sonnet run2"
uv run --no-sync tau2 run --domain telecom --task-ids "${TASK_ARGS[@]}" \
    --user user_simulator --max-steps $MAX_STEPS --max-concurrency $MAX_CONCURRENCY \
    --user-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --user-llm-args '{"aws_region_name":"us-west-2"}' \
    --agent-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --agent-llm-args '{"aws_region_name":"us-west-2"}' \
    --save-to "${PREFIX}_revealall_sonnet_sonnet_run2" --auto-resume &

# Run2 - GPT agent
echo ">>> revealall_sonnet_gpt54 run2"
uv run --no-sync tau2 run --domain telecom --task-ids "${TASK_ARGS[@]}" \
    --user user_simulator --max-steps $MAX_STEPS --max-concurrency $MAX_CONCURRENCY \
    --user-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --user-llm-args '{"aws_region_name":"us-west-2"}' \
    --agent-llm gpt-5.4 \
    --save-to "${PREFIX}_revealall_sonnet_gpt54_run2" --auto-resume &

wait
echo "Run2 complete. Starting run3..."

# Run3 - Sonnet agent
echo ""
echo ">>> revealall_sonnet_sonnet run3"
uv run --no-sync tau2 run --domain telecom --task-ids "${TASK_ARGS[@]}" \
    --user user_simulator --max-steps $MAX_STEPS --max-concurrency $MAX_CONCURRENCY \
    --user-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --user-llm-args '{"aws_region_name":"us-west-2"}' \
    --agent-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --agent-llm-args '{"aws_region_name":"us-west-2"}' \
    --save-to "${PREFIX}_revealall_sonnet_sonnet_run3" --auto-resume &

# Run3 - GPT agent
echo ">>> revealall_sonnet_gpt54 run3"
uv run --no-sync tau2 run --domain telecom --task-ids "${TASK_ARGS[@]}" \
    --user user_simulator --max-steps $MAX_STEPS --max-concurrency $MAX_CONCURRENCY \
    --user-llm bedrock/us.anthropic.claude-sonnet-4-6 \
    --user-llm-args '{"aws_region_name":"us-west-2"}' \
    --agent-llm gpt-5.4 \
    --save-to "${PREFIX}_revealall_sonnet_gpt54_run3" --auto-resume &

wait

echo ""
echo "============================================"
echo "  All reveal-all runs complete!"
echo "  $(date)"
echo "============================================"
