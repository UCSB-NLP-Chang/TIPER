#!/bin/bash
# Run naturalness optimization for all 5 GEPA-optimized personas
# Then test with 3 trials on the test set
set -e

cd "$(dirname "$0")/../.."
PYTHON=".venv/bin/python"
SCRIPT="experiments/persona_search/scripts/run_naturalness_opt.py"
LOG="/tmp/naturalness_opt_all.log"

echo "=== Naturalness Optimization: All 5 Seeds ==="
echo "Start: $(date)"
echo "Log: $LOG"

# Phase 1: Optimize all 5 seeds (sequential to avoid overloading API)
$PYTHON $SCRIPT \
    --seeds b00 b02 b03 b04 p03 \
    --max-iters 5 \
    --minibatch 5 \
    --concurrency 3 \
    --num-trials 1 \
    --difficulty-tolerance 0.05 \
    --mutation-model gpt-5.4 \
    2>&1 | tee "$LOG"

echo ""
echo "=== Phase 1 Complete: $(date) ==="
echo ""

# Phase 2: Test all on test set with 3 trials
$PYTHON $SCRIPT \
    --test-only \
    --seeds b00 b02 b03 b04 p03 \
    --test-trials 3 \
    --concurrency 3 \
    2>&1 | tee -a "$LOG"

echo ""
echo "=== All Done: $(date) ==="
