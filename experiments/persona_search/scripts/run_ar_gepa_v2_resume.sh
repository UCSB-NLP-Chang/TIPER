#!/bin/bash
# Resume unfinished + launch new seeds
# Unfinished: alaska_00 (9/10), gc_02 (8/10)
# New: alaska_02, gc_03, gc_04
set -euo pipefail

REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GEPA="${REPO}/.venv/bin/python3 ${REPO}/experiments/persona_search/scripts/parallel_gepa.py"
LOG_DIR="/tmp/gepa_ar_v2_logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    --user-type user_simulator
    --agent-llm gpt-5.4
    --user-llm gpt-5.4
    --effort high
    --iterations 10
    --minibatch 5
    --max-concurrency 5
    --num-trials 3
    --val-num-trials 3
    --mutation-model openai/gpt-5.5
    --mutation-effort xhigh
    --mutation-edit-pct "40-50%"
    --lint-persona
    --lint-model openai/gpt-5.4-mini
    --max-mutation-attempts 8
    --min-dissimilarity 0.3
    --max-semantic-similarity 0.9
    --print-pareto
    --max-steps 120
    --sim-timeout 600
)

AIRLINE_SPLIT="${REPO}/experiments/persona_search/configs/airline_alaska_split_v4.json"
RETAIL_SPLIT="${REPO}/experiments/persona_search/configs/retail_guitar_center_split_v2.json"

echo "========================================"
echo "[$(date)] Resuming + launching new seeds (5 parallel)"
echo "  Resume: alaska_00 (9/10), gc_02 (8/10)"
echo "  New:    alaska_02, gc_03, gc_04"
echo "========================================"

PIDS=()

# ── Resume alaska_00 (9/10) ──
echo "[$(date)] Resuming airline/alaska_00..."
$GEPA \
    --seed-persona /tmp/gepa_seed_airline_alaska_00.md \
    --run-dir "${REPO}/data/persona_opt/pgepa_runs/airline_v4/alaska_00" \
    --persona-id-base alaska_00 \
    --domains airline \
    --split-file "$AIRLINE_SPLIT" \
    --seed 42 --resume \
    "${COMMON_ARGS[@]}" \
    > "${LOG_DIR}/airline_alaska_00_resumed.log" 2>&1 &
PIDS+=($!)

# ── Resume gc_02 (8/10) ──
echo "[$(date)] Resuming retail/gc_02..."
$GEPA \
    --seed-persona /tmp/gepa_seed_retail_gc_02.md \
    --run-dir "${REPO}/data/persona_opt/pgepa_runs/retail_v2/gc_02" \
    --persona-id-base gc_02 \
    --domains retail \
    --split-file "$RETAIL_SPLIT" \
    --seed 43 --resume \
    "${COMMON_ARGS[@]}" \
    > "${LOG_DIR}/retail_gc_02_resumed.log" 2>&1 &
PIDS+=($!)

# ── New: alaska_02 ──
echo "[$(date)] Launching airline/alaska_02..."
mkdir -p "${REPO}/data/persona_opt/pgepa_runs/airline_v4/alaska_02"
$GEPA \
    --seed-persona /tmp/gepa_seed_airline_alaska_02.md \
    --run-dir "${REPO}/data/persona_opt/pgepa_runs/airline_v4/alaska_02" \
    --persona-id-base alaska_02 \
    --domains airline \
    --split-file "$AIRLINE_SPLIT" \
    --seed 44 \
    "${COMMON_ARGS[@]}" \
    > "${LOG_DIR}/airline_alaska_02.log" 2>&1 &
PIDS+=($!)

# ── New: gc_03 ──
echo "[$(date)] Launching retail/gc_03..."
mkdir -p "${REPO}/data/persona_opt/pgepa_runs/retail_v2/gc_03"
$GEPA \
    --seed-persona /tmp/gepa_seed_retail_gc_03.md \
    --run-dir "${REPO}/data/persona_opt/pgepa_runs/retail_v2/gc_03" \
    --persona-id-base gc_03 \
    --domains retail \
    --split-file "$RETAIL_SPLIT" \
    --seed 45 \
    "${COMMON_ARGS[@]}" \
    > "${LOG_DIR}/retail_gc_03.log" 2>&1 &
PIDS+=($!)

# ── New: gc_04 ──
echo "[$(date)] Launching retail/gc_04..."
mkdir -p "${REPO}/data/persona_opt/pgepa_runs/retail_v2/gc_04"
$GEPA \
    --seed-persona /tmp/gepa_seed_retail_gc_04.md \
    --run-dir "${REPO}/data/persona_opt/pgepa_runs/retail_v2/gc_04" \
    --persona-id-base gc_04 \
    --domains retail \
    --split-file "$RETAIL_SPLIT" \
    --seed 46 \
    "${COMMON_ARGS[@]}" \
    > "${LOG_DIR}/retail_gc_04.log" 2>&1 &
PIDS+=($!)

echo ""
echo "PIDs: ${PIDS[*]}"
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: PID $pid exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "[$(date)] ALL DONE (failed: $FAILED/5)"
echo "========================================"
