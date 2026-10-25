#!/bin/bash
# Airline (v4: 10/10/30) + Retail (v2: 15/15/84) GEPA optimization
# 6 seeds in parallel (3 airline + 3 retail), 10 iterations each, conc=5
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

# ── Airline seeds ──
AIRLINE_SPLIT="${REPO}/experiments/persona_search/configs/airline_alaska_split_v4.json"
AIRLINE_RUN="${REPO}/data/persona_opt/pgepa_runs/airline_v4"
AIRLINE_SEEDS=(alaska_01 alaska_00 alaska_03)
AIRLINE_SEED_FILES=(
    "${REPO}/data/persona_opt/pgepa_runs/alaska_airline_v3/alaska_01/candidates.json"
    "${REPO}/data/persona_opt/pgepa_runs/alaska_airline_v3/alaska_00/candidates.json"
    "${REPO}/data/persona_opt/pgepa_runs/alaska_airline_v3/alaska_03/candidates.json"
)

# ── Retail seeds ──
RETAIL_SPLIT="${REPO}/experiments/persona_search/configs/retail_guitar_center_split_v2.json"
RETAIL_RUN="${REPO}/data/persona_opt/pgepa_runs/retail_v2"
RETAIL_SEEDS=(gc_01 gc_02 gc_00)
RETAIL_SEED_FILES=(
    "${REPO}/data/persona_opt/pgepa_runs/retail_guitar_center_v1/gc_01/pgepa_state.json"
    "${REPO}/data/persona_opt/pgepa_runs/retail_guitar_center_v1/gc_02/pgepa_state.json"
    "${REPO}/data/persona_opt/pgepa_runs/retail_guitar_center_v1/gc_00/pgepa_state.json"
)

# Extract seed persona text to temp files
python3 << 'PYEOF'
import json, os

import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())

# Airline: seed text is idx0 in candidates.json
for s in ["alaska_01", "alaska_00", "alaska_03"]:
    cands = json.load(open(f"{repo}/data/persona_opt/pgepa_runs/alaska_airline_v3/{s}/candidates.json"))
    text = cands[0]["persona_text"]
    out = f"/tmp/gepa_seed_airline_{s}.md"
    with open(out, "w") as f:
        f.write(text)
    print(f"  {out}")

# Retail: seed text is idx0 in state candidates
for s in ["gc_01", "gc_02", "gc_00"]:
    state = json.load(open(f"{repo}/data/persona_opt/pgepa_runs/retail_guitar_center_v1/{s}/pgepa_state.json"))
    text = state["candidates"][0]["persona_text"]
    out = f"/tmp/gepa_seed_retail_{s}.md"
    with open(out, "w") as f:
        f.write(text)
    print(f"  {out}")
PYEOF

echo "========================================"
echo "[$(date)] Launching 6 GEPA seeds in parallel"
echo "  Airline: 3 seeds × 10 iter (split v4: train=10, val=10)"
echo "  Retail:  3 seeds × 10 iter (split v2: train=15, val=15)"
echo "  Concurrency: 5 per seed"
echo "========================================"

PIDS=()

# Launch airline seeds
for i in 0 1 2; do
    name="${AIRLINE_SEEDS[$i]}"
    seed_file="/tmp/gepa_seed_airline_${name}.md"
    run_dir="${AIRLINE_RUN}/${name}"
    log_file="${LOG_DIR}/airline_${name}.log"

    mkdir -p "$run_dir"
    echo "[$(date)] Launching airline/${name} → $log_file"

    $GEPA \
        --seed-persona "$seed_file" \
        --run-dir "$run_dir" \
        --persona-id-base "$name" \
        --domains airline \
        --split-file "$AIRLINE_SPLIT" \
        --seed $((42 + i)) \
        "${COMMON_ARGS[@]}" \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

# Launch retail seeds
for i in 0 1 2; do
    name="${RETAIL_SEEDS[$i]}"
    seed_file="/tmp/gepa_seed_retail_${name}.md"
    run_dir="${RETAIL_RUN}/${name}"
    log_file="${LOG_DIR}/retail_${name}.log"

    mkdir -p "$run_dir"
    echo "[$(date)] Launching retail/${name} → $log_file"

    $GEPA \
        --seed-persona "$seed_file" \
        --run-dir "$run_dir" \
        --persona-id-base "$name" \
        --domains retail \
        --split-file "$RETAIL_SPLIT" \
        --seed $((42 + i)) \
        "${COMMON_ARGS[@]}" \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/"
echo ""

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: PID $pid exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "[$(date)] ALL DONE (failed: $FAILED/6)"
echo "========================================"

# Summary
python3 << 'PYEOF'
import json, os

import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
print("\n===== GEPA RESULTS =====\n")

for domain, run_base, seeds in [
    ("airline", f"{repo}/data/persona_opt/pgepa_runs/airline_v4", ["alaska_01", "alaska_00", "alaska_03"]),
    ("retail", f"{repo}/data/persona_opt/pgepa_runs/retail_v2", ["gc_01", "gc_02", "gc_00"]),
]:
    print(f"── {domain} ──")
    print(f"{'seed':<15} {'best_idx':>8} {'best_agg':>9} {'candidates':>11} {'iters':>6}")
    print("-" * 52)
    for s in seeds:
        state_path = os.path.join(run_base, s, "pgepa_state.json")
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            best_idx = state.get("best_idx", "?")
            best_agg = state.get("best_agg", 0)
            n_cands = len(state.get("candidates", []))
            n_gens = len(state.get("generations", []))
            print(f"{s:<15} {best_idx:>8} {best_agg:>9.3f} {n_cands:>11} {n_gens:>6}")
        else:
            print(f"{s:<15} {'(not found)':>30}")
    print()
PYEOF
