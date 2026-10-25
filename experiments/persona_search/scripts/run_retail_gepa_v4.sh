#!/bin/bash
# Retail v4: search gc_00 + gc_01 on a LARGER split (train30/val30/test54) to give
# the minibatch more representative room. gc_00 was v2's best_agg (0.222) but never
# got a test eval; gc_01 stalled at seed in v2. edit 30-40% (between 25-35% near-dup
# and 50-60% instruction-sheet lint). enriched ctx, 2 trials, user xhigh + agent high.
set -euo pipefail

REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GEPA="${REPO}/.venv/bin/python3 ${REPO}/experiments/persona_search/scripts/parallel_gepa.py"
LOG_DIR="/tmp/gepa_retail_v4_logs"
mkdir -p "$LOG_DIR"

RETAIL_SPLIT="${REPO}/experiments/persona_search/configs/retail_guitar_center_split_v4.json"
RETAIL_RUN="${REPO}/data/persona_opt/pgepa_runs/retail_v4"
RETAIL_SEEDS=(gc_00 gc_01)

COMMON_ARGS=(
    --user-type user_simulator
    --agent-llm gpt-5.4
    --agent-llm-args '{"reasoning_effort":"high"}'
    --user-llm gpt-5.4
    --effort xhigh
    --iterations 15
    --minibatch 10
    --max-concurrency 13
    --num-trials 2
    --val-num-trials 2
    --mutation-model openai/gpt-5.5
    --mutation-effort xhigh
    --mutation-edit-pct "30-40%"
    --mutation-context enriched
    --lint-persona
    --lint-model openai/gpt-5.4-mini
    --judge-model gpt-5.2
    --max-mutation-attempts 8
    --min-dissimilarity 0.3
    --max-semantic-similarity 0.92
    --print-pareto
    --max-steps 120
    --sim-timeout 600
)

# Extract GEN0 seed persona (candidates[0]) from retail_guitar_center_v1 state.
python3 << 'PYEOF'
import json
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
v1 = f"{repo}/data/persona_opt/pgepa_runs/retail_guitar_center_v1"
for gc in ["gc_00", "gc_01"]:
    st = json.load(open(f"{v1}/{gc}/pgepa_state.json"))
    text = st["candidates"][0]["persona_text"]  # gen0 seed
    out = f"/tmp/retail_v4_seed_{gc}.md"
    open(out, "w").write(text)
    print(f"  {gc}: gen0 seed, {len(text)} chars")
PYEOF

echo "========================================"
echo "[$(date)] Launching 2 retail v4 GEPA islands (gc_00 + gc_01, larger split)"
echo "  split v4: train=30, val=30, test=54"
echo "  config: enriched ctx, 30-40% edit, iter 15, mb 10x2, val 30x2"
echo "  user reasoning=xhigh, agent reasoning=high"
echo "  max-semantic-similarity=0.92 (wider gate vs 0.9)"
echo "  concurrency: 13 per island = 26 total"
echo "========================================"

PIDS=()
for i in 0 1; do
    name="${RETAIL_SEEDS[$i]}"
    seed_file="/tmp/retail_v4_seed_${name}.md"
    run_dir="${RETAIL_RUN}/${name}"
    log_file="${LOG_DIR}/retail_${name}.log"

    mkdir -p "$run_dir"
    echo "[$(date)] Launching retail/${name} -> $log_file"

    $GEPA \
        --seed-persona "$seed_file" \
        --run-dir "$run_dir" \
        --persona-id-base "${name}_v4" \
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

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: PID $pid exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "[$(date)] ALL DONE (failed: $FAILED/2)"
echo "========================================"

python3 << 'PYEOF'
import json, os
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
run_base = f"{repo}/data/persona_opt/pgepa_runs/retail_v4"
v2 = f"{repo}/data/persona_opt/pgepa_runs/retail_v2"
print("\n===== RETAIL v4 vs v2 (larger split, gc_00+gc_01) =====\n")
print(f"{'seed':<8} {'v2 best_agg':>11} {'v4 best_idx':>12} {'v4 best_agg':>12} {'cands':>6} {'iters':>6}")
print("-" * 60)
for s in ["gc_00", "gc_01"]:
    v2agg = json.load(open(f"{v2}/{s}/pgepa_state.json")).get("best_agg", 0)
    sp = os.path.join(run_base, s, "pgepa_state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        print(f"{s:<8} {v2agg:>11.3f} {st.get('best_idx','?'):>12} "
              f"{st.get('best_agg',0):>12.3f} {len(st.get('candidates',[])):>6} {len(st.get('generations',[])):>6}")
    else:
        print(f"{s:<8} {v2agg:>11.3f} {'(not found)':>30}")
PYEOF
