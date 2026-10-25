#!/bin/bash
# Retail v3b: test whether enriched ctx alone (vs v2's basic) helps the WEAKEST
# v2 islands. 3 islands = the 3 lowest v2 best_agg (gc_01=0.133, gc_02=0.178,
# gc_04=0.200). Seeded from GEN0 seed persona (NOT v2 optim) for a clean
# basic-vs-enriched comparison. minibatch 10x2, val 30x2 (split v3).
# user reasoning xhigh, agent reasoning high. enriched ctx, 50-60% edit, iter 15.
set -euo pipefail

REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GEPA="${REPO}/.venv/bin/python3 ${REPO}/experiments/persona_search/scripts/parallel_gepa.py"
LOG_DIR="/tmp/gepa_retail_v3b_logs"
mkdir -p "$LOG_DIR"

RETAIL_SPLIT="${REPO}/experiments/persona_search/configs/retail_guitar_center_split_v3.json"
RETAIL_RUN="${REPO}/data/persona_opt/pgepa_runs/retail_v3b"
RETAIL_SEEDS=(gc_01 gc_02 gc_04)

COMMON_ARGS=(
    --user-type user_simulator
    --agent-llm gpt-5.4
    --agent-llm-args '{"reasoning_effort":"high"}'
    --user-llm gpt-5.4
    --effort xhigh
    --iterations 15
    --minibatch 10
    --max-concurrency 8
    --num-trials 2
    --val-num-trials 2
    --mutation-model openai/gpt-5.5
    --mutation-effort xhigh
    --mutation-edit-pct "50-60%"
    --mutation-context enriched
    --lint-persona
    --lint-model openai/gpt-5.4-mini
    --judge-model gpt-5.2
    --max-mutation-attempts 8
    --min-dissimilarity 0.3
    --max-semantic-similarity 0.9
    --print-pareto
    --max-steps 120
    --sim-timeout 600
)

# Extract GEN0 seed persona (candidates[0]) from retail_guitar_center_v1 state.
python3 << 'PYEOF'
import json
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
v1 = f"{repo}/data/persona_opt/pgepa_runs/retail_guitar_center_v1"
for gc in ["gc_01", "gc_02", "gc_04"]:
    st = json.load(open(f"{v1}/{gc}/pgepa_state.json"))
    text = st["candidates"][0]["persona_text"]  # gen0 seed
    out = f"/tmp/retail_v3b_seed_{gc}.md"
    open(out, "w").write(text)
    print(f"  {gc}: gen0 seed, {len(text)} chars")
PYEOF

echo "========================================"
echo "[$(date)] Launching 3 retail v3b GEPA islands (weakest v2, gen0 seed)"
echo "  split v3: train=15, val=30, test=69"
echo "  config: enriched ctx, 50-60% edit, iter 15, mb 10x2, val 30x2"
echo "  user reasoning=xhigh, agent reasoning=high"
echo "  concurrency: 8 per island = 24 total"
echo "========================================"

PIDS=()
for i in 0 1 2; do
    name="${RETAIL_SEEDS[$i]}"
    seed_file="/tmp/retail_v3b_seed_${name}.md"
    run_dir="${RETAIL_RUN}/${name}"
    log_file="${LOG_DIR}/retail_${name}.log"

    mkdir -p "$run_dir"
    echo "[$(date)] Launching retail/${name} -> $log_file"

    $GEPA \
        --seed-persona "$seed_file" \
        --run-dir "$run_dir" \
        --persona-id-base "${name}_v3b" \
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
echo "[$(date)] ALL DONE (failed: $FAILED/3)"
echo "========================================"

python3 << 'PYEOF'
import json, os
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
run_base = f"{repo}/data/persona_opt/pgepa_runs/retail_v3b"
v2 = f"{repo}/data/persona_opt/pgepa_runs/retail_v2"
print("\n===== RETAIL v3b vs v2 (enriched ctx, gen0 seed) =====\n")
print(f"{'seed':<8} {'v2 best_agg':>11} {'v3b best_idx':>12} {'v3b best_agg':>12} {'cands':>6} {'iters':>6}")
print("-" * 60)
for s in ["gc_01", "gc_02", "gc_04"]:
    v2agg = json.load(open(f"{v2}/{s}/pgepa_state.json")).get("best_agg", 0)
    sp = os.path.join(run_base, s, "pgepa_state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        print(f"{s:<8} {v2agg:>11.3f} {st.get('best_idx','?'):>12} "
              f"{st.get('best_agg',0):>12.3f} {len(st.get('candidates',[])):>6} {len(st.get('generations',[])):>6}")
    else:
        print(f"{s:<8} {v2agg:>11.3f} {'(not found)':>30}")
PYEOF
