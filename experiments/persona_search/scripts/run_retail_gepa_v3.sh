#!/bin/bash
# Retail (guitar_center) v3 GEPA optimization.
# Upgrade over v2: enriched mutation context + 25-35% edit + iter 15 + minibatch 8
# (mirrors airline v5 settings). Seeds from each gc's BEST persona (exploration_best
# or pgepa_state best candidate) so the enriched search continues from v2's converged
# point instead of restarting from gen0.
# 5 gc islands in parallel, conc=3 each = 15 total in-flight sims.
set -euo pipefail

REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GEPA="${REPO}/.venv/bin/python3 ${REPO}/experiments/persona_search/scripts/parallel_gepa.py"
LOG_DIR="/tmp/gepa_retail_v3_logs"
mkdir -p "$LOG_DIR"

RETAIL_SPLIT="${REPO}/experiments/persona_search/configs/retail_guitar_center_split_v2.json"
RETAIL_RUN="${REPO}/data/persona_opt/pgepa_runs/retail_v3"
RETAIL_SEEDS=(gc_00 gc_01 gc_02 gc_03 gc_04)

COMMON_ARGS=(
    --user-type user_simulator
    --agent-llm gpt-5.4
    --agent-llm-args '{"reasoning_effort":"high"}'
    --user-llm gpt-5.4
    --effort high
    --iterations 15
    --minibatch 8
    --max-concurrency 3
    --num-trials 2
    --val-num-trials 2
    --mutation-model openai/gpt-5.5
    --mutation-effort xhigh
    --mutation-edit-pct "25-35%"
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

# Extract each gc's BEST persona (from exploration_best_persona.md if present,
# else from pgepa_state best candidate) -> /tmp/retail_v3_seed_<gc>.md
python3 << 'PYEOF'
import json, os
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
v2 = f"{repo}/data/persona_opt/pgepa_runs/retail_v2"
for gc in ["gc_00", "gc_01", "gc_02", "gc_03", "gc_04"]:
    best_md = os.path.join(v2, gc, "exploration_best_persona.md")
    if os.path.exists(best_md):
        text = open(best_md).read()
        src = "exploration_best_persona.md"
    else:
        state = json.load(open(os.path.join(v2, gc, "pgepa_state.json")))
        bidx = state["best_idx"]
        text = state["candidates"][bidx]["persona_text"]
        src = f"state[best_idx={bidx}]"
    out = f"/tmp/retail_v3_seed_{gc}.md"
    with open(out, "w") as f:
        f.write(text)
    print(f"  {gc}: {len(text)} chars  ({src})")
PYEOF

echo "========================================"
echo "[$(date)] Launching 5 retail v3 GEPA islands in parallel"
echo "  split v2: train=15, val=15, test=84"
echo "  config: enriched ctx, 25-35% edit, iter 15, mb 8, 2 trials"
echo "  concurrency: 3 per island = 15 total"
echo "========================================"

PIDS=()
for i in 0 1 2 3 4; do
    name="${RETAIL_SEEDS[$i]}"
    seed_file="/tmp/retail_v3_seed_${name}.md"
    run_dir="${RETAIL_RUN}/${name}"
    log_file="${LOG_DIR}/retail_${name}.log"

    mkdir -p "$run_dir"
    echo "[$(date)] Launching retail/${name} -> $log_file"

    $GEPA \
        --seed-persona "$seed_file" \
        --run-dir "$run_dir" \
        --persona-id-base "${name}_v3" \
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
echo "[$(date)] ALL DONE (failed: $FAILED/5)"
echo "========================================"

python3 << 'PYEOF'
import json, os
import os; repo = os.environ.get("TAU2BENCH_DATA_ROOT", os.getcwd())
run_base = f"{repo}/data/persona_opt/pgepa_runs/retail_v3"
print("\n===== RETAIL v3 GEPA RESULTS =====\n")
print(f"{'seed':<10} {'best_idx':>8} {'best_agg':>9} {'candidates':>11} {'iters':>6}")
print("-" * 48)
for s in ["gc_00", "gc_01", "gc_02", "gc_03", "gc_04"]:
    sp = os.path.join(run_base, s, "pgepa_state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        print(f"{s:<10} {st.get('best_idx','?'):>8} {st.get('best_agg',0):>9.3f} "
              f"{len(st.get('candidates',[])):>11} {len(st.get('generations',[])):>6}")
    else:
        print(f"{s:<10} {'(not found)':>30}")
PYEOF
