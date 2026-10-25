#!/bin/bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Airline (v4 split: 10/10/30) + Retail (v2 split: 15/15/84) test eval
# Agent: gpt-5.4 high | User: gpt-5.4 high | 3 trials | serial conditions
set -e
cd "$REPO"

TAU2=".venv/bin/tau2"
NUM_TRIALS=3
CONC=5

# ── Test task IDs ──
AIRLINE_TEST="1 2 5 6 7 8 13 14 15 17 18 20 22 27 28 30 32 33 34 36 37 40 41 42 43 44 46 47 48 49"
RETAIL_TEST="0 2 3 4 5 6 10 11 12 13 14 15 17 18 19 20 22 23 24 25 27 28 29 31 33 35 36 37 38 39 40 42 43 44 45 46 48 49 53 54 55 56 57 58 59 60 61 62 64 65 66 68 69 71 72 75 76 77 81 83 85 86 87 91 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 109 110 111 112 113"

# ── Save dirs ──
AIRLINE_SAVE="gepa_opt/airline_v4_test"
RETAIL_SAVE="gepa_opt/retail_v2_test"

run_condition() {
    local domain=$1
    local name=$2
    local persona_file=$3
    local task_ids=$4
    local save_base=$5

    echo "[$(date)] Running ${domain}/${name} ..."

    local persona_args=""
    if [ -n "$persona_file" ]; then
        local pjson
        pjson=$(cat "$persona_file")
        persona_args="--user-persona ${pjson}"
    fi

    if [ -n "$persona_file" ]; then
        $TAU2 run --domain "$domain" \
            --user user_simulator --user-llm gpt-5.4 \
            --user-llm-args '{"reasoning_effort":"high"}' \
            --agent-llm gpt-5.4 --agent-llm-args '{"reasoning_effort":"high"}' \
            --num-trials $NUM_TRIALS --max-concurrency $CONC \
            --save-to "${save_base}/test_${name}/${domain}" \
            --auto-resume --max-steps 120 \
            --user-persona "$(cat "$persona_file")" \
            --task-ids $task_ids
    else
        $TAU2 run --domain "$domain" \
            --user user_simulator --user-llm gpt-5.4 \
            --user-llm-args '{"reasoning_effort":"high"}' \
            --agent-llm gpt-5.4 --agent-llm-args '{"reasoning_effort":"high"}' \
            --num-trials $NUM_TRIALS --max-concurrency $CONC \
            --save-to "${save_base}/test_${name}/${domain}" \
            --auto-resume --max-steps 120 \
            --task-ids $task_ids
    fi

    echo "[$(date)] ${domain}/${name} done."
}

echo "========================================"
echo "[$(date)] Airline + Retail test eval"
echo "  Airline: 30 test × 3 trials × 4 conditions = 360 sims"
echo "  Retail:  84 test × 3 trials × 4 conditions = 1008 sims"
echo "  Total: 1368 sims (serial, conc=$CONC)"
echo "========================================"

# ── Airline (30 test tasks) ──
echo ""
echo "===== AIRLINE (v4 split, 30 test tasks) ====="
run_condition "airline" "no_persona"  "" "$AIRLINE_TEST" "$AIRLINE_SAVE"
run_condition "airline" "seed_alaska_01" "/tmp/airline_persona_alaska_01.json" "$AIRLINE_TEST" "$AIRLINE_SAVE"
run_condition "airline" "seed_alaska_00" "/tmp/airline_persona_alaska_00.json" "$AIRLINE_TEST" "$AIRLINE_SAVE"
run_condition "airline" "seed_alaska_03" "/tmp/airline_persona_alaska_03.json" "$AIRLINE_TEST" "$AIRLINE_SAVE"

# ── Retail (84 test tasks) ──
echo ""
echo "===== RETAIL (v2 split, 84 test tasks) ====="
run_condition "retail" "no_persona"  "" "$RETAIL_TEST" "$RETAIL_SAVE"
run_condition "retail" "seed_gc_01" "/tmp/retail_persona_gc_01.json" "$RETAIL_TEST" "$RETAIL_SAVE"
run_condition "retail" "seed_gc_02" "/tmp/retail_persona_gc_02.json" "$RETAIL_TEST" "$RETAIL_SAVE"
run_condition "retail" "seed_gc_00" "/tmp/retail_persona_gc_00.json" "$RETAIL_TEST" "$RETAIL_SAVE"

echo ""
echo "========================================"
echo "[$(date)] ALL DONE"
echo "========================================"

# ── Summary ──
python3 << 'PYEOF'
import json, os, glob
from collections import defaultdict

def load_results(base_dir, domain, conditions):
    rows = []
    for name in conditions:
        sim_dir = os.path.join("data/simulations", base_dir, f"test_{name}", domain)
        files = glob.glob(os.path.join(sim_dir, "eval_*.json"))
        rewards = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            ri = d.get("reward_info")
            if ri:
                rewards.append(ri.get("reward", 0))
            else:
                rewards.append(0)
        n = len(rewards)
        avg = sum(rewards) / n if n else 0
        rows.append((name, n, avg))
    return rows

print("\n===== SUMMARY =====\n")

print("── Airline (v4, 30 test × 3 trials = 90) ──")
print(f"{'condition':<25} {'n':>5} {'avg':>8}")
print("-" * 40)
for name, n, avg in load_results("gepa_opt/airline_v4_test", "airline",
    ["no_persona", "seed_alaska_01", "seed_alaska_00", "seed_alaska_03"]):
    print(f"{name:<25} {n:>5} {avg:>8.3f}")

print()
print("── Retail (v2, 84 test × 3 trials = 252) ──")
print(f"{'condition':<25} {'n':>5} {'avg':>8}")
print("-" * 40)
for name, n, avg in load_results("gepa_opt/retail_v2_test", "retail",
    ["no_persona", "seed_gc_01", "seed_gc_02", "seed_gc_00"]):
    print(f"{name:<25} {n:>5} {avg:>8.3f}")

PYEOF
