#!/bin/bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Live monitor for cross-domain GEPA
# Usage: bash experiments/persona_search/scripts/monitor_gepa.sh [log_file] [interval_sec]
LOG="${1:-/tmp/crossdomain_gepa.log}"
INTERVAL="${2:-30}"

STATE_FILE="$REPO/data/persona_opt/pgepa_runs/crossdomain_haiku_5seed/pgepa_state.json"

while true; do
    clear
    echo "=== GEPA Monitor $(date '+%H:%M:%S') ==="
    echo ""

    # Process alive?
    PID=$(pgrep -f 'crossdomain_gepa.py' | head -1)
    if [ -n "$PID" ]; then
        echo "Process: RUNNING (pid $PID)"
    else
        echo "Process: STOPPED"
    fi

    # Active claude subprocesses
    N_CLAUDE=$(ps aux | grep '[c]laude.*--print' | wc -l | tr -d ' ')
    echo "Active claude procs: $N_CLAUDE"
    echo ""

    # Seeds
    echo "--- Gen 0: Seeds ---"
    grep -a 'seed [0-9]*: score=' "$LOG" 2>/dev/null || echo "(still evaluating seeds)"
    echo ""

    # Iterations
    ITER_LINES=$(grep -a '\[iter [0-9]' "$LOG" 2>/dev/null)
    if [ -n "$ITER_LINES" ]; then
        TOTAL=$(echo "$ITER_LINES" | wc -l | tr -d ' ')
        ACCEPTED=$(echo "$ITER_LINES" | grep -c 'idx[0-9]')
        REJECTED=$(echo "$ITER_LINES" | grep -c 'rejected')
        echo "--- Iterations: $TOTAL done ($ACCEPTED accepted, $REJECTED rejected) ---"
        echo "$ITER_LINES" | tail -5
    else
        echo "--- Iterations: 0 ---"
    fi
    echo ""

    # Current activity
    echo "--- Current ---"
    grep -a 'Status:' "$LOG" 2>/dev/null | tail -1 || echo "(no status)"
    echo ""

    # Best candidate
    BEST=$(grep -a 'best=' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        echo "--- Best ---"
        echo "$BEST"
        echo ""
    fi

    # State file summary
    if [ -f "$STATE_FILE" ]; then
        python3 -c "
import json
s = json.load(open('$STATE_FILE'))
cands = s.get('candidates', [])
gens = s.get('generations', [])
print('--- State file ---')
print(f'Candidates: {len(cands)}  Generations: {len(gens)}')
feasible = [c for c in cands if c.get('feasible')]
print(f'Feasible: {len(feasible)}/{len(cands)}')
if cands:
    best = max(cands, key=lambda c: c.get('agg') or 0)
    print(f'Best: idx{best[\"idx\"]} agg={best.get(\"agg\",\"?\")} difficulty={best.get(\"difficulty_agg\",\"?\")}')
" 2>/dev/null
    fi

    if [ -z "$PID" ]; then
        echo ""
        echo "Process stopped. Exiting monitor."
        break
    fi

    echo ""
    echo "(refreshing every ${INTERVAL}s, Ctrl-C to stop)"
    sleep "$INTERVAL"
done
