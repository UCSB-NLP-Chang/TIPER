#!/usr/bin/env bash
# Watch both dual-arm parallel_gepa runs (p0003 + p0002) in one screen.
#
#   bash experiments/persona_search/scripts/pgepa_watch_pair.sh            # default arms, 15s refresh
#   bash experiments/persona_search/scripts/pgepa_watch_pair.sh 30         # 30s refresh
#   ARMS="p0003 p0002 p0001" bash experiments/persona_search/scripts/pgepa_watch_pair.sh
#
# Shows each arm's pgepa_monitor board stacked, plus a cumulative infra_error
# count per arm (the number to watch when gen phases overlap under global conc~40).
set -u
cd "$(dirname "$0")/../.." || exit 1

INTERVAL="${1:-15}"
ARMS="${ARMS:-p0003 p0002}"
PY=.venv/bin/python

while true; do
  clear
  echo "$(date '+%H:%M:%S')  pgepa dual-arm  global conc~40  refresh=${INTERVAL}s  (Ctrl-C to stop)"
  alive=$(pgrep -fl parallel_gepa | grep -v grep | wc -l | tr -d ' ')
  echo "  parallel_gepa processes alive: ${alive}"
  echo
  for b in $ARMS; do
    "$PY" experiments/persona_search/scripts/pgepa_monitor.py \
      --run-dir "data/persona_opt/pgepa_runs/pgepa_pair_$b" \
      --base "pgepa_pair_$b" 2>/dev/null | sed -n '1,14p'
    infra=$(grep -rl infra_error data/simulations/gepa_opt/pgepa_pair_${b}_g* 2>/dev/null | wc -l | tr -d ' ')
    echo "    cumulative infra_error sims (all gens): ${infra}"
    echo
  done
  sleep "$INTERVAL"
done
