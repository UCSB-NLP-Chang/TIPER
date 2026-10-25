#!/usr/bin/env bash
# Lightweight change-detecting watcher for the Brinks pGEPA pair.
# Prints a heartbeat TICK every INTERVAL seconds, and a CHANGE/COMPLETE/ERROR
# line only when something meaningful moves (gen advances, run finishes, or an
# error/traceback shows up). Meant to be backgrounded and watched via a regex on
# CHANGE|COMPLETE|ERROR|ALLDONE. Written for macOS bash 3.2 (no assoc arrays).
set -u
cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"
INTERVAL="${INTERVAL:-300}"
R1="pgepa_brinks_03_verbose_polite"
R2="pgepa_brinks_04_confused_drip_feeding"
LOG_DIR="data/persona_opt/pgepa_runs/logs"

prev1=""
prev2=""

read_state() {  # $1=run -> "gens=<n> best=idx<n> agg=<f>" or gen0 marker
  local st="data/persona_opt/pgepa_runs/$1/pgepa_state.json"
  if [ -f "$st" ]; then
    "$PY" - "$st" <<'PYEOF' 2>/dev/null || echo "gens=? (state unreadable)"
import json, sys
d = json.load(open(sys.argv[1]))
gl = d.get("gen_logs", [])
print(f"gens={len(gl)} best=idx{d.get('best_idx')} agg={round(float(d.get('best_agg') or 0),3)}")
PYEOF
  else
    echo "gens=0 (gen0 running)"
  fi
}

run_status() {  # $1=run ; echoes "<state>|done=<0/1>|err=<n>"
  local r="$1" log="$LOG_DIR/$1.log" state done err
  state=$(read_state "$r")
  done=0; err=0
  if [ -f "$log" ]; then
    grep -q "PARALLEL-GEPA COMPLETE" "$log" && done=1
    err=$(grep -cE "Traceback \(most recent call last\)|CRITICAL|FATAL" "$log" 2>/dev/null)
  fi
  echo "$state|done=$done|err=$err"
}

while true; do
  ts=$(date '+%m-%d %H:%M:%S')
  s1=$(run_status "$R1")
  s2=$(run_status "$R2")
  summary="TICK $ts | 03_verbose_polite: $s1 | 04_confused_drip_feeding: $s2"
  change=0
  for pair in "1:$s1" "2:$s2"; do
    idx="${pair%%:*}"; cur="${pair#*:}"
    eval "old=\$prev$idx"
    if [ "$old" != "$cur" ]; then
      change=1
      case "$cur" in
        *"done=1"*) echo "COMPLETE $ts run$idx -> $cur" ;;
      esac
      errn="${cur##*err=}"
      [ "${errn:-0}" -gt 0 ] 2>/dev/null && echo "ERROR $ts run$idx $cur"
    fi
    eval "prev$idx=\"\$cur\""
  done
  echo "$summary"
  [ "$change" = "1" ] && echo "CHANGE $ts $summary"
  case "$s1|$s2" in
    *"done=1"*"|"*"done=1"*)
      echo "ALLDONE $ts both runs complete"; break ;;
  esac
  sleep "$INTERVAL"
done
