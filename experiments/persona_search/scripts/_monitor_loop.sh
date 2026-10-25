#!/bin/bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 30-min monitor for p0001/p0002 GEPA runs. Authoritative budget from gepa_state.bin.
# Flags TRIGGER when total_num_evals >= 700 (or process already stopped).
cd "$REPO"
LOG=/tmp/gepa_monitor.log
TRIG=/tmp/gepa_trigger
while true; do
  now=$(date -u +%H:%M:%SZ)
  echo "===== CHECK @ $now =====" >> $LOG
  for sid in p0001 p0002; do
    rd="data/persona_opt/gepa_runs/${sid}_aggressive50_20260606_151059"
    read evals iter cands <<< $(PYTHONPATH=experiments/persona_search/scripts .venv/bin/python -c "import pickle;s=pickle.load(open('$rd/gepa_state.bin','rb'));print(s['total_num_evals'],s['i'],len(s['program_candidates']))" 2>/dev/null)
    alive=$(pgrep -f "persona-id gepa_${sid}_v2_50t" >/dev/null && echo OK || echo DOWN)
    err=$(tail -50 "$rd/run_log.txt" 2>/dev/null | grep -c 'returned non-zero')
    sso=$(tail -50 "$rd/run_log.txt" 2>/dev/null | grep -c 'UnauthorizedSSOToken\|SSO session associated')
    trig=""
    if [ -n "$evals" ] && [ "$evals" -ge 700 ] 2>/dev/null; then trig=" *** TRIGGER:700 ***"; echo "$sid evals=$evals iter=$iter" >> $TRIG; fi
    echo "  $sid[$alive]: $evals/1000  iter=$iter cands=$cands  exit1=$err sso=$sso$trig" >> $LOG
  done
  exp=$(for f in ~/.aws/sso/cache/*.json; do python3 -c "import json;print(json.load(open('$f')).get('expiresAt',''))" 2>/dev/null; done | grep -v '^$' | sort | head -1)
  echo "  SSO expiry=$exp  claude=$(pgrep -f 'claude --print'|wc -l|tr -d ' ') tau2=$(pgrep -f 'bin/tau2 run'|wc -l|tr -d ' ')" >> $LOG
  sleep 1800
done
