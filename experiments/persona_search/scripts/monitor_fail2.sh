#!/usr/bin/env bash
# Monitor fail²/avg@2 GEPA run.
# Usage:
#   bash experiments/persona_search/scripts/monitor_fail2.sh            # one-shot
#   bash experiments/persona_search/scripts/monitor_fail2.sh --watch     # refresh every 60s
#   bash experiments/persona_search/scripts/monitor_fail2.sh --watch 30  # refresh every 30s
set -euo pipefail

WATCH=0
INTERVAL=60
if [[ "${1:-}" == "--watch" ]]; then
  WATCH=1
  INTERVAL="${2:-60}"
fi

PY="${PY:-.venv/bin/python}"
STATE="data/persona_opt/pgepa_runs/fail2_seed0_gpt54user/f2_brinks_00/pgepa_state.json"
LOG="data/persona_opt/pgepa_runs/fail2_seed0_gpt54user/logs/f2_brinks_00.log"

show_status() {
  PID=$(pgrep -f "parallel_gepa.*f2_brinks_00" 2>/dev/null || echo "")

  clear
  echo "════════════════════════════════════════════════════"
  echo "  avg@2 GEPA Monitor (brinks_00 seed)"
  echo "════════════════════════════════════════════════════"

  if [[ -z "$PID" ]]; then
    echo "⚠  Process not found (finished or killed)"
  else
    elapsed=$(ps -p "$PID" -o etime= 2>/dev/null || echo "?")
    echo "  PID: $PID  elapsed: $elapsed"
  fi
  echo

  # latest progress line
  if [[ -f "$LOG" ]]; then
    latest=$(grep "complete\. Avg" "$LOG" | tail -1)
    echo "Latest: $latest"
    echo
  fi

  # state file
  if [[ ! -f "$STATE" ]]; then
    echo "State file not yet created (still in seed eval)"
    echo
    echo "─── Live log tail ───"
    tail -5 "$LOG" 2>/dev/null
    return
  fi

  "$PY" -c "
import json
d = json.load(open('$STATE'))
cands = d.get('candidates', [])
gens = d.get('generations', [])

print(f'Iterations: {len(gens)}/20    Candidates: {len(cands)}')
print()

# best
if cands:
    best = max(cands, key=lambda c: c.get('agg', 0))
    print(f'Best: idx{best[\"idx\"]} gen={best.get(\"gen\",0)} agg={best.get(\"agg\",0):.3f} diff={best.get(\"difficulty_agg\",0):.3f}')
    print()

# candidates table
print('Candidates:')
print(f'  {\"idx\":>3} {\"gen\":>3} {\"agg\":>6} {\"diff\":>6} {\"train\":>6} {\"feas\":>5}')
for c in cands:
    ts = c.get('train_subscores', {})
    tr = sum(ts.values())/len(ts) if ts else 0
    print(f'  {c[\"idx\"]:3d} {c.get(\"gen\",0):3d} {c.get(\"agg\",0):6.3f} {c.get(\"difficulty_agg\",0):6.3f} {tr:6.3f} {str(c.get(\"feasible\",\"?\")):>5}')

# iteration log
print()
acc = sum(1 for g in gens if g.get('accepted'))
print(f'Acceptance rate: {acc}/{len(gens)} ({100*acc/len(gens):.0f}%)' if gens else '')
print()
print('Recent iterations:')
for g in gens[-8:]:
    a = '✅' if g.get('accepted') else '❌'
    mb = g.get('minibatch_mean', 0) or 0
    pmb = g.get('parent_minibatch_mean', 0) or 0
    diff = g.get('difficulty_agg')
    ds = f'{diff:.3f}' if diff is not None else '  -  '
    print(f'  gen{g[\"gen\"]:2d} {a} mb={mb:.3f}(p={pmb:.3f}) diff={ds}  {g.get(\"accept_reason\",\"\")}')

# pareto frontier candidates
if cands:
    val_ids = list(cands[0].get('subscores', {}).keys())
    non_dom = []
    for i, a in enumerate(cands):
        sa = {k: a.get('subscores', {}).get(k, 0) for k in val_ids}
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            sb = {k: b.get('subscores', {}).get(k, 0) for k in val_ids}
            if all(sb[k] >= sa[k] for k in val_ids) and any(sb[k] > sa[k] for k in val_ids):
                dominated = True
                break
        if not dominated:
            non_dom.append(i)
    print(f'\nPareto frontier: {len(non_dom)}/{len(cands)} candidates')
    print(f'  {\"idx\":>3} {\"gen\":>3} {\"agg\":>6} {\"diff\":>6} {\"win\":>4}')
    for i in non_dom:
        c = cands[i]
        ss = c.get('subscores', {})
        wins = sum(1 for v in ss.values() if v > 0)
        print(f'  {c[\"idx\"]:3d} {c.get(\"gen\",0):3d} {c.get(\"agg\",0):6.3f} {c.get(\"difficulty_agg\",0):6.3f} {wins:3d}/{len(ss)}')
"
}

# main
if [[ "$WATCH" -eq 1 ]]; then
  trap 'echo; echo "Stopped."; exit 0' INT
  while true; do
    show_status
    echo
    echo "── refreshing in ${INTERVAL}s (Ctrl-C to stop) ──"
    sleep "$INTERVAL"
  done
else
  show_status
fi
