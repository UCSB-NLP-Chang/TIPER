#!/bin/bash
# Full benchmark pipeline:
#   Telecom trials 1-2 + Vitabench delivery 3 trials
#   -> Offline validity judge on ALL
#   -> Per-task worst persona (fair-only) for each domain
set -uo pipefail  # no -e: continue even if a step fails

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
VB_VENV="$ROOT/../vitabench/.venv/bin/python"
ENV_FILE="$ROOT/../.env"
OUT_DIR="$ROOT/persona_policies/outputs"
JUDGE_TELECOM="$ROOT/persona_policies/scripts/offline_judge.py"
JUDGE_VITA="$ROOT/persona_policies/scripts/offline_judge_vitabench.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT_DIR/pipeline_logs/pipeline_$TIMESTAMP.log"
mkdir -p "$OUT_DIR/pipeline_logs" "$OUT_DIR/tau2_evals" "$OUT_DIR/vitabench_evals"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "========== PIPELINE START =========="

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Telecom trial 1 (seed=43)
# ═══════════════════════════════════════════════════════════════════════════
TELECOM_T1_OUT="$OUT_DIR/tau2_evals/tau2_telecom_pgepa_test_n10_t1_trial1.json"
log "[1/8] Telecom trial 1 (seed=43)..."
$VENV -u persona_policies/scripts/run_tau2_eval.py \
  --persona-json "$OUT_DIR/generated_personas/tau2_telecom_pgepa_test_n10.json" \
  --domain telecom \
  --agent-llm gpt-5.4 --user-llm gpt-5.4 --reasoning-effort high \
  --max-steps 120 --seed 43 --num-trials 1 --max-workers 30 \
  --save-trajectories --env-file "$ENV_FILE" \
  --out "$TELECOM_T1_OUT" 2>&1 | tee -a "$LOG" || log "[1/8] FAILED (continuing)"
log "[1/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Telecom trial 2 (seed=44)
# ═══════════════════════════════════════════════════════════════════════════
TELECOM_T2_OUT="$OUT_DIR/tau2_evals/tau2_telecom_pgepa_test_n10_t1_trial2.json"
log "[2/8] Telecom trial 2 (seed=44)..."
$VENV -u persona_policies/scripts/run_tau2_eval.py \
  --persona-json "$OUT_DIR/generated_personas/tau2_telecom_pgepa_test_n10.json" \
  --domain telecom \
  --agent-llm gpt-5.4 --user-llm gpt-5.4 --reasoning-effort high \
  --max-steps 120 --seed 44 --num-trials 1 --max-workers 30 \
  --save-trajectories --env-file "$ENV_FILE" \
  --out "$TELECOM_T2_OUT" 2>&1 | tee -a "$LOG" || log "[2/8] FAILED (continuing)"
log "[2/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Vitabench delivery 3 trials (seed 42,43,44)
# ═══════════════════════════════════════════════════════════════════════════
VITA_OUT="$OUT_DIR/vitabench_evals/vitabench_delivery_pgepa_test_n10_t3.json"
log "[3/8] Vitabench delivery 3 trials..."
$VB_VENV -u persona_policies/scripts/run_vitabench_eval.py \
  --persona-json "$OUT_DIR/generated_personas/vitabench_delivery_pgepa_test_n10.json" \
  --domain delivery --language english \
  --agent-llm gpt-5.4 --user-llm gpt-5.4 --reasoning-effort high \
  --evaluator-llm gpt-4.1 --max-steps 100 \
  --seed 42 --num-trials 3 --max-workers 20 \
  --save-trajectories --env-file "$ENV_FILE" \
  --out "$VITA_OUT" 2>&1 | tee -a "$LOG" || log "[3/8] FAILED (continuing)"
log "[3/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Offline judge on telecom trial 1
# ═══════════════════════════════════════════════════════════════════════════
log "[4/8] Offline judge: telecom trial 1..."
TELECOM_T1_TRAJ="${TELECOM_T1_OUT}_trajectories.json"
if [ -f "$TELECOM_T1_OUT" ] && [ -f "$TELECOM_T1_TRAJ" ]; then
  $VENV -u "$JUDGE_TELECOM" \
    --results "$TELECOM_T1_OUT" \
    --trajectories "$TELECOM_T1_TRAJ" \
    --judge-llm gpt-4.1 --max-workers 30 \
    --env-file "$ENV_FILE" 2>&1 | tee -a "$LOG" || log "[4/8] FAILED (continuing)"
else
  log "[4/8] SKIP (missing results or trajectories)"
fi
log "[4/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Offline judge on telecom trial 2
# ═══════════════════════════════════════════════════════════════════════════
log "[5/8] Offline judge: telecom trial 2..."
TELECOM_T2_TRAJ="${TELECOM_T2_OUT}_trajectories.json"
if [ -f "$TELECOM_T2_OUT" ] && [ -f "$TELECOM_T2_TRAJ" ]; then
  $VENV -u "$JUDGE_TELECOM" \
    --results "$TELECOM_T2_OUT" \
    --trajectories "$TELECOM_T2_TRAJ" \
    --judge-llm gpt-4.1 --max-workers 30 \
    --env-file "$ENV_FILE" 2>&1 | tee -a "$LOG" || log "[5/8] FAILED (continuing)"
else
  log "[5/8] SKIP (missing results or trajectories)"
fi
log "[5/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Offline judge on vitabench
# ═══════════════════════════════════════════════════════════════════════════
log "[6/8] Offline judge: vitabench delivery..."
VITA_TRAJ="${VITA_OUT}_trajectories.json"
if [ -f "$VITA_OUT" ] && [ -f "$VITA_TRAJ" ]; then
  $VENV -u "$JUDGE_VITA" \
    --results "$VITA_OUT" \
    --trajectories "$VITA_TRAJ" \
    --judge-llm gpt-4.1 --max-workers 30 \
    --env-file "$ENV_FILE" 2>&1 | tee -a "$LOG" || log "[6/8] FAILED (continuing)"
else
  log "[6/8] SKIP (missing results or trajectories)"
fi
log "[6/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Telecom per-task worst persona (fair-only, all 3 trials)
# ═══════════════════════════════════════════════════════════════════════════
log "[7/8] Telecom per-task worst persona (fair-only)..."
$VENV -c "
import json
from collections import defaultdict
from pathlib import Path

judged_files = [
    '$OUT_DIR/tau2_evals/tau2_telecom_pgepa_test_n10_t1.judged.json',
    '$TELECOM_T1_OUT'.replace('.json', '.judged.json'),
    '$TELECOM_T2_OUT'.replace('.json', '.judged.json'),
]

all_eps = []
for jf in judged_files:
    p = Path(jf)
    if p.exists():
        d = json.loads(p.read_text())
        all_eps.extend(d['results'])
    else:
        print(f'[warn] missing: {jf}')

fair = [r for r in all_eps if r.get('judge_fair') == True]
unfair = [r for r in all_eps if r.get('judge_fair') == False]
unknown = [r for r in all_eps if r.get('judge_fair') is None]
print(f'[TELECOM] Total: {len(all_eps)} | Fair: {len(fair)} | Unfair: {len(unfair)} | Unknown: {len(unknown)}')

for r in fair:
    r['eff'] = r['judge_reward'] if r['judge_reward'] is not None else r.get('reward', 0)

# Per (task, persona): avg over trials
tp = defaultdict(list)
for r in fair:
    tp[(r['task_id'], r['persona_id'])].append(r['eff'])

persona_avg = {k: sum(v)/len(v) for k, v in tp.items()}

# Per task: worst persona
tw = defaultdict(list)
for (tid, pid), avg_r in persona_avg.items():
    tw[tid].append((pid, avg_r))

worst = []
for tid, entries in tw.items():
    w = min(entries, key=lambda x: x[1])
    worst.append((tid, w[0], w[1], len(entries)))

worst.sort(key=lambda x: x[2])
rewards = [w[2] for w in worst]
sr = sum(1 for r in rewards if r >= 1.0) / len(rewards)
print(f'Tasks: {len(worst)}')
print(f'Worst-persona mean reward: {sum(rewards)/len(rewards):.3f}')
print(f'Worst-persona success rate: {sr:.1%}')
print()
print('Distribution:')
for lo, hi, label in [(0,0.2,'0.0-0.2'), (0.2,0.4,'0.2-0.4'), (0.4,0.6,'0.4-0.6'), (0.6,0.8,'0.6-0.8'), (0.8,1.0,'0.8-1.0')]:
    n = sum(1 for r in rewards if lo <= r < hi)
    print(f'  {label}: {n} tasks')
print()
print('Worst 10 tasks:')
for tid, pid, avg_r, n_personas in worst[:10]:
    short = tid.split('|')[0].replace('[[','').replace(']','')[:60]
    print(f'  {short} | {pid:<30} | avg={avg_r:.3f} (n={n_personas})')
" 2>&1 | tee -a "$LOG"
log "[7/8] Done."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Vitabench per-task worst persona (fair-only, all 3 trials)
# ═══════════════════════════════════════════════════════════════════════════
log "[8/8] Vitabench per-task worst persona (fair-only)..."
$VENV -c "
import json
from collections import defaultdict
from pathlib import Path

vita_judged = '$VITA_OUT'.replace('.json', '.judged.json')
if not Path(vita_judged).exists():
    print(f'[warn] vitabench judged file not found: {vita_judged}')
    exit(0)

vd = json.loads(Path(vita_judged).read_text())
all_eps = vd['results']

fair = [r for r in all_eps if r.get('judge_fair') == True]
unfair = [r for r in all_eps if r.get('judge_fair') == False]
unknown = [r for r in all_eps if r.get('judge_fair') is None]
print(f'[VITABENCH] Total: {len(all_eps)} | Fair: {len(fair)} | Unfair: {len(unfair)} | Unknown: {len(unknown)}')

for r in fair:
    r['eff'] = r['judge_reward'] if r['judge_reward'] is not None else r.get('reward', 0)

# Per (task, persona): avg over trials
tp = defaultdict(list)
for r in fair:
    tp[(str(r['task_id']), str(r['persona_id']))].append(r['eff'])

persona_avg = {k: sum(v)/len(v) for k, v in tp.items()}

tw = defaultdict(list)
for (tid, pid), avg_r in persona_avg.items():
    tw[tid].append((pid, avg_r))

worst = []
for tid, entries in tw.items():
    w = min(entries, key=lambda x: x[1])
    worst.append((tid, w[0], w[1], len(entries)))

worst.sort(key=lambda x: x[2])
rewards = [w[2] for w in worst]
sr = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0
print(f'Tasks: {len(worst)}')
print(f'Worst-persona mean reward: {sum(rewards)/len(rewards):.3f}' if rewards else 'No rewards')
print(f'Worst-persona success rate: {sr:.1%}')
print()
print('Distribution:')
for lo, hi, label in [(0,0.2,'0.0-0.2'), (0.2,0.4,'0.2-0.4'), (0.4,0.6,'0.4-0.6'), (0.6,0.8,'0.6-0.8'), (0.8,1.0,'0.8-1.0')]:
    n = sum(1 for r in rewards if lo <= r < hi)
    print(f'  {label}: {n} tasks')
print()
print('Worst 10 tasks:')
for tid, pid, avg_r, n_personas in worst[:10]:
    short = str(tid)[:60]
    print(f'  {short} | {pid:<30} | avg={avg_r:.3f} (n={n_personas})')
" 2>&1 | tee -a "$LOG"
log "[8/8] Done."

log "========== PIPELINE DONE =========="
log "Output files:"
log "  Telecom: $TELECOM_T1_OUT, $TELECOM_T2_OUT (+ judged variants)"
log "  Vitabench: $VITA_OUT (+ judged variant)"
log "  Log: $LOG"
