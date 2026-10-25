#!/usr/bin/env bash
# Monitor 3 enriched GEPA runs (brinks_02/03/04) side by side.
# Usage:
#   bash experiments/persona_search/scripts/monitor_enriched_3seeds.sh            # one-shot
#   bash experiments/persona_search/scripts/monitor_enriched_3seeds.sh --watch     # refresh every 60s
#   bash experiments/persona_search/scripts/monitor_enriched_3seeds.sh --watch 30  # refresh every 30s
set -euo pipefail

WATCH=0
INTERVAL=60
if [[ "${1:-}" == "--watch" ]]; then
  WATCH=1
  INTERVAL="${2:-60}"
fi

PY="${PY:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

SEEDS=("brinks_02" "brinks_03" "brinks_04")

show_status() {
  clear
  echo "══════════════════════════════════════════════════════════════════"
  echo "  GEPA Monitor — enriched mutation (brinks_02 / 03 / 04)"
  echo "══════════════════════════════════════════════════════════════════"
  echo

  "$PY" << 'PYEOF'
import json, subprocess, os

seeds = ["brinks_02", "brinks_03", "brinks_04"]
run_root = os.environ.get("RUN_ROOT", "data/persona_opt/pgepa_runs")
log_dir = os.environ.get("LOG_DIR", f"{run_root}/logs")

for seed in seeds:
    tag = f"enriched_{seed}_gpt54user"
    state_path = f"{run_root}/{tag}/enriched_{seed}/pgepa_state.json"
    log_path = f"{log_dir}/enriched_{seed}.log"

    print(f"┌─── {seed} ───────────────────────────────────────────")

    # PID
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"parallel_gepa.*enriched_{seed}"],
            stderr=subprocess.DEVNULL).decode().strip()
        pid = out.split("\n")[0]
        try:
            elapsed = subprocess.check_output(
                ["ps", "-p", pid, "-o", "etime="],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            elapsed = "?"
        print(f"│ PID: {pid}  elapsed: {elapsed}")
    except Exception:
        print(f"│ Process not found (finished or not started)")

    # Latest log line
    if os.path.exists(log_path):
        try:
            out = subprocess.check_output(
                ["grep", "-aE", r"\[iter |\[gen |\[twophase\]|EARLY.STOP|parent traces|val_confirm|gate\] attempt",
                 log_path], stderr=subprocess.DEVNULL).decode().strip()
            lines = out.split("\n")
            last = lines[-1] if lines else ""
            if len(last) > 100:
                last = last[:97] + "..."
            print(f"│ Latest: {last}")
        except Exception:
            pass

    # State
    if not os.path.exists(state_path):
        print(f"│ State: not yet created (seed eval in progress)")
        if os.path.exists(log_path):
            try:
                out = subprocess.check_output(["tail", "-3", log_path],
                    stderr=subprocess.DEVNULL).decode().strip()
                for ln in out.split("\n"):
                    if len(ln) > 90:
                        ln = ln[:87] + "..."
                    print(f"│   {ln}")
            except Exception:
                pass
        print(f"└────────────────────────────────────────────────────")
        print()
        continue

    try:
        d = json.load(open(state_path))
    except Exception as e:
        print(f"│ State: error reading ({e})")
        print(f"└────────────────────────────────────────────────────")
        print()
        continue

    cands = d.get("candidates", [])
    gens = d.get("generations", d.get("gen_logs", []))
    n_acc = sum(1 for g in gens if g.get("accepted"))

    best = max(cands, key=lambda c: c.get("difficulty_agg", 0)) if cands else None
    if best:
        best_str = f"idx{best['idx']} gen={best.get('gen',0)} diff={best.get('difficulty_agg',0):.3f}"
    else:
        best_str = "-"

    print(f"│ Iters: {len(gens)}/20  Candidates: {len(cands)}  "
          f"Accepted: {n_acc}/{len(gens)}  Best: {best_str}")

    # Candidate table (compact)
    print(f"│")
    hdr = f"{'idx':>3} {'gen':>3} {'par':>3} {'diff':>6} {'val':>5} {'feas':>5}"
    print(f"│ {hdr}")
    for c in cands:
        ci = c.get("idx", 0)
        cg = c.get("gen", 0)
        cp = c.get("parent")
        cp_s = f"{cp:3d}" if cp is not None else "  -"
        cdiff = c.get("difficulty_agg", 0) or 0
        cval = c.get("validity_agg")
        val_s = f"{cval:.2f}" if cval is not None else "  -  "
        feas = c.get("feasible", "?")
        marker = " *" if best and ci == best["idx"] else ""
        print(f"│ {ci:3d} {cg:3d} {cp_s} {cdiff:6.3f} {val_s:>5} {str(feas):>5}{marker}")

    # Recent iterations (last 5)
    print(f"│")
    print(f"│ Recent:")
    for g in gens[-5:]:
        a = "ACC" if g.get("accepted") else "REJ"
        mb = g.get("minibatch_mean", 0) or 0
        pmb = g.get("parent_minibatch_mean", 0) or 0
        gdiff = g.get("difficulty_agg")
        ds = f"{gdiff:.3f}" if gdiff is not None else "  -  "
        att = g.get("mutation_attempts", "?")
        reason = g.get("accept_reason", "") or ""
        if len(reason) > 25:
            reason = reason[:22] + "..."
        gg = g.get("gen", 0)
        print(f"│   gen{gg:2d} {a} att={att} mb={mb:.3f}(p={pmb:.3f}) diff={ds} {reason}")

    print(f"└────────────────────────────────────────────────────")
    print()
PYEOF
}

# main
if [[ "$WATCH" -eq 1 ]]; then
  trap 'echo; echo "Stopped."; exit 0' INT
  while true; do
    show_status
    echo "── refreshing in ${INTERVAL}s (Ctrl-C to stop) ──"
    sleep "$INTERVAL"
  done
else
  show_status
fi
