#!/usr/bin/env python3
"""Real-time monitor for lineage-based persona optimization.

.venv/bin/python experiments/persona_search/scripts/monitor.py
.venv/bin/python experiments/persona_search/scripts/monitor.py --watch 10   # refresh every 10s
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

STATE_PATH = DATA_ROOT / "data/persona_opt/lineage_state.json"
SIM_PATTERN = str(
    DATA_ROOT / "data/simulations/persona_opt/round_*/p*/telecom/results.json"
)
LOG_PATH = Path("/tmp/lineage_loop.log")

_INFRA_MARKERS = [
    "cc_user_con: unrecoverable",
    "INFRASTRUCTURE_ERROR",
    "persistent CC session",
]


def _is_infra(sim):
    if sim.get("termination_reason") == "infrastructure_error":
        return True
    for m in sim.get("messages", []):
        c = m.get("content")
        if isinstance(c, str) and any(mk in c for mk in _INFRA_MARKERS):
            return True
    return False


def load_state():
    if not STATE_PATH.exists():
        return None
    return json.loads(STATE_PATH.read_text())


def check_running():
    import subprocess

    r = subprocess.run(["pgrep", "-f", "lineage_loop"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def fmt_score(v):
    if v is None:
        return "   -   "
    return f"  {v:.3f}"


def print_status():
    os.system("clear" if os.name != "nt" else "cls")
    running = check_running()
    state = load_state()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          LINEAGE PERSONA OPTIMIZATION MONITOR          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Process: {'🟢 RUNNING' if running else '🔴 STOPPED'}")

    if not state:
        print("  No state file found.")
        return

    print(f"  Round:   {state['current_round']}")
    print()

    # Lineage table
    print("  ┌────────┬─────┬─────────┬─────────┬─────────┬──────────┐")
    print("  │ Lineage│ Gen │  Latest │   Best  │ Stalled │  Status  │")
    print("  ├────────┼─────┼─────────┼─────────┼─────────┼──────────┤")
    for pid in ["p0001", "p0002", "p0003"]:
        l = state["lineages"][pid]
        h = l["history"]
        latest_score = h[-1].get("score") if h else None
        print(
            f"  │ {pid}  │ {l['generation']:>3} │"
            f"{fmt_score(latest_score)} │"
            f"{fmt_score(l['best_score'])} │"
            f"   {l['stalled_rounds']}   │"
            f" {l['status']:<8} │"
        )
    print("  └────────┴─────┴─────────┴─────────┴─────────┴──────────┘")

    # Score history per lineage
    for pid in ["p0001", "p0002", "p0003"]:
        h = state["lineages"][pid]["history"]
        if not h:
            continue
        scores = [
            f"{e['score']:.3f}" if e.get("score") is not None else "  -  " for e in h
        ]
        dirs = [e.get("mutation_direction", "") for e in h]
        print(f"\n  {pid} history:")
        for i, e in enumerate(h):
            s = f"{e['score']:.3f}" if e.get("score") is not None else "  -  "
            d = e.get("mutation_direction", "")
            j = e.get("status", "")
            marker = " *" if e.get("improved") else ""
            extra = ""
            if e.get("retry_attempt"):
                extra = f" retry#{e['retry_attempt']}"
            if e.get("reverted_to") is not None:
                marker += f" →gen{e['reverted_to']}"
            sr = e.get("severe_rate")
            sr_str = f" sr={sr:.2f}" if sr and sr > 0 else ""
            print(
                f"    gen {e['gen']:>2} │ score={s} │ judge={j:<7} │ {d}{marker}{sr_str}{extra}"
            )

    # In-progress eval — show all lineages for current round side by side
    in_progress = []
    for f in sorted(glob.glob(SIM_PATTERN)):
        r = json.loads(Path(f).read_text())
        sims = r.get("simulations", [])
        done = [s for s in sims if s.get("reward_info")]
        total = len(r.get("tasks", []))
        avg = sum(s["reward_info"]["reward"] for s in done) / len(done) if done else 0

        # Failure breakdown
        n_infra = 0
        n_timeout = 0
        n_maxsteps = 0
        n_normal_fail = 0
        for s in sims:
            ri = s.get("reward_info")
            tr = s.get("termination_reason", "")
            if tr == "infrastructure_error" or _is_infra(s):
                n_infra += 1
            elif ri and ri.get("reward", 1) < 1.0:
                if tr == "timeout":
                    n_timeout += 1
                elif tr == "max_steps":
                    n_maxsteps += 1
                else:
                    n_normal_fail += 1

        parts = f.split("/")
        rnd, pid = parts[-4], parts[-3]
        in_progress.append(
            (
                rnd,
                pid,
                len(done),
                total,
                avg,
                len(done) >= total,
                n_infra,
                n_timeout,
                n_maxsteps,
                n_normal_fail,
            )
        )

    if in_progress:
        # Group by round
        by_round = {}
        for entry in in_progress:
            rnd = entry[0]
            by_round.setdefault(rnd, []).append(entry)

        # Show latest round(s) with any activity
        for rnd in sorted(by_round)[-1:]:
            entries = sorted(by_round[rnd], key=lambda x: x[1])
            running = [e for e in entries if not e[5]]
            done_list = [e for e in entries if e[5]]
            if running:
                print(f"\n  ⏳ In-progress evals ({rnd}):")
                for rnd, pid, done, total, avg, _, ni, nt, nm, nf in running:
                    bar_len = 20
                    filled = int(bar_len * done / total) if total else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"    {pid}: [{bar}] {done}/{total}  avg={avg:.3f}")
            if done_list:
                print(f"\n  ✅ Completed evals ({rnd}):")
                for rnd, pid, done, total, avg, _, ni, nt, nm, nf in done_list:
                    n_success = done - ni - nt - nm - nf
                    fail_parts = []
                    if nf:
                        fail_parts.append(f"fail={nf}")
                    if nt:
                        fail_parts.append(f"timeout={nt}")
                    if nm:
                        fail_parts.append(f"maxstep={nm}")
                    if ni:
                        fail_parts.append(f"infra={ni}")
                    detail = (
                        f"  ok={n_success} {' '.join(fail_parts)}" if fail_parts else ""
                    )
                    print(f"    {pid}: {done}/{total}  avg={avg:.3f}{detail}")

    # Log tail
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text().splitlines()
        key_lines = [
            l
            for l in lines
            if any(
                k in l
                for k in [
                    "score=",
                    "ROUND",
                    "judge=",
                    "mutation:",
                    "CONVERGED",
                    "ERROR",
                    "Traceback",
                ]
            )
        ]
        if key_lines:
            print(f"\n  📋 Recent log events:")
            for l in key_lines[-5:]:
                print(f"    {l.strip()[:90]}")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--watch", type=int, default=0, help="refresh interval in seconds (0=once)"
    )
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                print_status()
                print(f"  Refreshing every {args.watch}s... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        print_status()


if __name__ == "__main__":
    main()
