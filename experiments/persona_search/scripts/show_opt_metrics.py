#!/usr/bin/env python3
"""
GEPA Optimization Metrics Analyzer
Shows detailed optimization data: difficulty, validity, outcome distribution,
error types/rates per iteration and per candidate.

Usage:
    python3 experiments/persona_search/scripts/show_opt_metrics.py             # all runs, summary
    python3 experiments/persona_search/scripts/show_opt_metrics.py p0003_v2    # one run, detailed
    python3 experiments/persona_search/scripts/show_opt_metrics.py --all       # ALL runs, full detail (one shot)
    python3 experiments/persona_search/scripts/show_opt_metrics.py --all --watch            # ^ refresh every 120s
    python3 experiments/persona_search/scripts/show_opt_metrics.py --all --watch --interval 60
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

# ANSI
G, R, Y, B, C, BOLD, DIM, NC = (
    "\033[0;32m",
    "\033[0;31m",
    "\033[1;33m",
    "\033[0;34m",
    "\033[0;36m",
    "\033[1m",
    "\033[2m",
    "\033[0m",
)

RUNS = {
    "p0003_v2": {
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0003_aggressive_20260606_145857"
        ),
        "simdir": str(DATA_ROOT / "data/simulations/gepa_opt/gepa_p0003_v2"),
        "tasks": 35,
        "seed": "Distracted & Rambling",
        "pid": 11084,
    },
    "p0001_v2": {
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0001_aggressive50_20260606_151059"
        ),
        "simdir": str(DATA_ROOT / "data/simulations/gepa_opt/gepa_p0001_v2_50t"),
        "tasks": 50,
        "seed": "Suspicious & Withholding",
        "pid": 18162,
    },
    "p0002_v2": {
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0002_aggressive50_20260606_151059"
        ),
        "simdir": str(DATA_ROOT / "data/simulations/gepa_opt/gepa_p0002_v2_50t"),
        "tasks": 50,
        "seed": "Concise & Trusting",
        "pid": 18216,
    },
}

# The 7 objective dimensions GEPA tracks, with human descriptions
OBJ_DIMS = {
    "difficulty": "agent failure rate (higher = harder persona)",
    "validity": "fraction NOT severe (illegal behavior)",
    "validity_known": "fraction of sims actually judged",
    "timeout_safety": "1 - timeout rate",
    "semantic": "semantic coherence",
    "infra_error": "infra failure rate (lower better)",
    "feasible": "task feasibility",
}

# Outcome types = the "error type" taxonomy
OUTCOMES = {
    "agent_success": ("✓ agent solved", G),
    "genuine_agent_failure": ("✗ agent failed (GOOD)", Y),
    "illegal_user_behavior": ("⚠ SEVERE (persona cheated)", R),
    "timeout_or_max_steps": ("⏱ timeout/max-steps", DIM),
    "infra_error": ("🔧 infra error", DIM),
    "unknown": ("? unknown", DIM),
}


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def live_pid(cfg):
    """Find the CURRENT gepa_runner pid for this run via pgrep on its persona-id.

    The hardcoded cfg['pid'] goes stale across restarts; the persona-id
    (basename of simdir, e.g. gepa_p0001_v2_50t) is stable, so match on that.
    """
    persona_id = os.path.basename(cfg["simdir"])
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"persona-id {persona_id}"],
            capture_output=True,
            text=True,
        )
        pids = [int(x) for x in out.stdout.split()]
        return pids[0] if pids else None
    except Exception:
        return None


def read_state_budget(rundir):
    """Authoritative budget/iteration from gepa_state.bin (survives restarts).

    Returns (total_num_evals, iteration, num_candidates) or (None, None, None).
    This is GEPA's own counter (engine stops at max_metric_calls - total_num_evals),
    NOT an estimate — prefer it over the log-parsed formula which double-counts
    across append-mode restarts.
    """
    import pickle

    f = REPO / rundir / "gepa_state.bin"
    if not f.exists():
        return None, None, None
    try:
        s = pickle.load(open(f, "rb"))
        return (
            s.get("total_num_evals"),
            s.get("i"),
            len(s.get("program_candidates", [])),
        )
    except Exception:
        return None, None, None


def parse_progress(rundir):
    """Authoritative live progress from the tqdm bar in run_log_stderr.txt.

    Returns dict {rollouts, total, elapsed, eta, rate} or None.
    This is the REAL rollout counter (the budget formula elsewhere is an estimate).
    """
    f = REPO / rundir / "run_log_stderr.txt"
    if not f.exists():
        return None
    txt = f.read_text()
    # tqdm bars are separated by \r; grab the very last one with a count.
    bars = re.findall(
        r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)s/rollouts\]", txt
    )
    if not bars:
        # maybe only the 0% bar exists
        m0 = re.search(r"(\d+)/(\d+)\s*\[00:00<\?", txt)
        if m0:
            return {
                "rollouts": int(m0.group(1)),
                "total": int(m0.group(2)),
                "elapsed": "00:00",
                "eta": "?",
                "rate": None,
            }
        return None
    r, t, el, eta, rate = bars[-1]
    return {
        "rollouts": int(r),
        "total": int(t),
        "elapsed": el,
        "eta": eta,
        "rate": float(rate),
    }


def parse_objective_line(line):
    """Parse a GEPA objective-scores dict from a log line."""
    m = re.search(r"\{.*\}", line)
    if not m:
        return None
    txt = m.group(0).replace("'", '"').replace("None", "null")
    try:
        return json.loads(txt)
    except Exception:
        return None


def get_log_metrics(rundir):
    """Extract per-iteration objective scores from run log."""
    logfile = REPO / rundir / "run_log.txt"
    if not logfile.exists():
        return None

    lines = logfile.read_text().splitlines()

    # Per-iteration aggregate objective (the NEW candidate that iteration)
    iter_objs = []  # (iter_num, obj_dict)
    pareto_objs = []  # (iter_num, obj_dict) -- best-so-far across pool
    best_history = []  # (iter_num, best_difficulty)

    for ln in lines:
        if "Objective aggregate scores for new program" in ln:
            it = re.search(r"Iteration (\d+):", ln)
            obj = parse_objective_line(ln)
            if it and obj:
                iter_objs.append((int(it.group(1)), obj))
        elif "Objective pareto front scores" in ln:
            it = re.search(r"Iteration (\d+):", ln)
            obj = parse_objective_line(ln)
            if it and obj:
                pareto_objs.append((int(it.group(1)), obj))
        elif "Best score on valset:" in ln:
            it = re.search(r"Iteration (\d+):", ln)
            sc = re.search(r"Best score on valset:\s*([\d.]+)", ln)
            if it and sc:
                best_history.append((int(it.group(1)), float(sc.group(1))))

    last_iter = 0
    for ln in lines:
        m = re.search(r"Iteration (\d+):", ln)
        if m:
            last_iter = max(last_iter, int(m.group(1)))

    full_evals = sum(1 for l in lines if "Continue to full eval" in l)

    return {
        "iter_objs": iter_objs,
        "pareto_objs": pareto_objs,
        "best_history": best_history,
        "last_iter": last_iter,
        "full_evals": full_evals,
    }


def get_outcome_distribution(simdir, eval_name=None):
    """Aggregate outcome types + judge violations across a sim eval dir.

    If eval_name is None, uses the latest eval_* dir.
    Returns (outcome_counter, violation_counter, severe_rate, n_judged).
    """
    base = REPO / simdir
    if not base.exists():
        return Counter(), Counter(), None, 0

    eval_dirs = sorted(base.glob("eval_*"))
    if not eval_dirs:
        return Counter(), Counter(), None, 0

    target = base / eval_name if eval_name else eval_dirs[-1]
    if not target.exists():
        target = eval_dirs[-1]

    # Read results.json for outcomes (via reward + termination)
    outcomes = Counter()
    results = list(target.rglob("results.json"))
    # Exclude judge subdirs
    results = [r for r in results if "judge" not in str(r)]

    severe_rate = None
    n_judged = 0
    violations = Counter()

    # Read judge summary if present
    judge_summary = list(target.rglob("summary.json"))
    for js in judge_summary:
        if "judge" in str(js):
            try:
                d = json.loads(js.read_text())
                n_judged += d.get("n_sims_judged", 0)
                for vt, cnt in d.get("violation_type_counts", {}).items():
                    violations[vt] += cnt
                vc = d.get("verdict_counts", {})
                sev = d.get("severe_failure_count", 0)
                if n_judged:
                    severe_rate = sev / n_judged
            except Exception:
                pass

    # Outcome from results
    for rf in results:
        try:
            data = json.loads(rf.read_text())
            sims = data.get("simulations", data.get("tasks", []))
            for sim in sims:
                # Approximate outcome from reward/termination
                ri = sim.get("reward_info") or {}
                reward = ri.get("reward")
                term = sim.get("termination_reason", "")
                if term == "infrastructure_error":
                    outcomes["infra_error"] += 1
                elif reward == 1.0:
                    outcomes["agent_success"] += 1
                elif reward == 0.0:
                    outcomes["genuine_agent_failure"] += 1
                elif term in ("max_steps", "timeout"):
                    outcomes["timeout_or_max_steps"] += 1
                else:
                    outcomes["unknown"] += 1
        except Exception:
            pass

    return outcomes, violations, severe_rate, n_judged


def bar(value, width=20, color=G):
    """Horizontal bar for a 0-1 value."""
    filled = int(round(value * width))
    return f"{color}{'█' * filled}{NC}{DIM}{'░' * (width - filled)}{NC}"


def sparkline(values):
    if not values:
        return "─"
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return chars[3] * len(values)
    return "".join(chars[min(int((v - lo) / (hi - lo) * 7), 7)] for v in values)


def show_run_summary(name, cfg):
    """One-run compact summary with all dimensions."""
    rundir = cfg["rundir"]
    alive = live_pid(cfg) is not None
    status = f"{G}● RUNNING{NC}" if alive else f"{R}○ STOPPED{NC}"

    print(f"{BOLD}━━━ {name} {DIM}({cfg['seed']}){NC}  {status}")

    m = get_log_metrics(rundir)
    if not m or not m["best_history"]:
        print(f"  {Y}initializing — no metrics yet{NC}\n")
        return

    # Budget — prefer GEPA's own counter (gepa_state.bin); it survives restarts
    # and is what the engine stops on. Fall back to the log estimate if absent.
    state_calls, state_iter, _ = read_state_budget(rundir)
    if state_calls is not None:
        calls, it = state_calls, state_iter
    else:
        calls = (
            cfg["tasks"] * 2 + m["last_iter"] * 10 + m["full_evals"] * cfg["tasks"] * 2
        )
        it = m["last_iter"]
    print(
        f"  Budget: {bar(calls / 1000)} {calls}/1000  "
        f"iter {it}, {m['full_evals']} accepted"
    )

    # Pareto front (best across pool) — all 7 dims
    if m["pareto_objs"]:
        _, obj = m["pareto_objs"][-1]
        print(f"  {BOLD}Pareto-front objectives (best across candidate pool):{NC}")
        for dim, desc in OBJ_DIMS.items():
            v = obj.get(dim)
            if v is None:
                continue
            # Color: difficulty high=good, infra_error low=good
            if dim == "infra_error":
                col = G if v < 0.05 else (Y if v < 0.15 else R)
            elif dim in (
                "validity",
                "validity_known",
                "timeout_safety",
                "semantic",
                "feasible",
            ):
                col = G if v > 0.9 else (Y if v > 0.6 else R)
            else:  # difficulty
                col = G if v > 0.4 else (Y if v > 0.2 else DIM)
            print(f"    {dim:15s} {bar(v, 16, col)} {v * 100:5.1f}%  {DIM}{desc}{NC}")

    # Difficulty trend
    diffs = [d for _, d in m["best_history"]]
    print(
        f"  {BOLD}Difficulty trend:{NC} {sparkline(diffs)} "
        f"({DIM}iter {m['best_history'][0][0]}→{m['best_history'][-1][0]}{NC})  "
        f"best={G}{max(diffs) * 100:.1f}%{NC}"
    )

    print()


def show_run_detailed(name, cfg):
    """Detailed single-run view with outcome distribution + error analysis."""
    rundir = cfg["rundir"]
    alive = live_pid(cfg) is not None
    status = f"{G}● RUNNING{NC}" if alive else f"{R}○ STOPPED{NC}"
    print(f"{BOLD}{C}{'=' * 70}{NC}")
    print(
        f"{BOLD}{name} — {cfg['seed']}{NC}  {status}  {DIM}({cfg['tasks']} tasks){NC}"
    )
    print(f"{BOLD}{C}{'=' * 70}{NC}\n")

    # Real rollout progress (authoritative — from tqdm bar)
    prog = parse_progress(rundir)
    if prog:
        frac = prog["rollouts"] / prog["total"] if prog["total"] else 0
        eta = f"ETA {prog['eta']}" if prog["rate"] else "warming up"
        rate = f"{prog['rate']:.0f}s/sim" if prog["rate"] else ""
        print(
            f"{BOLD}Progress:{NC} {bar(frac, 24)} "
            f"{prog['rollouts']}/{prog['total']} rollouts  "
            f"{DIM}elapsed {prog['elapsed']} · {eta} · {rate}{NC}"
        )

    m = get_log_metrics(rundir)
    if not m:
        print(f"{Y}No metrics available yet.{NC}\n")
        return

    print(
        f"{BOLD}Iterations:{NC} {m['last_iter']}  |  "
        f"{BOLD}Accepted (full evals):{NC} {m['full_evals']}\n"
    )

    # === Per-iteration objective evolution table ===
    print(f"{BOLD}Per-iteration objective scores (accepted candidates):{NC}")
    print(
        f"  {'iter':>4} {'difficulty':>10} {'validity':>9} {'val_known':>9} "
        f"{'timeout_sf':>10} {'infra_err':>9}"
    )
    print(f"  {'-' * 4} {'-' * 10} {'-' * 9} {'-' * 9} {'-' * 10} {'-' * 9}")
    for it, obj in m["iter_objs"]:
        d = obj.get("difficulty", 0)
        v = obj.get("validity", 0)
        vk = obj.get("validity_known", 0)
        ts = obj.get("timeout_safety", 0)
        ie = obj.get("infra_error", 0)
        dc = G if d > 0.4 else (Y if d > 0.2 else DIM)
        print(
            f"  {it:>4} {dc}{d * 100:>9.1f}%{NC} {v * 100:>8.1f}% {vk * 100:>8.1f}% "
            f"{ts * 100:>9.1f}% {ie * 100:>8.1f}%"
        )
    print()

    # === Outcome distribution (latest eval) ===
    outcomes, violations, severe_rate, n_judged = get_outcome_distribution(
        cfg["simdir"]
    )
    total = sum(outcomes.values())
    if total:
        print(f"{BOLD}Outcome distribution (latest eval, n={total}):{NC}")
        for oc, (label, col) in OUTCOMES.items():
            cnt = outcomes.get(oc, 0)
            if cnt == 0:
                continue
            frac = cnt / total
            print(
                f"  {col}{label:28s}{NC} {bar(frac, 20, col)} "
                f"{cnt:>3} ({frac * 100:.1f}%)"
            )
        print()

    # === Error / violation analysis ===
    if violations or severe_rate is not None:
        print(f"{BOLD}Validity / error analysis (judge results):{NC}")
        if severe_rate is not None:
            col = G if severe_rate <= 0.05 else (Y if severe_rate <= 0.10 else R)
            print(
                f"  Severe-failure rate: {col}{severe_rate * 100:.1f}%{NC} "
                f"{DIM}(gate: ≤5%){NC}  judged={n_judged}"
            )
        if violations:
            print(f"  {BOLD}Violation types:{NC}")
            for vt, cnt in violations.most_common():
                print(f"    {R}• {vt}{NC}: {cnt}")
        print()


def show_baseline():
    """no_persona baseline (cooperative-user ceiling, full 114-task eval)."""
    pids = []
    try:
        import subprocess

        out = subprocess.run(
            ["pgrep", "-f", "opt_no_persona_baseline"], capture_output=True, text=True
        ).stdout
        pids = [p for p in out.split() if p]
    except Exception:
        pass
    status = f"{G}● RUNNING{NC}" if pids else f"{R}○ not running{NC}"
    print(
        f"{BOLD}no_persona baseline{NC} {DIM}(cooperative ceiling, full 114){NC}  "
        f"{status}  {DIM}pid {','.join(pids) or '-'}{NC}\n"
    )


def main():
    args = sys.argv[1:]
    watch = "--watch" in args
    show_all = "--all" in args
    interval = 120
    if "--interval" in args:
        i = args.index("--interval")
        try:
            interval = int(args[i + 1])
        except (IndexError, ValueError):
            pass
    positional = [a for a in args if not a.startswith("--") and not a.isdigit()]

    def render():
        if watch:
            os.system("clear")
        print(
            f"{BOLD}{C}GEPA Optimization Metrics{NC}  "
            f"{DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{NC}\n"
        )

        if positional and positional[0] in RUNS:
            show_run_detailed(positional[0], RUNS[positional[0]])
            show_baseline()
        elif show_all:
            for name, cfg in RUNS.items():
                show_run_detailed(name, cfg)
            show_baseline()
        else:
            for name, cfg in RUNS.items():
                show_run_summary(name, cfg)
            show_baseline()
            print(
                f"{DIM}Tip: --all for full detail on every run; "
                f"add --watch to refresh every {interval}s{NC}"
            )

    if watch:
        try:
            while True:
                render()
                print(f"\n{DIM}Refresh: {interval}s  |  Ctrl+C to exit{NC}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{G}✓{NC} stopped")
    else:
        render()


if __name__ == "__main__":
    main()
