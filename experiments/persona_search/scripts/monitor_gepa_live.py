#!/usr/bin/env python3
"""
GEPA Optimization Live Monitor
Real-time dashboard for tracking multiple GEPA persona optimization runs
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)


# ANSI colors
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    NC = "\033[0m"  # No Color


# Configuration
RUNS = {
    "p0003_v2": {
        "pid": 11084,
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0003_aggressive_20260606_145857"
        ),
        "tasks": 35,
        "seed": "Distracted & Rambling",
    },
    "p0001_v2": {
        "pid": 18162,
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0001_aggressive50_20260606_151059"
        ),
        "tasks": 50,
        "seed": "Suspicious & Withholding",
    },
    "p0002_v2": {
        "pid": 18216,
        "rundir": str(
            DATA_ROOT / "data/persona_opt/gepa_runs/p0002_aggressive50_20260606_151059"
        ),
        "tasks": 50,
        "seed": "Concise & Trusting",
    },
}

REPO_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)


def check_process_alive(pid: int) -> bool:
    """Check if process is running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_process_runtime(pid: int) -> str:
    """Get process elapsed time"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="], capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def draw_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Draw a progress bar"""
    filled = int(current * width / total)
    empty = width - filled
    percent = int(current * 100 / total)

    bar = f"{Colors.GREEN}{'█' * filled}{Colors.NC}{'░' * empty}"
    return f"[{bar}] {percent:3d}% ({current}/{total})"


def draw_sparkline(values: List[float]) -> str:
    """Draw a mini sparkline graph"""
    if not values:
        return "─"

    bars = "▁▂▃▄▅▆▇█"
    min_val, max_val = min(values), max(values)

    if max_val == min_val:
        return bars[3] * len(values)

    result = ""
    for val in values:
        norm = (val - min_val) / (max_val - min_val)
        idx = min(int(norm * 7), 7)
        result += bars[idx]

    return result


def parse_run_log(logfile: Path) -> Dict:
    """Extract key metrics from run log"""
    if not logfile.exists():
        return {
            "last_iter": 0,
            "full_evals": 0,
            "best_score": None,
            "seed_score": None,
            "score_history": [],
            "activity": "initializing...",
        }

    with open(logfile) as f:
        lines = f.readlines()

    # Parse iterations
    iter_lines = [l for l in lines if "Iteration" in l and ":" in l]
    last_iter = 0
    if iter_lines:
        try:
            last_iter = int(iter_lines[-1].split("Iteration")[1].split(":")[0].strip())
        except:
            pass

    # Count full evals
    full_evals = sum(1 for l in lines if "Continue to full eval" in l)

    # Get best score
    best_lines = [l for l in lines if "Best score on valset:" in l]
    best_score = None
    if best_lines:
        try:
            best_score = float(best_lines[-1].split(":")[-1].strip())
        except:
            pass

    # Get seed score (iter 0)
    seed_score = None
    for i, line in enumerate(lines):
        if "Iteration 0:" in line:
            # Look for valset score in next few lines
            for j in range(i, min(i + 5, len(lines))):
                if "Valset score" in lines[j]:
                    try:
                        seed_score = float(lines[j].split(":")[-1].strip())
                        break
                    except:
                        pass
            break

    # Score history (last 15)
    score_history = []
    for line in best_lines[-15:]:
        try:
            score = float(line.split(":")[-1].strip())
            score_history.append(score)
        except:
            pass

    # Current activity
    last_20 = lines[-20:]
    activity = "iterating..."
    for line in reversed(last_20):
        if "Continue to full eval" in line:
            activity = "running full valset eval"
            break
        elif "New subsample score" in line:
            activity = "evaluating minibatch"
            break
        elif "Selected program" in line:
            activity = "selecting next parent"
            break

    return {
        "last_iter": last_iter,
        "full_evals": full_evals,
        "best_score": best_score,
        "seed_score": seed_score,
        "score_history": score_history,
        "activity": activity,
    }


def get_candidate_count(rundir: Path) -> int:
    """Get number of candidates in pool"""
    cands_file = rundir / "candidates.json"
    if not cands_file.exists():
        return 0

    try:
        with open(cands_file) as f:
            return len(json.load(f))
    except:
        return 0


def get_latest_mutation(rundir: Path) -> Optional[str]:
    """Get preview of latest mutation"""
    cands_file = rundir / "candidates.json"
    if not cands_file.exists():
        return None

    try:
        with open(cands_file) as f:
            cands = json.load(f)

        if len(cands) <= 1:  # Only seed
            return None

        # Get latest (non-seed)
        latest = cands[-1]["persona_text"]
        lines = [l.strip() for l in latest.split("\n") if l.strip().startswith("- ")]

        # Get first behavior that's not in seed (typically line 6+)
        if len(lines) > 5:
            return lines[5][:85] + "..." if len(lines[5]) > 85 else lines[5]

        return None
    except:
        return None


def get_system_stats() -> Dict:
    """Get system-level statistics"""
    try:
        gepa = subprocess.run(
            ["pgrep", "-f", "gepa_runner.py"], capture_output=True, text=True
        )
        tau2 = subprocess.run(
            ["pgrep", "-f", "tau2 run"], capture_output=True, text=True
        )
        claude = subprocess.run(
            ["pgrep", "-f", "claude --print"], capture_output=True, text=True
        )

        uptime_out = subprocess.run(["uptime"], capture_output=True, text=True)
        load = uptime_out.stdout.strip().split("load averages:")[-1].strip()

        return {
            "gepa_count": len([l for l in gepa.stdout.split("\n") if l]),
            "tau2_count": len([l for l in tau2.stdout.split("\n") if l]),
            "claude_count": len([l for l in claude.stdout.split("\n") if l]),
            "load": load,
        }
    except:
        return {"gepa_count": 0, "tau2_count": 0, "claude_count": 0, "load": "unknown"}


def check_recent_errors(rundir: Path) -> int:
    """Count recent errors in log"""
    logfile = rundir / "run_log.txt"
    if not logfile.exists():
        return 0

    try:
        with open(logfile) as f:
            lines = f.readlines()

        # Check last 100 lines for errors (excluding benign LiteLLM)
        last_100 = lines[-100:]
        errors = [
            l
            for l in last_100
            if any(word in l.lower() for word in ["error", "exception", "failed"])
            and "LiteLLM" not in l
        ]
        return len(errors)
    except:
        return 0


def display_run(name: str, config: Dict, stats: Dict):
    """Display info for one run"""
    pid = config["pid"]
    rundir = REPO_ROOT / config["rundir"]
    tasks = config["tasks"]
    seed_type = config["seed"]

    print(
        f"{Colors.BOLD}━━━ {name} {Colors.DIM}(Seed: {seed_type}){Colors.NC} "
        f"{Colors.BOLD}{'━' * (75 - len(name) - len(seed_type) - 10)}{Colors.NC}"
    )

    # Check process
    if not check_process_alive(pid):
        print(f"  {Colors.RED}✗ STOPPED{Colors.NC} (PID {pid} not found)")
        print()
        return

    runtime = get_process_runtime(pid)
    print(
        f"  {Colors.GREEN}✓ RUNNING{Colors.NC}  "
        f"Runtime: {Colors.CYAN}{runtime}{Colors.NC}  PID: {pid}"
    )

    # Parse log
    logfile = rundir / "run_log.txt"
    log_data = parse_run_log(logfile)

    # Calculate budget
    seed_cost = tasks * 2
    minibatch_cost = log_data["last_iter"] * 10
    fulleval_cost = log_data["full_evals"] * tasks * 2
    total_calls = seed_cost + minibatch_cost + fulleval_cost
    remaining = 1000 - total_calls

    # Budget bar
    print()
    print(f"  {Colors.BOLD}Budget:{Colors.NC} {draw_progress_bar(total_calls, 1000)}")
    print(f"          Remaining: {Colors.YELLOW}{remaining}{Colors.NC} calls")

    # Progress
    cand_count = get_candidate_count(rundir)
    print()
    print(f"  {Colors.BOLD}Progress:{Colors.NC}")
    print(
        f"    Iterations: {Colors.CYAN}{log_data['last_iter']}{Colors.NC}  │  "
        f"Full evals: {Colors.CYAN}{log_data['full_evals']}{Colors.NC}  │  "
        f"{cand_count} candidates"
    )

    # Performance
    if log_data["best_score"] is not None:
        difficulty = log_data["best_score"] * 100
        seed_diff = (log_data["seed_score"] or 0) * 100
        delta = difficulty - seed_diff

        print()
        print(f"  {Colors.BOLD}Performance:{Colors.NC}")
        print(
            f"    Best difficulty: {Colors.GREEN}{difficulty:.1f}%{Colors.NC}  "
            f"(seed: {Colors.DIM}{seed_diff:.1f}%{Colors.NC}, "
            f"Δ: {Colors.YELLOW}{delta:+.1f}pp{Colors.NC})"
        )

        # Sparkline
        if log_data["score_history"]:
            sparkline = draw_sparkline(log_data["score_history"])
            iter_start = max(
                0, log_data["last_iter"] - len(log_data["score_history"]) + 1
            )
            iter_end = log_data["last_iter"]
            print(f"    Trend (last {len(log_data['score_history'])}): {sparkline}")
            print(
                f"                     {Colors.DIM}iter {iter_start} → {iter_end}{Colors.NC}"
            )

    # Activity
    print()
    print(
        f"  {Colors.BOLD}Status:{Colors.NC} {Colors.BLUE}{log_data['activity']}{Colors.NC}"
    )

    # Latest mutation preview
    latest_mut = get_latest_mutation(rundir)
    if latest_mut and log_data["full_evals"] > 0:
        print(f"  {Colors.DIM}Latest mutation:{Colors.NC}")
        print(f"    {Colors.DIM}{latest_mut}{Colors.NC}")

    print()


def display_dashboard():
    """Main dashboard display"""
    os.system("clear")

    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * 76}╗{Colors.NC}")
    print(
        f"{Colors.BOLD}{Colors.CYAN}║{Colors.NC}  "
        f"{Colors.BOLD}GEPA Persona Optimization - Live Monitor{Colors.NC}"
        f"{'':34}{Colors.BOLD}{Colors.CYAN}║{Colors.NC}"
    )
    print(
        f"{Colors.BOLD}{Colors.CYAN}║{Colors.NC}  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"{'':54}{Colors.BOLD}{Colors.CYAN}║{Colors.NC}"
    )
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * 76}╝{Colors.NC}")
    print()

    # Display each run
    stats = get_system_stats()
    for name in ["p0003_v2", "p0001_v2", "p0002_v2"]:
        display_run(name, RUNS[name], stats)

    # System summary
    print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 78}{Colors.NC}")
    print(f"{Colors.BOLD}System Status:{Colors.NC}")
    print(
        f"  GEPA runners: {Colors.CYAN}{stats['gepa_count']}{Colors.NC}  │  "
        f"tau2: {Colors.CYAN}{stats['tau2_count']}{Colors.NC}  │  "
        f"claude workers: {Colors.CYAN}{stats['claude_count']}{Colors.NC}"
    )
    print(f"  System load: {Colors.DIM}{stats['load']}{Colors.NC}")

    # Check errors
    total_errors = sum(
        check_recent_errors(REPO_ROOT / RUNS[name]["rundir"]) for name in RUNS
    )
    if total_errors > 0:
        print(
            f"  {Colors.YELLOW}⚠{Colors.NC} Recent errors: {total_errors} (check logs)"
        )
    else:
        print(f"  {Colors.GREEN}✓{Colors.NC} No recent errors")

    # Completion estimate
    max_remaining = 0
    for name in RUNS:
        rundir = REPO_ROOT / RUNS[name]["rundir"]
        logfile = rundir / "run_log.txt"
        log_data = parse_run_log(logfile)
        tasks = RUNS[name]["tasks"]

        total_calls = (
            tasks * 2 + log_data["last_iter"] * 10 + log_data["full_evals"] * tasks * 2
        )
        remaining = 1000 - total_calls
        max_remaining = max(max_remaining, remaining)

    print()
    print(f"{Colors.BOLD}Estimated completion:{Colors.NC}")
    if max_remaining > 500:
        print(f"  {Colors.YELLOW}~6-8 hours{Colors.NC} (early stage)")
    elif max_remaining > 200:
        print(f"  {Colors.YELLOW}~2-4 hours{Colors.NC}")
    elif max_remaining > 0:
        print(f"  {Colors.CYAN}~1-2 hours{Colors.NC}")
    else:
        print(f"  {Colors.GREEN}All runs complete!{Colors.NC}")

    print()
    print(f"{Colors.DIM}Press Ctrl+C to exit  │  Refresh: 30s{Colors.NC}")


def main():
    """Main loop"""
    print("Starting GEPA Live Monitor...")
    print("Monitoring 3 runs: p0003_v2, p0001_v2, p0002_v2")
    print()
    time.sleep(2)

    try:
        while True:
            display_dashboard()
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}✓{Colors.NC} Monitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
