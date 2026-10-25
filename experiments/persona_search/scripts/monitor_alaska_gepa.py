#!/usr/bin/env python3
"""Monitor parallel GEPA optimization runs.

Usage:
  python experiments/persona_search/scripts/monitor_alaska_gepa.py                  # one-shot
  python experiments/persona_search/scripts/monitor_alaska_gepa.py --watch          # auto-refresh
  python experiments/persona_search/scripts/monitor_alaska_gepa.py --run-root DIR   # custom root
  python experiments/persona_search/scripts/monitor_alaska_gepa.py --seeds a,b,c    # custom seeds
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

PRESETS = {
    "airline": {
        "seeds": ["alaska_00", "alaska_01", "alaska_02", "alaska_03"],
        "root": str(DATA_ROOT / "data/persona_opt/pgepa_runs/alaska_airline_v3"),
        "log_dir": None,
        "max_iters": 20,
    },
    "airline_v4": {
        "seeds": ["alaska_01", "alaska_00", "alaska_03", "alaska_02"],
        "root": str(DATA_ROOT / "data/persona_opt/pgepa_runs/airline_v4"),
        "log_dir": "/tmp/gepa_ar_v2_logs",
        "log_prefix": "airline_",
        "max_iters": 10,
    },
    "retail": {
        "seeds": ["gc_00", "gc_01", "gc_02", "gc_03", "gc_04"],
        "root": str(DATA_ROOT / "data/persona_opt/pgepa_runs/retail_guitar_center_v1"),
        "log_dir": None,
        "max_iters": 20,
    },
    "retail_v2": {
        "seeds": ["gc_01", "gc_02", "gc_00", "gc_03", "gc_04"],
        "root": str(DATA_ROOT / "data/persona_opt/pgepa_runs/retail_v2"),
        "log_dir": "/tmp/gepa_ar_v2_logs",
        "log_prefix": "retail_",
        "max_iters": 10,
    },
    "banking": {
        "seeds": ["bk_00"],
        "root": str(DATA_ROOT / "data/persona_opt/pgepa_runs/banking_v2"),
        "log_dir": None,
        "max_iters": 20,
    },
}
DEFAULT_PRESET = "airline_v4"

PRESET_GROUPS = {
    "all": ["airline_v4", "retail_v2"],
}


# ── log parsing ──────────────────────────────────────────────────────────


def parse_log(log_path: Path) -> dict:
    """Extract structured progress from parallel_gepa stdout log."""
    info = {
        "exists": False,
        "start_time": None,  # datetime of first loguru line
        "seed_score": None,  # float after [gen 0] seed score=
        "seed_validity": None,
        "completed_iters": [],  # list of {iter, accepted, idx, diff, best_idx, best_diff}
        "current_phase": None,  # "seed_val" | "seed_train" | "mutation" | "mb_eval" | "val_eval" | "done"
        "current_iter": None,  # int — which iter is in progress
        "last_status": None,  # latest "Status:" line parsed
        "gate_attempts": None,  # latest [gate] attempt count
        "last_line": "",
    }
    if not log_path.exists():
        return info
    info["exists"] = True
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return info

    if lines:
        info["last_line"] = lines[-1].rstrip()

    current_eval_tag = None  # "seed" | "seed_train" | "gNN_kNN"

    for line in lines:
        line_s = line.strip()

        # First loguru timestamp → start time
        if info["start_time"] is None:
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line_s)
            if m:
                try:
                    info["start_time"] = datetime.strptime(
                        m.group(1), "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass

        # Save: tag → tells us which eval phase
        m = re.search(r"Save: gepa_opt/\S+_(g\d+_\w+)/eval_\d+/", line_s)
        if m:
            current_eval_tag = m.group(1)
            if "g00_seed_train" in current_eval_tag:
                info["current_phase"] = "seed_train"
            elif "g00_seed" in current_eval_tag:
                info["current_phase"] = "seed_val"
            else:
                gm = re.match(r"g(\d+)", current_eval_tag)
                if gm:
                    info["current_iter"] = int(gm.group(1))
                    if "full_diff" in current_eval_tag:
                        info["current_phase"] = "val_eval"
                    else:
                        info["current_phase"] = "mb_eval"

        # Status: N/M complete. Avg reward: X.XX (N=Y). K running: ...
        m = re.match(
            r"Status:\s+(\d+)/(\d+)\s+complete\.\s+Avg reward:\s+([\d.]+|N/A)\s+\(N=(\d+)\)\.\s+(\d+)\s+running",
            line_s,
        )
        if m:
            info["last_status"] = {
                "done": int(m.group(1)),
                "total": int(m.group(2)),
                "avg_reward": None if m.group(3) == "N/A" else float(m.group(3)),
                "n": int(m.group(4)),
                "running": int(m.group(5)),
            }

        # [gen 0] seed score=0.433  raw_difficulty=0.229  validity=0.971
        m = re.search(
            r"\[gen 0\] seed score=([\d.]+)\s+raw_difficulty=([\d.]+)\s+validity=([\d.]+)",
            line_s,
        )
        if m:
            info["seed_score"] = float(m.group(1))
            info["seed_validity"] = float(m.group(3))
            info["current_phase"] = "iterating"

        # [gen 0] confirming seed on ...
        if "[gen 0] confirming seed" in line_s:
            info["current_phase"] = "seed_val"

        # [gen 0] scoring seed on N train tasks
        if "[gen 0] scoring seed" in line_s:
            info["current_phase"] = "seed_train"

        # [iter N] pareto_size=... parent=idxP — iteration start
        m = re.match(r"\s*\[iter (\d+)\] pareto_size=", line_s)
        if m:
            info["current_iter"] = int(m.group(1))
            info["current_phase"] = "mutation"
            info["gate_attempts"] = None
            info["last_status"] = None

        # [gate] attempt N/M
        m = re.search(r"\[gate\] attempt (\d+)/(\d+)", line_s)
        if m:
            info["gate_attempts"] = (int(m.group(1)), int(m.group(2)))
            info["current_phase"] = "mutation"

        # running N task(s) → entering eval within iteration
        m = re.search(r"running (\d+) task\(s\)", line_s)
        if m and info["current_phase"] == "mutation":
            info["current_phase"] = "mb_eval"

        # [iter N] parent=idxM ... -> result — iteration complete
        m = re.match(
            r"\s*\[iter (\d+)\] parent=idx(\d+)\s+lex_dissim=(\d+%\S*)\s+"
            r"(?:sem=([\d.]+)\s+)?lint=(\S+)\s+mb=([\d.]+)\(p=([\d.]+)\)\s+->\s+(\S+)\s+"
            r"best=idx(\d+)\s+diff=([\d.]+)",
            line_s,
        )
        if m:
            it = int(m.group(1))
            result_str = m.group(8)
            accepted = not result_str.startswith("rejected")
            child_idx = None
            reject_reason = None
            if accepted:
                child_idx = int(result_str.replace("idx", ""))
            else:
                rm = re.match(r"rejected\((\w+)\)", result_str)
                reject_reason = rm.group(1) if rm else result_str
            info["completed_iters"].append(
                {
                    "iter": it,
                    "accepted": accepted,
                    "child_idx": child_idx,
                    "reject_reason": reject_reason,
                    "mb_score": float(m.group(6)),
                    "parent_mb_score": float(m.group(7)),
                    "best_idx": int(m.group(9)),
                    "best_diff": float(m.group(10)),
                }
            )
            info["current_phase"] = "iterating"
            info["current_iter"] = it
            info["gate_attempts"] = None
            info["last_status"] = None

        if "SERIAL-PARETO-GEPA COMPLETE" in line_s:
            info["current_phase"] = "done"
        if "EARLY STOP" in line_s:
            info["current_phase"] = "early_stop"

    return info


def parse_log_multi(log_paths: list[Path]) -> dict:
    """Parse and merge multiple log files (original + resumed) in order."""
    if not log_paths:
        return parse_log(Path("/dev/null"))
    merged = parse_log(log_paths[0])
    for extra in log_paths[1:]:
        overlay = parse_log(extra)
        if not overlay["exists"]:
            continue
        for key in (
            "current_phase",
            "current_iter",
            "last_status",
            "gate_attempts",
            "last_line",
        ):
            if overlay[key] is not None:
                merged[key] = overlay[key]
        if overlay["seed_score"] is not None:
            merged["seed_score"] = overlay["seed_score"]
        if overlay["seed_validity"] is not None:
            merged["seed_validity"] = overlay["seed_validity"]
        merged["completed_iters"].extend(overlay["completed_iters"])
    return merged


def load_state(run_dir: Path) -> dict | None:
    state_file = run_dir / "pgepa_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── formatting ───────────────────────────────────────────────────────────


def _sparkline(vals: list[float]) -> str:
    if not vals:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx > mn else 1.0
    return "".join(
        chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in vals
    )


def _elapsed_str(start: datetime | None) -> str:
    if start is None:
        return ""
    delta = datetime.now() - start
    mins = int(delta.total_seconds() / 60)
    if mins < 60:
        return f"{mins}m"
    return f"{mins // 60}h{mins % 60:02d}m"


def _format_now(log: dict) -> str:
    """Describe what is happening RIGHT NOW."""
    phase = log.get("current_phase")
    it = log.get("current_iter")
    status = log.get("last_status")

    status_detail = ""
    if status:
        rew = (
            f"avg_diff={1 - status['avg_reward']:.2f}"
            if status["avg_reward"] is not None
            else ""
        )
        status_detail = (
            f" {status['done']}/{status['total']} sims{' ' + rew if rew else ''}"
        )

    if phase == "seed_val":
        return f"seed val eval{status_detail}"
    if phase == "seed_train":
        return f"seed train eval{status_detail}"
    if phase == "mutation":
        gate = log.get("gate_attempts")
        gate_s = f" (gate attempt {gate[0]}/{gate[1]})" if gate else ""
        return f"iter {it} — mutation{gate_s}"
    if phase == "mb_eval":
        return f"iter {it} — minibatch eval{status_detail}"
    if phase == "val_eval":
        return f"iter {it} — val set eval{status_detail}"
    if phase == "iterating":
        return f"between iterations (iter {it} done)"
    if phase and phase.startswith("gen_"):
        gn = phase.split("_")[1]
        return f"iter {gn} — eval{status_detail}"
    if phase in ("done", "early_stop"):
        return phase.upper()
    return "initializing..."


def _format_history(iters: list[dict], limit: int = 8) -> str:
    """Compact iteration history: 1✓idx2(0.53) 2✗ 3✓idx3(0.57)"""
    if not iters:
        return ""
    recent = iters[-limit:]
    parts = []
    for it in recent:
        n = it["iter"]
        if it["accepted"]:
            parts.append(f"{n}✓idx{it['child_idx']}")
        else:
            parts.append(f"{n}✗")
    return " ".join(parts)


def format_seed(name: str, state: dict | None, log: dict, max_iters: int) -> str:
    lines = []

    n_completed = len(log["completed_iters"])
    is_done = (
        log["current_phase"] in ("done", "early_stop")
        if log["current_phase"]
        else False
    )
    elapsed = _elapsed_str(log.get("start_time"))

    # Progress bar
    pct = n_completed / max_iters if max_iters > 0 else 0
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    iter_label = (
        f"{n_completed}/{max_iters}" if n_completed > 0 or is_done else f"—/{max_iters}"
    )
    status_tag = "DONE" if is_done else ""
    elapsed_s = f"  {elapsed}" if elapsed else ""

    lines.append(f"  {name}  [{bar}]  {iter_label}  {status_tag}{elapsed_s}")

    # Scores line
    if log["seed_score"] is not None:
        seed_s = f"seed={log['seed_score']:.3f}"
        if n_completed > 0:
            best = log["completed_iters"][-1]
            best_diff = best["best_diff"]
            best_idx = best["best_idx"]
            delta = best_diff - log["seed_score"]
            n_acc = sum(1 for i in log["completed_iters"] if i["accepted"])
            lines.append(
                f"  ├─ {seed_s} → best={best_diff:.3f} (idx{best_idx})  "
                f"Δ{delta:+.3f}    accept {n_acc}/{n_completed}"
            )
        else:
            val_s = (
                f"  validity={log['seed_validity']:.3f}"
                if log["seed_validity"] is not None
                else ""
            )
            lines.append(f"  ├─ {seed_s}{val_s}")

    # Current activity
    if not is_done:
        now = _format_now(log)
        connector = "└─" if n_completed == 0 and not log["completed_iters"] else "├─"
        lines.append(f"  {connector} now: {now}")

    # Iteration history
    if log["completed_iters"]:
        hist = _format_history(log["completed_iters"])
        trace = [it["best_diff"] for it in log["completed_iters"]]
        spark = _sparkline(trace) if len(trace) >= 2 else ""
        spark_s = f"  {spark}" if spark else ""
        lines.append(f"  └─ history: {hist}{spark_s}")
    elif not log["exists"]:
        lines.append(f"  └─ (no log file)")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────


def run_monitor(
    root: Path,
    seeds: list[str],
    max_iters: int,
    log_dir_override: Path | None = None,
    log_prefix: str = "",
    label: str = "",
):
    log_dir = log_dir_override or (root / "logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detect config from any available state
    config_line = ""
    for name in seeds:
        state = load_state(root / name)
        if state and state.get("config"):
            cfg = state["config"]
            config_line = (
                f"  {', '.join(cfg.get('domains', ['?']))} | "
                f"{len(seeds)} seeds × {max_iters} iters | "
                f"{cfg.get('agent_llm', '?')} agent + {cfg.get('user_llm', '?')} user"
            )
            break

    title = f"GEPA Monitor{' — ' + label if label else ''}"
    print(f"\n{'=' * 64}")
    print(f"  {title} — {now}")
    if config_line:
        print(config_line)
    print(f"  root: {root}")
    print(f"{'=' * 64}")

    all_done = True
    summary_rows = []
    earliest_start = None

    for name in seeds:
        # Read all log files for this seed (original + resumed)
        glob_pattern = f"{log_prefix}{name}*.log"
        log_files = sorted(log_dir.glob(glob_pattern))
        log = parse_log_multi(log_files)
        state = load_state(root / name)

        # Merge state info into log if state is ahead
        if state:
            gens = state.get("generations", [])
            if len(gens) > len(log["completed_iters"]):
                # State has more data — trust it
                log["completed_iters"] = []
                for g in gens:
                    accepted = g.get("accepted", False)
                    log["completed_iters"].append(
                        {
                            "iter": g.get("gen", 0),
                            "accepted": accepted,
                            "child_idx": g.get("idx") if accepted else None,
                            "reject_reason": g.get("accept_reason")
                            if not accepted
                            else None,
                            "mb_score": g.get("minibatch_mean", 0),
                            "parent_mb_score": g.get("parent_minibatch_mean", 0),
                            "best_idx": g.get("best_idx", 0),
                            "best_diff": g.get("best_agg", 0),
                        }
                    )
            if state.get("best_agg") is not None and log["seed_score"] is None:
                seed_c = next(
                    (c for c in state.get("candidates", []) if c.get("gen") == 0), None
                )
                if seed_c:
                    log["seed_score"] = seed_c.get("difficulty_agg", seed_c.get("agg"))
            if len(gens) >= max_iters and log["current_phase"] not in (
                "done",
                "early_stop",
            ):
                log["current_phase"] = "done"

        print()
        print(format_seed(name, state, log, max_iters))

        is_done = (
            log["current_phase"] in ("done", "early_stop")
            if log["current_phase"]
            else False
        )
        if not is_done:
            all_done = False

        if log.get("start_time") and (
            earliest_start is None or log["start_time"] < earliest_start
        ):
            earliest_start = log["start_time"]

        # Summary row
        n_iters = len(log["completed_iters"])
        seed_s = log["seed_score"] or 0
        best_s = (
            log["completed_iters"][-1]["best_diff"] if log["completed_iters"] else 0
        )
        n_acc = sum(1 for i in log["completed_iters"] if i["accepted"])
        summary_rows.append((name, n_iters, seed_s, best_s, n_acc))

    # Summary table
    has_data = any(r[2] > 0 for r in summary_rows)
    if has_data:
        print(f"\n{'─' * 64}")
        print(
            f"  {'Seed':<12} {'Iters':>7} {'Seed':>7} {'Best':>7} {'Delta':>7} {'Accept':>8}"
        )
        print(f"  {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8}")
        for name, iters, seed, best, acc in summary_rows:
            if seed > 0:
                delta = best - seed if best > 0 else 0
                acc_s = f"{acc}/{iters}" if iters > 0 else "—"
                print(
                    f"  {name:<12} {iters:>4}/{max_iters}  {seed:>7.3f} {best:>7.3f} {delta:>+7.3f} {acc_s:>8}"
                )
            else:
                print(
                    f"  {name:<12} {'—':>4}/{max_iters}  {'—':>7} {'—':>7} {'—':>7} {'—':>8}"
                )
        valid = [r for r in summary_rows if r[2] > 0 and r[3] > 0]
        if len(valid) > 1:
            avg_seed = sum(r[2] for r in valid) / len(valid)
            avg_best = sum(r[3] for r in valid) / len(valid)
            print(f"  {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8}")
            print(
                f"  {'MEAN':<12} {'':>7}  {avg_seed:>7.3f} {avg_best:>7.3f} {avg_best - avg_seed:>+7.3f}"
            )

    # Process check & footer
    result = subprocess.run(
        ["pgrep", "-f", "parallel_gepa"], capture_output=True, text=True
    )
    pids = [p for p in result.stdout.strip().split("\n") if p]
    elapsed_total = _elapsed_str(earliest_start) if earliest_start else "?"

    print(f"\n  {'─' * 60}")
    if pids:
        print(
            f"  {len(pids)} processes alive (pids: {', '.join(pids)}) | elapsed: {elapsed_total}"
        )
    else:
        print(f"  no processes running | elapsed: {elapsed_total}")

    if all_done:
        print(f"  STATUS: ALL COMPLETE")
        run_dirs = " ".join(str(root / n) for n in seeds)
        print(
            f"\n  Next: python experiments/persona_search/scripts/certify_gepa_candidates.py --run-dirs {run_dirs}"
        )
    else:
        # Estimate remaining time
        total_done = sum(r[1] for r in summary_rows)
        total_target = len(seeds) * max_iters
        if total_done > 0 and earliest_start:
            elapsed_min = (datetime.now() - earliest_start).total_seconds() / 60
            rate = elapsed_min / total_done  # min per iter across all seeds
            remaining = (total_target - total_done) * rate / len(seeds)  # parallel
            print(f"  STATUS: IN PROGRESS  |  ~{remaining:.0f}m remaining (est.)")
        else:
            print(f"  STATUS: IN PROGRESS")

    print()
    return all_done


def run_preset(preset_name: str, max_iters_override: int | None = None):
    preset = PRESETS[preset_name]
    root = Path(preset["root"])
    seeds = preset["seeds"]
    max_iters = max_iters_override or preset.get("max_iters", 20)
    log_dir = Path(preset["log_dir"]) if preset.get("log_dir") else None
    log_prefix = preset.get("log_prefix", "")
    if not root.exists():
        print(f"\n  [SKIP] {preset_name}: root not found ({root})")
        return True
    return run_monitor(
        root,
        seeds,
        max_iters,
        log_dir_override=log_dir,
        log_prefix=log_prefix,
        label=preset_name,
    )


def main():
    parser = argparse.ArgumentParser(description="Monitor parallel GEPA runs")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS) + list(PRESET_GROUPS),
        default=DEFAULT_PRESET,
        help="Domain preset or group name (e.g. 'all' for airline_v4+retail_v2)",
    )
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed names")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh")
    parser.add_argument(
        "--interval", type=int, default=60, help="Refresh interval (seconds)"
    )
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    # Resolve preset group → list of preset names
    if args.preset in PRESET_GROUPS:
        preset_names = PRESET_GROUPS[args.preset]
    else:
        preset_names = [args.preset]

    # Single preset with manual overrides
    if len(preset_names) == 1 and (args.run_root or args.seeds):
        preset = PRESETS[preset_names[0]]
        root = Path(args.run_root or preset["root"])
        seeds = (
            [s.strip() for s in args.seeds.split(",")]
            if args.seeds
            else preset["seeds"]
        )
        max_iters = args.max_iters or preset.get("max_iters", 20)
        log_dir = Path(preset["log_dir"]) if preset.get("log_dir") else None
        log_prefix = preset.get("log_prefix", "")
        if not root.exists():
            print(f"Run root not found: {root}")
            sys.exit(1)

        if args.watch:
            try:
                while True:
                    os.system("clear")
                    done = run_monitor(
                        root,
                        seeds,
                        max_iters,
                        log_dir_override=log_dir,
                        log_prefix=log_prefix,
                    )
                    if done:
                        print("All runs complete. Exiting watch mode.")
                        break
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            run_monitor(
                root, seeds, max_iters, log_dir_override=log_dir, log_prefix=log_prefix
            )
        return

    # Multiple presets (group mode)
    if args.watch:
        try:
            while True:
                os.system("clear")
                all_done = True
                for pn in preset_names:
                    done = run_preset(pn, args.max_iters)
                    if not done:
                        all_done = False
                if all_done:
                    print("\nAll runs complete. Exiting watch mode.")
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        for pn in preset_names:
            run_preset(pn, args.max_iters)


if __name__ == "__main__":
    main()
