#!/usr/bin/env python3
"""Live monitor for the parallel generational persona search (parallel_gepa.py).

Reads <run_dir>/pgepa_state.json for the per-generation convergence curve, and
scans data/simulations/gepa_opt/<persona_id_base>_* for in-generation, real-time
sim progress (counts + difficulty + infra errors). Read-only; touches no other
file.

    .venv/bin/python experiments/persona_search/scripts/pgepa_monitor.py
    .venv/bin/python experiments/persona_search/scripts/pgepa_monitor.py --watch 10
    .venv/bin/python experiments/persona_search/scripts/pgepa_monitor.py \
        --run-dir data/persona_opt/pgepa_runs/pgepa_p0003seed_run1 --watch 15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SIMULATIONS_DIR, is_infra_broken, read_json  # noqa: E402, DATA_ROOT

DEFAULT_RUN_DIR = str(DATA_ROOT / "data/persona_opt/pgepa_runs/xdom_v5_3domain")
GEPA_OPT = SIMULATIONS_DIR / "gepa_opt"
# LIC (lost_in_conversation) sim artifacts live here (relative to the LIC cwd):
#   data/persona_opt/lic_eval/<persona_id>/eval_NNNN/<domain>/demo_runs.json
LIC_EVAL_DIR = DATA_ROOT / "data/persona_opt/lic_eval"
ACTIVE_WINDOW_S = 150  # a sim dir touched within this many seconds = "active"

C = {
    "g": "\033[0;32m",
    "y": "\033[1;33m",
    "b": "\033[0;34m",
    "c": "\033[0;36m",
    "r": "\033[0;31m",
    "d": "\033[2m",
    "bold": "\033[1m",
    "x": "\033[0m",
}
_SPARK = "▁▂▃▄▅▆▇█"


def col(s, k):
    return f"{C[k]}{s}{C['x']}"


def find_pid(run_dir: str) -> int | None:
    """PID of the *_gepa.py process for this run_dir, if alive."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "_gepa.py"], capture_output=True, text=True
        ).stdout
    except Exception:
        return None
    for pid in out.split():
        try:
            cl = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore")
        except Exception:
            cl = subprocess.run(
                ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
            ).stdout
        # The process may have been launched with a RELATIVE --run-dir while the
        # caller passes an absolute path (or vice versa); match on the last two
        # path components (e.g. "islands_telecom_v1/isl_brinks_00"), which is
        # present in both forms and unique per run_dir.
        suffix = "/".join(Path(run_dir).parts[-2:])
        if run_dir in cl or (suffix and suffix in cl):
            return int(pid)
    return None


def proc_uptime(pid: int) -> str:
    try:
        et = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="], capture_output=True, text=True
        ).stdout.strip()
        return et or "?"
    except Exception:
        return "?"


def sparkline(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return _SPARK[3] * len(vals)
    return "".join(_SPARK[min(7, int((v - lo) / (hi - lo) * 7))] for v in vals)


def scan_sim_dir(d: Path) -> dict | None:
    """Aggregate the (possibly partial) results.json under one persona sim dir.

    When a resume creates a new eval_NNNN, the old one is stale. To avoid
    double-counting (e.g. "35/60" instead of "5/30"), only use results.json
    files from the LATEST eval_NNNN dir per domain.
    """
    files = glob.glob(str(d / "**" / "results.json"), recursive=True)
    if not files:
        return None
    # Group by domain (parent of results.json), keep only the latest eval_NNNN
    # per domain. Path example: .../eval_0002/telecom/results.json
    by_domain: dict[str, list[str]] = {}
    for f in files:
        domain = Path(f).parent.name  # e.g. "telecom"
        by_domain.setdefault(domain, []).append(f)
    latest_files = []
    for domain, domain_files in by_domain.items():
        latest_files.append(max(domain_files, key=lambda f: os.path.getmtime(f)))

    total_tasks = 0
    n_sims = infra = 0
    task_ids_done: set[str] = set()
    rewards: list[float] = []
    mtime = 0.0
    for f in latest_files:
        try:
            mtime = max(mtime, os.path.getmtime(f))
            data = json.load(open(f))
        except Exception:
            continue
        total_tasks += len(data.get("tasks", []) or [])
        for s in data.get("simulations", []) or []:
            n_sims += 1
            tid = s.get("task_id")
            if is_infra_broken(s):
                infra += 1
                continue
            ri = s.get("reward_info") or {}
            r = ri.get("reward")
            if r is not None:
                rewards.append(float(r))
                if tid:
                    task_ids_done.add(tid)
    done_tasks = len(task_ids_done)
    diff = (1.0 - sum(rewards) / len(rewards)) if rewards else None
    return {
        "name": d.name,
        "done": done_tasks,
        "total": total_tasks,
        "n_sims": n_sims,
        "infra": infra,
        "diff": diff,
        "mtime": mtime,
    }


def live_sims(base: str) -> list[dict]:
    dirs = sorted(GEPA_OPT.glob(f"{base}_*"))
    rows = [scan_sim_dir(d) for d in dirs]
    return [r for r in rows if r]


def scan_lic_sim_dir(d: Path) -> dict | None:
    """LIC: aggregate demo_runs.json under lic_eval/<persona_id>/eval_NNNN/<domain>/.

    demo_runs.json: {"summary": [...], "runs": [{task_id, trial_idx, is_correct,
    shards, trace, error?}, ...]}. Multi-trial: n_sims counts every run, done
    counts unique task_ids. difficulty = mean(1-is_correct) over non-infra runs.
    """
    files = glob.glob(str(d / "**" / "demo_runs.json"), recursive=True)
    if not files:
        return None
    by_domain: dict[str, list[str]] = {}
    for f in files:
        domain = Path(f).parent.name  # .../eval_NNNN/<domain>/demo_runs.json
        by_domain.setdefault(domain, []).append(f)
    latest_files = [
        max(fs, key=lambda f: os.path.getmtime(f)) for fs in by_domain.values()
    ]
    task_ids: set[str] = set()
    n_sims = infra = 0
    diffs: list[float] = []
    mtime = 0.0
    for f in latest_files:
        try:
            mtime = max(mtime, os.path.getmtime(f))
            data = json.load(open(f))
        except Exception:
            continue
        for r in data.get("runs", []):
            n_sims += 1
            if "error" in r:
                infra += 1
                continue
            tid = r.get("task_id")
            if tid:
                task_ids.add(tid)
            ic = r.get("is_correct")
            if ic is not None:
                diffs.append(0.0 if ic else 1.0)
    return {
        "name": d.name,
        "done": len(task_ids),
        "total": None,
        "n_sims": n_sims,
        "infra": infra,
        "diff": (sum(diffs) / len(diffs)) if diffs else None,
        "mtime": mtime,
    }


def live_sims_lic(base: str) -> list[dict]:
    if not LIC_EVAL_DIR.exists():
        return []
    dirs = sorted(LIC_EVAL_DIR.glob(f"{base}_*"))
    rows = [scan_lic_sim_dir(d) for d in dirs]
    return [r for r in rows if r]


def fmt_diff(d) -> str:
    return f"{d:.3f}" if d is not None else "  -  "


def _detect_adapter(run_dir: str, adapter: str) -> str:
    if adapter != "auto":
        return adapter
    return (
        "lic"
        if "/lic_" in run_dir
        or run_dir.rstrip("/").endswith(
            tuple(f"lic_{d}" for d in ["math", "code", "database", "actions"])
        )
        or "lic_" in Path(run_dir).name
        else "tau"
    )


def render(
    run_dir: str, base_override: str | None = None, adapter: str = "auto"
) -> str:
    adapter = _detect_adapter(run_dir, adapter)
    is_lic = adapter == "lic"
    sim_root = LIC_EVAL_DIR if is_lic else GEPA_OPT
    rd = Path(run_dir)
    state_p = rd / "pgepa_state.json"
    L = []
    name = rd.name
    pid = find_pid(run_dir)
    alive = col("[running]", "g") if pid else col("[stopped]", "r")
    up = f"up {proc_uptime(pid)}" if pid else ""
    L.append(col(f"╭─ pgepa monitor — {name} " + "─" * max(0, 46 - len(name)), "c"))
    L.append(f"  process  {alive}  " + (f"pid {pid}  {up}" if pid else ""))

    state = read_json(state_p) if state_p.exists() else None
    base = None
    if state:
        cfg = state.get("config", {})
        base = None  # persona_id_base not stored; infer from sim dirs below
        L.append(
            "  config   "
            + col(
                f"G{cfg.get('generations')} K{cfg.get('children_per_gen')} "
                f"b{cfg.get('parents_per_gen')} mb{cfg.get('minibatch')} "
                f"t{cfg.get('top_t')}",
                "d",
            )
            + f"  domains {','.join(cfg.get('domains', []))}  "
            + col(f"valset {len(state.get('instance_ids', []))}", "d")
        )
    else:
        L.append(
            "  config   "
            + col("(gen0 in progress — pgepa_state.json not written yet)", "y")
        )

    # resolve persona_id_base: --base override > state config > newest g00_seed dir
    if base_override:
        base = base_override
    elif state and state.get("config", {}).get("persona_id_base"):
        base = state["config"]["persona_id_base"]
    else:
        # LIC: persona_id_base == run-dir name (lic_math -> base "lic_math"), so
        # glob("lic_math_*") only matches that domain's sim dirs. tau: fall back
        # to the newest g00_seed dir under the tau sim root.
        if is_lic:
            base = rd.name
        else:
            cands = list(sim_root.glob("*_g00_seed"))
            if cands:
                newest = max(cands, key=lambda p: p.stat().st_mtime)
                base = newest.name[: -len("_g00_seed")]

    # ---- per-generation convergence table ----
    if state and state.get("generations"):
        gens = state["generations"]
        is_pgepa = "accepted" in gens[0]
        L.append("")
        if is_pgepa:
            L.append(
                col("  GEN PAR  MB     >PAR   DIFF   VAL   FEAS ATT  RESULT", "bold")
            )
            for gl in gens:
                g = gl.get("gen", "?")
                par = gl.get("parent", "?")
                mb = gl.get("minibatch_mean")
                pmb = gl.get("parent_minibatch_mean")
                diff = gl.get("difficulty_agg")
                val = gl.get("validity_agg")
                feas = gl.get("feasible")
                att = gl.get("mutation_attempts", 1)
                acc = gl.get("accepted", False)
                reason = gl.get("accept_reason", "?")
                dirn = str(gl.get("mutation_direction", ""))[:50]
                mb_s = f"{mb:.3f}" if mb is not None else "  -  "
                pmb_s = f"{pmb:.3f}" if pmb is not None else "  -  "
                diff_s = f"{diff:.3f}" if diff is not None else "  -  "
                val_s = f"{val:.3f}" if val is not None else "  -  "
                feas_s = "✓" if feas else ("✗" if feas is not None else "-")
                if acc:
                    result = col(f"✓ idx{gl.get('idx', '?')}", "g")
                else:
                    result = col(f"✗ {reason}", "r")
                L.append(
                    f"  {g:>3} {par:>3}  {mb_s} {pmb_s} {diff_s} {val_s}  {feas_s}   {att:>2}  {result}"
                )
                if dirn:
                    L.append(col(f"                dir: {dirn}", "d"))
            diffs = [gl.get("difficulty_agg") for gl in gens if gl.get("accepted")]
            if diffs:
                L.append("  trend  " + col(sparkline(diffs), "c"))
        else:
            L.append(
                col(
                    "  GEN  PARENTS        PARETO  SURVIVORS(idx:agg,gate)            BEST",
                    "bold",
                )
            )
            for gl in gens:
                surv = "  ".join(
                    f"{s['idx']}:{s['agg']:.3f}"
                    + ("" if s.get("feasible", True) else col("✗", "r"))
                    for s in gl.get("survivors", [])
                )
                bagg = gl.get("best_agg") or 0.0
                L.append(
                    f"  {gl['gen']:>3}  {str(gl.get('parents', '')):<13}  "
                    f"{gl.get('pareto_size', 0):>5}   {surv:<34}  "
                    + col(f"{bagg:.3f}", "g")
                )
            best_aggs = [gl.get("best_agg") for gl in gens]
            L.append("  trend  " + col(sparkline(best_aggs), "c"))

    # ---- best candidate so far ----
    if state and state.get("candidates"):
        bi = state.get("best_idx")
        bc = next((c for c in state["candidates"] if c["idx"] == bi), None)
        if bc:
            n_conf = sum(1 for c in state["candidates"] if c.get("subscores"))
            L.append("")
            L.append(
                "  best     "
                + col(
                    f"idx{bc['idx']} agg {bc['agg']:.3f} (gen {bc['gen']}, "
                    f"parent {bc['parent']})",
                    "g",
                )
                + col(f"   confirmed candidates: {n_conf}", "d")
            )

    # ---- live in-generation sims ----
    if base:
        rows = live_sims_lic(base) if is_lic else live_sims(base)
        now = time.time()
        active = [r for r in rows if now - r["mtime"] < ACTIVE_WINDOW_S]
        L.append("")
        L.append(
            col(f"  live sims (active dirs, <{ACTIVE_WINDOW_S}s):", "bold")
            if active
            else col("  live sims: none active", "d")
        )
        for r in sorted(active, key=lambda r: r["name"]):
            bar = (
                f"{r['done']:>2}/{r['total']:<2}tasks"
                if r["total"]
                else f"{r['done']:>2}/??tasks"
            )
            n_sims = r.get("n_sims", 0)
            sim_s = f"({n_sims}sims)" if n_sims != r.get("total", 0) else ""
            tag = (
                r["name"][len(base) + 1 :] if r["name"].startswith(base) else r["name"]
            )
            infra = col(f"infra {r['infra']}", "r") if r["infra"] else ""
            L.append(f"    {tag:<16} {bar}{sim_s}  diff {fmt_diff(r['diff'])}  {infra}")
    else:
        L.append("")
        L.append(col("  live sims: no sim dirs yet", "d"))

    L.append(col("╰" + "─" * 60, "c"))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        nargs="+",
        default=[DEFAULT_RUN_DIR],
        help="one or more run dirs to monitor (rendered stacked)",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="persona_id_base override (else read from state / newest seed dir)",
    )
    ap.add_argument(
        "--adapter",
        choices=["auto", "tau", "lic"],
        default="auto",
        help="sim backend to scan for live sims (auto = infer from run-dir name)",
    )
    ap.add_argument(
        "--lic-all-domains",
        action="store_true",
        help="monitor all 4 LIC domain runs: lic_math/code/database/actions",
    )
    ap.add_argument(
        "--lic-suffix",
        default="",
        help="suffix appended to lic_<domain> run-dir names with "
        "--lic-all-domains (e.g. _v2 -> lic_math_v2)",
    )
    ap.add_argument(
        "--watch", type=float, default=0.0, help="refresh every N seconds (0 = once)"
    )
    args = ap.parse_args()

    if args.lic_all_domains:
        args.run_dir = [
            f"{DATA_ROOT}/data/persona_opt/pgepa_runs/lic_{d}{args.lic_suffix}"
            for d in ["math", "code", "database", "actions"]
        ]
        args.adapter = "lic"

    def render_all() -> str:
        return "\n\n".join(render(rd, args.base, args.adapter) for rd in args.run_dir)

    if args.watch <= 0:
        print(render_all())
        return
    try:
        while True:
            os.system("clear")
            print(render_all())
            print(col(f"\n  refresh {args.watch:g}s — Ctrl-C to exit", "d"))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
