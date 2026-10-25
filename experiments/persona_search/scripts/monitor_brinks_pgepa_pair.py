#!/usr/bin/env python3
"""Monitor the two Brinks parallel_gepa runs launched by run_brinks_pgepa_pair.sh.

Examples:
  .venv/bin/python experiments/persona_search/scripts/monitor_brinks_pgepa_pair.py
  .venv/bin/python experiments/persona_search/scripts/monitor_brinks_pgepa_pair.py --watch 20
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

RUNS = [
    (
        "brinks_03_verbose_polite",
        str(DATA_ROOT / "data/persona_opt/pgepa_runs/pgepa_brinks_03_verbose_polite"),
        "pgepa_brinks_03_verbose_polite",
    ),
    (
        "brinks_04_confused_drip_feeding",
        str(
            DATA_ROOT
            / "data/persona_opt/pgepa_runs/pgepa_brinks_04_confused_drip_feeding"
        ),
        "pgepa_brinks_04_confused_drip_feeding",
    ),
]

GEPA_OPT = SIMULATIONS_DIR / "gepa_opt"
ACTIVE_WINDOW_S = 180

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


def col(s: str, k: str) -> str:
    return f"{C[k]}{s}{C['x']}"


def strip_ansi(s: str) -> str:
    # Good enough for width-free rendering; this monitor mostly avoids alignment
    # across colored substrings.
    for code in C.values():
        s = s.replace(code, "")
    return s


def find_pid(run_dir: str) -> int | None:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "parallel_gepa.py"], capture_output=True, text=True
        ).stdout
    except Exception:
        return None
    for pid in out.split():
        cmd = subprocess.run(
            ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
        ).stdout
        if run_dir in cmd:
            return int(pid)
    return None


def proc_uptime(pid: int | None) -> str:
    if not pid:
        return ""
    out = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etime="], capture_output=True, text=True
    ).stdout.strip()
    return out or "?"


def fmt(v, nd: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"


def persona_summary(text: str, max_rules: int = 5) -> str:
    """Short readable summary from the persona markdown."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    title = next(
        (ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")),
        "untitled persona",
    )
    bullets: list[str] = []
    in_behavior = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("## how you behave"):
            in_behavior = True
            continue
        if in_behavior and s.startswith("## "):
            break
        if in_behavior and s.startswith("- "):
            bullets.append(s[2:])
        if len(bullets) >= max_rules:
            break
    if bullets:
        return title + " | " + "; ".join(bullets)
    return title


def expected_rollouts(cfg: dict, n_instances: int) -> int:
    g = int(cfg.get("generations") or 0)
    k = int(cfg.get("children_per_gen") or 0)
    mb = int(cfg.get("minibatch") or 0)
    t = int(cfg.get("top_t") or 0)
    return n_instances + g * (k * mb + t * n_instances)


def completed_rollouts(state: dict | None) -> int:
    """Approximate simulator rollouts represented in completed state.

    This counts accepted/rejected completed generation records. It intentionally
    ignores active partial sim dirs; those are shown separately.
    """
    if not state:
        return 0
    n = len(state.get("instance_ids") or [])
    total = n if state.get("candidates") else 0  # seed confirm
    cfg = state.get("config", {})
    mb = int(cfg.get("minibatch") or 0)
    for gl in state.get("generations") or []:
        total += len(gl.get("children") or []) * mb
        # every promoted survivor runs the full difficulty screen once
        total += len(gl.get("survivors") or []) * n
    return total


def scan_sim_dir(d: Path) -> dict | None:
    files = glob.glob(str(d / "**" / "results.json"), recursive=True)
    if not files:
        return None
    done = total = infra = 0
    rewards: list[float] = []
    mtime = 0.0
    for f in files:
        try:
            mtime = max(mtime, os.path.getmtime(f))
            data = json.load(open(f))
        except Exception:
            continue
        total += len(data.get("tasks") or [])
        for sim in data.get("simulations") or []:
            done += 1
            if is_infra_broken(sim):
                infra += 1
                continue
            r = (sim.get("reward_info") or {}).get("reward")
            if r is not None:
                rewards.append(float(r))
    diff = 1.0 - sum(rewards) / len(rewards) if rewards else None
    return {
        "name": d.name,
        "done": done,
        "total": total,
        "infra": infra,
        "diff": diff,
        "mtime": mtime,
    }


def live_sims(base: str) -> list[dict]:
    rows = [scan_sim_dir(d) for d in sorted(GEPA_OPT.glob(f"{base}_*"))]
    return [r for r in rows if r]


def render_one(label: str, run_dir: str, base: str) -> str:
    rd = Path(run_dir)
    state_p = rd / "pgepa_state.json"
    state = read_json(state_p) if state_p.exists() else None
    pid = find_pid(run_dir)
    status = col("running", "g") if pid else col("stopped", "r")

    out: list[str] = []
    out.append(
        col(f"## {label}", "bold")
        + f"  {status}"
        + (f" pid={pid} up={proc_uptime(pid)}" if pid else "")
    )
    out.append(f"run_dir: {run_dir}")
    out.append(f"base:    {base}")

    if not state:
        out.append(
            col(
                "state:   waiting for pgepa_state.json (gen0 may still be running)", "y"
            )
        )
    else:
        cfg = state.get("config", {})
        n = len(state.get("instance_ids") or [])
        done = completed_rollouts(state)
        exp = expected_rollouts(cfg, n)
        pct = (100.0 * done / exp) if exp else 0.0
        out.append(
            "config:  "
            f"G={cfg.get('generations')} K={cfg.get('children_per_gen')} "
            f"b={cfg.get('parents_per_gen')} mb={cfg.get('minibatch')} "
            f"top_t={cfg.get('top_t')} valset={n} "
            f"score='{cfg.get('search_score')}'"
        )
        out.append(
            "budget:  "
            f"completed~{done}/{exp} rollouts ({pct:.1f}%)  "
            f"accepted_candidates={len(state.get('candidates') or [])}"
        )

        best = next(
            (
                c
                for c in state.get("candidates", [])
                if c.get("idx") == state.get("best_idx")
            ),
            None,
        )
        if best:
            out.append(
                "best:    "
                + col(f"idx={best.get('idx')} diff={fmt(best.get('agg'))}", "g")
                + f" gen={best.get('gen')} parent={best.get('parent')} "
                + f"validity={fmt(best.get('validity_agg'))} feasible={best.get('feasible')} "
                + f"direction={best.get('mutation_direction')}"
            )
            out.append("desc:    " + persona_summary(best.get("persona_text") or ""))

        gens = state.get("generations") or []
        if gens:
            out.append("")
            out.append(col("gen summary:", "bold"))
            out.append(
                "  gen | parents      | accepted/rejected | best_diff | survivors"
            )
            for gl in gens:
                surv_bits = []
                for s in gl.get("survivors") or []:
                    ok = s.get("accepted")
                    marker = col("A", "g") if ok else col("R", "r")
                    reason = s.get("accept_reason", "")
                    idx = s.get("idx")
                    diff = s.get("difficulty_agg")
                    pdiff = s.get("parent_difficulty_agg")
                    valid = s.get("validity_agg")
                    surv_bits.append(
                        f"{marker} idx={idx} d={fmt(diff)} p={fmt(pdiff)} "
                        f"v={fmt(valid)} {reason}"
                    )
                acc = sum(1 for s in gl.get("survivors", []) if s.get("accepted"))
                rej = sum(1 for s in gl.get("survivors", []) if not s.get("accepted"))
                out.append(
                    f"  {gl.get('gen'):>3} | {str(gl.get('parents')):<12} | "
                    f"{acc}/{rej:<13} | {fmt(gl.get('best_agg')):<9} | "
                    + " ; ".join(surv_bits)
                )

    rows = live_sims(base)
    now = time.time()
    active = [r for r in rows if now - r["mtime"] < ACTIVE_WINDOW_S]
    if active:
        out.append("")
        out.append(col(f"active sims (<{ACTIVE_WINDOW_S}s):", "bold"))
        for r in sorted(active, key=lambda x: x["name"]):
            tag = (
                r["name"][len(base) + 1 :]
                if r["name"].startswith(base + "_")
                else r["name"]
            )
            infra = col(f"infra={r['infra']}", "r") if r["infra"] else "infra=0"
            out.append(
                f"  {tag:<20} {r['done']:>3}/{r['total']:<3} diff={fmt(r['diff'])} {infra}"
            )
    else:
        out.append("")
        out.append(col("active sims: none", "d"))

    return "\n".join(out)


def render() -> str:
    parts = [
        col(f"Brinks pGEPA pair monitor  {time.strftime('%Y-%m-%d %H:%M:%S')}", "c"),
        "",
    ]
    for label, run_dir, base in RUNS:
        parts.append(render_one(label, run_dir, base))
        parts.append("")
        parts.append("-" * 100)
        parts.append("")
    return "\n".join(parts).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--watch", type=float, default=0.0, help="refresh every N seconds (0 = once)"
    )
    args = ap.parse_args()

    if args.watch <= 0:
        print(render())
        return
    try:
        while True:
            os.system("clear")
            print(render())
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
