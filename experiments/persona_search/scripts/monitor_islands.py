#!/usr/bin/env python3
"""Aggregate live monitor for the telecom ISLAND-MODEL persona search.

The island run is 5 independent parallel_gepa.py processes (one run_dir each,
RUN_ROOT/isl_<seed>) plus ONE shared file-locked seen pool. pgepa_monitor.py only
watches a single run_dir, so this wraps it: it parses run_islands_telecom.sh for
RUN_ROOT / SEEDS / SHARED_POOL (env overrides honored, same as
show_island_settings.py), prints a one-line-per-island fleet overview + the shared
pool's cross-island uniqueness breakdown, and can expand any/all islands to the
full pgepa_monitor view. Read-only.

    .venv/bin/python experiments/persona_search/scripts/monitor_islands.py
    .venv/bin/python experiments/persona_search/scripts/monitor_islands.py --watch 15
    .venv/bin/python experiments/persona_search/scripts/monitor_islands.py --full
    RUN_TAG=islands_telecom_v2 .venv/bin/python experiments/persona_search/scripts/monitor_islands.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pgepa_monitor as pm  # noqa: E402
from show_island_settings import (  # noqa: E402
    LAUNCHER,
    parse_launcher_defaults,
    parse_launcher_seeds,
)

C, col = pm.C, pm.col
REPO = Path(__file__).resolve().parents[2]


def pool_breakdown(pool_path: Path) -> tuple[int, Counter]:
    """Total personas + per-island counts from the shared seen pool (tolerant of a
    half-written trailing line, same as SeenPool._read_all)."""
    if not pool_path.exists():
        return 0, Counter()
    by_island: Counter = Counter()
    total = 0
    for line in pool_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail line
        total += 1
        by_island[row.get("island", "?")] += 1
    return total, by_island


def island_row(name: str, run_dir: Path) -> str:
    """One compact line: status / gen progress / best_agg / trend / active sims."""
    base = run_dir.name  # == isl_<name>, matches --persona-id-base
    pid = pm.find_pid(str(run_dir))
    dot = col("●", "g") if pid else col("○", "r")
    up = pm.proc_uptime(pid) if pid else "stopped"

    state_p = run_dir / "pgepa_state.json"
    state = pm.read_json(state_p) if state_p.exists() else None
    if state and state.get("generations"):
        gens = state["generations"]
        last = gens[-1]
        gen_s = f"g{last['gen']}"
        best = state.get("best_agg")
        best_s = col(f"{best:.3f}", "g") if best is not None else "  -  "
        # For pgepa format, sparkline over accepted children's difficulty_agg
        is_pgepa = "accepted" in gens[0]
        if is_pgepa:
            diffs = [g.get("difficulty_agg") for g in gens if g.get("accepted")]
            trend = col(pm.sparkline(diffs), "c") if diffs else ""
        else:
            trend = col(pm.sparkline([g.get("best_agg") for g in gens]), "c")
        n_conf = sum(1 for c in state.get("candidates", []) if c.get("subscores"))
        cands = state.get("candidates", [])
        # Seed difficulty
        seed_diff = cands[0].get("difficulty_agg") if cands else None
        seed_s = f"seed={seed_diff:.3f}" if seed_diff is not None else ""
        # Last gen result summary
        n_acc = sum(1 for g in gens if g.get("accepted"))
        last_dir = str(last.get("mutation_direction", ""))[:30]
        last_acc = "✓" if last.get("accepted") else "✗"
        last_reason = (
            str(last.get("accept_reason", ""))[:20] if not last.get("accepted") else ""
        )
        last_s = f"last:{last_acc}{last_reason}"
        meta = col(f"{n_conf}cand({n_acc}acc) {seed_s}", "d")
        if last_dir:
            meta += "  " + col(f"{last_s} {last_dir}", "d")
    elif state:
        gen_s, best_s, trend, meta = "g0", col("eval", "y"), "", col("seed", "d")
    else:
        gen_s, best_s, trend = "--", col("init", "y"), ""
        meta = col("no state yet", "d")

    # active in-generation sim dirs — show only the LATEST one (by mtime) to
    # avoid confusing sums like "30/80" from stale + current dirs overlapping.
    # Also infer the actual running gen from the dir name (e.g. "g03_k00").
    now = time.time()
    rows = pm.live_sims(base)
    active = [r for r in rows if now - r["mtime"] < pm.ACTIVE_WINDOW_S]
    live = ""
    if active:
        latest = max(active, key=lambda r: r["mtime"])
        done, tot = latest["done"], latest["total"]
        infra = latest["infra"]
        # Infer running gen from dir name: isl_brinks_XX_g03_k00_full_diff -> g3
        import re as _re

        gm = _re.search(r"_g(\d+)_", latest["name"])
        if gm:
            running_gen = int(gm.group(1))
            gen_s = f"g{running_gen}"
        # Label the eval stage from dir name
        stage = ""
        if "_full_diff" in latest["name"] or "_full_" in latest["name"]:
            stage = "val"
        elif "_train" in latest["name"]:
            stage = "trn"
        else:
            stage = "mb"
        n_sims = latest.get("n_sims", 0)
        sim_detail = f"({n_sims}sims)" if n_sims != tot else ""
        live = col(f"  ▶ {stage} {done}/{tot}tasks{sim_detail}", "c") + (
            col(f" infra{infra}", "r") if infra else ""
        )

    return (
        f"  {dot} {name:<10} {up:<12} {gen_s:<4} best {best_s} {trend:<22} {meta}{live}"
    )


def render(defaults: dict, seeds: list[str], full: bool) -> str:
    run_root = Path(defaults.get("RUN_ROOT", ""))
    if not run_root.is_absolute():
        run_root = REPO / run_root
    pool_path = Path(
        defaults.get("SHARED_POOL", str(run_root / "shared_seen_pool.jsonl"))
    )
    if not pool_path.is_absolute():
        pool_path = REPO / pool_path

    L = []
    title = f"ISLAND FLEET — {run_root.name}"
    L.append(col(f"╔═ {title} " + "═" * max(0, 56 - len(title)), "c"))
    L.append("  " + col(f"{len(seeds)} islands  root={run_root}", "d"))
    L.append("")

    n_alive = 0
    for name in seeds:
        # Find the actual run_dir: try known prefixes or any dir ending with the seed name
        run_dir = None
        for prefix in [
            "isl_ds_",
            "isl_",
            "ev2v4_",
            "ev2v3_",
            "ev2v2_",
            "ev2_",
            "g52_",
            "g54_",
            "",
        ]:
            candidate = run_root / f"{prefix}{name}"
            if candidate.is_dir():
                run_dir = candidate
                break
        if run_dir is None:
            run_dir = run_root / f"isl_{name}"  # fallback
        if pm.find_pid(str(run_dir)):
            n_alive += 1
        L.append(island_row(name, run_dir))

    # shared pool: cross-island uniqueness
    total, by_isl = pool_breakdown(pool_path)
    L.append("")
    if total:
        dist = "  ".join(
            f"{k.replace('brinks_', 'b')}:{v}" for k, v in sorted(by_isl.items())
        )
        L.append(
            "  "
            + col(f"shared pool: {total} unique personas", "bold")
            + col(f"   [{dist}]", "d")
        )
    else:
        L.append("  " + col("shared pool: empty (no mutations written yet)", "d"))
    L.append("  " + col(f"alive {n_alive}/{len(seeds)}   pool={pool_path}", "d"))
    L.append(col("╚" + "═" * 60, "c"))

    if full:
        for name in seeds:
            run_dir = None
            for prefix in [
                "isl_ds_",
                "isl_",
                "ev2v4_",
                "ev2v3_",
                "ev2v2_",
                "ev2_",
                "g52_",
                "g54_",
                "",
            ]:
                candidate = run_root / f"{prefix}{name}"
                if candidate.is_dir():
                    run_dir = candidate
                    break
            if run_dir is None:
                run_dir = run_root / f"isl_{name}"
            L.append("")
            L.append(pm.render(str(run_dir), base_override=run_dir.name))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--launcher",
        default=None,
        help="path to launcher .sh (default: run_islands_telecom.sh)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="also print the full pgepa_monitor view for every island",
    )
    ap.add_argument(
        "--watch", type=float, default=0.0, help="refresh every N seconds (0 = once)"
    )
    ap.add_argument(
        "--extra-seeds",
        nargs="+",
        default=[],
        help="additional island names to monitor (e.g. p0003)",
    )
    args = ap.parse_args()

    launcher = Path(args.launcher) if args.launcher else LAUNCHER
    defaults = parse_launcher_defaults(launcher)
    seeds = parse_launcher_seeds(launcher)
    for s in args.extra_seeds:
        if s not in seeds:
            seeds.append(s)

    def once():
        return render(defaults, seeds, args.full)

    if args.watch <= 0:
        print(once())
        return
    try:
        while True:
            os.system("clear")
            print(once())
            print(col(f"\n  refresh {args.watch:g}s — Ctrl-C to exit", "d"))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
