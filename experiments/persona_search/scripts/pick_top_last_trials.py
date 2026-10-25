#!/usr/bin/env python3
"""Pick best/worst 3 trials from a 4-trial fullset and compute avg@3.

For each (domain, island, condition) cell in a fullset directory:
  - seed condition: keep TOP 3 trials (highest reward) per task → avg@3
  - opt  condition: keep LAST 3 trials (lowest reward) per task → avg@3

This gives a tighter estimate: seed shows the agent's best-case performance,
opt shows the adversarial persona's strongest effect.

Usage:
    python experiments/persona_search/scripts/pick_top_last_trials.py --prefix fullset_sonnet46
    python experiments/persona_search/scripts/pick_top_last_trials.py --prefix fullset_ds_max
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
SIM = REPO / "data/simulations"


def load_cell(cell_dir: Path) -> list[dict]:
    rf = cell_dir / "results.json"
    if rf.is_file():
        data = json.loads(rf.read_text())
        # tau2 auto-resume format: simulations[].reward_info.reward
        sims = data.get("simulations", [])
        if sims:
            out = []
            for s in sims:
                ri = s.get("reward_info") or {}
                reward = ri.get("reward") if isinstance(ri, dict) else None
                if reward is not None:
                    out.append({"task_id": s.get("task_id"), "reward": reward})
            if out:
                return out
        # fullset_build assembled format: results[].reward
        results = data.get("results", [])
        if results:
            return results
    # fallback: individual sim directories
    sim_dirs = sorted(cell_dir.glob("sim_*/results.json"))
    out = []
    for s in sim_dirs:
        try:
            r = json.loads(s.read_text())
            out.append(r)
        except Exception:
            pass
    return out


def pick_trials(results: list[dict], mode: str, keep: int = 3) -> dict:
    by_task = defaultdict(list)
    for r in results:
        tid = r.get("task_id") or r.get("id", "?")
        reward = r.get("reward")
        if reward is None:
            continue
        by_task[tid].append(reward)

    task_avgs = []
    n_tasks = 0
    n_tasks_short = 0
    for tid, rewards in sorted(by_task.items()):
        if len(rewards) < keep:
            n_tasks_short += 1
        if mode == "top":
            picked = sorted(rewards, reverse=True)[:keep]
        else:
            picked = sorted(rewards)[:keep]
        task_avgs.append(sum(picked) / len(picked))
        n_tasks += 1

    avg = sum(task_avgs) / len(task_avgs) if task_avgs else 0.0
    return {
        "avg@3": round(avg, 4),
        "n_tasks": n_tasks,
        "n_tasks_short": n_tasks_short,
        "total_sims": len(results),
        "mode": mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix", required=True, help="fullset save prefix (e.g. fullset_sonnet46)"
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=3,
        help="number of trials to keep per task (default 3)",
    )
    args = ap.parse_args()

    base = SIM / args.prefix
    if not base.is_dir():
        print(f"ERROR: {base} not found")
        return

    print(
        f"{'domain':<12} {'island':<12} {'cond':<6} {'pick':<6} "
        f"{'avg@3':>8} {'tasks':>6} {'sims':>6} {'short':>6}"
    )
    print("-" * 70)

    rows = []
    for dom_dir in sorted(base.iterdir()):
        if not dom_dir.is_dir():
            continue
        domain = dom_dir.name
        for cell_dir in sorted(dom_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            name = cell_dir.name
            if "_seed" in name:
                cond = "seed"
                island = name.replace("_seed", "")
                mode = "top"
            elif "_opt" in name:
                cond = "opt"
                island = name.replace("_opt", "")
                mode = "last"
            elif "_no_persona" in name:
                cond = "no_p"
                island = name.replace("_no_persona", "")
                mode = "top"
            else:
                continue

            results = load_cell(cell_dir)
            if not results:
                print(
                    f"{domain:<12} {island:<12} {cond:<6} {mode:<6} "
                    f"{'N/A':>8} {'0':>6} {'0':>6} {'':>6}"
                )
                continue

            info = pick_trials(results, mode, args.keep)
            print(
                f"{domain:<12} {island:<12} {cond:<6} {mode:<6} "
                f"{info['avg@3']:>8.4f} {info['n_tasks']:>6} "
                f"{info['total_sims']:>6} {info['n_tasks_short']:>6}"
            )
            rows.append(
                {
                    "domain": domain,
                    "island": island,
                    "cond": cond,
                    **info,
                }
            )

    print()
    # summary table: Δ(seed−opt)
    print("=== Summary: Δ(seed − opt) ===")
    print(
        f"{'domain':<12} {'island':<12} {'seed_avg@3':>12} {'opt_avg@3':>12} {'Δ':>8}"
    )
    print("-" * 60)
    by_key = {}
    for r in rows:
        by_key[(r["domain"], r["island"], r["cond"])] = r

    for dom_dir in sorted(base.iterdir()):
        if not dom_dir.is_dir():
            continue
        domain = dom_dir.name
        islands_seen = set()
        for cell_dir in sorted(dom_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            name = cell_dir.name
            for suffix in ("_seed", "_opt", "_no_persona"):
                if name.endswith(suffix):
                    islands_seen.add(name[: -len(suffix)])
        for island in sorted(islands_seen):
            seed = by_key.get((domain, island, "seed"))
            opt = by_key.get((domain, island, "opt"))
            if seed and opt:
                delta = seed["avg@3"] - opt["avg@3"]
                print(
                    f"{domain:<12} {island:<12} {seed['avg@3']:>12.4f} "
                    f"{opt['avg@3']:>12.4f} {delta:>+8.4f}"
                )
            else:
                print(f"{domain:<12} {island:<12} {'?':>12} {'?':>12} {'?':>8}")


if __name__ == "__main__":
    main()
