#!/usr/bin/env python3
"""Random-sampling representative subset: remove floor/ceiling, sample N, check ranking.

Simplest possible recipe (per user request):
  1. drop floor (pooled_rate==0) and ceiling (pooled_rate==1) tasks  -> "informative" pool
  2. randomly sample N tasks per domain (default 25 banking + 25 telecom = 50)
  3. check whether the subset PRESERVES the pool's per-agent ranking

Because step 2 is random, we run a Monte Carlo over many seeds and report:
  - fraction of random subsets that preserve the agent ranking
  - distribution of max per-agent rate deviation (subset vs pool)
and then emit ONE concrete subset (the best low-deviation, rank-preserving draw)
to data/tau2/representative_tasks_random.json.

Reuses cell loading / stat computation from select_representative_tasks_mainmatrix.py.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
from select_representative_tasks_mainmatrix import (
    AGENTS,
    BANKING_CELLS,
    TELECOM_CELLS,
    compute_task_stats,
    get_difficulty_tier,
)

OUTPUT_PATH = DATA_ROOT / "data/tau2/representative_tasks_random.json"
BANKING_TARGET = 25
TELECOM_TARGET = 25
N_TRIALS = 5000


def informative_pool(stats: dict) -> dict:
    return {t: s for t, s in stats.items() if 0.0 < s["pooled_rate"] < 1.0}


def per_agent_rates(pool: dict, tids) -> dict:
    """Mean per-agent rate over the given task ids (skips tasks lacking that agent)."""
    out = {}
    for a in AGENTS:
        vals = [
            pool[t]["per_agent_rate"][a] for t in tids if a in pool[t]["per_agent_rate"]
        ]
        out[a] = float(np.mean(vals)) if vals else None
    return out


def ranking(rates: dict) -> list:
    present = [a for a in AGENTS if rates[a] is not None]
    return sorted(present, key=lambda a: rates[a], reverse=True)


def max_dev(pool_rates: dict, sub_rates: dict) -> float:
    devs = [
        abs(sub_rates[a] - pool_rates[a])
        for a in AGENTS
        if pool_rates[a] is not None and sub_rates[a] is not None
    ]
    return max(devs) if devs else 0.0


def monte_carlo(pool: dict, target: int, domain: str):
    tids = list(pool.keys())
    n = min(target, len(tids))
    pool_rates = per_agent_rates(pool, tids)
    pool_rank = ranking(pool_rates)

    floor_ceil_note = ""
    print(f"\n=== {domain}: informative pool = {len(tids)}, sampling {n} ===")
    print(
        f"  pool per-agent rate: "
        + ", ".join(
            f"{a}={pool_rates[a]:.1%}" for a in AGENTS if pool_rates[a] is not None
        )
    )
    print(f"  pool ranking: {pool_rank}{floor_ceil_note}")

    rank_ok = 0
    maxdevs = []
    best = None  # (maxdev, sorted task ids) among rank-preserving draws
    for seed in range(N_TRIALS):
        rng = np.random.RandomState(seed)
        sample = list(rng.choice(tids, size=n, replace=False))
        sub_rates = per_agent_rates(pool, sample)
        md = max_dev(pool_rates, sub_rates)
        maxdevs.append(md)
        preserved = ranking(sub_rates) == pool_rank
        if preserved:
            rank_ok += 1
            cand = (md, sorted(sample))
            if best is None or cand[0] < best[0]:
                best = cand

    maxdevs = np.array(maxdevs)
    print(f"  ranking preserved: {rank_ok}/{N_TRIALS} = {rank_ok / N_TRIALS:.1%}")
    print(
        "  max per-agent dev:  "
        f"mean={maxdevs.mean():.1%}  median={np.median(maxdevs):.1%}  "
        f"p90={np.percentile(maxdevs, 90):.1%}  worst={maxdevs.max():.1%}"
    )
    frac_5pp = float((maxdevs <= 0.05).mean())
    print(f"  fraction with maxdev <= 5pp: {frac_5pp:.1%}")

    if best is None:
        print(
            "  !! NO random draw preserved the ranking -> falling back to lowest-maxdev draw"
        )
        # recompute a best-by-maxdev regardless of ranking
        for seed in range(N_TRIALS):
            rng = np.random.RandomState(seed)
            sample = list(rng.choice(tids, size=n, replace=False))
            sub_rates = per_agent_rates(pool, sample)
            md = max_dev(pool_rates, sub_rates)
            cand = (md, sorted(sample))
            if best is None or cand[0] < best[0]:
                best = cand

    chosen = best[1]
    sub_rates = per_agent_rates(pool, chosen)
    print(
        f"  -> emitted draw: maxdev={best[0]:.1%}, ranking={ranking(sub_rates)} ({'OK' if ranking(sub_rates) == pool_rank else 'CHANGED'})"
    )
    tiers = defaultdict(int)
    for t in chosen:
        tiers[get_difficulty_tier(pool[t]["pooled_rate"])] += 1
    print(f"  difficulty coverage: {dict(tiers)}")

    return chosen, {
        "informative_pool_size": len(tids),
        "sampled": n,
        "pool_ranking": pool_rank,
        "pool_per_agent_rate": pool_rates,
        "ranking_preserved_fraction": rank_ok / N_TRIALS,
        "maxdev_mean": float(maxdevs.mean()),
        "maxdev_median": float(np.median(maxdevs)),
        "maxdev_p90": float(np.percentile(maxdevs, 90)),
        "maxdev_worst": float(maxdevs.max()),
        "frac_maxdev_le_5pp": frac_5pp,
        "emitted_draw_maxdev": best[0],
        "emitted_draw_ranking": ranking(sub_rates),
        "n_trials": N_TRIALS,
    }


def main():
    print("=" * 64)
    print("Random Subset + Ranking Preservation (Monte Carlo)")
    print("=" * 64)

    b_stats, b_cov = compute_task_stats(BANKING_CELLS)
    t_stats, t_cov = compute_task_stats(TELECOM_CELLS)
    b_pool = informative_pool(b_stats)
    t_pool = informative_pool(t_stats)

    b_floor = sum(1 for s in b_stats.values() if s["pooled_rate"] == 0.0)
    b_ceil = sum(1 for s in b_stats.values() if s["pooled_rate"] == 1.0)
    t_floor = sum(1 for s in t_stats.values() if s["pooled_rate"] == 0.0)
    t_ceil = sum(1 for s in t_stats.values() if s["pooled_rate"] == 1.0)
    print(
        f"\nBanking: {len(b_stats)} total, {b_floor} floor, {b_ceil} ceiling, {len(b_pool)} informative"
    )
    print(
        f"Telecom: {len(t_stats)} total, {t_floor} floor, {t_ceil} ceiling, {len(t_pool)} informative"
    )

    b_sel, b_report = monte_carlo(b_pool, BANKING_TARGET, "Banking")
    t_sel, t_report = monte_carlo(t_pool, TELECOM_TARGET, "Telecom")

    out = {
        "banking": sorted(b_sel),
        "telecom": sorted(t_sel),
        "metadata": {
            "selection_method": "floor_ceiling_removal + uniform_random_sample",
            "source": "main_agent_matrix",
            "agents": AGENTS,
            "n_trials": N_TRIALS,
            "banking_report": b_report,
            "telecom_report": t_report,
            "banking_coverage": b_cov,
            "telecom_coverage": t_cov,
            "banking_caveat": "RAGGED: sonnet-4.5 only under cc_con; agent x user confounded for banking.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"\nSelected {len(b_sel)} banking + {len(t_sel)} telecom = {len(b_sel) + len(t_sel)}"
    )
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
