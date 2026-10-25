#!/usr/bin/env python3
"""How small can the representative subset get?  (both domains AGGREGATED)

Pools banking + telecom tasks into ONE informative pool, then sweeps the subset
size N downward. For each N we Monte-Carlo random-sample N tasks and measure
whether the aggregate per-agent ranking AND the absolute per-agent scores stay
close to the full-pool values.

Reference (full combined informative pool):
  rank*  = agents ordered by mean per-agent reward over ALL pooled tasks
  rate*  = the per-agent mean rewards themselves

For each candidate N we report, over N_TRIALS random draws:
  - rank_ok      : % of draws whose agent ranking == rank*
  - top1_ok      : % of draws whose #1 agent == rank*[0]
  - maxdev mean  : mean over draws of max_a |rate_sub[a] - rate*[a]|
  - p_le_5pp     : % of draws with maxdev <= 5pp  (scores "approximately unchanged")
  - gap_ok       : % of draws preserving every adjacent-agent gap SIGN (ordering margin)

The smallest N meeting the thresholds (rank_ok>=RANK_THRESH and p_le_5pp>=DEV_THRESH)
is reported, and one concrete best draw at that N is emitted to JSON.

Reuses cell loading + stats from select_representative_tasks_mainmatrix.py.
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

OUTPUT_PATH = DATA_ROOT / "data/tau2/representative_tasks_agg_min.json"
N_TRIALS = 3000
SIZES = [60, 50, 40, 30, 25, 20, 16, 12, 10, 8, 6, 5, 4, 3]
RANK_THRESH = 0.95  # >=95% of random draws must preserve the full ranking
DEV_THRESH = 0.80  # >=80% of random draws must keep maxdev <= 5pp
REQUIRE_ALL_AGENTS = (
    True  # only pool tasks that have all 3 agents (well-defined ranking)
)


def build_pool():
    """Combined informative pool across both domains: {tid: {rate, per_agent_rate, domain}}."""
    b_stats, _ = compute_task_stats(BANKING_CELLS)
    t_stats, _ = compute_task_stats(TELECOM_CELLS)
    pool = {}
    for domain, stats in (("banking", b_stats), ("telecom", t_stats)):
        for tid, s in stats.items():
            if not (0.0 < s["pooled_rate"] < 1.0):
                continue
            if REQUIRE_ALL_AGENTS and s["n_agents"] < len(AGENTS):
                continue
            # namespace tid by domain to avoid cross-domain id collisions
            pool[f"{domain}::{tid}"] = {
                "domain": domain,
                "pooled_rate": s["pooled_rate"],
                "per_agent_rate": dict(s["per_agent_rate"]),
            }
    return pool


def per_agent_rates(pool, tids):
    out = {}
    for a in AGENTS:
        vals = [
            pool[t]["per_agent_rate"][a] for t in tids if a in pool[t]["per_agent_rate"]
        ]
        out[a] = float(np.mean(vals)) if vals else None
    return out


def ranking(rates):
    present = [a for a in AGENTS if rates[a] is not None]
    return sorted(present, key=lambda a: rates[a], reverse=True)


def gap_signs(rates, rank):
    """Sign of each adjacent gap along the reference ranking order."""
    return [
        int(np.sign(rates[rank[i]] - rates[rank[i + 1]])) for i in range(len(rank) - 1)
    ]


def maxdev(ref, sub):
    devs = [
        abs(sub[a] - ref[a])
        for a in AGENTS
        if ref[a] is not None and sub[a] is not None
    ]
    return max(devs) if devs else 0.0


def greedy_refine(pool, tids, ref_rates, ref_rank, selected, max_iters=400):
    """Swap tasks in/out to drive maxdev down while keeping the reference ranking.

    Returns (refined_selected, maxdev). A swap is accepted only if it lowers maxdev
    AND does not break the reference ranking — so the result both preserves ordering
    and tightens the absolute scores.
    """
    selected = list(selected)
    sset = set(selected)
    unsel = [t for t in tids if t not in sset]
    cur = maxdev(ref_rates, per_agent_rates(pool, selected))
    for _ in range(max_iters):
        if cur <= 1e-9:
            break
        improved = False
        # try swapping each selected task for an unselected one (bounded scan)
        for o in list(selected):
            for i in unsel:
                trial = [t for t in selected if t != o] + [i]
                tr = per_agent_rates(pool, trial)
                if ranking(tr) != ref_rank:
                    continue
                md = maxdev(ref_rates, tr)
                if md < cur - 1e-9:
                    selected = trial
                    unsel = [t for t in tids if t not in set(selected)]
                    cur = md
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return selected, cur


def sweep():
    pool = build_pool()
    tids = list(pool.keys())
    dom_counts = defaultdict(int)
    for t in tids:
        dom_counts[pool[t]["domain"]] += 1

    ref_rates = per_agent_rates(pool, tids)
    ref_rank = ranking(ref_rates)
    ref_gaps = gap_signs(ref_rates, ref_rank)

    print("=" * 72)
    print("AGGREGATED (banking + telecom) — subset size sweep")
    print("=" * 72)
    print(f"combined informative pool: {len(tids)} tasks  ({dict(dom_counts)})")
    print(
        "full-pool per-agent rate: "
        + ", ".join(f"{a}={ref_rates[a]:.1%}" for a in AGENTS)
    )
    print(f"full-pool ranking: {ref_rank}")
    margins = [
        ref_rates[ref_rank[i]] - ref_rates[ref_rank[i + 1]]
        for i in range(len(ref_rank) - 1)
    ]
    print(
        "adjacent margins: "
        + ", ".join(
            f"{ref_rank[i]}>{ref_rank[i + 1]} by {margins[i]:.1%}"
            for i in range(len(margins))
        )
    )
    print()
    print("RANDOM robustness (left)  vs  best CONSTRUCTED subset achievable (right)\n")
    header = (
        f"{'N':>4} | {'rank_ok':>8} {'top1_ok':>8} {'maxdev~':>8} {'p<=5pp':>7} || "
        f"{'built_rank':>10} {'built_dev':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    best_draw_by_n = {}
    for n in SIZES:
        if n > len(tids):
            continue
        rank_ok = top1_ok = gap_ok = le5 = 0
        devs = []
        best = None  # (maxdev, sorted tids) among rank-preserving draws
        dom_split = defaultdict(int)
        for seed in range(N_TRIALS):
            rng = np.random.RandomState(seed)
            sample = list(rng.choice(tids, size=n, replace=False))
            sr = per_agent_rates(pool, sample)
            md = maxdev(ref_rates, sr)
            devs.append(md)
            r = ranking(sr)
            preserved = r == ref_rank
            if preserved:
                rank_ok += 1
                if best is None or md < best[0]:
                    best = (md, sorted(sample))
            if r and r[0] == ref_rank[0]:
                top1_ok += 1
            if gap_signs(sr, ref_rank) == ref_gaps:
                gap_ok += 1
            if md <= 0.05:
                le5 += 1
            for t in sample:
                dom_split[pool[t]["domain"]] += 1
        # fallback best: lowest maxdev regardless of ranking
        if best is None:
            for seed in range(N_TRIALS):
                rng = np.random.RandomState(seed)
                sample = list(rng.choice(tids, size=n, replace=False))
                md = maxdev(ref_rates, per_agent_rates(pool, sample))
                if best is None or md < best[0]:
                    best = (md, sorted(sample))
        # CONSTRUCTED track: refine the best rank-preserving random draw
        refined, built_dev = greedy_refine(pool, tids, ref_rates, ref_rank, best[1])
        built_rank_ok = ranking(per_agent_rates(pool, refined)) == ref_rank
        best_draw_by_n[n] = (built_dev, sorted(refined), built_rank_ok)
        avg_b = dom_split["banking"] / N_TRIALS
        avg_t = dom_split["telecom"] / N_TRIALS
        rows.append(
            {
                "N": n,
                "rank_ok": rank_ok / N_TRIALS,
                "top1_ok": top1_ok / N_TRIALS,
                "gap_ok": gap_ok / N_TRIALS,
                "maxdev_mean": float(np.mean(devs)),
                "p_le_5pp": le5 / N_TRIALS,
                "built_dev": built_dev,
                "built_rank_ok": built_rank_ok,
                "avg_banking": avg_b,
                "avg_telecom": avg_t,
            }
        )
        print(
            f"{n:>4} | {rank_ok / N_TRIALS:>8.1%} {top1_ok / N_TRIALS:>8.1%} "
            f"{np.mean(devs):>8.1%} {le5 / N_TRIALS:>7.1%} || "
            f"{('OK' if built_rank_ok else 'BROKE'):>10} {built_dev:>9.1%}"
        )

    # (1) smallest N that is ROBUST under pure random sampling
    robust = [
        r for r in rows if r["rank_ok"] >= RANK_THRESH and r["p_le_5pp"] >= DEV_THRESH
    ]
    robust_n = min((r["N"] for r in robust), default=None)
    # (2) smallest N where a CONSTRUCTED subset still preserves ranking AND maxdev<=5pp.
    # Guard against overfit: require the signal to be genuinely present (random top1_ok
    # >= 0.70) — below that a tiny subset just curve-fits 3 means and won't generalize.
    built = [r for r in rows if r["built_rank_ok"] and r["built_dev"] <= 0.05]
    built_n = min((r["N"] for r in built), default=None)
    built_safe = [r for r in built if r["top1_ok"] >= 0.70]
    built_safe_n = min((r["N"] for r in built_safe), default=None)

    print()
    if robust_n is not None:
        print(
            f"==> smallest RANDOM-robust N = {robust_n} "
            f"(>= {RANK_THRESH:.0%} draws keep ranking AND >= {DEV_THRESH:.0%} within 5pp)"
        )
    else:
        print(
            f"==> no N is random-robust at rank>={RANK_THRESH:.0%} & p5pp>={DEV_THRESH:.0%} "
            "(gpt-5.4/gpt-5.2 margin too tight for blind sampling)"
        )
    if built_n is not None:
        print(
            f"==> smallest CONSTRUCTED N = {built_n} "
            "(preserves ranking & scores within 5pp — but tiny values OVERFIT the 3 agents)"
        )
    if built_safe_n is not None:
        print(
            f"==> smallest DEFENSIBLE constructed N = {built_safe_n} "
            "(constructed AND random top1_ok>=70%, i.e. the signal is genuinely there)"
        )

    force_n = os.environ.get("AGG_FORCE_N")
    chosen_n = (
        int(force_n)
        if force_n
        else (built_safe_n or robust_n or built_n or max(r["N"] for r in rows))
    )
    best = best_draw_by_n[chosen_n]
    chosen = best[1]
    print(
        f"\nEmitting CONSTRUCTED subset at N={chosen_n} (built_dev={best[0]:.1%}, "
        f"rank {'OK' if best[2] else 'BROKE'})"
    )
    sr = per_agent_rates(pool, chosen)
    dom = defaultdict(int)
    tier = defaultdict(int)
    for t in chosen:
        dom[pool[t]["domain"]] += 1
        tier[get_difficulty_tier(pool[t]["pooled_rate"])] += 1
    print(
        f"  emitted draw: N={chosen_n} maxdev={best[0]:.1%} ranking={ranking(sr)} "
        f"({'OK' if ranking(sr) == ref_rank else 'CHANGED'})"
    )
    print(
        "  per-agent rate: "
        + ", ".join(f"{a}={sr[a]:.1%}(Δ{sr[a] - ref_rates[a]:+.1%})" for a in AGENTS)
    )
    print(f"  domain split: {dict(dom)}  difficulty: {dict(tier)}")

    out = {
        "selection_method": "both_domains_aggregated + random-robustness sweep + greedy-constructed floor",
        "random_robust_N": robust_n,
        "constructed_min_N": built_n,
        "constructed_defensible_N": built_safe_n,
        "chosen_N": chosen_n,
        "tasks": sorted(chosen),  # namespaced "domain::task_id"
        "banking": sorted(
            t.split("::", 1)[1] for t in chosen if pool[t]["domain"] == "banking"
        ),
        "telecom": sorted(
            t.split("::", 1)[1] for t in chosen if pool[t]["domain"] == "telecom"
        ),
        "full_pool_ranking": ref_rank,
        "full_pool_per_agent_rate": ref_rates,
        "subset_per_agent_rate": sr,
        "emitted_maxdev": best[0],
        "sweep": rows,
        "n_trials": N_TRIALS,
        "thresholds": {"rank_ok": RANK_THRESH, "p_le_5pp": DEV_THRESH},
        "pool_size": len(tids),
        "pool_domain_counts": dict(dom_counts),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    sweep()
