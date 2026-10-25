"""Does behavioral diversity HOLD across GEPA optimization?

Compares the conversation-behavior diversity of the Pareto frontier BEFORE
optimization (the initial seed pool) vs AFTER (the frontier at the current
generation). "Holds" = the after-frontier's within-group diversity is not
materially lower than the before-frontier's.

Frontier = the non-dominated candidate set, computed exactly as GEPA does:
build the per-instance winner mapping over the VAL subscores, then prune with
gepa.gepa_utils.remove_dominated_programs. The frontier members are the union
of the pruned per-instance winner sets.

Diversity of a frontier = mean PAIRWISE Jensen-Shannon distance over the 19
Sim2Real behavioral metrics (D1-D4), measured on the personas' val rollouts.
Histograms are binned on edges shared across ALL personas (both frontiers) so
the before/after numbers are directly comparable.

Task confound is held fixed: every persona is evaluated on the same val task
set with the same #trials.

Usage:
  python persona_frontier_diversity.py \
      --state data/persona_opt/pgepa_runs/xdom_v7_3domain/pgepa_state.json \
      --seed-sims "data/simulations/gepa_opt/xdom6_g00_seed{n}_val" \
      --opt-sims  "data/simulations/gepa_opt/xdom7_g{gen:02d}_k00_full_diff" \
      --out-prefix data/persona_opt/frontier_div_v7
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from gepa.gepa_utils import remove_dominated_programs

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
from persona_behavior_jsd import (
    DIMENSIONS,
    METRIC_ORDER,
    build_histograms,
    heatmap,
    pairwise,
    persona_samples,
)


def per_instance_frontier(cands, idx_subset, tie_tol=1e-9):
    """{instance: {idx}} winners among idx_subset, over val subscores."""
    sel = [
        c
        for c in cands
        if c["idx"] in idx_subset and c.get("subscores") and c.get("feasible")
    ]
    insts = set()
    for c in sel:
        insts.update(c["subscores"].keys())
    front = {}
    for inst in insts:
        best = max((c["subscores"].get(inst, 0.0) for c in sel), default=0.0)
        winners = {
            c["idx"] for c in sel if c["subscores"].get(inst, 0.0) >= best - tie_tol
        }
        if winners:
            front[inst] = winners
    return front


def frontier_idxs(cands, idx_subset):
    """Non-dominated candidate idx set (GEPA's remove_dominated_programs)."""
    front = per_instance_frontier(cands, idx_subset)
    if not front:
        return set()
    pruned = remove_dominated_programs(front)
    out = set()
    for ws in pruned.values():
        out.update(ws)
    return out


def sim_dir(c, seed_tmpl, opt_tmpl):
    if c["gen"] == 0:
        return seed_tmpl.format(n=c["idx"] + 1)  # idx0->seed1 (brinks01)
    return opt_tmpl.format(gen=c["gen"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument(
        "--seed-sims",
        default=str(DATA_ROOT / "data/simulations/gepa_opt/xdom6_g00_seed{n}_val"),
    )
    ap.add_argument(
        "--opt-sims",
        default=str(
            DATA_ROOT / "data/simulations/gepa_opt/xdom7_g{gen:02d}_k00_full_diff"
        ),
    )
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument(
        "--out-prefix", default=str(DATA_ROOT / "data/persona_opt/frontier_div")
    )
    args = ap.parse_args()

    state = json.load(open(args.state))
    cands = state["candidates"]
    by_idx = {c["idx"]: c for c in cands}

    seed_idxs = {c["idx"] for c in cands if c["gen"] == 0 and c.get("feasible")}
    all_idxs = {c["idx"] for c in cands if c.get("feasible")}

    before = frontier_idxs(cands, seed_idxs)  # initial pool = seeds only
    after = frontier_idxs(cands, all_idxs)  # current full pool

    def short(c):
        nm = c["persona_text"].splitlines()[0].replace("# Persona:", "").strip()
        nm = nm.split("(effect")[0].strip()[:24].replace(" ", "_")
        return f"i{c['idx']}_g{c['gen']:02d}_{nm}"

    print(
        f"BEFORE frontier (seed pool, {len(seed_idxs)} cands):"
        f" {sorted(before)}  agg={[by_idx[i]['agg'] for i in sorted(before)]}"
    )
    print(
        f"AFTER  frontier (full pool, {len(all_idxs)} cands):"
        f" {sorted(after)}  agg={[by_idx[i]['agg'] for i in sorted(after)]}"
    )

    # collect samples for every persona that appears in either frontier
    union = sorted(before | after)
    persona_vecs, label_of, group_member = {}, {}, {"before": [], "after": []}
    print(f"\n{'label':34s} {'#convs':>7s}  in")
    for i in union:
        c = by_idx[i]
        d = sim_dir(c, args.seed_sims, args.opt_sims)
        samples, _ = persona_samples(d)
        if not samples:
            print(f"  !! idx{i}: no convs from {d}")
            continue
        lab = short(c)
        label_of[i] = lab
        persona_vecs[lab] = samples
        where = []
        if i in before:
            group_member["before"].append(lab)
            where.append("before")
        if i in after:
            group_member["after"].append(lab)
            where.append("after")
        print(f"{lab:34s} {len(samples):7d}  {'+'.join(where)}")

    # shared binning across all personas -> comparable distances
    hists, degenerate = build_histograms(persona_vecs, args.bins, args.alpha)
    if degenerate:
        print(f"\n[degenerate metrics excluded: {', '.join(degenerate)}]")

    labels = list(persona_vecs)
    overall = pairwise(labels, hists, METRIC_ORDER)
    per_dim = {d: pairwise(labels, hists, ms) for d, ms in DIMENSIONS.items()}

    def grp_agg(mat, members, fn):
        idx = [labels.index(m) for m in members]
        vals = [mat[i, j] for a, i in enumerate(idx) for j in idx[a + 1 :]]
        return fn(vals) if vals else float("nan")

    def grp_mean_for(mat, members):
        return grp_agg(mat, members, np.mean)

    nb, na = len(group_member["before"]), len(group_member["after"])
    b = grp_mean_for(overall, group_member["before"])
    a = grp_mean_for(overall, group_member["after"])
    bd = grp_agg(overall, group_member["before"], np.max)  # diameter = coverage
    ad = grp_agg(overall, group_member["after"], np.max)

    def verdict(bv, av):
        if np.isnan(av) or np.isnan(bv):
            return "n/a"
        return "HOLDS" if av >= bv * 0.9 else "DROPS"

    print("\n=== Pareto frontier diversity: BEFORE (seeds) vs AFTER (current) ===")
    print(f"  {'metric':24s}{'before':>9s}{'after':>9s}   verdict")
    print(f"  {'mean pairwise JS (density)':24s}{b:>9.4f}{a:>9.4f}   {verdict(b, a)}")
    print(
        f"  {'max  pairwise JS (coverage)':24s}{bd:>9.4f}{ad:>9.4f}   {verdict(bd, ad)}"
    )
    print(
        f"  (frontier size {nb} -> {na}; seeds retained on frontier means "
        "the behavioral extremes are kept)"
    )

    print("\n=== per-dimension (within-frontier mean) ===")
    print("  " + "dim".ljust(12) + f"{'before':>9s}{'after':>9s}")
    for d, mat in per_dim.items():
        print(
            "  "
            + d.ljust(12)
            + f"{grp_mean_for(mat, group_member['before']):>9.4f}"
            + f"{grp_mean_for(mat, group_member['after']):>9.4f}"
        )

    # heatmap over union, tag group membership
    groups = {}
    for lab in labels:
        in_b = lab in group_member["before"]
        in_a = lab in group_member["after"]
        groups[lab] = "both" if (in_b and in_a) else ("before" if in_b else "after")
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    heatmap(
        overall,
        labels,
        groups,
        "Frontier behavior JS distance (19 metrics)",
        f"{args.out_prefix}_heatmap.png",
    )
    print(f"\nwrote {args.out_prefix}_heatmap.png")


if __name__ == "__main__":
    main()
