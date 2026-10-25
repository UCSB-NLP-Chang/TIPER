"""Behavioral-fingerprint pool DISPERSION across the GEPA progression.

Each persona -> ONE fingerprint vector = mean over its rollouts of the 19
Sim2Real behavioral metrics (D1-D4). Metrics are z-scored (per-metric std over
ALL rollouts) so each contributes comparably to Euclidean distance.

Pool dispersion is reported two ways:
  * mean pairwise Euclidean distance between fingerprints
  * mean distance to the pool centroid (behavioral "spread")

Reference points:
  * seed frontier  vs  optimized frontier  -> retention% = optimized / seed
  * noise floor: expected dispersion IF all personas were behaviorally identical
    (only rollout sampling noise), derived from within-persona variance. A pool
    whose dispersion >> floor has genuinely distinct personas (not collapsed).
    When a no_persona rollout set is supplied (--no-persona DIR), its rollouts
    are bootstrapped into pseudo-personas for an EMPIRICAL floor.

Groups are taken from a GEPA state's Pareto frontier (before = seed pool,
after = current full pool), exactly as in persona_frontier_diversity.py, and/or
from explicit --persona LABEL GROUP DIR specs.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from persona_behavior_jsd import (
    METRIC_ORDER,
    metrics_for_interaction,
    persona_samples,
    user_turns_tau2,
)
from persona_frontier_diversity import frontier_idxs, sim_dir

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)


def no_persona_samples(root, val_map):
    """no_persona rollout metric vectors, restricted to the val-25 (domain,task).

    val_map: {domain: [task_id,...]}. Keeps only sims whose domain+task_id are in
    the val set, so the floor is measured on the SAME tasks as the pools.
    """
    allow = {(d, t) for d, ts in val_map.items() for t in ts}
    out = []
    for f in glob.glob(str(Path(root) / "**" / "results.json"), recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        dom = s.get("info", {}).get("environment_info", {}).get("domain_name")
        for sm in s.get("simulations", []):
            if (dom, sm.get("task_id")) not in allow:
                continue
            turns = user_turns_tau2(sm.get("messages", []) or [])
            if turns:
                m = metrics_for_interaction(turns)
                if m is not None:
                    out.append(m)
    return out


def fingerprint_matrix(persona_rollouts, metrics):
    """Return (labels, FP[n_persona, n_metric], per-metric rollout std, within-var).

    FP[i] = mean rollout vector of persona i (raw units).
    within_var[m] = mean over personas of within-persona variance of metric m.
    """
    labels = list(persona_rollouts)
    fp, within = [], []
    for lab in labels:
        arr = np.array(
            [[v[m] for m in metrics] for v in persona_rollouts[lab]], dtype=float
        )  # [n_roll, n_metric]
        fp.append(arr.mean(axis=0))
        within.append(
            arr.var(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(len(metrics))
        )
    fp = np.array(fp)
    within_var = np.array(within).mean(axis=0)
    # scale = std across ALL rollouts pooled (stable per-metric unit)
    allroll = np.array(
        [[v[m] for m in metrics] for lab in labels for v in persona_rollouts[lab]],
        dtype=float,
    )
    scale = allroll.std(axis=0, ddof=1)
    return labels, fp, scale, within_var


def dispersion(fp_z, idx):
    """mean pairwise + mean-to-centroid on standardized fingerprints fp_z[idx]."""
    pts = fp_z[idx]
    if len(pts) < 2:
        return float("nan"), float("nan")
    pair = [
        np.linalg.norm(pts[a] - pts[b])
        for a in range(len(pts))
        for b in range(a + 1, len(pts))
    ]
    cen = pts.mean(axis=0)
    tocen = [np.linalg.norm(p - cen) for p in pts]
    return float(np.mean(pair)), float(np.mean(tocen))


def analytic_floor(within_var, scale, n_per_persona, k, metrics_keep):
    """Expected dispersion if k personas were identical (rollout noise only).

    Per standardized coordinate the persona-mean has variance (within_var/scale^2)/n.
    Distance-to-centroid^2 ~ sum_m var_coord * (k-1)/k ; pairwise^2 ~ 2*sum_m var_coord.
    """
    var_coord = (within_var / np.where(scale > 0, scale**2, 1.0))[
        metrics_keep
    ] / n_per_persona
    s = var_coord.sum()
    to_cen = float(np.sqrt(s * (k - 1) / k)) if k > 1 else float("nan")
    pair = float(np.sqrt(2 * s))
    return pair, to_cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state", default=None, help="GEPA state -> before/after frontier groups"
    )
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
    ap.add_argument(
        "--persona",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABEL", "GROUP", "DIR"),
        help="explicit persona; repeatable",
    )
    ap.add_argument(
        "--no-persona", default=None, help="no_persona rollout dir -> empirical floor"
    )
    ap.add_argument(
        "--no-persona-val",
        default=str(DATA_ROOT / "data/persona_opt/val25_taskids.json"),
        help="val task-id map; restricts no_persona floor to val-25 tasks",
    )
    ap.add_argument(
        "--boot", type=int, default=500, help="bootstrap reps for empirical floor"
    )
    args = ap.parse_args()

    # ---- gather rollouts per persona + group membership ---------------------
    persona_rollouts, group_idx = {}, {"seed": [], "optim": []}
    order = []

    if args.state:
        st = json.load(open(args.state))
        cands = st["candidates"]
        by_idx = {c["idx"]: c for c in cands}
        seed_idxs = {c["idx"] for c in cands if c["gen"] == 0 and c.get("feasible")}
        all_idxs = {c["idx"] for c in cands if c.get("feasible")}
        before = frontier_idxs(cands, seed_idxs)
        after = frontier_idxs(cands, all_idxs)
        print(f"BEFORE frontier (seeds): {sorted(before)}")
        print(f"AFTER  frontier (full) : {sorted(after)}")
        for i in sorted(before | after):
            c = by_idx[i]
            d = sim_dir(c, args.seed_sims, args.opt_sims)
            s, _ = persona_samples(d)
            if not s:
                print(f"  !! idx{i}: no convs from {d}")
                continue
            lab = f"i{i}_g{c['gen']:02d}"
            persona_rollouts[lab] = s
            order.append(
                (lab, "seed" if i in before else "optim_only", i in before, i in after)
            )
    for lab, grp, d in args.persona:
        s, _ = persona_samples(d)
        if not s:
            print(f"  !! {lab}: no convs from {d}")
            continue
        persona_rollouts[lab] = s
        order.append((lab, grp, grp == "seed", grp != "seed"))

    if len(persona_rollouts) < 2:
        ap.error("need >=2 personas with data")

    labels, fp, scale, within_var = fingerprint_matrix(persona_rollouts, METRIC_ORDER)
    metrics_keep = np.where(scale > 0)[0]  # drop degenerate metrics
    degen = [METRIC_ORDER[m] for m in range(len(METRIC_ORDER)) if m not in metrics_keep]
    scale_safe = np.where(scale > 0, scale, 1.0)
    fp_z = (fp - fp.mean(axis=0)) / scale_safe
    fp_z = fp_z[:, metrics_keep]

    lab_to_row = {lab: r for r, lab in enumerate(labels)}
    before_rows = [lab_to_row[l] for (l, _, b, _) in order if b and l in lab_to_row]
    after_rows = [lab_to_row[l] for (l, _, _, a) in order if a and l in lab_to_row]
    n_per = int(np.median([len(persona_rollouts[l]) for l in labels]))

    if degen:
        print(f"[degenerate metrics dropped: {', '.join(degen)}]")
    print(f"\n{'persona':14s}{'rollouts':>9s}  groups")
    for lab, grp, b, a in order:
        tag = "+".join([g for g, on in [("before", b), ("after", a)] if on]) or grp
        print(f"{lab:14s}{len(persona_rollouts[lab]):>9d}  {tag}")

    # ---- dispersion --------------------------------------------------------
    bp, bc = dispersion(fp_z, before_rows)
    apw, ac = dispersion(fp_z, after_rows)
    # analytic noise floor sized to the AFTER frontier
    fl_pair, fl_cen = analytic_floor(
        within_var, scale, n_per, max(len(after_rows), 2), metrics_keep
    )

    print("\n=== fingerprint pool dispersion (z-scored 19-dim space) ===")
    print(f"  {'pool':26s}{'pairwise':>10s}{'to-centroid':>13s}")
    print(f"  {'BEFORE frontier (seeds)':26s}{bp:>10.3f}{bc:>13.3f}")
    print(f"  {'AFTER frontier (optim)':26s}{apw:>10.3f}{ac:>13.3f}")
    print(f"  {'noise floor (identical)':26s}{fl_pair:>10.3f}{fl_cen:>13.3f}")

    if not (np.isnan(bp) or np.isnan(apw)):
        print("\n=== retention & ratios ===")
        print(f"  optimized retains {apw / bp * 100:5.1f}% of seed pairwise dispersion")
        print(
            f"  optimized retains {ac / bc * 100:5.1f}% of seed to-centroid dispersion"
        )
        print(
            f"  AFTER  dispersion / noise floor: pairwise {apw / fl_pair:.1f}x, "
            f"centroid {ac / fl_cen:.1f}x"
        )
        print(
            f"  BEFORE dispersion / noise floor: pairwise {bp / fl_pair:.1f}x, "
            f"centroid {bc / fl_cen:.1f}x"
        )

    # ---- empirical floor from no_persona rollouts (Tier 2) -----------------
    if args.no_persona:
        val_map = json.load(open(args.no_persona_val))
        s = no_persona_samples(args.no_persona, val_map)
        if not s:
            print(f"\n[no_persona] no val convs from {args.no_persona}")
        else:
            R = np.array([[v[m] for m in METRIC_ORDER] for v in s], dtype=float)
            R = ((R - fp.mean(axis=0)) / scale_safe)[:, metrics_keep]
            k = max(len(after_rows), 2)
            rng = np.random.RandomState(0)
            pair_s, cen_s = [], []
            for _ in range(args.boot):
                # k pseudo-personas, each averaging n_per resampled no_persona rollouts
                pts = np.array(
                    [R[rng.randint(0, len(R), n_per)].mean(axis=0) for _ in range(k)]
                )
                p, c = dispersion(pts, list(range(k)))
                pair_s.append(p)
                cen_s.append(c)
            ep, ec = np.mean(pair_s), np.mean(cen_s)
            print(
                f"\n=== empirical no_persona floor (k={k}, n={n_per}, {args.boot} boot) ==="
            )
            print(
                f"  pairwise {ep:.3f}  to-centroid {ec:.3f}  "
                f"(#no_persona rollouts={len(s)})"
            )
            if not np.isnan(apw):
                print(
                    f"  AFTER dispersion / no_persona floor: "
                    f"pairwise {apw / ep:.1f}x, centroid {ac / ec:.1f}x"
                )


if __name__ == "__main__":
    main()
