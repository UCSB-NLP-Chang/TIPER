#!/usr/bin/env python3
"""Sim2Real realism as PRECISION / RECALL (coverage), geometric version (NO LLM).

Motivation: the existing ``behMean`` (sim2real_full_table.py) compares only the
*population mean* of each behavioral metric (Dice on two scalars). That is blind
to the distribution -- a simulator that emits the human average every turn scores
~100 (mode collapse), and opposite per-interaction errors cancel. Precision/recall
on the per-interaction point cloud fixes both:

    point        = one interaction -> its 19 behavioral metrics (rule features)
    human set    = 495 sessions ; sim set = a cell's N interactions
    P_geom (precision) = frac of SIM points inside the HUMAN manifold  (naturalness)
    R_geom (recall)    = frac of HUMAN points inside the SIM manifold  (coverage)

Manifold = union of per-point k-NN hyperspheres (Kynkaeaenniemi et al. 2019).
R_geom is the one behMean cannot give: it drops to ~0 under mode collapse.

This is the FREE path: rule features only, recomputed in milliseconds, no API.

    uv run python sim2real_pr.py            # p0003 lineage + collapse/ceiling checks
    uv run python sim2real_pr.py --mode faithful --k 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
from pathlib import Path

import sim2real_metrics as M
import sim2real_run_tau2 as R
from sim2real_full_table import DIMS_NOIDS, dice, load_human

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

CERT = str(DATA_ROOT / "data/simulations/gepa_certify")
LINEAGE = [  # (label, cert dir under CERT)
    ("no_persona", "progcert_no_persona_fullbase228"),
    ("seed", "progcert_seed"),
    ("idx1", "progcert_idx1"),
    ("idx3 (best)", "progcert_candidate"),
]


# --------------------------------------------------------------------------- #
# Feature extraction: interaction -> 19-metric vector (rule)
# --------------------------------------------------------------------------- #
def feature_matrix(turn_lists):
    rows = []
    for turns in turn_lists:
        m = M.metrics_for_interaction(turns)
        if m:
            rows.append([m[k] for k in M.METRIC_ORDER])
    return np.asarray(rows, float)


def human_turn_lists(rows):
    return [M.user_turns(s["conversation"]) for s in rows]


def load_human_features(mode="strict"):
    """Convenience for external callers (plots): -> human 19-metric matrix."""
    M.set_mode(mode)
    rows, _ = load_human()
    return feature_matrix(human_turn_lists(rows))


def cell_permetric_pr(results_path, Hx_raw):
    """Overall + per-dimension per-metric 1-D coverage for one sim cell."""
    Sx = feature_matrix(sim_turn_lists(results_path))
    per, dims, overall = permetric_pr(Hx_raw, Sx)
    return {"overall": overall, "dims": dims, "n": len(Sx)}


def sim_turn_lists(results_path):
    d = json.load(open(results_path))
    return [R.tau2_user_turns(s.get("messages", [])) for s in d.get("simulations", [])]


# --------------------------------------------------------------------------- #
# Spaces: standardize by HUMAN moments, then raw-19 / pca / dims-4
# --------------------------------------------------------------------------- #
def standardizer(Hx):
    mu = Hx.mean(0)
    sd = Hx.std(0) + 1e-9
    return mu, sd


def pca_fit(H, var_keep=0.90):
    c = H.mean(0)
    _, sv, Vt = np.linalg.svd(H - c, full_matrices=False)
    cum = (sv**2).cumsum() / (sv**2).sum()
    d = int(np.searchsorted(cum, var_keep) + 1)
    return c, Vt[:d].T, d


def dims4(Zmat):
    """Average standardized metrics within D1-D4 -> 4-D per-interaction vector."""
    idx = {k: i for i, k in enumerate(M.METRIC_ORDER)}
    cols = []
    for dim, ks in DIMS_NOIDS.items():
        cols.append(Zmat[:, [idx[k] for k in ks]].mean(1))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
# k-NN manifold precision / recall
# --------------------------------------------------------------------------- #
def _pdist(A, B):
    """Pairwise Euclidean distance, (len(A), len(B))."""
    a2 = (A**2).sum(1)[:, None]
    b2 = (B**2).sum(1)[None, :]
    d2 = np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)
    return np.sqrt(d2)


def radii(X, k):
    D = np.sort(_pdist(X, X), axis=1)  # col 0 = self = 0
    kk = min(k, D.shape[1] - 1)
    return D[:, kk]


def coverage(Y, X, rX):
    """Fraction of Y points inside ANY X-ball."""
    return float((_pdist(Y, X) <= rX[None, :]).any(axis=1).mean())


def precision_recall(H, S, k):
    rH, rS = radii(H, k), radii(S, k)
    P = coverage(S, H, rH)  # sim inside human  -> precision
    Rc = coverage(H, S, rS)  # human inside sim  -> recall
    F1 = 2 * P * Rc / (P + Rc) if (P + Rc) else 0.0
    return P, Rc, F1


# --------------------------------------------------------------------------- #
# Per-metric 1-D coverage (interpretable; ignores cross-metric correlation)
#   precision_k = frac of SIM values inside the human [p_lo, p_hi] interval
#   recall_k    = frac of HUMAN values inside the SIM [p_lo, p_hi] interval
# Averaged into D1-D4 and overall. Shows WHICH axis a persona violates.
# --------------------------------------------------------------------------- #
def permetric_pr(Hx_raw, Sx_raw, lo=2.5, hi=97.5):
    idx = {k: i for i, k in enumerate(M.METRIC_ORDER)}
    per = {}
    for k, i in idx.items():
        h, s = Hx_raw[:, i], Sx_raw[:, i]
        hlo, hhi = np.percentile(h, [lo, hi])
        slo, shi = np.percentile(s, [lo, hi])
        P = float(((s >= hlo) & (s <= hhi)).mean())
        Rc = float(((h >= slo) & (h <= shi)).mean())
        per[k] = (P, Rc)
    dims = {
        dim: (st.mean(per[k][0] for k in ks), st.mean(per[k][1] for k in ks))
        for dim, ks in DIMS_NOIDS.items()
    }
    allk = [k for ks in DIMS_NOIDS.values() for k in ks]
    overall = (st.mean(per[k][0] for k in allk), st.mean(per[k][1] for k in allk))
    return per, dims, overall


# --------------------------------------------------------------------------- #
# behMean (existing scalar Dice) for side-by-side reference
# --------------------------------------------------------------------------- #
def beh_mean(Sx_raw, pooled_human_raw):
    cell = {k: float(np.mean(Sx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    dd = {
        dim: st.mean(dice(cell[k], pooled_human_raw[k]) for k in ks)
        for dim, ks in DIMS_NOIDS.items()
    }
    return st.mean(dd.values())


# --------------------------------------------------------------------------- #
def all_spaces(Hx_raw, Sx_raw, mu, sd, pca):
    """Return {space_name: (H_proj, S_proj)} in raw19 / pca / dims4."""
    Hz = (Hx_raw - mu) / sd
    Sz = (Sx_raw - mu) / sd
    c, W, dpca = pca
    out = {
        "raw19": (Hz, Sz),
        f"pca{dpca}": ((Hz - c) @ W, (Sz - c) @ W),
        "dims4": (dims4(Hz), dims4(Sz)),
    }
    return out


def report_cell(label, Sx_raw, Hx_raw, mu, sd, pca, pooled, ks, spaces_to_show):
    bm = beh_mean(Sx_raw, pooled)
    sp = all_spaces(Hx_raw, Sx_raw, mu, sd, pca)
    line = f"{label:14s} n={len(Sx_raw):<4d} behMean={bm:5.1f} |"
    for sname in spaces_to_show:
        Hp, Sp = sp[sname]
        for k in ks:
            P, Rc, F1 = precision_recall(Hp, Sp, k)
            line += f"  [{sname} k{k}] P={P:.2f} R={Rc:.2f} F1={F1:.2f}"
    print(line)
    return {
        "label": label,
        "n": len(Sx_raw),
        "behMean": bm,
        **{
            f"{sname}_k{k}": dict(
                zip(("P", "R", "F1"), precision_recall(*sp[sname], k))
            )
            for sname in spaces_to_show
            for k in ks
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["faithful", "strict"], default="strict")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5])
    ap.add_argument(
        "--spaces",
        nargs="+",
        default=["pca", "raw19", "dims4"],
        help="which spaces to print; 'pca' matches the pca<N> column",
    )
    ap.add_argument(
        "--sim-glob",
        default=None,
        help="override: run on an arbitrary set of results.json cells",
    )
    ap.add_argument(
        "--no-checks",
        action="store_true",
        help="skip the human-ceiling + collapse stress tests",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    M.set_mode(args.mode)
    print(
        f"== sim2real Precision/Recall (geometric, rule features, mode={args.mode}) ==\n"
    )

    # ---- human reference -------------------------------------------------
    rows, batches = load_human()
    Hx_raw = feature_matrix(human_turn_lists(rows))
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    print(
        f"human: {len(Hx_raw)} interactions, 19 features; "
        f"pca keeps {pca[2]} dims (~90% var)\n"
    )

    # which space names to show
    pca_name = f"pca{pca[2]}"
    spaces_to_show = [pca_name if s == "pca" else s for s in args.spaces]

    hdr_spaces = "   ".join(spaces_to_show)
    print(
        f"{'cell':14s} {'':6s} {'':13s}   spaces: {hdr_spaces}  (P=precision/naturalness, "
        f"R=recall/coverage)"
    )
    print("-" * 110)

    results = []
    raws = {}
    # ---- p0003 lineage (or custom glob) ----------------------------------
    if args.sim_glob:
        cells = [(p, p) for p in sorted(glob.glob(args.sim_glob))]
    else:
        cells = [(lbl, f"{CERT}/{d}/telecom/results.json") for lbl, d in LINEAGE]
    for lbl, path in cells:
        Sx_raw = feature_matrix(sim_turn_lists(path))
        raws[lbl] = Sx_raw
        results.append(
            report_cell(
                lbl, Sx_raw, Hx_raw, mu, sd, pca, pooled, args.k, spaces_to_show
            )
        )

    # ---- per-metric 1-D coverage (interpretable: which axis breaks) ------
    print(
        "\n--- per-metric 1-D coverage (P=sim in human 95% range, R=human in sim range) ---"
    )
    print(
        f"{'cell':14s} | " + "  ".join(f"{d:>11s}" for d in DIMS_NOIDS) + " |   overall"
    )
    for lbl, Sx_raw in raws.items():
        per, dims, ov = permetric_pr(Hx_raw, Sx_raw)
        cells_s = "  ".join(f"{dims[d][0]:.2f}/{dims[d][1]:.2f}" for d in DIMS_NOIDS)
        print(f"{lbl:14s} | {cells_s} | {ov[0]:.2f}/{ov[1]:.2f}")
    # worst-offending metrics for the most adversarial cell
    last = list(raws)[-1]
    per, _, _ = permetric_pr(Hx_raw, raws[last])
    worst = sorted(per.items(), key=lambda kv: kv[1][0] + kv[1][1])[:6]
    print(
        f"\n  {last}: lowest-coverage metrics (P+R) -> "
        + ", ".join(f"{k}(P{p:.2f}/R{r:.2f})" for k, (p, r) in worst)
    )

    # ---- sanity checks ---------------------------------------------------
    if not args.no_checks:
        print("\n--- checks ---")
        # (1) ceiling: human batch B1 vs pooled-other batches
        bkeys = sorted(batches)
        if len(bkeys) >= 2:
            B1 = feature_matrix(
                [M.user_turns(s["conversation"]) for s in batches[bkeys[0]]]
            )
            Bo = feature_matrix(
                [M.user_turns(s["conversation"]) for b in bkeys[1:] for s in batches[b]]
            )
            sp = all_spaces(
                B1, Bo, mu, sd, pca
            )  # note: standardized by full-human mu/sd
            Hp, Sp = sp[pca_name]
            P, Rc, F1 = precision_recall(Hp, Sp, args.k[0])
            print(
                f"{'CEILING human':14s}      (B1 vs rest, {pca_name} k{args.k[0]})"
                f"        P={P:.2f} R={Rc:.2f} F1={F1:.2f}   (expect ~1.0)"
            )

        # (2) collapse stress test: a fake sim that always emits the human MEAN
        rng = np.random.default_rng(0)
        Fake = np.repeat(Hx_raw.mean(0)[None, :], 200, axis=0) + rng.normal(
            0, 1e-6, (200, 19)
        )
        bm = beh_mean(Fake, pooled)
        sp = all_spaces(Hx_raw, Fake, mu, sd, pca)
        Hp, Sp = sp[pca_name]
        P, Rc, F1 = precision_recall(Hp, Sp, args.k[0])
        print(
            f"{'COLLAPSE fake':14s}      (always = human mean, {pca_name} k{args.k[0]})  "
            f"behMean={bm:5.1f}  P={P:.2f} R={Rc:.2f} F1={F1:.2f}   "
            f"(expect behMean high, R~0)"
        )

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
