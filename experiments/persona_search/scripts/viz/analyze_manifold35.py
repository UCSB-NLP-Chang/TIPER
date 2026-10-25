"""Manifold (geometric P/R/F1) naturalness for the p0002 & brinks_04 lineages on the
35-task overlap, contrasted with the scalar behMean we have been plotting.

behMean compares only population MEANS of the 19 behavioral metrics (Dice on two
scalars): it is blind to the distribution, so a mode-collapsed simulator that emits
the human average every turn still scores ~100. The k-NN manifold P/R fixes this:

    P_geom (precision / naturalness) = frac of SIM points inside the HUMAN manifold
    R_geom (recall  / coverage)      = frac of HUMAN points inside the SIM manifold
    F1                               = harmonic mean

R_geom is exactly what behMean cannot see -- it collapses to ~0 under mode collapse.

We score each lineage's seed and per-gen best-born persona (the same selection as
analyze_overlap35.py, cert 2-trial sims preferred), plus the no_persona floor and a
human split-half CEILING (so P/R/F1 are interpretable -- even humans don't hit 1.0).

    .venv/bin/python experiments/persona_search/scripts/viz/analyze_manifold35.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HERE.parent))

import os
import sys

import _style as st  # noqa: E402
import analyze_overlap35 as A  # noqa: E402

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
import sim2real_run_tau2 as R  # noqa: E402
from sim2real_naturalness_brinks import brinks_human_turn_lists  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import (  # noqa: E402
    all_spaces,
    beh_mean,
    feature_matrix,
    pca_fit,
    precision_recall,
    standardizer,
)

OUT_JSON = A.PGEPA / "p0002_brinks04_manifold35.json"
K = 3
# k-NN recall/precision are sample-size dependent (fewer sim points -> larger radii
# -> inflated recall). Cells have different N (cert=70 turns, 1-trial search=35), so
# we subsample every sim cell to a common N and average over repeats before comparing.
N_TARGET = 35
N_REPEAT = 40
_RNG = np.random.default_rng(0)

# the persona each lineage actually SHIPS (peak genuine difficulty feasible candidate),
# both certified 2-trial so seed-vs-final is an apples-to-apples manifold comparison.
SEED_IDX = {"p0002": 0, "brinks_04": 0}
FINAL_IDX = {"p0002": 6, "brinks_04": 4}


def _metrics(Sx, sp_name, Hx_raw, mu, sd, pca, pooled, n_target=N_TARGET):
    """behMean (N-robust, uses the cell mean) + manifold P/R/F1 subsampled to a fixed
    sim-side N so cells with different trial counts are comparable."""
    sp = all_spaces(Hx_raw, Sx, mu, sd, pca)
    Hp, Sp = sp[sp_name]
    n = min(n_target, len(Sp))
    Ps, Rs, F1s = [], [], []
    for _ in range(N_REPEAT):
        idx = _RNG.choice(len(Sp), n, replace=False)
        P, Rc, F1 = precision_recall(Hp, Sp[idx], K)
        Ps.append(P)
        Rs.append(Rc)
        F1s.append(F1)
    return {
        "behMean": round(float(beh_mean(Sx, pooled)), 2),
        "P": round(float(np.mean(Ps)), 4),
        "R": round(float(np.mean(Rs)), 4),
        "F1": round(float(np.mean(F1s)), 4),
        "sub_n": n,
    }


def compute() -> dict:
    overlap = A.overlap_task_ids()
    M.set_mode("strict")
    Hx_raw = feature_matrix(brinks_human_turn_lists())
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"

    data = {"overlap_n": len(overlap), "k": K, "pca_dims": pca[2], "arms": {}}

    # human split-half ceiling: one half is the human manifold, the OTHER half plays
    # the "simulator" role, subsampled to the same N_TARGET as the personas. This is
    # the realistic upper bound a perfect 35-sample simulator could reach (not 1.0).
    Hall = brinks_human_turn_lists()
    half = len(Hall) // 2
    Ha, Hb = feature_matrix(Hall[:half]), feature_matrix(Hall[half:])
    cm = _metrics(Hb, pca_name, Ha, mu, sd, pca, pooled)
    data["ceiling"] = {"P": cm["P"], "R": cm["R"], "F1": cm["F1"], "n": cm["sub_n"]}

    # no_persona floor
    np_sims = A._sims_in(A.NO_PERSONA, overlap)
    np_Sx = feature_matrix([R.tau2_user_turns(s.get("messages", [])) for s in np_sims])
    data["floor"] = {
        **_metrics(np_Sx, pca_name, Hx_raw, mu, sd, pca, pooled),
        "n": len(np_sims),
    }

    for arm in A.RUNS:
        base, lineage = A.build_points(arm, "born")  # best-born persona each gen
        rows = []
        for pt in lineage:
            if not pt["valid"]:
                continue
            gen, idx, k = pt["gen"], pt["idx"], pt["k"]
            cert = A.CERT.get((arm, idx)) if idx is not None else None
            if cert and cert.exists():
                rp, trials = cert, 2
            else:
                rp, trials = A._best_results(base, pt["birth_gen"], k), 1
            if rp is None:
                continue
            sims = A._sims_in(rp, overlap)
            Sx = feature_matrix(
                [R.tau2_user_turns(s.get("messages", [])) for s in sims]
            )
            m = _metrics(Sx, pca_name, Hx_raw, mu, sd, pca, pooled)
            rows.append({"gen": gen, "idx": idx, "trials": trials, "n": len(sims), **m})
        data["arms"][arm] = rows
    OUT_JSON.write_text(json.dumps(data, indent=2))
    return data


def print_table(data: dict):
    print(
        f"\n=== Manifold naturalness, {data['overlap_n']} overlap tasks, "
        f"pca{data['pca_dims']} k={data['k']} ==="
    )
    cl, fl = data["ceiling"], data["floor"]
    print(
        f"  human split-half CEILING : P={cl['P']:.2f} R={cl['R']:.2f} F1={cl['F1']:.2f}"
    )
    print(
        f"  no_persona FLOOR         : behMean={fl['behMean']:.1f}  "
        f"P={fl['P']:.2f} R={fl['R']:.2f} F1={fl['F1']:.2f}  (n={fl['n']})"
    )
    for arm, rows in data["arms"].items():
        print(f"\n{A.ARM_LABEL[arm]}")
        hdr = f"  {'gen':<5}{'idx':<6}{'src':<8}{'behMean':<9}{'P':<7}{'R':<7}{'F1':<7}{'n':<5}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            src = f"cert" if r["trials"] >= 2 else "search"
            print(
                f"  {r['gen']:<5}{str(r['idx']):<6}{src:<8}{r['behMean']:<9}"
                f"{r['P']:<7}{r['R']:<7}{r['F1']:<7}{r['n']:<5}"
            )


def _seed_final(rows, arm):
    by_idx = {r["idx"]: r for r in rows}
    return by_idx[SEED_IDX[arm]], by_idx[FINAL_IDX[arm]]


def fig_bars(data, out_path):
    """Per-arm grouped bars: seed vs final vs no_persona vs human ceiling, for the
    four metrics on a common 0-1 scale (behMean shown as behMean/100)."""
    import matplotlib.pyplot as plt

    st.apply_style()
    arms = list(data["arms"])
    fig, axes = plt.subplots(1, len(arms), figsize=(12, 5.2), sharey=True)
    metrics = ["behMean", "P", "R", "F1"]
    labels = ["behMean/100", "P_geom\n(naturalness)", "R_geom\n(coverage)", "F1"]
    x = np.arange(len(metrics))
    w = 0.2
    cl = data["ceiling"]
    fl = data["floor"]
    for ax, arm in zip(axes, arms):
        seed, final = _seed_final(data["arms"][arm], arm)
        col = A.ARM_COLOR[arm]

        def vec(d):
            return [d["behMean"] / 100, d["P"], d["R"], d["F1"]]

        ax.bar(x - w, vec(fl), w, label="no_persona", color="0.75")
        ax.bar(x, vec(seed), w, label="seed (g0)", color=col, alpha=0.45)
        ax.bar(x + w, vec(final), w, label=f"final (g{final['gen']})", color=col)
        # human split-half ceiling as dashed caps on P/R/F1 only
        for xi, val in zip(x[1:], [cl["P"], cl["R"], cl["F1"]]):
            ax.hlines(
                val, xi - 1.5 * w, xi + 1.5 * w, color="0.2", ls="--", lw=1.5, zorder=5
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(A.ARM_LABEL[arm])
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("score (0-1 scale)")
    fig.suptitle(
        "Scalar behMean vs geometric manifold P/R/F1  (35-task overlap, "
        "dashed = human split-half ceiling)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return st.save(fig, out_path)


def fig_traj(data, out_path):
    """behMean (left axis) vs manifold F1 (right axis) over generations per arm."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    st.apply_style()
    fig, ax = plt.subplots(figsize=st.FIGSIZE)
    ax2 = ax.twinx()
    ax2.grid(False)
    has_prov = False
    for arm, rows in data["arms"].items():
        col = A.ARM_COLOR[arm]
        g = [r["gen"] for r in rows]
        ax.plot(
            g,
            [r["behMean"] for r in rows],
            color=col,
            lw=2.4,
            marker="o",
            ms=7,
            zorder=3,
        )
        # manifold F1: solid line through CERT points only (comparable); search
        # (1-trial) F1 is inflated by per-point noise -> drawn hollow, off the line
        cert = [r for r in rows if r["trials"] >= 2]
        ax2.plot(
            [r["gen"] for r in cert],
            [r["F1"] for r in cert],
            color=col,
            lw=2.0,
            ls="--",
            marker="s",
            ms=7,
            alpha=0.9,
            zorder=3,
        )
        for r in rows:
            if r["trials"] < 2:
                has_prov = True
                ax2.scatter(
                    r["gen"],
                    r["F1"],
                    marker="s",
                    s=55,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.6,
                    zorder=4,
                )
    ax.axhline(data["floor"]["behMean"], color="0.5", ls=":", lw=1.6)
    ax2.axhline(data["ceiling"]["F1"], color="0.2", ls="--", lw=1.2, alpha=0.6)
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("behMean (scalar, left)")
    ax2.set_ylabel("manifold F1 (geometric, right)")
    ax.set_ylim(30, 60)
    ax2.set_ylim(0, max(0.5, data["ceiling"]["F1"] * 1.15))
    gmax = max(r["gen"] for rows in data["arms"].values() for r in rows)
    ax.set_xticks(range(gmax + 1))
    ax.set_title("Scalar vs manifold naturalness over optimization  (35-task overlap)")
    handles = [
        Line2D([], [], color=A.ARM_COLOR[a], lw=2.4, label=A.ARM_LABEL[a])
        for a in data["arms"]
    ]
    handles += [
        Line2D([], [], color="0.3", lw=2.2, marker="o", label="behMean (solid, left)"),
        Line2D(
            [],
            [],
            color="0.3",
            lw=2.0,
            ls="--",
            marker="s",
            label="manifold F1 (dashed, right)",
        ),
        Line2D([], [], color="0.2", ls="--", lw=1.2, label="F1 human ceiling"),
    ]
    if has_prov:
        handles.append(
            Line2D(
                [],
                [],
                color="0.4",
                ls="",
                marker="s",
                mfc="white",
                mew=1.6,
                label="F1 1-trial (not comparable)",
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    return st.save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default=str(REPO / "data/persona_opt/figs/p0002_brinks04")
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = compute()
    print_table(data)
    print(fig_bars(data, out / "overlap35_manifold_bars.png"))
    print(fig_traj(data, out / "overlap35_manifold_traj.png"))


if __name__ == "__main__":
    main()
