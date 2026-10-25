#!/usr/bin/env python3
"""PCA scatter of the human behavioral manifold vs simulator point clouds,
plus a density-fair coverage table.

Projects every interaction (19 behavioral metrics, standardized by human moments)
onto the first two principal components of the HUMAN cloud, then overlays the
simulator clouds. The human 95% envelope (convex hull of the central 95% of human
points) shows where "real" behavior lives; a sim cloud "covers more" when it fills
more of that envelope and reaches regions no_persona never visits.

    # strict (rule-based, no API) on ALL sims:
    MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/persona_search/scripts/viz/plot_pca_coverage.py
    # hybrid (LLM judge, needs OPENAI_API_KEY) on failures only:
    ... plot_pca_coverage.py --mode hybrid --failures-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
from sim2real_naturalness import failure_sim_turn_lists, feat  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import (  # noqa: E402
    _pdist,
    human_turn_lists,
    load_human,
    radii,
    sim_turn_lists,
    standardizer,
)

SIM = REPO / "data/simulations/gepa_certify"
P0002SEED = (
    REPO
    / "data/simulations/20260607_112504_telecom_llm_agent_gpt-5.4_cc_user_con_claude-opus-4-7/results.json"
)
P0002 = [SIM / f"gepa_certify_c0{i}/telecom/results.json" for i in (4, 5, 6, 7)] + [
    P0002SEED
]
P0003 = [SIM / "gepa_certify_incumbent/telecom/results.json"] + [
    SIM / f"gepa_certify_c0{i}/telecom/results.json" for i in (0, 1, 2, 3, 8)
]
NOP = [SIM / "progcert_no_persona_fullbase228/telecom/results.json"]


def pca2_fit(H: np.ndarray):
    c = H.mean(0)
    _, _, Vt = np.linalg.svd(H - c, full_matrices=False)
    return c, Vt[:2].T


def hull_xy(P: np.ndarray, keep: float = 0.95) -> np.ndarray:
    d = np.linalg.norm(P - P.mean(0), axis=1)
    cut = np.quantile(d, keep)
    core = P[d <= cut]
    h = ConvexHull(core)
    return core[h.vertices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="strict", choices=["strict", "faithful", "hybrid"]
    )
    ap.add_argument(
        "--failures-only",
        action="store_true",
        help="restrict sims to agent-failed interactions (reward<1)",
    )
    a = ap.parse_args()
    turn_fn = (
        failure_sim_turn_lists if a.failures_only else (lambda p: sim_turn_lists(p))
    )
    scope = "failonly" if a.failures_only else "allsims"

    def raw(paths) -> np.ndarray:
        return np.vstack([feat(turn_fn(str(p)), a.mode) for p in paths])

    out_dir = REPO / "data/persona_opt/figs/pair"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_human()
    Hx = feat(human_turn_lists(rows), a.mode)
    mu, sd = standardizer(Hx)
    Hz = (Hx - mu) / sd
    c, W = pca2_fit(Hz)

    def project(X: np.ndarray) -> np.ndarray:
        return ((X - mu) / sd - c) @ W

    Hp = project(Hx)

    # ----- density-fair coverage table (human-scale recall + per-axis 1D recall) -----
    rH = radii(Hp, 3)  # each human point's own kNN(3) radius in 2D PCA
    easy = np.array([i for i, r in enumerate(rows) if (r.get("reward") or 0) >= 1])
    diff = np.array([i for i, r in enumerate(rows) if (r.get("reward") or 0) < 1])
    allidx = np.arange(len(rows))

    def cov_hs(Sp, idxs):
        nearest = _pdist(Hp[idxs], Sp).min(axis=1)
        return float((nearest <= rH[idxs]).mean())

    def cov_1d(Sx, idxs):
        vals = []
        for i in range(len(M.METRIC_ORDER)):
            slo, shi = np.percentile(Sx[:, i], [2.5, 97.5])
            h = Hx[idxs, i]
            vals.append(((h >= slo) & (h <= shi)).mean())
        return float(np.mean(vals))

    raws = {
        "no_persona": raw(NOP),
        "p0002 pool": raw(P0002),
        "p0003 pool": raw(P0003),
    }
    raws["p0002∪p0003"] = np.vstack([raws["p0002 pool"], raws["p0003 pool"]])
    raws["no_persona∪p0002∪p0003"] = np.vstack(list(raws.values())[:3])

    print(f"== mode={a.mode}  scope={scope} ==")
    print(f"{'simulator':26} {'n':>4} | human-scale recall      | per-axis 1D recall")
    print(
        f"{'':26} {'':>4} | {'all':>6}{'easy':>6}{'diff':>6} | {'all':>6}{'easy':>6}{'diff':>6}"
    )
    print("-" * 78)
    for name, Sx in raws.items():
        Sp = project(Sx)
        print(
            f"{name:26} {Sx.shape[0]:4d} | "
            f"{cov_hs(Sp, allidx):6.2f}{cov_hs(Sp, easy):6.2f}{cov_hs(Sp, diff):6.2f} | "
            f"{cov_1d(Sx, allidx):6.2f}{cov_1d(Sx, easy):6.2f}{cov_1d(Sx, diff):6.2f}"
        )

    # ----- PCA scatter -----
    sims = {
        "no_persona": (project(raws["no_persona"]), "#444444", 28, "o"),
        "p0002 pool": (project(raws["p0002 pool"]), "#1f77b4", 14, "^"),
        "p0003 pool": (project(raws["p0003 pool"]), "#d62728", 14, "s"),
    }

    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=150)
    ax.scatter(
        Hp[:, 0],
        Hp[:, 1],
        s=10,
        c="#cccccc",
        alpha=0.55,
        label=f"human (n={len(Hp)})",
        zorder=1,
        edgecolors="none",
    )
    hv = hull_xy(Hp, 0.95)
    ax.add_patch(
        Polygon(
            hv,
            closed=True,
            fill=False,
            edgecolor="#888888",
            lw=1.6,
            ls="--",
            zorder=2,
            label="human 95% envelope",
        )
    )
    for name, (P, color, size, marker) in sims.items():
        ax.scatter(
            P[:, 0],
            P[:, 1],
            s=size,
            c=color,
            marker=marker,
            alpha=0.65,
            label=f"{name} (n={len(P)})",
            zorder=3,
            edgecolors="none",
        )
    ax.set_xlabel("PC1 (standardized behavioral space)")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Behavioral manifold coverage ({a.mode} metrics, {scope})\n"
        "human envelope vs simulator clouds in PCA space"
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)

    out = out_dir / f"pca_coverage_{a.mode}_{scope}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
