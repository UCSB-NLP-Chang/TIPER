#!/usr/bin/env python3
"""GEPA persona-optimization difficulty convergence for p0001 / p0002 / p0003
in one figure (3 colors).

x = GEPA rollouts (num_metric_calls_by_discovery)
y = difficulty (= 1 - agent_success on the valset), the GEPA objective

Per run: faint dots = each discovered candidate's difficulty; bold step line =
best-difficulty-so-far (the actual optimization/convergence curve).

Reads each run's gepa_state.bin (needs the `gepa` package -> run via `uv run`).

    uv run python experiments/persona_search/scripts/plot_difficulty_optimization.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GR = str(DATA_ROOT / "data/persona_opt/gepa_runs")
RUNS = [  # (label, run dir, color, final-best note)
    ("p0001", "p0001_aggressive50_20260606_151059", "#4C72B0"),
    ("p0002", "p0002_aggressive50_20260606_151059", "#E1A024"),
    ("p0003", "p0003_aggressive_20260606_145857", "#2E8B4F"),
]
# Full-base-228 cert difficulty of each run's CERTIFIED best candidate
# (= 1 - mean reward on the held-out full-base telecom set; validity-passed).
# Note the valset->full-set reversal: p0003 overfits (0.57->0.46) while
# p0001/p0002 generalize up (0.28->0.32, 0.34->0.38).
FULLSET_BEST = {
    "p0001": 0.320,  # c02
    "p0002": 0.377,  # c04 (cert; c06 valset-best 0.386 failed validity)
    "p0003": 0.456,  # idx3
}
# Full-base-228 difficulty of each lineage's seed persona (idx0), and the
# cooperative no_persona baseline. These are the *starting points* the search
# departs from on the full set (the valset idx0 all sit ~0.12).
FULLSET_SEED = {
    "p0001": 0.285,  # "Suspicious & Withholding" — already hard before search
    "p0002": 0.211,
    "p0003": 0.167,
}
BASELINE = 0.149  # no_persona (cooperative), full-base 228
OUT = str(DATA_ROOT / "data/persona_opt/figs/difficulty_optimization_p123.png")


def load_traj(run_dir):
    S = pickle.load(open(f"{GR}/{run_dir}/gepa_state.bin", "rb"))
    diffs = [float(o["difficulty"]) for o in S["prog_candidate_objective_scores"]]
    rollouts = list(S["num_metric_calls_by_discovery"])
    best, b = [], -1.0
    for d in diffs:
        b = max(b, d)
        best.append(b)
    return rollouts, diffs, best


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.edgecolor": "#bbbbbb",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    for lbl, rd, col in RUNS:
        x, diffs, best = load_traj(rd)
        # faint candidate dots (the raw search)
        ax.scatter(x, diffs, s=42, color=col, alpha=0.30, zorder=2, edgecolor="none")
        # bold best-so-far step line (the valset convergence)
        ax.step(
            x,
            best,
            where="post",
            color=col,
            lw=2.8,
            zorder=4,
            label=f"{lbl}  (valset {best[-1]:.2f} → full-set {FULLSET_BEST[lbl]:.2f})",
        )
        ax.scatter(x, best, s=30, color=col, zorder=5, edgecolor="white", linewidth=1.0)
        # valset best label
        ax.annotate(
            f"{best[-1]:.2f}",
            (x[-1], best[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=10.5,
            color=col,
            alpha=0.7,
        )
        # full-set realized best: star + vertical connector (valset -> full-set)
        fs = FULLSET_BEST[lbl]
        ax.plot([x[-1], x[-1]], [best[-1], fs], color=col, lw=1.3, ls=":", zorder=4)
        ax.scatter(
            [x[-1]],
            [fs],
            marker="*",
            s=360,
            color=col,
            zorder=6,
            edgecolor="black",
            linewidth=1.1,
        )
        va = "bottom" if fs >= best[-1] else "top"
        dy = 9 if fs >= best[-1] else -9
        ax.annotate(
            f"{fs:.2f}",
            (x[-1], fs),
            textcoords="offset points",
            xytext=(10, dy),
            va=va,
            fontsize=12,
            fontweight="bold",
            color=col,
        )
        # full-set seed persona (diamond at x=0) + connector to valset seed
        sd = FULLSET_SEED[lbl]
        ax.plot(
            [x[0], x[0]], [diffs[0], sd], color=col, lw=1.1, ls=":", zorder=3, alpha=0.7
        )
        ax.scatter(
            [x[0]],
            [sd],
            marker="D",
            s=80,
            color=col,
            zorder=6,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.annotate(
            f"{sd:.2f}",
            (x[0], sd),
            textcoords="offset points",
            xytext=(-9, 0),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=col,
        )

    # no_persona cooperative baseline (full-set)
    ax.axhline(BASELINE, color="#888888", ls="--", lw=1.4, zorder=1)
    ax.annotate(
        f"no_persona baseline (full-set {BASELINE:.2f})",
        xy=(1, BASELINE),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#777777",
        xycoords=("axes fraction", "data"),
    )

    # p0003's jump callout
    ax.annotate(
        "p0003: +0.44 jump\n(idx1 → idx3)",
        xy=(145, 0.571),
        xytext=(195, 0.50),
        fontsize=10.5,
        color="#2E8B4F",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>", color="#2E8B4F", lw=1.8, connectionstyle="arc3,rad=0.2"
        ),
    )

    ax.set_xlabel("GEPA rollouts  (metric calls)", fontsize=12, fontweight="bold")
    ax.set_ylabel("difficulty  =  1 − agent success", fontsize=12, fontweight="bold")
    ax.set_title(
        "GEPA persona optimization — valset search vs full-set realized",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 0.63)
    ax.set_xlim(-15, None)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D(
            [0],
            [0],
            marker="D",
            color="#555555",
            lw=0,
            markersize=9,
            markeredgecolor="black",
            label="◆ seed persona (full-set)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="#555555",
            lw=0,
            markersize=15,
            markeredgecolor="black",
            label="★ best persona (full-set cert)",
        ),
        Line2D(
            [0],
            [0],
            color="#888888",
            ls="--",
            lw=1.4,
            label="no_persona baseline (full-set)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
        title="bold step = valset best-so-far",
        title_fontsize=10,
    )
    fig.text(
        0.5,
        0.005,
        "dots = each candidate's valset difficulty;  bold step = valset "
        "best-so-far;  ★ = that best persona's realized difficulty on the "
        "held-out full-base 228 (validity-passed).",
        ha="center",
        fontsize=8.5,
        color="#888888",
    )

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    for lbl, rd, _ in RUNS:
        x, d, best = load_traj(rd)
        print(
            f"  {lbl}: {len(d)} candidates, {x[-1]} rollouts, best difficulty {best[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
