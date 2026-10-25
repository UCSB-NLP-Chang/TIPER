#!/usr/bin/env python3
"""Figure: adversarial-difficulty convergence along the p0003 winning lineage,
on the FULL-SET (held-out full-base 114 tasks), with the sim2real behavioural
judge score overlaid on a secondary axis.

  left  y-axis : difficulty   = 1 − mean reward (full-base cert)
  right y-axis : sim2real score = behMean (mean Dice D1-D4 vs paper-human),
                 from sim2real_full_table.py (default: hybrid LLM judge)

Lineage: no_persona (cooperative) → idx0 seed → idx1 → idx3 (best).
Difficulty is read live from the cert results.json; the sim2real scores are read
from <run>/sim2real_behmean.json (written from sim2real_full_table runs).

    python experiments/persona_search/scripts/plot_difficulty_convergence.py \
        --run-dir data/persona_opt/gepa_runs/p0003_aggressive_20260606_145857 \
        --sim2real hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

import matplotlib

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402,F401

CERT = str(DATA_ROOT / "data/simulations/gepa_certify")
# idx4 is on the difficulty line only: it has no entry in sim2real_behmean.json,
# so sim.get("idx4") -> None and the sim2real line skips it automatically.
NODES = ["no_persona", "seed", "idx1", "idx3", "idx4"]
FULLSET_DIR = {
    "no_persona": "progcert_no_persona_fullbase228",
    "seed": "progcert_seed",
    "idx1": "progcert_idx1",
    "idx3": "progcert_candidate",
    "idx4": "progcert_idx4",
}
NODE_LABEL = {
    "no_persona": "no_persona\n(cooperative)",
    "seed": "seed\nDistracted &\nRambling",
    "idx1": "idx1\n+wrong-cause\n+careless verify",
    "idx3": "idx3 · BEST\n+fix-lobby\n+premature\nwrap-up",
    "idx4": "idx4\n(descendant\nof idx3)",
}

DIFF_COL = "#d1495b"  # difficulty bars (warm red)
SIM_COL = "#2e7d8a"  # sim2real line (teal)
BEST_EDGE = "#1b7a3d"


def fullset_difficulty(key: str):
    p = Path(f"{CERT}/{FULLSET_DIR[key]}/telecom/results.json")
    if not p.exists():
        return None
    sims = json.load(open(p)).get("simulations", [])
    r = [x["reward_info"]["reward"] for x in sims if x.get("reward_info")]
    if not r:
        return None
    return {"difficulty": 1 - st.mean(r), "n": len(sims), "partial": len(sims) < 228}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sim2real", choices=["hybrid", "strict"], default="hybrid")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "fullset_difficulty_sim2real.png"

    diffs = {k: fullset_difficulty(k) for k in NODES}
    sim = json.load(open(run_dir / "sim2real_behmean.json"))[args.sim2real]

    xs = list(range(len(NODES)))
    dvals = [diffs[k]["difficulty"] if diffs[k] else None for k in NODES]
    svals = [sim.get(k) for k in NODES]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axL = plt.subplots(figsize=(10.5, 6.6))
    axR = axL.twinx()

    # ---- left axis: difficulty as a line ------------------------------------
    dx = [x for x, v in zip(xs, dvals) if v is not None]
    dy = [v for v in dvals if v is not None]
    axL.plot(
        dx,
        dy,
        "-o",
        color=DIFF_COL,
        lw=2.8,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=1.6,
        zorder=4,
    )
    # emphasize the best node
    axL.scatter(
        [3],
        [dvals[3]],
        s=230,
        facecolor=DIFF_COL,
        edgecolor=BEST_EDGE,
        linewidth=2.6,
        zorder=5,
    )
    for x, d in zip(xs, dvals):
        if d is None:
            continue
        partial = diffs[NODES[x]]["partial"]
        axL.annotate(
            f"{d:.3f}" + ("*" if partial else ""),
            (x, d),
            textcoords="offset points",
            xytext=(0, -17),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=DIFF_COL,
        )

    axL.set_ylabel(
        "difficulty  =  1 − agent reward   (full-base 114 tasks)",
        color=DIFF_COL,
        fontsize=12,
        fontweight="bold",
    )
    axL.tick_params(axis="y", colors=DIFF_COL)
    axL.set_ylim(0, max(d for d in dvals if d is not None) * 1.35)
    axL.set_xticks(xs)
    axL.set_xticklabels([NODE_LABEL[k] for k in NODES], fontsize=10)
    axL.set_axisbelow(True)
    axL.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    for spine in ("top",):
        axL.spines[spine].set_visible(False)
        axR.spines[spine].set_visible(False)

    # ---- right axis: sim2real behMean as a line -----------------------------
    sx = [x for x, v in zip(xs, svals) if v is not None]
    sy = [v for v in svals if v is not None]
    axR.plot(
        sx,
        sy,
        "-o",
        color=SIM_COL,
        lw=2.6,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=1.6,
        zorder=5,
    )
    for x, v in zip(sx, sy):
        # nudge idx3's label right so it clears the difficulty-jump callout
        dx = 16 if x == 3 else 0
        axR.annotate(
            f"{v:.1f}",
            (x, v),
            textcoords="offset points",
            xytext=(dx, 13),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=SIM_COL,
        )

    axR.set_ylabel(
        f"sim2real score  =  behMean Dice vs human  ({args.sim2real} judge)",
        color=SIM_COL,
        fontsize=12,
        fontweight="bold",
    )
    axR.tick_params(axis="y", colors=SIM_COL)
    lo, hi = min(sy), max(sy)
    axR.set_ylim(lo - (hi - lo) * 0.8, hi + (hi - lo) * 0.6)

    # ---- legend / title -----------------------------------------------------
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=DIFF_COL,
            marker="o",
            lw=2.8,
            label="difficulty (left, ↑ = harder)",
        ),
        Line2D(
            [0],
            [0],
            color=SIM_COL,
            marker="o",
            lw=2.6,
            label="sim2real human-likeness (right, ↑ = more human)",
        ),
    ]
    axL.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
    axL.set_title(
        "p0003 lineage — adversarial difficulty climbs, human-likeness holds",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    partial_note = ""
    if any(diffs[k] and diffs[k]["partial"] for k in NODES):
        partial_note = "   (* partial cert, still running)"
    fig.text(
        0.5,
        0.005,
        "difficulty = 1 − mean reward on full-base telecom (114 tasks × 2 trials);  "
        f"sim2real behMean = mean Dice(D1–D4) vs paper-human ({args.sim2real})."
        + partial_note,
        ha="center",
        fontsize=8.5,
        color="#777777",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    print(
        "difficulty:", {k: round(diffs[k]["difficulty"], 3) for k in NODES if diffs[k]}
    )
    print(f"sim2real({args.sim2real}):", sim)


if __name__ == "__main__":
    main()
