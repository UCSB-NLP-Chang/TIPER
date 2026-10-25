"""Plot 2: difficulty vs validity Pareto scatter.

Each candidate is a point in (validity, difficulty) space. The Pareto-optimal
frontier (max difficulty for >= each validity) is highlighted. Goal region =
high difficulty AND high validity (top-right).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

PT_SIZE = 130


def _pareto_front(pts):
    """pts: list of (validity, difficulty, cand). Front = non-dominated (both maximized)."""
    front = []
    for p in sorted(pts, key=lambda t: (-t[0], -t[1])):
        if all(
            not (q[0] >= p[0] and q[1] > p[1]) and not (q[0] > p[0] and q[1] >= p[1])
            for q in front
        ):
            front.append(p)
    return sorted(front, key=lambda t: t[0])


def build(run_id: str, out_dir: Path) -> str:
    rows = [r for r in _data.candidate_table(run_id) if r["difficulty"] is not None]
    pts = [(r["validity"] or 0, r["difficulty"], r["cand"]) for r in rows]
    front = _pareto_front(pts)

    fig, ax = st.new_fig(figsize=(7.5, 6))
    # goal region shading
    ax.axhspan(0.3, 1.0, xmin=0.7, color=st.PALETTE["difficulty"], alpha=0.05)
    ax.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=PT_SIZE,
        color=st.PALETTE["neutral"],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label="candidates",
    )
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            "--",
            color=st.PALETTE["best"],
            lw=1.8,
            zorder=2,
            label="Pareto front",
        )
    for v, d, c in pts:
        ax.annotate(
            f"c{c}", (v, d), textcoords="offset points", xytext=(7, 4), fontsize=8
        )
    ax.set_xlabel("validity  (legal / in-character)")
    ax.set_ylabel("difficulty  (1 − agent reward)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"{run_id}: difficulty vs validity (Pareto)")
    ax.legend(loc="upper left")
    out = out_dir / f"{run_id}_pareto_difficulty_validity.png"
    return st.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["p0001", "p0002"])
    ap.add_argument("--out-dir", default=str(_data.DATA_ROOT / "data/persona_opt/figs"))
    a = ap.parse_args()
    for r in a.runs:
        d = Path(a.out_dir) / r
        d.mkdir(parents=True, exist_ok=True)
        print(build(r, d))


if __name__ == "__main__":
    main()
