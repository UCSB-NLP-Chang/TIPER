"""Plot 4: candidate lineage tree (the mutation genealogy).

Nodes = candidates laid out by discovery order; edges = parent -> child
mutations. Node color = difficulty, node label shows difficulty & validity.
Shows how mutation explored the space and where difficulty jumped.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)


def build(run_id: str, out_dir: Path) -> str:
    rows = _data.candidate_table(run_id)
    # depth = generation (distance from root); x = discovery order
    depth = {}
    for r in rows:
        p = r["parent"]
        depth[r["cand"]] = 0 if p is None else depth.get(p, 0) + 1
    # x position: order discovered; y position: -depth
    pos = {r["cand"]: (i, -depth[r["cand"]]) for i, r in enumerate(rows)}

    fig, ax = st.new_fig(figsize=(max(8, len(rows) * 1.5), 5))
    norm = mcolors.Normalize(
        vmin=0, vmax=max((r["difficulty"] or 0) for r in rows) or 1
    )
    cmap = cm.get_cmap(st.HEATMAP_CMAP)

    # edges
    for r in rows:
        if r["parent"] is not None and r["parent"] in pos:
            x0, y0 = pos[r["parent"]]
            x1, y1 = pos[r["cand"]]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.3),
            )
    # nodes
    for r in rows:
        x, y = pos[r["cand"]]
        diff = r["difficulty"] or 0
        ax.scatter(
            [x],
            [y],
            s=900,
            color=cmap(norm(diff)),
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f"c{r['cand']}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x,
            y - 0.28,
            f"d={diff:.2f}\nv={(r['validity'] or 0):.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("difficulty")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("discovery order  →  (depth = mutation generation)")
    ax.set_title(f"{run_id}: candidate lineage / mutation tree")
    ax.margins(0.15)
    out = out_dir / f"{run_id}_lineage_tree.png"
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
