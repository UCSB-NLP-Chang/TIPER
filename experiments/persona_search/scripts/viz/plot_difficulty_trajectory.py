"""Plot 1: difficulty trajectory over the optimization budget.

X = cumulative evals at which each candidate was discovered.
Y = candidate difficulty; line traces the running-best difficulty.
Marker color = validity (green=valid .. red=invalid). Best candidate starred.

Edit STYLE block below to restyle just this plot.
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

# --- per-plot STYLE (local overrides) --------------------------------------
LINE_KW = dict(color=st.PALETTE["difficulty"], lw=2, zorder=2, label="running best")
PT_SIZE = 90


def build(run_id: str, out_dir: Path) -> str:
    rows = _data.candidate_table(run_id)
    rows = [r for r in rows if r["difficulty"] is not None]
    rows.sort(key=lambda r: r["discovered_at"] or 0)
    xs = [r["discovered_at"] for r in rows]
    ys = [r["difficulty"] for r in rows]
    val = [r["validity"] if r["validity"] is not None else 0 for r in rows]
    best_i = max(range(len(rows)), key=lambda i: rows[i]["difficulty"])

    running = []
    cur = -1.0
    for y in ys:
        cur = max(cur, y)
        running.append(cur)

    fig, ax = st.new_fig()
    ax.step(xs, running, where="post", **LINE_KW)
    sc = ax.scatter(
        xs,
        ys,
        c=val,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=PT_SIZE,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for i, r in enumerate(rows):
        ax.annotate(
            f"c{r['cand']}",
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.scatter(
        [xs[best_i]],
        [ys[best_i]],
        marker="*",
        s=320,
        color="none",
        edgecolor=st.PALETTE["best"],
        linewidth=1.8,
        zorder=4,
        label=f"best (c{rows[best_i]['cand']}={ys[best_i]:.3f})",
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("validity")
    ax.set_xlabel("cumulative evals (budget) when discovered")
    ax.set_ylabel("difficulty  (1 − agent reward)")
    ax.set_title(f"{run_id}: difficulty trajectory over optimization budget")
    ax.legend(loc="lower right")
    out = out_dir / f"{run_id}_difficulty_trajectory.png"
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
