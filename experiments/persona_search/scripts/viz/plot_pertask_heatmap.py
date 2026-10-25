"""Plot 5: per-task difficulty heatmap (candidate x task).

Rows = candidates, cols = the 50 reprep tasks, cell = per-task difficulty
score (0 easy .. 1 hard). Reveals which tasks each persona makes hard and
whether difficulty is broad or concentrated on a few tasks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st
import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)


def build(run_id: str, out_dir: Path) -> str:
    mat = np.array(_data.subscores(run_id))  # cand x task
    rows = _data.candidate_table(run_id)
    # order tasks by mean difficulty (hardest left) for readability
    order = np.argsort(-mat.mean(axis=0))
    mat = mat[:, order]

    st.apply_style()
    fig, ax = st.new_fig(figsize=(13, max(3, len(rows) * 0.6)))
    im = ax.imshow(mat, aspect="auto", cmap=st.HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"c{r['cand']} (d={r['difficulty']:.2f})" for r in rows])
    ax.set_xlabel(f"task (n={mat.shape[1]}, sorted hardest→easiest)")
    ax.set_ylabel("candidate")
    ax.set_title(f"{run_id}: per-task difficulty heatmap")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("per-task difficulty")
    out = out_dir / f"{run_id}_pertask_heatmap.png"
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
