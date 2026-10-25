"""Plot 9: conversation length vs outcome.

Histogram of #messages per sim, split by agent outcome (success reward==1 vs
failure reward<1). A persona that wins by dragging conversations out shows a
long-tail in the failure group near the step cap.
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
    sims = _data.load_all_sims(run_id)
    cl = _data.conversation_lengths(sims)
    win = [n for n, r in cl if r is not None and r >= 1.0]
    loss = [n for n, r in cl if r is not None and r < 1.0]

    fig, ax = st.new_fig()
    alln = [n for n, _ in cl] or [1]
    bins = np.linspace(0, max(alln), 30)
    ax.hist(
        win,
        bins=bins,
        alpha=0.6,
        color=st.PALETTE["assistant"],
        label=f"agent success (n={len(win)})",
    )
    ax.hist(
        loss,
        bins=bins,
        alpha=0.6,
        color=st.PALETTE["difficulty"],
        label=f"agent failure (n={len(loss)})",
    )
    if win:
        ax.axvline(np.mean(win), color=st.PALETTE["assistant"], ls="--", lw=1)
    if loss:
        ax.axvline(np.mean(loss), color=st.PALETTE["difficulty"], ls="--", lw=1)
    ax.set_xlabel("messages per conversation")
    ax.set_ylabel("count")
    ax.set_title(f"{run_id}: conversation length by outcome")
    ax.legend()
    out = out_dir / f"{run_id}_conversation_length.png"
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
