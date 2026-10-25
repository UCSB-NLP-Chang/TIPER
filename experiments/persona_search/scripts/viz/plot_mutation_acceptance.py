"""Plot 6: mutation acceptance per iteration.

For each GEPA iteration: minibatch difficulty of the selected parent
(sum of subsample_scores) vs the proposed child (new_subsample_scores).
Child accepted when its minibatch sum strictly improves. Bars colored by
accept/reject; line shows the per-iteration improvement delta.

This is the closest signal to "mutation 方式 / effectiveness": which parents
were selected and whether the reflective mutation actually helped.
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
    trace = _data.iter_trace(run_id)
    its, old, new, parent = [], [], [], []
    for t in trace:
        its.append(t["i"])
        old.append(sum(t.get("subsample_scores") or []))
        new.append(sum(t.get("new_subsample_scores") or []))
        parent.append(t.get("selected_program_candidate"))
    its = np.array(its)
    old = np.array(old, float)
    new = np.array(new, float)
    delta = new - old
    accepted = delta > 0

    import matplotlib.pyplot as plt

    st.apply_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(9, len(its) * 0.9), 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    w = 0.4
    ax1.bar(
        its - w / 2,
        old,
        w,
        color=st.PALETTE["neutral"],
        label="parent (minibatch difficulty)",
    )
    ax1.bar(
        its + w / 2,
        new,
        w,
        color=[
            st.PALETTE["accepted"] if a else st.PALETTE["rejected"] for a in accepted
        ],
        label="child",
    )
    for i, p in zip(its, parent):
        ax1.annotate(f"←c{p}", (i - w / 2, 0.1), fontsize=7, ha="center", color="white")
    ax1.set_ylabel("Σ minibatch difficulty")
    ax1.set_title(f"{run_id}: mutation acceptance per iteration")
    ax1.legend(loc="upper left")

    ax2.bar(
        its,
        delta,
        color=[
            st.PALETTE["accepted"] if a else st.PALETTE["rejected"] for a in accepted
        ],
    )
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Δ (child−parent)")
    ax2.set_xlabel("GEPA iteration")
    n_acc = int(accepted.sum())
    ax2.text(
        0.99,
        0.9,
        f"accepted {n_acc}/{len(its)}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=st.PALETTE["accepted"],
    )
    out = out_dir / f"{run_id}_mutation_acceptance.png"
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
