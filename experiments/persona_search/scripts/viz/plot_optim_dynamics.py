"""Plot: optimization dynamics from the per-iteration program trace.

The single clearest view of *how* the search progressed. Two stacked panels
sharing the iteration axis:

  A (top)  — running-best valset difficulty per iteration (step line), markers
             at accepted candidates (annotated c{idx}:{difficulty}), and the
             background shaded by which PARENT each iteration selected. The
             shading exposes "front collapse": after a breakthrough the search
             keeps exploiting one champion.
  B (bottom) — per-iteration parent vs child minibatch sum on the SAME 10-task
             subsample (paired bars), colored by accept/reject. This is the
             local signal GEPA actually decides on; its volatility across iters
             (same parent, different task draw) exposes evaluation noise.

Why this exists: difficulty_trajectory plots only the ~8 accepted candidates
(sparse, hides rejects + noise). The trace has every iteration's decision.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

# local style knobs (edit just this plot)
ACCEPT_C = "#2e7d32"  # accepted mutation
REJECT_C = "#b0b0b0"  # rejected mutation
PARENT_C = "#1f77b4"  # parent bar
CHILD_C = "#d1495b"  # child bar
BEST_C = "#f0a500"  # best-candidate star


def build(run_id: str, out_dir: Path) -> str:
    trace = _data.iter_trace(run_id)
    cand = _data.candidate_table(run_id)  # idx -> objective scores
    diff_of = {r["cand"]: (r["difficulty"] or 0.0) for r in cand}

    iters = [e["i"] for e in trace]
    parents = [e["selected_program_candidate"] for e in trace]
    p_mb = [sum(e.get("subsample_scores", [])) for e in trace]
    c_mb = [sum(e.get("new_subsample_scores", [])) for e in trace]
    new_idx = [e.get("new_program_idx") for e in trace]
    accepted = [n is not None for n in new_idx]

    # running-best valset difficulty after each iteration
    best, run_best = diff_of.get(0, 0.0), []
    for n in new_idx:
        if n is not None:
            best = max(best, diff_of.get(n, 0.0))
        run_best.append(best)

    st.apply_style()
    fig, (axA, axB) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    # ---- Panel A: running-best difficulty + parent-selection shading ----
    # shade contiguous spans of the same selected parent
    uniq_parents = sorted(set(parents))
    cmap = plt.get_cmap("tab10")
    pcolor = {p: cmap(k % 10) for k, p in enumerate(uniq_parents)}
    start = 0
    for j in range(1, len(iters) + 1):
        if j == len(iters) or parents[j] != parents[start]:
            p = parents[start]
            axA.axvspan(
                iters[start] - 0.5,
                iters[j - 1] + 0.5,
                color=pcolor[p],
                alpha=0.10,
                lw=0,
            )
            axA.text(
                (iters[start] + iters[j - 1]) / 2,
                0.02,
                f"parent c{p}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=pcolor[p],
                transform=axA.get_xaxis_transform(),
            )
            start = j

    axA.step(
        iters,
        run_best,
        where="post",
        color=st.PALETTE["difficulty"],
        lw=2.2,
        label="running-best difficulty",
    )
    for it, n, acc in zip(iters, new_idx, accepted):
        if acc:
            d = diff_of.get(n, 0.0)
            axA.scatter(
                [it],
                [d],
                s=70,
                color=ACCEPT_C,
                zorder=5,
                edgecolor="white",
                linewidth=0.8,
            )
            axA.annotate(
                f"c{n}:{d:.2f}",
                (it, d),
                textcoords="offset points",
                xytext=(4, 7),
                fontsize=8,
                color=ACCEPT_C,
            )
    # mark the global best
    bi = max(range(len(cand)), key=lambda i: diff_of.get(i, 0.0))
    axA.scatter(
        [],
        [],
        marker="*",
        s=180,
        color=BEST_C,
        label=f"best = c{bi} ({diff_of[bi]:.2f})",
    )
    axA.set_ylabel("valset difficulty")
    axA.set_ylim(0, 1)
    axA.set_title(
        f"{run_id}: optimization dynamics  (accept {sum(accepted)}/{len(trace)} iters)"
    )
    axA.legend(loc="upper left", fontsize=9)

    # ---- Panel B: parent vs child minibatch sum, paired bars ----
    x = np.arange(len(iters))
    w = 0.38
    axB.bar(
        x - w / 2, p_mb, w, color=PARENT_C, alpha=0.85, label="parent (same minibatch)"
    )
    childbars = axB.bar(x + w / 2, c_mb, w, color=CHILD_C, alpha=0.85, label="child")
    # outline accepted children in green, rejected in grey
    for b, acc in zip(childbars, accepted):
        b.set_edgecolor(ACCEPT_C if acc else REJECT_C)
        b.set_linewidth(2.0 if acc else 1.2)
    axB.set_xticks(x)
    axB.set_xticklabels(iters)
    axB.set_xlabel("iteration")
    axB.set_ylabel("minibatch difficulty sum")
    # noise annotation for the dominant parent
    dom = max(uniq_parents, key=lambda p: parents.count(p))
    dvals = [p_mb[k] for k in range(len(parents)) if parents[k] == dom]
    if len(dvals) > 1:
        axB.text(
            0.99,
            0.95,
            f"parent c{dom} minibatch sum: "
            f"μ={np.mean(dvals):.1f} σ={np.std(dvals):.1f} "
            f"[{min(dvals):.0f}–{max(dvals):.0f}]  → eval noise",
            transform=axB.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=st.PALETTE["difficulty"],
        )
    axB.legend(loc="upper left", fontsize=9)

    out = out_dir / f"{run_id}_optim_dynamics.png"
    return st.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["p0001", "p0002", "p0003"])
    ap.add_argument("--out-dir", default=str(_data.DATA_ROOT / "data/persona_opt/figs"))
    a = ap.parse_args()
    for r in a.runs:
        d = Path(a.out_dir) / r
        d.mkdir(parents=True, exist_ok=True)
        print(build(r, d))


if __name__ == "__main__":
    main()
