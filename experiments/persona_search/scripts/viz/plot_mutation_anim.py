"""Animated GIF of the optimization from the MUTATION angle.

Replays the per-iteration program trace as a growing lineage tree. Each
iteration plays as two frames:

  1. PROPOSE — the selected parent pulses; a tentative child appears (grey);
     the minibatch panel shows parent vs child difficulty sum on the SAME
     10-task subsample ("the signal the mutation is judged on").
  2. VERDICT — ACCEPT: child joins the tree, colored by its valset difficulty
     (magma); REJECT: child turns red ✗ and is dropped.

The left panel is the lineage tree (accepted candidates persist, colored by
difficulty). The right panels track running-best difficulty and the current
minibatch decision. Watch for: the early breakthrough mutation, then the
"front collapse" where the search keeps mutating the single champion.

Output: <out_dir>/<run>_mutation.gif   (PillowWriter, no ffmpeg needed)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

# local style knobs (edit just this animation)
FPS = 1.6
HOLD_LAST = 4  # repeat final frame N times so the GIF lingers
GHOST_C = "#bdbdbd"  # tentative / pending child
REJECT_C = "#c0392b"
PARENT_RING = "#1f77b4"
CMAP = "magma"


def _tree_layout(parents: dict[int, int | None]) -> dict[int, tuple[float, float]]:
    """Assign (x=depth, y=spread-within-depth) to every candidate node."""
    depth: dict[int, int] = {}

    def d(n):
        if n in depth:
            return depth[n]
        p = parents.get(n)
        depth[n] = 0 if p is None else d(p) + 1
        return depth[n]

    for n in parents:
        d(n)
    by_depth: dict[int, list[int]] = {}
    for n in sorted(parents):
        by_depth.setdefault(depth[n], []).append(n)
    pos = {}
    for dep, nodes in by_depth.items():
        k = len(nodes)
        for j, n in enumerate(nodes):
            y = j - (k - 1) / 2.0
            pos[n] = (dep, y)
    return pos


def build(run_id: str, out_dir: Path) -> str:
    trace = _data.iter_trace(run_id)
    rows = _data.candidate_table(run_id)
    diff_of = {r["cand"]: (r["difficulty"] or 0.0) for r in rows}
    parents = {r["cand"]: r["parent"] for r in rows}
    pos = _tree_layout(parents)
    vmax = max(0.1, max(diff_of.values()))

    # ---- build the frame script ----
    frames = []
    visible = {0}  # seed always present
    best = diff_of.get(0, 0.0)
    best_hist = [(0, best)]  # (iter, running-best) revealed progressively
    for e in trace:
        it = e["i"]
        par = e["selected_program_candidate"]
        child = e.get("new_program_idx")
        p_mb = sum(e.get("subsample_scores", []))
        c_mb = sum(e.get("new_subsample_scores", []))
        acc = child is not None
        frames.append(
            dict(
                it=it,
                parent=par,
                child=child,
                p_mb=p_mb,
                c_mb=c_mb,
                visible=set(visible),
                best_hist=list(best_hist),
                phase="propose",
                acc=acc,
            )
        )
        if acc:
            best = max(best, diff_of.get(child, 0.0))
            visible = visible | {child}
        best_hist = best_hist + [(it + 1, best)]
        frames.append(
            dict(
                it=it,
                parent=par,
                child=child,
                p_mb=p_mb,
                c_mb=c_mb,
                visible=set(visible),
                best_hist=list(best_hist),
                phase="verdict",
                acc=acc,
            )
        )
    frames += [frames[-1]] * HOLD_LAST  # linger on the end state

    st.apply_style()
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.7, 1.0], height_ratios=[1, 1], hspace=0.55, wspace=0.25
    )
    axT = fig.add_subplot(gs[:, 0])  # tree (spans both rows)
    axD = fig.add_subplot(gs[0, 1])  # running-best difficulty
    axM = fig.add_subplot(gs[1, 1])  # minibatch bars
    cmap = plt.get_cmap(CMAP)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xlim = (min(xs) - 0.6, max(xs) + 0.9)
    ylim = (min(ys) - 0.8, max(ys) + 0.8)

    def draw(fr):
        for ax in (axT, axD, axM):
            ax.clear()

        # ---- tree ----
        axT.set_xlim(*xlim)
        axT.set_ylim(*ylim)
        axT.axis("off")
        vis = fr["visible"]
        # edges among visible nodes
        for n in vis:
            p = parents.get(n)
            if p is not None and p in vis:
                axT.plot(
                    [pos[p][0], pos[n][0]],
                    [pos[p][1], pos[n][1]],
                    "-",
                    color="#999",
                    lw=1.3,
                    zorder=1,
                )
        # nodes
        for n in vis:
            x, y = pos[n]
            d = diff_of.get(n, 0.0)
            axT.scatter(
                [x],
                [y],
                s=520,
                color=cmap(d / vmax),
                zorder=3,
                edgecolor="white",
                linewidth=1.5,
            )
            txtc = "black" if d / vmax > 0.55 else "white"  # contrast on bright magma
            axT.text(
                x,
                y,
                f"c{n}\n{d:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txtc,
                zorder=4,
                fontweight="bold",
            )
        # highlight selected parent
        par = fr["parent"]
        if par in vis:
            px, py = pos[par]
            axT.scatter(
                [px],
                [py],
                s=900,
                facecolors="none",
                edgecolors=PARENT_RING,
                linewidths=2.5,
                zorder=5,
            )
        # pending child
        child = fr["child"]
        if child is not None and child in pos:
            cx, cy = pos[child]
            if fr["phase"] == "propose":
                axT.plot(
                    [pos[par][0], cx],
                    [pos[par][1], cy],
                    "--",
                    color=GHOST_C,
                    lw=1.4,
                    zorder=1,
                )
                axT.scatter(
                    [cx],
                    [cy],
                    s=520,
                    color=GHOST_C,
                    zorder=2,
                    edgecolor="white",
                    linewidth=1.2,
                )
                axT.text(
                    cx,
                    cy,
                    "?",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                    zorder=4,
                    fontweight="bold",
                )
        elif not fr["acc"]:
            # rejected: ghost ✗ off the parent
            if par in pos:
                gx, gy = pos[par][0] + 0.55, pos[par][1] + 0.45
                axT.plot(
                    [pos[par][0], gx],
                    [pos[par][1], gy],
                    ":",
                    color=REJECT_C,
                    lw=1.2,
                    zorder=1,
                )
                axT.scatter(
                    [gx],
                    [gy],
                    s=360,
                    color=REJECT_C,
                    alpha=0.65,
                    zorder=2,
                    edgecolor="white",
                    linewidth=1.0,
                )
                axT.text(
                    gx,
                    gy,
                    "✗",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                    zorder=4,
                    fontweight="bold",
                )

        verdict = (
            "proposing mutation…"
            if fr["phase"] == "propose"
            else (
                f"ACCEPT → c{child} (diff {diff_of.get(child, 0):.2f})"
                if fr["acc"]
                else "REJECT (minibatch not improved)"
            )
        )
        vc = (
            st.PALETTE.get("difficulty", "#d1495b")
            if fr["phase"] == "propose"
            else ("#2e7d32" if fr["acc"] else REJECT_C)
        )
        axT.set_title(
            f"{run_id}  ·  iter {fr['it']}  ·  parent c{par}\n{verdict}",
            fontsize=12,
            color=vc,
        )

        # ---- running-best difficulty ----
        bh = fr["best_hist"]
        axD.step(
            [i for i, _ in bh],
            [b for _, b in bh],
            where="post",
            color=st.PALETTE.get("difficulty", "#d1495b"),
            lw=2.2,
        )
        axD.scatter(
            [bh[-1][0]],
            [bh[-1][1]],
            s=60,
            color=st.PALETTE.get("difficulty", "#d1495b"),
            zorder=5,
        )
        axD.set_xlim(-0.5, len(trace) + 0.5)
        axD.set_ylim(0, 1)
        axD.set_ylabel("running-best\ndifficulty")
        axD.set_title("progress", fontsize=10)
        axD.grid(True, alpha=0.3)

        # ---- minibatch decision ----
        axM.bar([0], [fr["p_mb"]], 0.55, color=PARENT_RING, alpha=0.85, label="parent")
        cb = axM.bar(
            [1],
            [fr["c_mb"]],
            0.55,
            color=cmap(diff_of.get(child, 0.0) / vmax) if fr["acc"] else GHOST_C,
            alpha=0.9,
            label="child",
        )
        cb[0].set_edgecolor("#2e7d32" if fr["acc"] else REJECT_C)
        cb[0].set_linewidth(2.4)
        axM.set_xticks([0, 1])
        axM.set_xticklabels(["parent", "child"])
        axM.set_ylim(0, max(10, fr["p_mb"], fr["c_mb"]) * 1.1)
        axM.set_ylabel("minibatch\ndiff sum")
        axM.set_title("same 10-task subsample", fontsize=10)
        axM.grid(True, axis="y", alpha=0.3)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=int(1000 / FPS), blit=False)
    out = out_dir / f"{run_id}_mutation.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return str(out)


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
