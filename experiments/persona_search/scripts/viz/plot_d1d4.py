"""Decompose the 35-overlap naturalness into its four sim2real dimensions D1-D4
(behMean = mean(D1..D4)) and plot each dimension's trajectory over generations for
both lineages, so we can see WHICH dimension drives the naturalness curve.

Uses the same per-gen best-born persona selection and cert-preferred sims as
analyze_overlap35.py (mode='born').

--ref selects the HUMAN reference distribution the dimensions are scored against:
  brinks : real Brinks Home support chats          (user_simulator_dataset_v2.json)
  paper  : the Sim2Real paper tau-bench human-as-user baseline
           (tau_bench_tasks_unified.json, 495 telecom sessions) -- the "original" data

    .venv/bin/python experiments/persona_search/scripts/viz/plot_d1d4.py --ref paper
    .venv/bin/python experiments/persona_search/scripts/viz/plot_d1d4.py --ref brinks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HERE.parent))

import os
import sys

import _style as st  # noqa: E402
import analyze_overlap35 as A  # noqa: E402

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
import sim2real_run_tau2 as R  # noqa: E402
from sim2real_naturalness_brinks import brinks_human_turn_lists, d1_d4  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import beh_mean, feature_matrix, load_human_features  # noqa: E402

DIM_TITLE = {
    "D1": "D1  style  (verbosity / politeness / formality)",
    "D2": "D2  info structuring  (front-loading / opening length)",
    "D3": "D3  stance  (uncertainty / pushback / questions)",
    "D4": "D4  emotion  (frustration / accusation / pivot)",
}
REF_LABEL = {"brinks": "Brinks human chats", "paper": "tau-bench paper human"}


def _pooled(ref: str):
    M.set_mode("strict")
    if ref == "paper":
        Hx = load_human_features("strict")
    else:
        Hx = feature_matrix(brinks_human_turn_lists())
    return {k: float(np.nanmean(Hx[:, i])) for i, k in enumerate(M.METRIC_ORDER)}


def _row_metrics(sims, pooled):
    Sx = feature_matrix([R.tau2_user_turns(s.get("messages", [])) for s in sims])
    dd = d1_d4(Sx, pooled)
    return {
        "behMean": round(float(beh_mean(Sx, pooled)), 2),
        **{d: round(dd[d], 1) for d in ("D1", "D2", "D3", "D4")},
    }


def compute(ref: str) -> dict:
    overlap = A.overlap_task_ids()
    pooled = _pooled(ref)
    data = {"ref": ref, "overlap_n": len(overlap), "arms": {}}

    np_sims = A._sims_in(A.NO_PERSONA, overlap)
    data["floor"] = {
        **_row_metrics(np_sims, pooled),
        "difficulty": round(A.difficulty_on(np_sims), 4),
    }

    for arm in A.RUNS:
        state = __import__("json").loads(A.RUNS[arm][1].read_text())
        cg = {c["idx"]: c for c in state["candidates"]}
        overlap_inst = set(
            __import__("json").loads(A.RUNS["brinks_04"][1].read_text())["instance_ids"]
        )
        base, lineage = A.build_points(arm, "born")
        rows = []
        for pt in lineage:
            gen, idx, k, valid = pt["gen"], pt["idx"], pt["k"], pt["valid"]
            cert = A.CERT.get((arm, idx)) if idx is not None else None
            if cert and cert.exists():
                rp, trials = cert, 2
            else:
                rp, trials = A._best_results(base, pt["birth_gen"], k), 1
            if rp is None:
                continue
            sims = A._sims_in(rp, overlap)
            ss = cg.get(idx, {}).get("subscores") if idx is not None else None
            if ss:
                vals = [ss[i] for i in overlap_inst if i in ss]
                diff = sum(vals) / len(vals) if vals else float("nan")
            else:
                diff = pt.get("screen_diff", float("nan"))
            rows.append(
                {
                    "gen": gen,
                    "idx": idx,
                    "valid": valid,
                    "trials": trials,
                    "difficulty": round(diff, 4),
                    **_row_metrics(sims, pooled),
                }
            )
        data["arms"][arm] = rows
    return data


def draw(data: dict, out_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    st.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    dims = ["D1", "D2", "D3", "D4"]
    floor = data["floor"]
    gmax = max(r["gen"] for rows in data["arms"].values() for r in rows)

    for ax, dim in zip(axes.flat, dims):
        ax.axhline(floor[dim], ls=":", lw=1.8, color="0.4", zorder=1)
        for arm, rows in data["arms"].items():
            col = A.ARM_COLOR[arm]
            valid = [r for r in rows if r["valid"]]
            vx = [r["gen"] for r in valid]
            vy = [r[dim] for r in valid]
            ax.plot(
                vx,
                vy,
                lw=2.4,
                color=col,
                zorder=3,
                label=f"{A.ARM_LABEL[arm]} ({vy[0]:.0f}→{vy[-1]:.0f})",
            )
            ax.scatter(vx, vy, marker="o", s=60, color=col, zorder=4)
            for r in rows:
                if r["valid"]:
                    continue
                prev = [v for v in valid if v["gen"] < r["gen"]]
                if prev:
                    p = prev[-1]
                    ax.plot(
                        [p["gen"], r["gen"]],
                        [p[dim], r[dim]],
                        color=col,
                        lw=1.5,
                        ls="--",
                        zorder=2,
                    )
                ax.scatter(
                    r["gen"],
                    r[dim],
                    marker="X",
                    s=90,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.8,
                    zorder=4,
                )
        ax.set_title(DIM_TITLE[dim], fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(gmax + 1))
        ax.legend(loc="lower left", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("optimization generation")
    for ax in axes[:, 0]:
        ax.set_ylabel("dimension score (Dice vs human)")
    handles = [
        Line2D([], [], ls=":", lw=1.8, color="0.4", label="no_persona floor"),
        Line2D(
            [],
            [],
            marker="X",
            ls="",
            mfc="white",
            mec="0.4",
            mew=1.8,
            label="validity-gated (rejected)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"Naturalness decomposed into D1-D4  —  reference: {REF_LABEL[data['ref']]}"
        f"  (35-task overlap, best persona born each gen)",
        fontsize=12.5,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return st.save(fig, out_path)


def draw_agg(out_path: Path):
    """Aggregate naturalness = behMean = mean(D1..D4), both references overlaid so the
    reference-dependent flip is visible in a single curve per arm."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    st.apply_style()
    refs = {"paper": ("-", "o"), "brinks": ("--", "s")}
    datas = {r: compute(r) for r in refs}
    fig, ax = plt.subplots(figsize=(10, 6))
    gmax = 0
    for ref, (ls, mk) in refs.items():
        d = datas[ref]
        for arm, rows in d["arms"].items():
            col = A.ARM_COLOR[arm]
            v = [r for r in rows if r["valid"]]
            gmax = max(gmax, max(r["gen"] for r in v))
            ax.plot(
                [r["gen"] for r in v],
                [r["behMean"] for r in v],
                ls=ls,
                marker=mk,
                ms=7,
                lw=2.4,
                color=col,
                zorder=3,
            )
        ax.axhline(d["floor"]["behMean"], ls=ls, lw=1.3, color="0.55", zorder=1)
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("naturalness  (behMean = mean of D1-D4)")
    ax.set_title(
        "Aggregate naturalness over optimization  —  paper vs Brinks human "
        "reference  (35-task overlap)"
    )
    ax.set_xticks(range(gmax + 1))
    ax.set_ylim(30, 70)
    handles = [
        Line2D([], [], color=A.ARM_COLOR[a_], lw=2.4, label=A.ARM_LABEL[a_])
        for a_ in datas["paper"]["arms"]
    ]
    handles += [
        Line2D(
            [], [], color="0.3", ls="-", marker="o", label="ref: tau-bench paper human"
        ),
        Line2D(
            [], [], color="0.3", ls="--", marker="s", label="ref: Brinks human chats"
        ),
        Line2D(
            [], [], color="0.55", ls="-", lw=1.3, label="no_persona floor (per ref)"
        ),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    print(f"\naggregate behMean (seed -> final, born):")
    for ref in refs:
        for arm, rows in datas[ref]["arms"].items():
            v = [r for r in rows if r["valid"]]
            print(
                f"  [{ref:6}] {A.ARM_LABEL[arm]:<26} {v[0]['behMean']:.1f} -> "
                f"{v[-1]['behMean']:.1f}   (floor {datas[ref]['floor']['behMean']:.1f})"
            )
    return st.save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", choices=["brinks", "paper"], default="paper")
    ap.add_argument(
        "--agg",
        action="store_true",
        help="draw the aggregate behMean figure (both refs overlaid)",
    )
    a = ap.parse_args()
    out = REPO / "data/persona_opt/figs/p0002_brinks04"
    out.mkdir(parents=True, exist_ok=True)
    if a.agg:
        print(draw_agg(out / "overlap35_naturalness_aggref.png"))
        return
    data = compute(a.ref)
    sfx = "_paperref" if a.ref == "paper" else "_brinksref"
    print(draw(data, out / f"overlap35_d1d4{sfx}.png"))

    dims = ["D1", "D2", "D3", "D4"]
    print(f"\nreference = {REF_LABEL[a.ref]}   seed -> final (born) per dimension:")
    for arm, rows in data["arms"].items():
        v = [r for r in rows if r["valid"]]
        s, f = v[0], v[-1]
        ds = "  ".join(f"{d}:{s[d]:.0f}→{f[d]:.0f}" for d in dims)
        print(
            f"  {A.ARM_LABEL[arm]:<26} behMean:{s['behMean']:.0f}→{f['behMean']:.0f}"
            f"   {ds}"
        )
    fl = data["floor"]
    print(
        f"  {'no_persona floor':<26} behMean:{fl['behMean']:.0f}        "
        + "  ".join(f"{d}:{fl[d]:.0f}" for d in dims)
    )


if __name__ == "__main__":
    main()
