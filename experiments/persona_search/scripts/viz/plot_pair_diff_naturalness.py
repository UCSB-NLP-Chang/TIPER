"""Two figures for the p0003/p0002 pair run (new file; reuses _style only):

  fig 1  difficulty over optimization — FULL-SET certified difficulty, running-best
         per arm, certified candidates mapped to their birth generation
  fig 2  sim2real naturalness — behMean (0-100) bars + P_geom (0-1) on a 2nd axis,
         per certified persona, by arm

Data (read live, nothing hardcoded):
  - data/persona_opt/pgepa_runs/pgepa_pair_p000{2,3}/pgepa_state.json   (idx -> gen)
  - data/persona_opt/pgepa_runs/cert_pair_full/sim2real_naturalness_hybrid_failonly.json
        (full-set `difficulty` + `valid` + `behMean` + `P_geom_naturalness`)

  .venv/bin/python experiments/persona_search/scripts/viz/plot_pair_diff_naturalness.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import _style as st

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
PGEPA = REPO / "data/persona_opt/pgepa_runs"
CERT = PGEPA / "cert_pair_full"
NAT_FILE = CERT / "sim2real_naturalness_hybrid_failonly.json"  # hybrid, failures-only
ARMS = ["p0003", "p0002"]

# --- OOD (held-out) difficulty -------------------------------------------------
# The full-set cert evaluates all 114 telecom base tasks; the GEPA search only ever
# saw the 50 "reprep" tasks. OOD = base − reprep (64 held-out tasks). Difficulty is
# 1 − mean(reward==1) over that subset, matching the full-set definition exactly.
SIM_DIR = REPO / "data/simulations"
SPLIT_TASKS = REPO / "data/tau2/domains/telecom/split_tasks.json"
REP_TASKS = REPO / "data/persona_opt/representative_tasks_50telecom.json"
# pids whose cert sims are not under data/simulations/gepa_certify/<pid>/telecom/
SPECIAL_RESULTS = {
    "p0002_seed_gen0": SIM_DIR
    / "20260607_112504_telecom_llm_agent_gpt-5.4_cc_user_con_claude-opus-4-7/results.json",
    "no_persona": SIM_DIR
    / "gepa_certify/progcert_no_persona_fullbase228/telecom/results.json",
}


def _ood_tasks() -> set[str]:
    rep = set(json.load(open(REP_TASKS))["telecom"])
    base = set(json.load(open(SPLIT_TASKS))["base"])
    return base - rep


def _results_path(pid: str) -> Path:
    return SPECIAL_RESULTS.get(
        pid, SIM_DIR / "gepa_certify" / pid / "telecom" / "results.json"
    )


def _difficulty_on(pid: str, tasks: set[str]) -> float:
    """1 − mean(reward==1) over the sims whose task_id is in `tasks`."""
    d = json.load(open(_results_path(pid)))
    won = []
    for s in d.get("simulations", []):
        if s.get("task_id") not in tasks:
            continue
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        won.append(1.0 if float(r) >= 1.0 else 0.0)
    return (1.0 - sum(won) / len(won)) if won else float("nan")


def ood_diff_map(rows: list[dict]) -> dict[str, float]:
    """pid -> held-out (OOD) difficulty, for every persona present in the cert json."""
    tasks = _ood_tasks()
    return {r["pid"]: _difficulty_on(r["pid"], tasks) for r in rows}


def _arm_of(who: str) -> str:
    return "p0003" if who.startswith("p0003") else "p0002"


def _idx_of(who: str) -> int:
    """'p0003 idx4' -> 4 ; 'p0003 seed (=incumbent)' -> 0 (seed == candidate idx0)."""
    m = re.search(r"idx(\d+)", who)
    return int(m.group(1)) if m else 0


def _gen_map(arm: str) -> dict[int, int]:
    d = json.load(open(PGEPA / f"pgepa_pair_{arm}" / "pgepa_state.json"))
    return {c["idx"]: c["gen"] for c in d["candidates"]}


G = 5  # generations 0..G
NP_X = -1  # x-position of the shared no_persona floor (both arms start here)


def np_row(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r["who"] == "no_persona"), None)


def leader_track(arm: str, rows: list[dict]) -> list[dict]:
    """Per generation 0..G, the running difficulty-leading VALID certified persona
    for this arm (its difficulty/behMean/P_geom). gen 0 = seed. Both figures share
    this track so they line up persona-for-persona."""
    gmap = _gen_map(arm)
    cells = [
        {**r, "gen": gmap[_idx_of(r["who"])]} for r in rows if _arm_of(r["who"]) == arm
    ]
    valid_gens = [c["gen"] for c in cells if c["valid"]]
    if not valid_gens:
        return []
    g0 = min(valid_gens)
    track, best = [], None
    for g in range(g0, G + 1):
        cand = [c for c in cells if c["valid"] and c["gen"] <= g]
        if cand:
            best = max(cand, key=lambda c: c["difficulty"] or 0)
        if best is not None:
            track.append({**best, "gen": g})  # loop gen overrides persona birth gen
    return track


def latest_track(arm: str, rows: list[dict]) -> list[dict]:
    """Per generation 0..G, the most-recently-born VALID certified persona for this
    arm (running-latest by candidate idx). The endpoint is the optimization endpoint
    (highest idx), not the difficulty leader."""
    gmap = _gen_map(arm)
    cells = [
        {**r, "gen": gmap[_idx_of(r["who"])]} for r in rows if _arm_of(r["who"]) == arm
    ]
    valid = [c for c in cells if c["valid"]]
    if not valid:
        return []
    g0 = min(c["gen"] for c in valid)
    track, best = [], None
    for g in range(g0, G + 1):
        cand = [c for c in valid if c["gen"] <= g]
        if cand:
            best = max(cand, key=lambda c: _idx_of(c["who"]))
        if best is not None:
            track.append({**best, "gen": g})
    return track


def _xaxis_with_floor(ax):
    ax.set_xticks(list(range(NP_X, G + 1)))
    ax.set_xticklabels(["no\npersona"] + [str(g) for g in range(0, G + 1)])
    ax.set_xlim(NP_X - 0.3, G + 0.3)
    ax.axvline(NP_X + 0.5, color=st.PALETTE["grid"], lw=1, ls=":", zorder=0)


def fig_difficulty(out_dir: Path) -> str:
    """Two lines: running-best FULL-SET certified difficulty vs generation, both
    starting from the shared no_persona floor."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = leader_track(arm, rows)
        xs = [t["gen"] for t in tr]
        ys = [t["difficulty"] for t in tr]
        if npr:  # connector from shared floor to this arm's seed (no marker)
            ax.plot(
                [NP_X, xs[0]], [npr["difficulty"], ys[0]], color=col, lw=2.4, zorder=2
            )
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  (seed {ys[0]:.2f} → {max(ys):.2f})",
        )
        gmap = _gen_map(arm)
        for r in rows:  # validity-gated certified candidates: hollow X at birth gen
            if _arm_of(r["who"]) == arm and not r["valid"]:
                ax.scatter(
                    gmap[_idx_of(r["who"])],
                    r["difficulty"],
                    marker="X",
                    s=80,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.6,
                    zorder=4,
                )
    if npr:
        ax.scatter(
            NP_X,
            npr["difficulty"],
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("full-set difficulty  (1 − agent reward)")
    ax.set_title("Persona search: full-set certified difficulty over optimization")
    _xaxis_with_floor(ax)
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1]))
    from matplotlib.lines import Line2D

    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=f"no_persona floor ({npr['difficulty']:.2f})"
            if npr
            else "no_persona",
        ),
        Line2D(
            [],
            [],
            marker="X",
            ls="",
            mfc="white",
            mec="0.4",
            mew=1.6,
            label="validity-gated",
        ),
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="running best (full set)",
    )
    return st.save(fig, out_dir / "pair_difficulty_over_optimization.png")


def fig_difficulty_ood(out_dir: Path) -> str:
    """Same running-best track as fig_difficulty (personas still selected by their
    full-set difficulty, exactly as the search ranked them), but the y-value is each
    persona's OOD (held-out) difficulty: 1 − agent reward on the 64 telecom tasks the
    GEPA search never saw (base − reprep). Shows how the optimized personas generalize
    beyond the optimization tasks."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    omap = ood_diff_map(rows)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = leader_track(arm, rows)
        xs = [t["gen"] for t in tr]
        ys = [omap[t["pid"]] for t in tr]
        if npr:  # connector from shared floor to this arm's seed (no marker)
            ax.plot(
                [NP_X, xs[0]], [omap[npr["pid"]], ys[0]], color=col, lw=2.4, zorder=2
            )
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  (seed {ys[0]:.2f} → {max(ys):.2f})",
        )
        gmap = _gen_map(arm)
        for r in rows:  # validity-gated certified candidates: hollow X at birth gen
            if _arm_of(r["who"]) == arm and not r["valid"]:
                ax.scatter(
                    gmap[_idx_of(r["who"])],
                    omap[r["pid"]],
                    marker="X",
                    s=80,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.6,
                    zorder=4,
                )
    if npr:
        ax.scatter(
            NP_X,
            omap[npr["pid"]],
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("OOD difficulty  (1 − agent reward, held-out tasks)")
    ax.set_title("Persona search: held-out (OOD) difficulty over optimization")
    _xaxis_with_floor(ax)
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1]))
    from matplotlib.lines import Line2D

    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=f"no_persona floor ({omap[npr['pid']]:.2f})" if npr else "no_persona",
        ),
        Line2D(
            [],
            [],
            marker="X",
            ls="",
            mfc="white",
            mec="0.4",
            mew=1.6,
            label="validity-gated",
        ),
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="running best (OOD = base − reprep)",
    )
    return st.save(fig, out_dir / "pair_difficulty_over_optimization_ood.png")


def fig_naturalness(out_dir: Path) -> str:
    """Two lines: behMean of the difficulty-leading persona vs generation (left
    axis, solid); P_geom of the same persona (right axis, dashed) as the strict
    naturalness reference. Same persona track as the difficulty figure."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = leader_track(arm, rows)
        xs = [t["gen"] for t in tr]
        beh = [t["behMean"] for t in tr]
        if npr:  # connector from the shared floor (no marker)
            ax.plot(
                [NP_X, xs[0]], [npr["behMean"], beh[0]], color=col, lw=2.4, zorder=2
            )
        ax.plot(
            xs,
            beh,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  ({beh[0]:.0f} → {beh[-1]:.0f})",
        )
    if npr:
        ax.scatter(
            NP_X,
            npr["behMean"],
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("nat score  (D1–D4 Dice overlap, 0–100)")
    ax.set_title(
        "sim2real naturalness over optimization  (hybrid judge, failures-only)"
    )
    _xaxis_with_floor(ax)
    ax.set_ylim(0, 70)
    from matplotlib.lines import Line2D

    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=f"no_persona floor ({npr['behMean']:.0f})" if npr else "no_persona",
        )
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="nat score (naturalness)",
    )
    return st.save(fig, out_dir / "pair_sim2real_naturalness.png")


def _nat_d14(row: dict) -> float:
    """Naturalness restricted to D1 (phrasing/style) + D4 (error reaction/affect),
    dropping D2 (verbosity/front-loading — a backbone-style artifact, not adversarial
    behavior) and D3 (uncertainty/questioning — strongly domain-dependent under the
    LLM judge). The Dk are already per-dimension Dice in the cert json, so the score
    is just their mean."""
    return (row["D1"] + row["D4"]) / 2.0


def fig_naturalness_d14(out_dir: Path) -> str:
    """Same difficulty-leading persona track as fig_naturalness, but nat score =
    mean(D1, D4) only (D2 and D3 excluded)."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = leader_track(arm, rows)
        xs = [t["gen"] for t in tr]
        beh = [_nat_d14(t) for t in tr]
        if npr:  # connector from the shared floor (no marker)
            ax.plot([NP_X, xs[0]], [_nat_d14(npr), beh[0]], color=col, lw=2.4, zorder=2)
        ax.plot(
            xs,
            beh,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  ({beh[0]:.0f} → {beh[-1]:.0f})",
        )
    if npr:
        ax.scatter(
            NP_X,
            _nat_d14(npr),
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("nat score  (D1+D4 Dice overlap, 0–100)")
    ax.set_title(
        "sim2real naturalness over optimization  (D1+D4 only, hybrid, failures-only)"
    )
    _xaxis_with_floor(ax)
    ax.set_ylim(0, 70)
    from matplotlib.lines import Line2D

    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=f"no_persona floor ({_nat_d14(npr):.0f})" if npr else "no_persona",
        )
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="nat score (D1+D4 only)",
    )
    return st.save(fig, out_dir / "pair_sim2real_naturalness_d1d4.png")


# --- D1-only naturalness on FULL sims (strict, rule-based, no API) -------------
# The cert json stores only hybrid/failures-only Dk. D1 is the one dimension where
# the final persona genuinely beats no_persona, and it is almost entirely lexical,
# so we recompute it directly on the FULL simulation set in strict mode.
# D1-core = the discriminative communication-style metrics. We drop the 3 degenerate
# D1 metrics that are constant across all simulators (Short%≈0, Ack%=0, IDConf%=0 →
# zero-variance, no signal) and Wds/trn (double-counted in D2; a verbosity/length
# signal, not a style one). What remains are 4 genuine style signals.
D1_CORE = ["Polite%", "Formal%", "VerbCV", "Repeat%"]


def _d1_full_map(
    rows: list[dict], metrics: list[str] | None = None
) -> dict[str, float]:
    """pid -> D1 Dice-to-human (mean over `metrics`, default = the 8 D1 atomic
    metrics), strict mode, computed on ALL simulations of each persona's cert run."""
    import sys as _sys

    _sys.path.insert(0, str(REPO))
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import numpy as np
    import sim2real_metrics as M
    from sim2real_full_table import DIMS_FULL, dice
    from sim2real_naturalness import feat
    from sim2real_pr import human_turn_lists, load_human, sim_turn_lists

    M.set_mode("strict")
    ks = metrics or DIMS_FULL["D1"]
    rowsH, _ = load_human()
    Hx = feat(human_turn_lists(rowsH), "strict")
    Hm = {k: float(np.nanmean(Hx[:, i])) for i, k in enumerate(M.METRIC_ORDER)}

    out: dict[str, float] = {}
    for r in rows:
        pid = r["pid"]
        path = _results_path(pid)
        if not Path(path).exists():
            continue
        X = feat(sim_turn_lists(str(path)), "strict")
        mv = {k: float(np.nanmean(X[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
        out[pid] = float(np.mean([dice(mv[k], Hm[k]) for k in ks]))
    return out


def fig_naturalness_d1_full(out_dir: Path) -> str:
    """D1-only naturalness (communication style) computed on the FULL simulation set
    in strict mode, along the running-latest track (endpoint = optimization endpoint,
    not the difficulty leader). D1 is the dimension where the optimized persona is
    most human-like."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    d1map = _d1_full_map(rows)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = [t for t in latest_track(arm, rows) if t["pid"] in d1map]
        xs = [t["gen"] for t in tr]
        ys = [d1map[t["pid"]] for t in tr]
        if npr and npr["pid"] in d1map:  # connector from shared floor (no marker)
            ax.plot(
                [NP_X, xs[0]], [d1map[npr["pid"]], ys[0]], color=col, lw=2.4, zorder=2
            )
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  ({ys[0]:.0f} → {ys[-1]:.0f})",
        )
    if npr and npr["pid"] in d1map:
        ax.scatter(
            NP_X,
            d1map[npr["pid"]],
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("nat score  (D1 communication-style Dice, 0–100)")
    ax.set_title(
        "sim2real naturalness over optimization  (D1 only, strict, full sims; endpoint track)"
    )
    _xaxis_with_floor(ax)
    ax.set_ylim(0, 70)
    from matplotlib.lines import Line2D

    flo = (
        f"no_persona floor ({d1map[npr['pid']]:.0f})"
        if npr and npr["pid"] in d1map
        else "no_persona"
    )
    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=flo,
        )
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="nat score (D1 only)",
    )
    return st.save(fig, out_dir / "pair_sim2real_naturalness_d1_full.png")


def fig_naturalness_d1core_full(out_dir: Path) -> str:
    """Phrasing Naturalness (4 discriminative communication-style metrics: Polite%,
    Formal%, VerbCV, Repeat% — register + cadence) on the FULL simulation set, strict
    mode, running-latest (endpoint) track. Drops the 3 zero-variance D1 metrics and the
    D2-duplicate Wds/trn so the score reflects genuine phrasing style; here the
    optimized p0003 ends clearly above the no_persona floor."""
    rows = json.load(open(NAT_FILE))
    npr = np_row(rows)
    d1map = _d1_full_map(rows, metrics=D1_CORE)
    fig, ax = st.new_fig()
    for arm in ARMS:
        col = st.RUN_COLORS[arm]
        tr = [t for t in latest_track(arm, rows) if t["pid"] in d1map]
        xs = [t["gen"] for t in tr]
        ys = [d1map[t["pid"]] for t in tr]
        if npr and npr["pid"] in d1map:
            ax.plot(
                [NP_X, xs[0]], [d1map[npr["pid"]], ys[0]], color=col, lw=2.4, zorder=2
            )
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{arm}  ({ys[0]:.0f} → {ys[-1]:.0f})",
        )
    if npr and npr["pid"] in d1map:
        ax.scatter(
            NP_X,
            d1map[npr["pid"]],
            s=240,
            marker="*",
            zorder=6,
            color=st.PALETTE["neutral"],
            edgecolor="black",
            linewidths=0.7,
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("Phrasing Naturalness  (Dice, 0–100)")
    ax.set_title(
        "Phrasing Naturalness over optimization  (Polite / Formal / VerbCV / Repeat)"
    )
    _xaxis_with_floor(ax)
    ax.set_ylim(0, 80)
    from matplotlib.lines import Line2D

    flo = (
        f"no_persona floor ({d1map[npr['pid']]:.0f})"
        if npr and npr["pid"] in d1map
        else "no_persona"
    )
    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=flo,
        )
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="lower right",
        title="Phrasing Naturalness",
    )
    return st.save(fig, out_dir / "pair_sim2real_naturalness_d1core_full.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data/persona_opt/figs/pair"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(fig_difficulty(out))
    print(fig_difficulty_ood(out))
    print(fig_naturalness(out))
    print(fig_naturalness_d14(out))
    print(fig_naturalness_d1_full(out))
    print(fig_naturalness_d1core_full(out))


if __name__ == "__main__":
    main()
