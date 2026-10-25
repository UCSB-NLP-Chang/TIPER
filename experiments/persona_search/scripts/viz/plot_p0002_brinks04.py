"""Two figures comparing the p0002 (telecom) and brinks_04 lineages, in the same
style as the earlier pair figures:

  fig 1  optimization difficulty over generations (full-set, 1 - agent reward)
  fig 2  sim2real naturalness (behMean, 0-100) over generations

Both arms are scored on the SAME basis so the two lines are comparable:
  - difficulty: full-set (228 sims) 1 - mean(reward==1), recomputed from each
    persona's results.json
  - naturalness: strict mode, scored against the SAME real Brinks human dataset
    (user_simulator_dataset_v2.json), reusing the sim2real_naturalness_brinks
    machinery. A shared no_persona floor anchors both arms.

  .venv/bin/python experiments/persona_search/scripts/viz/plot_p0002_brinks04.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import sys

import _style as st  # noqa: E402

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
from sim2real_naturalness_brinks import brinks_human_turn_lists, score_one  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import feature_matrix, pca_fit, standardizer  # noqa: E402

SIM = REPO / "data/simulations"
PGEPA = REPO / "data/persona_opt/pgepa_runs"
OUT_JSON = PGEPA / "p0002_brinks04_naturalness_strict.json"

# Search-state files: rep-set (35 telecom representative tasks) per-generation
# difficulty. `agg` is the search difficulty (== difficulty on the rep set) for
# feasible candidates; infeasible candidates carry agg 0.0 and are validity-gated.
STATE = {
    "p0002": PGEPA / "pgepa_pair_p0002/pgepa_state.json",
    "brinks_04": PGEPA / "pgepa_brinks_04_confused_drip_feeding/pgepa_state.json",
}

# arm -> ordered (generation, label, results.json) ; gen 0 == seed.
CELLS = {
    "p0002": [
        (
            0,
            True,
            "p0002 seed",
            SIM
            / "20260607_112504_telecom_llm_agent_gpt-5.4_cc_user_con_claude-opus-4-7/results.json",
        ),
        (
            1,
            True,
            "p0002 idx1",
            SIM / "gepa_certify/gepa_certify_c04/telecom/results.json",
        ),
        (
            2,
            True,
            "p0002 idx4",
            SIM / "gepa_certify/gepa_certify_c05/telecom/results.json",
        ),
        (
            3,
            True,
            "p0002 idx6",
            SIM / "gepa_certify/gepa_certify_c06/telecom/results.json",
        ),
        (
            4,
            False,
            "p0002 idx8",
            SIM / "gepa_certify/gepa_certify_c07/telecom/results.json",
        ),
    ],
    "brinks_04": [
        (
            0,
            True,
            "brinks_04 seed",
            SIM
            / "persona_opt/fullset_brinks/fullset_brinks_04_seed/telecom/results.json",
        ),
        (
            3,
            True,
            "brinks_04 optimized",
            SIM
            / "persona_opt/fullset_brinks/fullset_brinks_04_optimized/telecom/results.json",
        ),
    ],
}
NO_PERSONA = SIM / "gepa_certify/progcert_no_persona_fullbase228/telecom/results.json"

ARM_COLOR = {"p0002": st.RUN_COLORS["p0002"], "brinks_04": "#e36414"}
ARM_LABEL = {"p0002": "p0002 (telecom)", "brinks_04": "brinks_04"}
G = 5  # x-axis upper bound
NP_X = -1  # shared no_persona floor position


def compute() -> dict:
    """Score every persona + no_persona against the Brinks human reference (strict)
    and cache to OUT_JSON. Returns the in-memory dict."""
    M.set_mode("strict")
    h_turns = brinks_human_turn_lists()
    Hx_raw = feature_matrix(h_turns)
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"

    def sc(label, path):
        r = score_one(label, path, Hx_raw, pooled, mu, sd, pca, pca_name, 3)
        return r

    out = {"no_persona": sc("no_persona", NO_PERSONA), "arms": {}}
    for arm, cells in CELLS.items():
        rows = []
        for gen, valid, label, path in cells:
            if not path.exists():
                print(f"  [skip] {label}: missing {path}")
                continue
            r = sc(label, path)
            r["gen"], r["valid"] = gen, valid
            rows.append(r)
            print(
                f"  {label:<22} gen{gen} valid={valid} "
                f"diff={r['difficulty']:.3f} behMean={r['behMean']:.1f}"
            )
        out["arms"][arm] = rows
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"wrote {OUT_JSON}")
    return out


def _xaxis(ax):
    ax.set_xticks(list(range(NP_X, G)))
    ax.set_xticklabels(["no\npersona"] + [str(g) for g in range(0, G)])
    ax.set_xlim(NP_X - 0.3, G - 1 + 0.3)
    ax.axvline(NP_X + 0.5, color=st.PALETTE["grid"], lw=1, ls=":", zorder=0)


def rep_difficulty_track(arm: str):
    """Per generation, the running-best rep-set difficulty among FEASIBLE candidates
    (the search's own convergence curve). Returns (xs, ys, last_gen) plus the list of
    validity-gated (infeasible) candidates as (gen, _) so they can be flagged."""
    d = json.loads(STATE[arm].read_text())
    cands = d["candidates"]
    gens = sorted({c["gen"] for c in cands})
    feas = [
        (c["gen"], c["agg"])
        for c in cands
        if c.get("feasible") and c.get("agg") is not None
    ]
    gated = [c["gen"] for c in cands if not c.get("feasible")]
    xs, ys, best = [], [], None
    for g in range(min(gens), max(gens) + 1):
        seen = [a for (gg, a) in feas if gg <= g]
        if seen:
            best = max(seen)
        if best is not None:
            xs.append(g)
            ys.append(best)
    return xs, ys, gated


def fig_rep_difficulty(out_path: Path) -> str:
    """Difficulty over optimization on the REP SET (35 telecom representative tasks =
    the search valset), so every generation has a point. Running-best feasible
    difficulty per generation; infeasible (validity-gated) candidates are not plotted
    (their search difficulty is set to 0)."""
    fig, ax = st.new_fig()
    gmax = 0
    for arm in CELLS:  # keep arm order/colors consistent with the naturalness fig
        col = ARM_COLOR[arm]
        xs, ys, gated = rep_difficulty_track(arm)
        gmax = max(gmax, xs[-1])
        n_gated = sum(1 for g in gated)
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{ARM_LABEL[arm]}  ({ys[0]:.2f} → {ys[-1]:.2f}"
            + (f", {n_gated} gated)" if n_gated else ")"),
        )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel("rep-set difficulty  (1 − agent reward, 35 tasks)")
    ax.set_title("Persona search: rep-set difficulty over optimization")
    ax.set_xticks(list(range(0, gmax + 1)))
    ax.set_xlim(-0.3, gmax + 0.3)
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    ax.legend(loc="upper left", title="lineage (running best)")
    return st.save(fig, out_path)


def _draw(data, key, ylabel, title, out_path, ymax):
    from matplotlib.lines import Line2D

    npr = data["no_persona"]
    fig, ax = st.new_fig()
    for arm, rows in data["arms"].items():
        col = ARM_COLOR[arm]
        valid = [r for r in rows if r["valid"]]
        xs = [r["gen"] for r in valid]
        ys = [r[key] for r in valid]
        # connector from the shared no_persona floor to this arm's seed (no marker)
        ax.plot([NP_X, xs[0]], [npr[key], ys[0]], color=col, lw=2.4, zorder=2)
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=2.4,
            ms=8,
            color=col,
            zorder=3,
            label=f"{ARM_LABEL[arm]}  ({ys[0]:.2f} → {ys[-1]:.2f})"
            if ymax <= 1
            else f"{ARM_LABEL[arm]}  ({ys[0]:.0f} → {ys[-1]:.0f})",
        )
        for r in rows:  # validity-gated candidates: hollow X at birth gen
            if not r["valid"]:
                ax.scatter(
                    r["gen"],
                    r[key],
                    marker="X",
                    s=85,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.7,
                    zorder=4,
                )
    ax.scatter(
        NP_X,
        npr[key],
        s=240,
        marker="*",
        zorder=6,
        color=st.PALETTE["neutral"],
        edgecolor="black",
        linewidths=0.7,
    )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _xaxis(ax)
    ax.set_ylim(0, ymax)
    floor = f"{npr[key]:.2f}" if ymax <= 1 else f"{npr[key]:.0f}"
    extra = [
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc=st.PALETTE["neutral"],
            mec="black",
            ms=13,
            label=f"no_persona floor ({floor})",
        ),
        Line2D(
            [],
            [],
            marker="X",
            ls="",
            mfc="white",
            mec="0.4",
            mew=1.7,
            label="validity-gated",
        ),
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + extra,
        loc="upper left",
        title="lineage",
    )
    return st.save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default=str(REPO / "data/persona_opt/figs/p0002_brinks04")
    )
    ap.add_argument("--cache", action="store_true", help="reuse OUT_JSON if present")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.cache and OUT_JSON.exists():
        data = json.load(open(OUT_JSON))
    else:
        data = compute()

    f1 = fig_rep_difficulty(out / "p0002_brinks04_difficulty.png")
    f2 = _draw(
        data,
        "behMean",
        "naturalness  (sim2real behMean, 0–100)",
        "sim2real naturalness over optimization  (strict, Brinks human ref)",
        out / "p0002_brinks04_naturalness.png",
        ymax=70,
    )
    print(f1)
    print(f2)


if __name__ == "__main__":
    main()
