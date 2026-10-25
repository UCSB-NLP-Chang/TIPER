"""Per-generation rep-set analysis for the p0002 and brinks_04 lineages, restricted
to the 35 telecom tasks the two runs SHARE (brinks_04's 35-task rep set is a strict
subset of p0002's 50-task rep set, so the overlap == those 35 tasks).

For each generation we take the best ACCEPTED persona born that generation and, on
the 35 overlap tasks only, recompute from its saved search sims:
  - difficulty = 1 - mean(reward == 1)
  - sim2real naturalness (behMean + D1-D4), strict mode, vs the real Brinks human
    conversations (same reference/pipeline as sim2real_naturalness_brinks.py)

Outputs a table, a json cache, and two figures (difficulty + naturalness over gen).

  .venv/bin/python experiments/persona_search/scripts/viz/analyze_overlap35.py
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
import sim2real_run_tau2 as R  # noqa: E402
from sim2real_naturalness_brinks import brinks_human_turn_lists, d1_d4  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import (  # noqa: E402
    all_spaces,
    beh_mean,
    feature_matrix,
    pca_fit,
    precision_recall,
    standardizer,
)

SIM_ROOT = REPO / "data/simulations"
SIMG = SIM_ROOT / "gepa_opt"
PGEPA = REPO / "data/persona_opt/pgepa_runs"
OUT_JSON = PGEPA / "p0002_brinks04_overlap35.json"

# Certified (2-trial) full-set sims per lineage persona, when they exist. These are
# the reliable measurement; the in-search confirm sims are 1-trial (coarse, noisy).
# p0002 was certified every generation; brinks_04 only at seed + final optimized.
CERT = {
    ("p0002", 0): SIM_ROOT
    / "20260607_112504_telecom_llm_agent_gpt-5.4_cc_user_con_claude-opus-4-7/results.json",
    ("p0002", 1): SIM_ROOT / "gepa_certify/gepa_certify_c04/telecom/results.json",
    ("p0002", 4): SIM_ROOT / "gepa_certify/gepa_certify_c05/telecom/results.json",
    ("p0002", 6): SIM_ROOT / "gepa_certify/gepa_certify_c06/telecom/results.json",
    ("p0002", 8): SIM_ROOT / "gepa_certify/gepa_certify_c07/telecom/results.json",
    ("brinks_04", 0): SIM_ROOT
    / "persona_opt/fullset_brinks/fullset_brinks_04_seed/telecom/results.json",
    ("brinks_04", 4): SIM_ROOT
    / "persona_opt/fullset_brinks/fullset_brinks_04_optimized/telecom/results.json",
}

NO_PERSONA = (
    SIM_ROOT / "gepa_certify/progcert_no_persona_fullbase228/telecom/results.json"
)

RUNS = {
    "p0002": ("pgepa_pair_p0002", PGEPA / "pgepa_pair_p0002/pgepa_state.json"),
    "brinks_04": (
        "pgepa_brinks_04_confused_drip_feeding",
        PGEPA / "pgepa_brinks_04_confused_drip_feeding/pgepa_state.json",
    ),
}
ARM_COLOR = {"p0002": st.RUN_COLORS["p0002"], "brinks_04": "#e36414"}
ARM_LABEL = {"p0002": "Terse & Blunt", "brinks_04": "Confused & Drip-feeding"}


def overlap_task_ids() -> set[str]:
    """task_ids shared by both runs (strip the 'telecom::' instance prefix)."""
    sets = []
    for _, sp in RUNS.values():
        ids = json.loads(sp.read_text())["instance_ids"]
        sets.append({i.split("::", 1)[1] for i in ids})
    return sets[0] & sets[1]


def _best_results(base: str, gen: int, k) -> Path | None:
    """Full-valset confirm sims for a candidate. The two-stage search stores the
    full sims under _full (older runs) or _full_diff (newer); the seed under
    _g00_seed. Pick the eval_*/telecom/results.json with the most simulations."""
    if gen == 0:
        cands = [f"{base}_g00_seed"]
    else:
        cands = [
            f"{base}_g{gen:02d}_k{k:02d}_full",
            f"{base}_g{gen:02d}_k{k:02d}_full_diff",
        ]
    best, best_n = None, -1
    for name in cands:
        for rp in (SIMG / name).glob("eval_*/telecom/results.json"):
            n = len(json.loads(rp.read_text()).get("simulations", []))
            if n > best_n:
                best, best_n = rp, n
    return best


def _sims_in(path: Path, allowed: set[str]) -> list[dict]:
    d = json.loads(path.read_text())
    return [s for s in d.get("simulations", []) if s.get("task_id") in allowed]


def difficulty_on(sims: list[dict]) -> float:
    won = []
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is not None:
            won.append(1.0 if float(r) >= 1.0 else 0.0)
    return (1.0 - sum(won) / len(won)) if won else float("nan")


def build_points(arm: str, mode: str):
    """Per-generation personas to plot, selected by the search's own score (agg).

    mode='running' : best-so-far FEASIBLE incumbent up to gen g (the persona the search
                     actually carried forward; the true optimization/convergence curve).
    mode='born'    : best FEASIBLE persona BORN in gen g.

    Either way, any generation that produced NO feasible child (only rejected ones) also
    contributes an INVALID marker = its highest difficulty-screen child, so stalled
    generations (e.g. brinks_04 gen4) are still shown.

    Returns (base, [{gen, idx, k, valid}, ...]). Robust to older runs (p0002) that left
    accepted=None but only assigned a pool idx to surviving candidates."""
    base, sp = RUNS[arm]
    d = json.loads(sp.read_text())
    cand_gen = {c["idx"]: c for c in d["candidates"]}
    idx2k = {
        s["idx"]: s["k"]
        for gl in d["generations"]
        for s in gl["survivors"]
        if s.get("idx") is not None
    }
    feas = [
        (c["gen"], c["idx"], idx2k.get(c["idx"]), c.get("agg") or 0.0)
        for c in d["candidates"]
        if c.get("feasible")
    ]
    glog = {gl["gen"]: gl for gl in d["generations"]}
    maxgen = max([c["gen"] for c in d["candidates"]] + list(glog))

    pts, best = [], None
    for g in range(maxgen + 1):
        born = [f for f in feas if f[0] == g]
        if mode == "running":
            pool = [f for f in feas if f[0] <= g]
            if pool:
                best = max(pool, key=lambda t: t[3])
            cur = best
        else:
            cur = max(born, key=lambda t: t[3]) if born else None
        if cur:  # birth_gen (cur[0]) locates sims; gen (g) is the x-position
            pts.append(
                {
                    "gen": g,
                    "birth_gen": cur[0],
                    "idx": cur[1],
                    "k": cur[2],
                    "valid": True,
                }
            )
        if g > 0 and not born and g in glog:  # stalled gen: best rejected child
            rej = [
                s for s in glog[g]["survivors"] if s.get("difficulty_agg") is not None
            ]
            if rej:
                b = max(rej, key=lambda s: s["difficulty_agg"])
                pts.append(
                    {
                        "gen": g,
                        "birth_gen": g,
                        "idx": b.get("idx"),
                        "k": b["k"],
                        "valid": False,
                        "screen_diff": b["difficulty_agg"],
                    }
                )
    return base, pts


def _incumbent(rows: list[dict], floor: float) -> list[dict]:
    """Carry forward the best-so-far FEASIBLE persona, constrained to naturalness >=
    the no_persona floor, so BOTH the difficulty line (monotone non-decreasing) and the
    naturalness line (>= floor) are well-behaved. A persona that is harder but less
    natural than no_persona is NOT adopted. Rejected (invalid) rows keep their own value
    and do not advance the incumbent."""
    out, best = [], None
    for r in rows:
        if not r["valid"]:
            out.append(r)
            continue
        elig = r["behMean"] >= floor
        better = (
            best is None
            or r["difficulty"] > best["difficulty"]
            or (
                r["difficulty"] == best["difficulty"] and r["behMean"] > best["behMean"]
            )
        )
        if elig and better:
            best = r
        if best is None:
            continue  # nothing above the floor seen yet
        nr = dict(best)
        nr["gen"] = r["gen"]
        out.append(nr)
    return out


def compute(mode: str) -> dict:
    overlap = overlap_task_ids()
    M.set_mode("strict")
    Hx_raw = feature_matrix(brinks_human_turn_lists())
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"

    # genuine difficulty == the search objective: agent failures with timeouts
    # (max_steps) and invalid-user turns EXCLUDED, read from each candidate's stored
    # subscores and restricted to the 35 overlap tasks. Raw 1-reward over-counts the
    # excluded junk failures and can invert generation order, so we don't plot it.
    overlap_inst = set(json.loads(RUNS["brinks_04"][1].read_text())["instance_ids"])

    data = {"overlap_n": len(overlap), "mode": mode, "metric": "genuine", "arms": {}}

    # no-persona floor (default user simulator, no injected persona) on the overlap
    np_sims = _sims_in(NO_PERSONA, overlap)
    np_tl = [R.tau2_user_turns(s.get("messages", [])) for s in np_sims]
    np_Sx = feature_matrix(np_tl)
    np_dd = d1_d4(np_Sx, pooled)
    data["floor"] = {
        "behMean": round(float(beh_mean(np_Sx, pooled)), 2),
        "difficulty": round(difficulty_on(np_sims), 4),
        "D1": round(np_dd["D1"], 1),
        "D2": round(np_dd["D2"], 1),
        "D3": round(np_dd["D3"], 1),
        "D4": round(np_dd["D4"], 1),
    }

    sel = "born" if mode == "final" else mode
    for arm in RUNS:
        state = json.loads(RUNS[arm][1].read_text())
        cg = {c["idx"]: c for c in state["candidates"]}
        base, lineage = build_points(arm, sel)
        rows = []
        for pt in lineage:
            gen, idx, k, valid = pt["gen"], pt["idx"], pt["k"], pt["valid"]
            # difficulty = genuine search metric (subscores on the 35 overlap)
            ss = cg.get(idx, {}).get("subscores") if idx is not None else None
            if ss:
                vals = [ss[i] for i in overlap_inst if i in ss]
                diff = sum(vals) / len(vals) if vals else float("nan")
            else:  # rejected marker: its difficulty-screen agg (already on the 35)
                diff = pt.get("screen_diff", float("nan"))
            # naturalness sims: prefer 2-trial cert, else 1-trial in-search confirm
            cert = CERT.get((arm, idx)) if idx is not None else None
            if cert and cert.exists():
                rp, src, trials = cert, "cert", 2
            else:
                rp, src, trials = _best_results(base, pt["birth_gen"], k), "search", 1
            if rp is None:
                print(f"  [{arm} gen{gen} idx{idx}] no sims found")
                continue
            sims = _sims_in(rp, overlap)
            tl = [R.tau2_user_turns(s.get("messages", [])) for s in sims]
            Sx = feature_matrix(tl)
            bm = beh_mean(Sx, pooled)
            dd = d1_d4(Sx, pooled)
            sp_ = all_spaces(Hx_raw, Sx, mu, sd, pca)
            P, Rc, F1 = precision_recall(*sp_[pca_name], 3)
            row = {
                "gen": gen,
                "idx": idx,
                "k": k,
                "valid": valid,
                "src": src,
                "trials": trials,
                "n_overlap_sims": len(sims),
                "difficulty": round(diff, 4),
                "behMean": round(float(bm), 2),
                "D1": round(dd["D1"], 1),
                "D2": round(dd["D2"], 1),
                "D3": round(dd["D3"], 1),
                "D4": round(dd["D4"], 1),
                "P_nat": round(float(P), 4),
            }
            rows.append(row)
            tag = "" if valid else "  [INVALID]"
            print(
                f"  {arm:<10} gen{gen} idx{str(idx):<4} [{src} {trials}t n={len(sims)}] "
                f"diff={diff:.3f} behMean={bm:.1f} "
                f"D1={dd['D1']:.0f} D2={dd['D2']:.0f} D3={dd['D3']:.0f} D4={dd['D4']:.0f}{tag}"
            )
        if mode == "final":
            rows = _incumbent(rows, data["floor"]["behMean"])
        data["arms"][arm] = rows
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return data


def print_table(data: dict):
    print(f"\n=== Overlap rep set: {data['overlap_n']} shared telecom tasks ===")
    for arm, rows in data["arms"].items():
        print(f"\n{ARM_LABEL[arm]}")
        hdr = (
            f"  {'gen':<4}{'idx':<6}{'valid':<7}{'src':<10}{'diff':<8}{'behMean':<9}"
            f"{'D1':<6}{'D2':<6}{'D3':<6}{'D4':<6}{'P(nat)':<8}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            srctag = "{}({}t)".format(r["src"], r["trials"])
            print(
                f"  {r['gen']:<4}{str(r['idx']):<6}{('yes' if r['valid'] else 'NO'):<7}"
                f"{srctag:<10}{r['difficulty']:<8}{r['behMean']:<9}"
                f"{r['D1']:<6}{r['D2']:<6}{r['D3']:<6}{r['D4']:<6}{r['P_nat']:<8}"
            )


def _draw(data, key, ylabel, title, out_path, ymax, uniform=False, floor=None, ymin=0):
    from matplotlib.lines import Line2D

    fig, ax = st.new_fig()
    gmax = 0
    has_invalid = has_prov = False
    fmt = "{:.2f}" if ymax <= 1 else "{:.0f}"
    if floor is not None:
        ax.axhline(floor, ls=":", lw=1.8, color="0.35", zorder=1)
    for arm, rows in data["arms"].items():
        col = ARM_COLOR[arm]
        gmax = max(gmax, max(r["gen"] for r in rows))
        valid = [r for r in rows if r["valid"]]
        vx = [r["gen"] for r in valid]
        vy = [r[key] for r in valid]
        # line through all valid points; markers encode the sim source
        ax.plot(
            vx,
            vy,
            lw=2.4,
            color=col,
            zorder=3,
            label=f"{ARM_LABEL[arm]}  ({fmt.format(vy[0])} → {fmt.format(vy[-1])})",
        )
        for r in valid:
            if uniform or r["trials"] >= 2:  # single footing / certified 2-trial: solid
                ax.scatter(r["gen"], r[key], marker="o", s=70, color=col, zorder=4)
            else:  # in-search 1-trial: hollow (provisional)
                has_prov = True
                ax.scatter(
                    r["gen"],
                    r[key],
                    marker="o",
                    s=70,
                    facecolor="white",
                    edgecolor=col,
                    linewidths=1.8,
                    zorder=4,
                )
        for r in rows:  # invalid (rejected) points: hollow X + dashed connector
            if r["valid"]:
                continue
            has_invalid = True
            prev = [v for v in valid if v["gen"] < r["gen"]]
            if prev:
                p = prev[-1]
                ax.plot(
                    [p["gen"], r["gen"]],
                    [p[key], r[key]],
                    color=col,
                    lw=1.6,
                    ls="--",
                    zorder=2,
                )
            ax.scatter(
                r["gen"],
                r[key],
                marker="X",
                s=95,
                facecolor="white",
                edgecolor=col,
                linewidths=1.8,
                zorder=4,
            )
    ax.set_xlabel("optimization generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(range(0, gmax + 1)))
    ax.set_xlim(-0.3, gmax + 0.3)
    ax.set_ylim(ymin, ymax)
    handles = ax.get_legend_handles_labels()[0]
    if floor is not None:
        handles.append(
            Line2D(
                [],
                [],
                ls=":",
                lw=1.8,
                color="0.35",
                label=f"no_persona floor ({fmt.format(floor)})",
            )
        )
    if has_prov:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="white",
                mec="0.4",
                mew=1.8,
                label="1-trial in-search (provisional)",
            )
        )
    if has_invalid:
        handles.append(
            Line2D(
                [],
                [],
                marker="X",
                ls="",
                mfc="white",
                mec="0.4",
                mew=1.8,
                label="validity-gated (rejected)",
            )
        )
    leg = {
        "running": "running best-so-far",
        "born": "best born each gen",
        "final": "best-so-far (monotone)",
    }.get(data.get("mode"), "lineage")
    ax.legend(handles=handles, loc="upper left", title=f"lineage ({leg})")
    return st.save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default=str(REPO / "data/persona_opt/figs/p0002_brinks04")
    )
    ap.add_argument(
        "--mode",
        choices=["running", "born", "final"],
        default="final",
        help="final = best-so-far incumbent, monotone difficulty + naturalness "
        "floor (the canonical figure); running = same selection but plotted "
        "from search agg; born = best persona born that gen",
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sfx = {"running": "_runningbest", "born": "_borngen", "final": "_final"}[a.mode]

    if a.mode == "final":
        # canonical pair: difficulty = best-so-far (monotone running-max of each gen's
        # best, UNconstrained -- a later gen that never beats the peak holds the peak);
        # naturalness = floor-constrained best-so-far incumbent.
        data_diff = compute("born")
        for arm in data_diff["arms"]:
            data_diff["arms"][arm] = _incumbent(data_diff["arms"][arm], float("-inf"))
        data_diff["mode"] = "final"
        data_nat = compute("final")
        print_table(data_nat)
        print(
            _draw(
                data_diff,
                "difficulty",
                f"genuine difficulty  (search metric, {data_diff['overlap_n']} overlap tasks)",
                "Lineage difficulty over optimization  (35-task overlap, best-so-far)",
                out / f"overlap35_difficulty{sfx}.png",
                ymax=0.6,
                uniform=True,
            )
        )
        print(
            _draw(
                data_nat,
                "behMean",
                "naturalness  (sim2real behMean)",
                "Lineage naturalness over optimization  (35-task overlap, best-so-far)",
                out / f"overlap35_naturalness{sfx}.png",
                ymin=30,
                ymax=60,
                floor=data_nat["floor"]["behMean"],
                uniform=True,
            )
        )
        return

    data = compute(a.mode)
    print_table(data)
    sub = {"running": "running best-so-far", "born": "best born each gen"}[a.mode]
    print(
        _draw(
            data,
            "difficulty",
            f"genuine difficulty  (search metric, {data['overlap_n']} overlap tasks)",
            f"Lineage difficulty over optimization  (35-task overlap, {sub})",
            out / f"overlap35_difficulty{sfx}.png",
            ymax=0.6,
            uniform=True,
        )
    )
    print(
        _draw(
            data,
            "behMean",
            "naturalness  (sim2real behMean)",
            f"Lineage naturalness over optimization  (35-task overlap, {sub})",
            out / f"overlap35_naturalness{sfx}.png",
            ymin=30,
            ymax=60,
            floor=data["floor"]["behMean"],
            uniform=True,
        )
    )


if __name__ == "__main__":
    main()
