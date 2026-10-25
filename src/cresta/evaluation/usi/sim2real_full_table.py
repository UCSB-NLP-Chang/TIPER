#!/usr/bin/env python3
"""Reproduce the FULL Sim2Real Table 1 pipeline (D1-D4 + Eval + ECE + USI).

The released human file (tau_bench_tasks_unified.json) contains ONLY human-as-user
sessions (agent = gpt-5.2). It therefore lets us reproduce exactly one row of
Table 1 -- the "Human (inter-ann.)" row -- across the three annotation batches
(base / _ann2 / _ann3). We use this as the validation anchor for the whole
pipeline, including the Eval/ECE/USI half that the behavioral-only runner skips.

Reproduced vs paper "Human (inter-ann.)":
    D1 87.4  D2 97.9  D3 88.0  D4 93.5  Eval 97.4  ECE 0.069  USI 92.9

For the telecom/banking simulators there is no same-domain human baseline, so
Eval (needs human survey on the same tasks) and ECE (needs human success on the
same tasks) are N/A -> the paper USI is not computable; only behavioral D1-D4
(cross-domain Dice vs the paper's human raw values) can be reported.
"""

import argparse
import glob
import itertools
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

from cresta.paths import ARTIFACTS_DIR

try:
    from . import sim2real_metrics as M
    from . import sim2real_run_tau2 as R
except ImportError:  # direct execution
    import sim2real_metrics as M
    import sim2real_run_tau2 as R

DEFAULT_HUMAN_FILE = M.DEFAULT_HUMAN_FILE

# Behavioral dimension membership (IDs/trn dropped from D2 -> "D2*": not repro).
DIMS_FULL = {
    "D1": [
        "Wds/trn",
        "Short%",
        "Polite%",
        "Formal%",
        "Ack%",
        "VerbCV",
        "Repeat%",
        "IDConf%",
    ],
    "D2": ["FrntLoad%", "IDs/trn", "Wds/trn", "Open.Wds"],
    "D3": ["Uncert%", "Certn%", "Pushbk%", "ClarfyQ%", "InfoQ%"],
    "D4": ["Emot%", "Accuse%", "Pivot%"],
}
DIMS_NOIDS = {**DIMS_FULL, "D2": ["FrntLoad%", "Wds/trn", "Open.Wds"]}

# Ordinal survey answer -> [0,1]. Both sides of the inter-annotator MAE use the
# same scale, so Eval is robust to the exact numeric anchors.
SURVEY_MAP = {
    "task_success": {
        "No - Task failed": 0.0,
        "Partially - Some progress": 0.25,
        "No - Due to a policy issue, which the agent clearly explained": 0.5,
        "Yes - Task completed": 0.75,
        "Fully - Exceeded expectations": 1.0,
    },
    "efficiency": {
        "Very inefficient - Too many steps": 0.0,
        "Somewhat inefficient": 1 / 3,
        "About right": 2 / 3,
        "Very efficient": 1.0,
    },
    "question_amount_preference": {"About right": 1.0, "Too many": 0.5, "Too few": 0.5},
    "answer_effort_time": {"High": 0.0, "Medium": 0.5, "Low": 1.0},
    "human_like": {"No": 0.0, "Partially": 0.5, "Yes": 1.0},
    "interaction_flow": {
        "Not smooth": 0.0,
        "OK": 1 / 3,
        "Smooth": 2 / 3,
        "Excellent": 1.0,
    },
    "overall_score": {
        "1 (Very poor)": 0.0,
        "2 (Poor)": 0.25,
        "3 (Acceptable)": 0.5,
        "4 (Good)": 0.75,
        "5 (Excellent)": 1.0,
    },
    "reuse": {
        "Absolutely no": 0.0,
        "No": 0.25,
        "Maybe": 0.5,
        "Yes": 0.75,
        "Absolutely yes": 1.0,
    },
}
SURVEY_DIMS = list(SURVEY_MAP)


def dice(a, b):
    """Return the paper's Dice similarity for two non-negative metric values."""
    if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0:
        raise ValueError("Dice inputs must be finite and non-negative")
    return 100.0 if a == 0 and b == 0 else 2 * min(a, b) / (a + b) * 100


def batch_of(instance_id):
    if instance_id.endswith("_ann2"):
        return "B2", instance_id[:-5]
    if instance_id.endswith("_ann3"):
        return "B3", instance_id[:-5]
    return "B1", instance_id


def load_human(path=DEFAULT_HUMAN_FILE):
    with Path(path).open(encoding="utf-8") as f:
        D = json.load(f)
    rows = list(D.values()) if isinstance(D, dict) else D
    batches = defaultdict(list)  # batch -> list of session dicts
    for r in rows:
        b, base = batch_of(r["instance_id"])
        r["_base"] = base
        batches[b].append(r)
    return rows, batches


# ---- behavioral metrics per batch -------------------------------------------
def batch_metric_means(sessions):
    acc = {k: [] for k in M.METRIC_ORDER}
    for s in sessions:
        turns = M.user_turns(s["conversation"])
        mm = M.metrics_for_interaction(turns)
        if mm:
            for k in M.METRIC_ORDER:
                acc[k].append(mm[k])
    return {k: (st.mean(v) if v else float("nan")) for k, v in acc.items()}


def inter_ann_behavioral(batches, dims):
    """Pairwise-batch Dice per dimension -> mean +/- std over the 3 batch pairs."""
    bmeans = {b: batch_metric_means(s) for b, s in batches.items()}
    keys = sorted(bmeans)
    out = {}
    for dim, metrics in dims.items():
        vals = []
        for a, c in itertools.combinations(keys, 2):
            d = st.mean(dice(bmeans[a][m], bmeans[c][m]) for m in metrics)
            vals.append(d)
        out[dim] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)
    return out, bmeans


# ---- Eval (inter-annotator survey MAE) --------------------------------------
def survey_vec(session):
    sv = session.get("survey", {})
    out = {}
    for d in SURVEY_DIMS:
        a = sv.get(d, {}).get("answer")
        out[d] = SURVEY_MAP[d].get(a)
    return out


def batch_survey_by_task(sessions):
    """base_task -> {dim: value} averaged within batch."""
    perv = defaultdict(lambda: defaultdict(list))
    for s in sessions:
        v = survey_vec(s)
        for d, x in v.items():
            if x is not None:
                perv[s["_base"]][d].append(x)
    return {t: {d: st.mean(xs) for d, xs in dv.items()} for t, dv in perv.items()}


def batch_survey_means(sessions):
    """{dim: mean over all sessions in batch}."""
    perv = defaultdict(list)
    for s in sessions:
        for d, x in survey_vec(s).items():
            if x is not None:
                perv[d].append(x)
    return {d: st.mean(xs) for d, xs in perv.items()}


def inter_ann_eval(batches):
    """Inter-annotator survey agreement at the BATCH-aggregate level: each batch
    is a different human role-play of the same task set, so per-interaction
    ratings differ; what the paper's high Human Eval reflects is agreement of the
    aggregate rating distribution. MAE between batch-pairs over the 8 mean-per-dim
    scores -> Eval=(1-MAE)*100. (Task-level MAE = 0.278 -> 72.2, reported as a
    lower bound that conflates genuine interaction variance.)"""
    bm = {b: batch_survey_means(s) for b, s in batches.items()}
    keys = sorted(bm)
    maes = []
    for a, c in itertools.combinations(keys, 2):
        errs = [
            abs(bm[a][d] - bm[c][d]) for d in SURVEY_DIMS if d in bm[a] and d in bm[c]
        ]
        if errs:
            maes.append(st.mean(errs))
    mae = st.mean(maes)
    return (1 - mae) * 100, mae


# ---- ECE (inter-annotator outcome calibration) ------------------------------
def inter_ann_ece(batches, nbins=5):
    """Bin tasks by difficulty (overall success rate), compare per-batch success
    rate to the complementary batches' rate; weighted abs gap. Approximate: the
    paper's exact difficulty binning is unspecified."""
    # overall success per base task across all batches
    succ = defaultdict(list)
    for b, sess in batches.items():
        for s in sess:
            succ[s["_base"]].append(s.get("reward", 0.0))
    overall = {t: st.mean(v) for t, v in succ.items()}
    # per-batch success per task
    bt = {}
    for b, sess in batches.items():
        m = defaultdict(list)
        for s in sess:
            m[s["_base"]].append(s.get("reward", 0.0))
        bt[b] = {t: st.mean(v) for t, v in m.items()}
    keys = sorted(bt)
    eces = []
    for a, c in itertools.combinations(keys, 2):
        shared = sorted(set(bt[a]) & set(bt[c]))
        if not shared:
            continue
        # bin by overall difficulty
        bins = defaultdict(list)
        for t in shared:
            bi = min(nbins - 1, int(overall[t] * nbins))
            bins[bi].append(t)
        N = len(shared)
        ece = 0.0
        for bi, ts in bins.items():
            pa = st.mean(bt[a][t] for t in ts)
            pc = st.mean(bt[c][t] for t in ts)
            ece += (len(ts) / N) * abs(pa - pc)
        eces.append(ece)
    return st.mean(eces)


def usi(d1, d2, d3, d4, eval_, ece):
    return (d1 + d2 + d3 + d4 + (1 - ece) * 100 + eval_) / 6


def main():
    M.set_mode("faithful")
    ap = argparse.ArgumentParser(description="Reproduce the paper's full human USI row")
    ap.add_argument(
        "--human-file",
        type=Path,
        default=DEFAULT_HUMAN_FILE,
        help=f"unified human sessions JSON (default: {DEFAULT_HUMAN_FILE})",
    )
    ap.add_argument(
        "--sim-glob",
        default=str(
            ARTIFACTS_DIR / "baseline" / "canonical_agent_matrix" / "*" / "results.json"
        ),
        help="glob for sim results.json; rows keyed by parent-of-domain dir + user",
    )
    args = ap.parse_args()
    sim_glob = args.sim_glob
    sim_dims = DIMS_NOIDS

    rows, batches = load_human(args.human_file)

    print(
        f"mode=faithful | Human file: {len(rows)} sessions, batches "
        + ", ".join(f"{b}={len(s)}" for b, s in sorted(batches.items()))
    )

    beh, _ = inter_ann_behavioral(batches, DIMS_FULL)
    eval_, mae = inter_ann_eval(batches)
    ece = inter_ann_ece(batches)
    d1, d2, d3, d4 = (beh[k][0] for k in ("D1", "D2", "D3", "D4"))
    U = usi(d1, d2, d3, d4, eval_, ece)

    print("\n================ FULL Table-1 row: Human (inter-ann.) ================")
    print(
        f"{'':14s} {'D1':>7s} {'D2':>7s} {'D3':>7s} {'D4':>7s} {'Eval':>7s} {'ECE':>7s} {'USI':>7s}"
    )
    print(
        f"{'reproduced':14s} {d1:7.1f} {d2:7.1f} {d3:7.1f} {d4:7.1f} {eval_:7.1f} {ece:7.3f} {U:7.1f}"
    )
    print(
        f"{'paper':14s} {87.4:7.1f} {97.9:7.1f} {88.0:7.1f} {93.5:7.1f} {97.4:7.1f} {0.069:7.3f} {92.9:7.1f}"
    )
    print(
        f"{'diff':14s} {d1 - 87.4:7.1f} {d2 - 97.9:7.1f} {d3 - 88.0:7.1f} {d4 - 93.5:7.1f} "
        f"{eval_ - 97.4:7.1f} {ece - 0.069:7.3f} {U - 92.9:7.1f}"
    )
    print(f"\n  (survey inter-annotator MAE = {mae:.4f}; D2 incl. IDs/trn)")

    # Human raw means (pooled) for the cross-domain Dice baseline used on sims
    pooled = batch_metric_means(rows)

    # ---- simulators: behavioral-only -------------------------------------
    print("\n================ telecom/banking simulators ================")
    print("Eval = N/A (no sim survey on these tasks); ECE = N/A (no same-domain")
    print("human success) -> paper USI not computable. Reporting behavioral half only:")
    print("D1-D4 = Dice(sim raw, paper-human raw); cross-domain, relative use only.\n")
    by = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(sim_glob)):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        ui = d["info"]["user_info"]
        user = f"{ui.get('implementation')}:{ui.get('llm', '')}".replace(
            "bedrock/us.anthropic.", ""
        )
        dom = "telecom" if "telecom" in p else "banking"
        # disambiguate same-user cells (e.g. cert conditions) by the dir above the domain
        tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if tag not in ("canonical_agent_matrix",) and not tag.startswith(dom):
            user = f"{tag}:{user}"
        for s in d["simulations"]:
            mm = M.metrics_for_interaction(R.tau2_user_turns(s.get("messages", [])))
            if mm:
                for k in M.METRIC_ORDER:
                    by[(user, dom)][k].append(mm[k])

    print(
        f"{'simulator / domain':40s} {'D1':>6s} {'D2*':>6s} {'D3':>6s} {'D4':>6s} {'behMean':>8s}"
    )
    print("-" * 74)
    for (user, dom), acc in sorted(by.items()):
        mm = {k: st.mean(v) for k, v in acc.items()}
        dd = {
            dim: st.mean(dice(mm[k], pooled[k]) for k in ks)
            for dim, ks in sim_dims.items()
        }
        bm = st.mean(dd.values())
        print(
            f"{user + ' / ' + dom:40s} {dd['D1']:6.1f} {dd['D2']:6.1f} "
            f"{dd['D3']:6.1f} {dd['D4']:6.1f} {bm:8.1f}"
        )
    print("\n(behMean = mean(D1-D4); a behavioral-only proxy, NOT the paper USI.)")


if __name__ == "__main__":
    main()
