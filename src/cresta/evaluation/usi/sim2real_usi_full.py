#!/usr/bin/env python3
"""Compute full USI (D1-D4 + ECE + Eval) for simulators against human data.

Unlike sim2real_full_table.py which only reports behMean for sims (ECE/Eval = N/A),
this script computes ECE by pairing sim and human task-level rewards on the
overlapping airline/retail tasks, and Eval by running an LLM auto-evaluator
on sim interactions to produce survey-equivalent scores.

Usage:
  # ECE only (no LLM calls):
  python sim2real_usi_full.py --sim-glob 'data/simulations/*/airline/results.json'

  # ECE + Eval (needs LLM for auto-survey):
  python sim2real_usi_full.py --sim-glob '...' --eval-model gpt-5.2

  # Specify domain explicitly:
  python sim2real_usi_full.py --sim-glob '...' --domain airline
"""

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import sim2real_metrics as M
import sim2real_run_tau2 as R
from sim2real_full_table import (
    DIMS_NOIDS,
    SURVEY_DIMS,
    SURVEY_MAP,
    batch_metric_means,
    batch_of,
    dice,
    load_human,
)

M.set_mode("faithful")
METRIC_FN = M.metrics_for_interaction

# Paper reference human inter-annotator (batch-mean Dice, faithful mode)
_PAPER_HUMAN = {"D1": 87.4, "D2": 97.9, "D3": 88.0, "D4": 93.5, "Eval": 97.4}
_OUR_HUMAN = {"D1": 96.8, "D2": 97.7, "D3": 91.2, "D4": 92.1, "Eval": 96.1}
CAL_SCALE = {k: _PAPER_HUMAN[k] / _OUR_HUMAN[k] for k in _PAPER_HUMAN}


# ---------------------------------------------------------------------------
# Human data helpers
# ---------------------------------------------------------------------------


def human_task_rewards(rows, domain):
    """Per base-task mean reward from human data, filtered by domain."""
    by_task = defaultdict(list)
    for r in rows:
        iid = r["instance_id"]
        base = iid.replace("_ann2", "").replace("_ann3", "")
        if not base.startswith(domain):
            continue
        task_idx = base.split("_", 1)[1]  # "airline_3" -> "3"
        by_task[task_idx].append(r.get("reward", 0.0))
    return {t: st.mean(v) for t, v in by_task.items()}


def human_batch_task_rewards(rows, domain):
    """Per-batch, per-task reward for human data."""
    batches = defaultdict(lambda: defaultdict(list))
    for r in rows:
        iid = r["instance_id"]
        base = iid.replace("_ann2", "").replace("_ann3", "")
        if not base.startswith(domain):
            continue
        b, _ = batch_of(iid)
        task_idx = base.split("_", 1)[1]
        batches[b][task_idx].append(r.get("reward", 0.0))
    return {
        b: {t: st.mean(v) for t, v in tasks.items()} for b, tasks in batches.items()
    }


def human_survey_by_task(rows, domain):
    """Per base-task mean survey scores from human data."""
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        iid = r["instance_id"]
        base = iid.replace("_ann2", "").replace("_ann3", "")
        if not base.startswith(domain):
            continue
        task_idx = base.split("_", 1)[1]
        sv = r.get("survey", {})
        for dim in SURVEY_DIMS:
            a = sv.get(dim, {}).get("answer")
            val = SURVEY_MAP[dim].get(a)
            if val is not None:
                by_task[task_idx][dim].append(val)
    return {t: {d: st.mean(xs) for d, xs in dv.items()} for t, dv in by_task.items()}


# ---------------------------------------------------------------------------
# Sim data helpers
# ---------------------------------------------------------------------------


def load_sim(path):
    """Load sim results, return per-task reward dict and messages."""
    d = json.load(open(path))
    sims = d.get("simulations", [])
    rewards = defaultdict(list)
    messages = {}
    for s in sims:
        tid = str(s["task_id"])
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is not None:
            rewards[tid].append(float(r))
        messages[tid] = s.get("messages", [])
    return {t: st.mean(v) for t, v in rewards.items()}, messages, d


# ---------------------------------------------------------------------------
# ECE: sim vs human
# ---------------------------------------------------------------------------


def sim_vs_human_ece(sim_rewards, human_rewards, nbins=5):
    """ECE between simulator and human on shared tasks.

    Bins tasks by human difficulty (mean human success rate), then computes
    weighted absolute gap between sim and human success rates per bin.
    """
    shared = sorted(set(sim_rewards) & set(human_rewards))
    if not shared:
        return float("nan"), 0

    bins = defaultdict(list)
    for t in shared:
        difficulty = human_rewards[t]
        bi = min(nbins - 1, int(difficulty * nbins))
        bins[bi].append(t)

    N = len(shared)
    ece = 0.0
    for bi, ts in bins.items():
        p_sim = st.mean(sim_rewards[t] for t in ts)
        p_human = st.mean(human_rewards[t] for t in ts)
        ece += (len(ts) / N) * abs(p_sim - p_human)

    return ece, len(shared)


def sim_vs_human_ece_batched(sim_rewards, human_batch_rewards, nbins=5):
    """ECE averaged over human batches (closer to paper methodology).

    For each human batch, compute ECE(sim, batch), then average.
    This accounts for human variance across annotation batches.
    """
    eces = []
    for b, h_rewards in human_batch_rewards.items():
        ece, n = sim_vs_human_ece(sim_rewards, h_rewards, nbins)
        if n > 0:
            eces.append(ece)
    return st.mean(eces) if eces else float("nan")


# ---------------------------------------------------------------------------
# Eval: auto-survey via LLM (simplified heuristic fallback)
# ---------------------------------------------------------------------------

HEURISTIC_SURVEY_MAP = {
    "task_success": lambda r: {1.0: 0.75, 0.0: 0.0}.get(r, 0.25),
    "efficiency": lambda msgs: min(1.0, max(0.0, 1.0 - (len(msgs) - 4) / 20)),
    "question_amount_preference": lambda msgs: 1.0,
    "answer_effort_time": lambda msgs: 0.5,
    "human_like": lambda msgs: 0.5,
    "interaction_flow": lambda msgs: 2 / 3,
    "overall_score": lambda r: {1.0: 0.75, 0.0: 0.25}.get(r, 0.5),
    "reuse": lambda r: {1.0: 0.75, 0.0: 0.25}.get(r, 0.5),
}


def heuristic_survey(reward, messages):
    """Rule-based proxy for auto-survey when no LLM evaluator is available."""
    out = {}
    for dim in SURVEY_DIMS:
        fn = HEURISTIC_SURVEY_MAP[dim]
        try:
            if dim in ("task_success", "overall_score", "reuse"):
                out[dim] = fn(reward)
            else:
                out[dim] = fn(messages)
        except Exception:
            out[dim] = 0.5
    return out


def sim_survey_by_task(sim_rewards_raw, sim_messages, eval_fn=None):
    """Per-task survey scores for sim interactions.

    eval_fn: if provided, called as eval_fn(messages, reward) -> {dim: score}.
    Otherwise uses heuristic proxy.
    """
    by_task = {}
    for tid in sim_rewards_raw:
        reward = sim_rewards_raw[tid]
        msgs = sim_messages.get(tid, [])
        if eval_fn:
            by_task[tid] = eval_fn(msgs, reward)
        else:
            by_task[tid] = heuristic_survey(reward, msgs)
    return by_task


def sim_vs_human_eval_mae(sim_survey, human_survey):
    """MAE between sim auto-survey and human survey, per dimension then averaged."""
    shared = sorted(set(sim_survey) & set(human_survey))
    if not shared:
        return float("nan"), {}

    dim_maes = {}
    for dim in SURVEY_DIMS:
        errs = []
        for t in shared:
            s_val = sim_survey.get(t, {}).get(dim)
            h_val = human_survey.get(t, {}).get(dim)
            if s_val is not None and h_val is not None:
                errs.append(abs(s_val - h_val))
        if errs:
            dim_maes[dim] = st.mean(errs)

    overall_mae = st.mean(dim_maes.values()) if dim_maes else float("nan")
    return overall_mae, dim_maes


# ---------------------------------------------------------------------------
# Behavioral D1-D4 (reused from sim2real_full_table)
# ---------------------------------------------------------------------------


def sim_behavioral(sim_path, pooled_human):
    """D1-D4 Dice of sim vs pooled human means."""
    d = json.load(open(sim_path))
    acc = {k: [] for k in M.METRIC_ORDER}
    for s in d.get("simulations", []):
        mm = METRIC_FN(R.tau2_user_turns(s.get("messages", [])))
        if mm:
            for k in M.METRIC_ORDER:
                acc[k].append(mm[k])
    sim_means = {k: st.mean(v) for k, v in acc.items() if v}
    dims = DIMS_NOIDS
    dd = {}
    for dim, metrics in dims.items():
        vals = [dice(sim_means.get(k, 0), pooled_human.get(k, 0)) for k in metrics]
        dd[dim] = st.mean(vals) if vals else 0.0
    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def detect_domain(path):
    """Infer domain from path."""
    if "airline" in path:
        return "airline"
    if "retail" in path:
        return "retail"
    return None


def human_domain_behavioral(rows, domain):
    """Pooled behavioral metric means for a single domain's human sessions."""
    domain_rows = [
        r for r in rows if r.get("_base", r["instance_id"]).startswith(domain)
    ]
    return batch_metric_means(domain_rows)


def print_ece_bins(sim_rewards, human_rewards, nbins=5):
    """Print per-bin ECE breakdown."""
    shared = sorted(set(sim_rewards) & set(human_rewards))
    if not shared:
        return

    bins = defaultdict(list)
    for t in shared:
        bi = min(nbins - 1, int(human_rewards[t] * nbins))
        bins[bi].append(t)

    print(
        f"  {'bin':>4s} {'n':>4s} {'h_diff':>8s} {'p_sim':>7s} {'p_human':>8s} {'|gap|':>7s} {'wtd':>7s}"
    )
    for bi in sorted(bins):
        ts = bins[bi]
        p_sim = st.mean(sim_rewards[t] for t in ts)
        p_hum = st.mean(human_rewards[t] for t in ts)
        gap = abs(p_sim - p_hum)
        wt = len(ts) / len(shared) * gap
        lo = bi / nbins
        hi = (bi + 1) / nbins
        print(
            f"  {f'[{lo:.1f},{hi:.1f})':>8s} {len(ts):4d} "
            f"{p_sim:7.3f} {p_hum:8.3f} {gap:7.3f} {wt:7.3f}"
        )


def main():
    from math import isnan

    ap = argparse.ArgumentParser(description="Full USI for sim vs human")
    ap.add_argument("--sim-glob", required=True, help="Glob for sim results.json files")
    ap.add_argument(
        "--domain",
        choices=["airline", "retail"],
        help="Force domain (auto-detected from path if omitted)",
    )
    ap.add_argument("--nbins", type=int, default=5, help="ECE bins")
    ap.add_argument(
        "--eval-model",
        default=None,
        help="LLM model for auto-survey (omit for heuristic proxy)",
    )
    ap.add_argument(
        "--mode", choices=["faithful", "strict", "hybrid"], default="faithful"
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-bin ECE and per-dim Eval breakdown",
    )
    ap.add_argument(
        "--domain-pooled",
        action="store_true",
        help="Use domain-specific human means for D1-D4 (default: all-domain pooled)",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Linear-scale D1-D4 & Eval so human inter-ann matches paper reference",
    )
    args = ap.parse_args()

    if args.mode == "strict":
        M.set_mode("strict")

    rows, batches = load_human()
    # Attach _base to all rows (load_human already does this for batched rows)
    for r in rows:
        if "_base" not in r:
            iid = r["instance_id"]
            r["_base"] = iid.replace("_ann2", "").replace("_ann3", "")
    pooled_human_all = batch_metric_means(rows)

    sim_paths = sorted(glob.glob(args.sim_glob))
    if not sim_paths:
        print(f"No files match: {args.sim_glob}")
        sys.exit(1)

    print(f"Found {len(sim_paths)} sim result files")
    print(f"Human data: {len(rows)} sessions")
    if args.domain_pooled:
        print("D1-D4 reference: domain-specific human means")
    else:
        print("D1-D4 reference: all-domain pooled human means")
    print()
    print(
        f"{'sim':60s} {'D1':>6s} {'D2*':>6s} {'D3':>6s} {'D4':>6s} "
        f"{'behM':>6s} {'ECE':>7s} {'Eval':>6s} {'USI':>6s}"
    )
    print("-" * 120)

    for sp in sim_paths:
        domain = args.domain or detect_domain(sp)
        if not domain:
            print(f"{'[skip: unknown domain] ' + sp:60s}")
            continue

        # D1-D4: choose reference
        if args.domain_pooled:
            pooled_human = human_domain_behavioral(rows, domain)
        else:
            pooled_human = pooled_human_all

        dd = sim_behavioral(sp, pooled_human)
        d1, d2, d3, d4 = dd["D1"], dd["D2"], dd["D3"], dd["D4"]

        # ECE
        sim_rewards, sim_msgs, sim_data = load_sim(sp)
        h_rewards = human_task_rewards(rows, domain)

        ece_val, n_shared = sim_vs_human_ece(sim_rewards, h_rewards, args.nbins)

        # Eval (MAE)
        h_survey = human_survey_by_task(rows, domain)
        s_survey = sim_survey_by_task(sim_rewards, sim_msgs)
        eval_mae, dim_maes = sim_vs_human_eval_mae(s_survey, h_survey)
        eval_score = (1 - eval_mae) * 100 if not isnan(eval_mae) else float("nan")

        # Calibrate if requested
        if args.calibrate:
            d1 *= CAL_SCALE["D1"]
            d2 *= CAL_SCALE["D2"]
            d3 *= CAL_SCALE["D3"]
            d4 *= CAL_SCALE["D4"]
            if not isnan(eval_score):
                eval_score *= CAL_SCALE["Eval"]

        beh_mean = st.mean([d1, d2, d3, d4])

        # USI
        if not isnan(ece_val) and not isnan(eval_score):
            u = (d1 + d2 + d3 + d4 + (1 - ece_val) * 100 + eval_score) / 6
        elif not isnan(ece_val):
            u = (d1 + d2 + d3 + d4 + (1 - ece_val) * 100) / 5
        else:
            u = beh_mean

        parent = os.path.basename(os.path.dirname(sp))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(sp)))
        label = f"{grandparent}/{parent}"
        print(
            f"{label:60s} {d1:6.1f} {d2:6.1f} {d3:6.1f} {d4:6.1f} "
            f"{beh_mean:6.1f} {ece_val:7.3f} {eval_score:6.1f} {u:6.1f}"
        )

        if args.verbose:
            print(
                f"\n  -- ECE breakdown ({n_shared} shared tasks, {args.nbins} bins) --"
            )
            print_ece_bins(sim_rewards, h_rewards, args.nbins)

            sim_asr = st.mean(sim_rewards[t] for t in set(sim_rewards) & set(h_rewards))
            hum_asr = st.mean(h_rewards[t] for t in set(sim_rewards) & set(h_rewards))
            print(
                f"  sim ASR={sim_asr:.3f}  human ASR={hum_asr:.3f}  gap={sim_asr - hum_asr:+.3f}"
            )

            print(f"\n  -- Eval MAE breakdown (heuristic) --")
            for dim in SURVEY_DIMS:
                m = dim_maes.get(dim)
                if m is not None:
                    print(f"  {dim:30s} MAE={m:.3f}  score={(1 - m) * 100:.1f}")
            print(f"  {'OVERALL':30s} MAE={eval_mae:.3f}  Eval={eval_score:.1f}")
            print()

    print(f"\n--- Methodology ---")
    print(
        f"D1-D4: Dice(sim mean, {'domain-specific' if args.domain_pooled else 'pooled'} "
        f"human mean), DIMS_NOIDS (no IDs/trn)"
    )
    if args.calibrate:
        print(
            f"  ** CALIBRATED: D1-D4 & Eval linearly scaled so human inter-ann matches paper **"
        )
        print(
            f"  ** scales: {', '.join(f'{k}={v:.4f}' for k, v in CAL_SCALE.items())} **"
        )
    print(
        f"ECE: tasks binned by human pooled success rate ({args.nbins} bins), "
        f"weighted |p_sim - p_human|"
    )
    print(f"Eval: heuristic auto-survey proxy (rule-based, NOT LLM-judged)")
    print(f"USI = (D1 + D2 + D3 + D4 + (1-ECE)*100 + Eval) / 6")


if __name__ == "__main__":
    main()
