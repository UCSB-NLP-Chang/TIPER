#!/usr/bin/env python3
"""Feasibility / calibration harness for the hybrid (rule + LLM-rubric) metrics.

Two checks:

1. HUMAN CALIBRATION -- run the LLM rubric on the 495 human sessions
   (tau_bench_tasks_unified.json) and compare the 10 LLM-owned metrics to the
   paper Figure-3 human reference (REF_HUMAN), side-by-side with the faithful
   rule baseline. Gate: LLM MAE <= rule MAE on these 10 (especially IDs/trn
   closer to 2.6, Accuse/Pivot closer to 2.7/8.4).

2. DOMAIN-FRAGILITY -- run on the p0003 cert candidate telecom trajectories and
   confirm the LLM removes the faithful rule's spurious IDConf 87.6% (-> ~0),
   while keeping Accuse/Pivot semantically sane.

Usage:
    .venv/bin/python sim2real_llm_calibrate.py                 # full 495 x maj 3
    .venv/bin/python sim2real_llm_calibrate.py --limit 30      # quick sample
    .venv/bin/python sim2real_llm_calibrate.py --maj 1 --skip-domain
"""

import argparse
import json
import os
import statistics as st
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from pathlib import Path

import sim2real_llm_metrics as L
import sim2real_metrics as M
import sim2real_run_tau2 as R

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

HUMAN_FILE = os.path.join(
    os.environ.get("PERSONA_DATA_DIR", "/Users/yutao.xie/Downloads"),
    "tau_bench_tasks_unified.json",
)
CANDIDATE = str(
    DATA_ROOT / "data/simulations/gepa_certify/progcert_candidate/telecom/results.json"
)


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]  # drop NaN/None
    return st.mean(xs) if xs else float("nan")


def human_turn_lists(limit=0):
    data = json.load(open(HUMAN_FILE))
    items = list(data.values()) if isinstance(data, dict) else data
    out = []
    for entry in items:
        turns = M.user_turns(entry.get("conversation", []))
        if turns:
            out.append(turns)
        if limit and len(out) >= limit:
            break
    return out


def rule_means(turn_lists, mode, keys):
    M.set_mode(mode)
    acc = {k: [] for k in keys}
    for turns in turn_lists:
        mm = M.metrics_for_interaction(turns)
        if mm:
            for k in keys:
                acc[k].append(mm[k])
    return {k: _mean(v) for k, v in acc.items()}


def llm_means(turn_lists, labels, keys):
    acc = {k: [] for k in keys}
    for turns, lab in zip(turn_lists, labels):
        mm = L.metrics_from_labels(turns, lab)
        for k in keys:
            acc[k].append(mm[k])
    return {k: _mean(v) for k, v in acc.items()}


def calibrate(args):
    keys = L.LLM_METRICS
    print(f"loading human sessions from {HUMAN_FILE} ...")
    tl = human_turn_lists(args.limit)
    print(
        f"  {len(tl)} interactions with >=1 user turn"
        + (f" (limited to {args.limit})" if args.limit else "")
    )

    rule_f = rule_means(tl, "faithful", keys)
    labels = L.label_interactions(
        tl,
        model=args.model,
        maj=args.maj,
        effort=args.effort,
        concurrency=args.concurrency,
    )
    llm = llm_means(tl, labels, keys)

    print(
        "\n================ HUMAN CALIBRATION (10 LLM-owned metrics) ================"
    )
    hdr = (
        f"{'metric':10s} {'human':>7s} {'rule(F)':>8s} {'LLM':>8s} "
        f"{'|rule-H|':>9s} {'|LLM-H|':>9s}  winner"
    )
    print(hdr)
    print("-" * len(hdr))
    rule_errs, llm_errs, wins = [], [], 0
    for k in keys:
        h = M.REF_HUMAN[k]
        re_, le_ = abs(rule_f[k] - h), abs(llm[k] - h)
        rule_errs.append(re_)
        llm_errs.append(le_)
        win = "LLM" if le_ < re_ - 1e-9 else ("rule" if re_ < le_ - 1e-9 else "tie")
        wins += win == "LLM"
        print(
            f"{k:10s} {h:7.2f} {rule_f[k]:8.2f} {llm[k]:8.2f} "
            f"{re_:9.2f} {le_:9.2f}  {win}"
        )
    print("-" * len(hdr))
    print(
        f"{'MAE':10s} {'':7s} {_mean(rule_errs):8.2f} {_mean(llm_errs):8.2f} "
        f"{'':9s} {'':9s}  LLM wins {wins}/{len(keys)}"
    )
    gate = _mean(llm_errs) <= _mean(rule_errs)
    print(
        f"\nGATE (LLM MAE <= rule MAE): {'PASS' if gate else 'FAIL'} "
        f"({_mean(llm_errs):.2f} vs {_mean(rule_errs):.2f})"
    )


def domain_fragility(args):
    print(
        "\n================ DOMAIN-FRAGILITY: p0003 cert candidate (telecom) ==========="
    )
    try:
        d = json.load(open(CANDIDATE))
    except FileNotFoundError:
        print(f"  [skip] {CANDIDATE} not found")
        return
    sims = d.get("simulations", [])
    tl = [R.tau2_user_turns(s.get("messages", [])) for s in sims]
    tl = [t for t in tl if t]
    print(f"  {len(tl)} interactions")

    rule_faith = rule_means(tl, "faithful", ["IDConf%", "Accuse%", "Pivot%"])
    rule_strict = rule_means(tl, "strict", ["IDConf%", "Accuse%", "Pivot%"])
    labels = L.label_interactions(
        tl,
        model=args.model,
        maj=args.maj,
        effort=args.effort,
        concurrency=args.concurrency,
    )
    llm = llm_means(tl, labels, ["IDConf%", "Accuse%", "Pivot%"])

    print(f"\n  {'metric':10s} {'rule(F)':>8s} {'rule(S)':>8s} {'LLM':>8s}")
    for k in ("IDConf%", "Accuse%", "Pivot%"):
        print(f"  {k:10s} {rule_faith[k]:8.2f} {rule_strict[k]:8.2f} {llm[k]:8.2f}")
    print(
        f"\n  faithful-rule IDConf = {rule_faith['IDConf%']:.1f}%  ->  "
        f"LLM IDConf = {llm['IDConf%']:.1f}%  "
        f"({'FIXED' if llm['IDConf%'] < 10 else 'still high'})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=0, help="cap human interactions (0=all 495)"
    )
    ap.add_argument("--maj", type=int, default=L.MAJ)
    ap.add_argument("--model", default=L.MODEL)
    ap.add_argument("--effort", default=L.EFFORT)
    ap.add_argument("--concurrency", type=int, default=L.CONC)
    ap.add_argument("--skip-domain", action="store_true")
    args = ap.parse_args()

    calibrate(args)
    if not args.skip_domain:
        domain_fragility(args)


if __name__ == "__main__":
    main()
