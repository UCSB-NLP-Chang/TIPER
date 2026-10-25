#!/usr/bin/env python3
"""Apply the Sim2Real D1-D4 behavioral metrics to tau2 simulation runs
(telecom/banking) to compare user simulators.

Reads tau2 results.json files (messages with role=="user" = the user simulator),
extracts user turns, computes the atomic metrics per simulation, averages per
cell, and prints a per-dimension comparison across cells.

No human baseline exists for telecom/banking, so we report RAW values for
cross-simulator comparison (no Dice). Per the credibility review:
  - IDs/trn is dropped (absolute value not reproducible).
  - strict mode uses cleaner Accuse/Pivot/IDConf lists for debugging.

Usage:
    python -m cresta.evaluation.usi.sim2real_run_tau2 [results.json | dir ...]
    python -m cresta.evaluation.usi.sim2real_run_tau2 --mode strict [paths ...]
    (defaults to the canonical_agent_matrix telecom/banking cells)

--mode:
    faithful (default) paper domain-blind lexicon
    strict             atomic-metric debug mode with cleaner Accuse/Pivot/IDConf
"""

import argparse
import glob
import json
import os
import re
import statistics as st

from cresta.paths import ARTIFACTS_DIR

try:
    from . import sim2real_metrics as M
except ImportError:  # direct execution
    import sim2real_metrics as M

RE_CONTROL = re.compile(
    r"#{2,}[A-Z][A-Z\- ]*#{2,}"
)  # ###STOP###, ###OUT-OF-SCOPE### ...

# Metrics shown for atomic comparison. IDs/trn is omitted because the paper does
# not specify enough detail to reproduce it exactly.
SHOW = [
    "Wds/trn",
    "Short%",
    "Polite%",
    "Formal%",
    "Ack%",
    "VerbCV",
    "Repeat%",
    "IDConf%",
    "FrntLoad%",
    "Open.Wds",
    "Uncert%",
    "Certn%",
    "Pushbk%",
    "ClarfyQ%",
    "InfoQ%",
    "Emot%",
    "Accuse%",
    "Pivot%",
]


def tau2_user_turns(messages):
    out = []
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if c is None:
            continue  # tool-only user turn
        c = str(c)
        c = RE_CONTROL.sub(" ", c).strip()  # strip control tokens
        if not c:
            continue
        out.append(c)
    return out


def cell_metrics(results_path, mode="faithful"):
    M.set_mode(mode)
    with open(results_path, encoding="utf-8") as f:
        d = json.load(f)
    sims = d.get("simulations", [])
    per = {k: [] for k in M.METRIC_ORDER}
    n = 0
    for s in sims:
        turns = tau2_user_turns(s.get("messages", []))
        mm = M.metrics_for_interaction(turns)
        if mm is None:
            continue
        n += 1
        for k in M.METRIC_ORDER:
            per[k].append(mm[k])
    means = {k: (st.mean(v) if v else float("nan")) for k, v in per.items()}
    info = d.get("info", {})
    ui = info.get("user_info", {})
    user = f"{ui.get('implementation', '?')}:{ui.get('llm', '?')}"
    reward = None
    rws = [s["reward_info"]["reward"] for s in sims if s.get("reward_info")]
    if rws:
        reward = st.mean(rws)
    return {
        "n": n,
        "user": user,
        "reward": reward,
        "means": means,
        "name": os.path.basename(os.path.dirname(results_path)),
    }


def discover():
    base = ARTIFACTS_DIR / "baseline" / "canonical_agent_matrix"
    out = []
    for dom in ("telecom", "banking"):
        for p in sorted(glob.glob(str(base / f"{dom}_*" / "results.json"))):
            out.append(p)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["faithful", "strict"],
        default="faithful",
        help="'strict' only changes atomic debug metrics; it is not a USI score",
    )
    ap.add_argument("paths", nargs="*", help="results.json files or dirs")
    args = ap.parse_args(argv)

    M.set_mode(args.mode)

    paths = []
    for a in args.paths:
        if os.path.isdir(a):
            paths += sorted(
                glob.glob(os.path.join(a, "**/results.json"), recursive=True)
            )
        elif a.endswith(".json"):
            paths.append(a)
    if not paths:
        paths = discover()

    cells = []
    for p in paths:
        try:
            cells.append(cell_metrics(p, mode=args.mode))
        except Exception as e:
            print(f"  [skip] {p}: {type(e).__name__} {e}")

    # print as a metric x cell matrix
    print(
        f"mode={args.mode} | {len(cells)} cells | metric values are RAW means over sims\n"
    )
    namew = max(len(c["name"]) for c in cells) if cells else 10
    # header: per cell
    for c in cells:
        rw = f"{c['reward']:.3f}" if c["reward"] is not None else "  -  "
        print(
            f"  [{c['name']:<{namew}}]  user={c['user']:<34} n={c['n']:<4} reward={rw}"
        )
    print()
    # matrix
    hdr = f"{'metric':10s} " + " ".join(f"{i:>7d}" for i in range(len(cells)))
    print(hdr)
    print("-" * len(hdr))
    for k in SHOW:
        row = " ".join(f"{c['means'][k]:7.2f}" for c in cells)
        print(f"{k:10s} {row}")
    print("\n(columns are the cells listed above, in order)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
