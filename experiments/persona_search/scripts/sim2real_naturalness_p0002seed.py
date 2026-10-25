#!/usr/bin/env python3
"""Supplement: sim2real naturalness for **p0002 seed (gen0)**, which was absent
from cert_pair_full (c09 = p0002 seed never finished in that batch, so it is not in
CELLS / FINAL_pair_cert.json).

NEW file — reuses the validated sim2real pipeline AND sim2real_naturalness.py by
import, with ZERO edits to either. Computes the same row schema as
sim2real_naturalness_*.json and (with --merge) appends it into the matching file so
the plots can pick up the p0002 gen0 point.

  .venv/bin/python experiments/persona_search/scripts/sim2real_naturalness_p0002seed.py \
      --mode hybrid --failures-only --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
DATA_ROOT = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))  # sim2real_*.py at repo root
sys.path.insert(0, str(REPO_ROOT / "experiments/persona_search/scripts"))

import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
from sim2real_naturalness import d1_d4, failure_sim_turn_lists, feat  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import (  # noqa: E402
    all_spaces,
    beh_mean,
    human_turn_lists,
    load_human,
    pca_fit,
    precision_recall,
    sim_turn_lists,
    standardizer,
)

# Extra cells absent from sim2real_naturalness.py's CELLS, each its own 228 full-set
# cert run.  (pid, who, results.json, full-set difficulty=1-ASR, severe_rate, valid)
EXTRA = {
    "p0002_seed": dict(
        pid="p0002_seed_gen0",
        who="p0002 seed",
        results=DATA_ROOT / "data/simulations/"
        "20260607_112504_telecom_llm_agent_gpt-5.4_cc_user_con_claude-opus-4-7/results.json",
        difficulty=0.2105,
        severe=0.0,
        valid=True,
    ),  # cert_p0002_seed_judge: 0 severe
    "no_persona": dict(
        pid="no_persona",
        who="no_persona",
        results=DATA_ROOT / "data/simulations/gepa_certify/"
        "progcert_no_persona_fullbase228/telecom/results.json",
        difficulty=0.1491,
        severe=0.026,
        valid=True,
    ),  # floor: ASR 0.851, severe 0.026
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["faithful", "strict", "hybrid"], default="hybrid"
    )
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--failures-only", action="store_true")
    ap.add_argument(
        "--targets",
        default="p0002_seed,no_persona",
        help="comma list of EXTRA keys to compute",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="append into sim2real_naturalness_<mode>[_failonly].json",
    )
    args = ap.parse_args()

    if args.mode != "hybrid":
        M.set_mode(args.mode)
    rows, _ = load_human()
    print(f"[human] features (mode={args.mode}, cached if seen before)...")
    Hx = feat(human_turn_lists(rows), args.mode)
    pooled = {k: float(np.nanmean(Hx[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx)
    pca = pca_fit((Hx - mu) / sd)
    pca_name = f"pca{pca[2]}"

    out_rows = []
    for key in [t.strip() for t in args.targets.split(",") if t.strip()]:
        c = EXTRA[key]
        tl = (
            failure_sim_turn_lists(str(c["results"]))
            if args.failures_only
            else sim_turn_lists(str(c["results"]))
        )
        print(
            f"[{c['who']}] {len(tl)} interactions "
            f"({'failures-only' if args.failures_only else 'all'}), mode={args.mode}..."
        )
        Sx = feat(tl, args.mode)
        Hp, Sp = all_spaces(Hx, Sx, mu, sd, pca)[pca_name]
        P, Rc, F1 = precision_recall(Hp, Sp, args.k)
        bm = beh_mean(Sx, pooled)
        dd = d1_d4(Sx, pooled)
        row = {
            "pid": c["pid"],
            "who": c["who"],
            "n_interactions": int(Sx.shape[0]),
            "P_geom_naturalness": round(float(P), 4),
            "R_geom_coverage": round(float(Rc), 4),
            "F1": round(float(F1), 4),
            "behMean": round(float(bm), 2),
            "D1": round(dd["D1"], 1),
            "D2": round(dd["D2"], 1),
            "D3": round(dd["D3"], 1),
            "D4": round(dd["D4"], 1),
            "difficulty": c["difficulty"],
            "severe": c["severe"],
            "valid": c["valid"],
        }
        print(json.dumps(row, indent=2))
        out_rows.append(row)

    if args.merge:
        suf = "_failonly" if args.failures_only else ""
        f = (
            DATA_ROOT
            / "data/persona_opt/pgepa_runs/cert_pair_full"
            / f"sim2real_naturalness_{args.mode}{suf}.json"
        )
        pids = {r["pid"] for r in out_rows}
        data = [r for r in json.load(open(f)) if r.get("pid") not in pids]
        data.extend(out_rows)
        json.dump(data, open(f, "w"), indent=2)
        print(f"\nmerged {len(out_rows)} row(s) -> {f}")


if __name__ == "__main__":
    main()
