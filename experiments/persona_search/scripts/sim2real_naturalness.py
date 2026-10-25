#!/usr/bin/env python3
"""Sim2Real (approach #2) naturalness for the pair-run cert personas.

Reuses the EXISTING, validated sim2real pipeline by import (zero edits to those
files): every user interaction -> 19 rule behavioral metrics; then TWO readings,
both vs the same human reference manifold (tau_bench_tasks_unified human-as-user;
cross-domain -> relative ranking only, which is all we need to compare our 10
telecom personas):

  * approach #2 (geometric PR, the one asked for):
      P_geom (precision = NATURALNESS) = frac of SIM points inside HUMAN manifold
      R_geom (recall    = COVERAGE)    = frac of HUMAN points inside SIM manifold
                                         (this is the mode-collapse detector)
  * the D1-D4 + behMean Dice "percent" reading (shown alongside for reference).

Merges in full-114 difficulty + validity from FINAL_pair_cert.json so the output
is the same difficulty x naturalness x validity table, with naturalness = #2.

  .venv/bin/python experiments/persona_search/scripts/sim2real_naturalness.py [--mode strict|faithful] [--k 3]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(REPO))  # import the existing sim2real_*.py at repo root

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
import sim2real_llm_metrics as LLM  # noqa: E402
import sim2real_metrics as M  # noqa: E402
from sim2real_full_table import DIMS_NOIDS, dice  # noqa: E402
from sim2real_pr import (  # noqa: E402
    all_spaces,
    beh_mean,
    feature_matrix,
    human_turn_lists,
    load_human,
    pca_fit,
    precision_recall,
    sim_turn_lists,
    standardizer,
)


def failure_sim_turn_lists(results_path):
    """Like sim_turn_lists, but ONLY interactions the agent FAILED (reward < 1) --
    answers 'is the difficulty natural or fabricated?' at ~1/3 the LLM cost."""
    import sim2real_run_tau2 as R

    d = json.load(open(results_path))
    out = []
    for s in d.get("simulations", []):
        rew = (s.get("reward_info") or {}).get("reward")
        if rew is not None and rew < 1:
            out.append(R.tau2_user_turns(s.get("messages", [])))
    return out


def feat(turn_lists, mode):
    """interaction -> 19-metric matrix. mode in {strict,faithful}=pure rule;
    mode=hybrid swaps the 10 semantic metrics for the gpt-5.2 rubric judge
    (cross-domain robust), keeping the SAME interaction set + dict shape so the
    PR/Dice math downstream is unchanged."""
    if mode != "hybrid":
        return feature_matrix(turn_lists)
    M.set_mode("strict")  # rule half of hybrid is mode-independent; fix a state
    kept = [t for t in turn_lists if M.metrics_for_interaction(t)]
    labels = LLM.label_interactions(kept)  # batched, concurrent, disk-cached
    rows = [
        [LLM.hybrid_metrics_for_interaction(t, labels=lab)[k] for k in M.METRIC_ORDER]
        for t, lab in zip(kept, labels)
    ]
    return np.asarray(rows, float)


CERT = REPO / "data" / "simulations" / "gepa_certify"
FINAL = (
    REPO
    / "data"
    / "persona_opt"
    / "pgepa_runs"
    / "cert_pair_full"
    / "FINAL_pair_cert.json"
)

# pid -> human-readable who (same labels as FINAL_pair_cert.json)
CELLS = [
    ("gepa_certify_incumbent", "p0003 seed (=incumbent)"),
    ("gepa_certify_c00", "p0003 idx4"),
    ("gepa_certify_c01", "p0003 idx5"),
    ("gepa_certify_c02", "p0003 idx6"),
    ("gepa_certify_c03", "p0003 idx10"),
    ("gepa_certify_c04", "p0002 idx1"),
    ("gepa_certify_c05", "p0002 idx4"),
    ("gepa_certify_c06", "p0002 idx6"),
    ("gepa_certify_c07", "p0002 idx8"),
    ("gepa_certify_c08", "p0003 idx9"),
]


def d1_d4(Sx_raw, pooled):
    """Per-dimension Dice(sim mean, human mean), 0-100 (the 'percent' reading)."""
    cell = {k: float(np.mean(Sx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    return {
        dim: st.mean(dice(cell[k], pooled[k]) for k in ks)
        for dim, ks in DIMS_NOIDS.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["faithful", "strict", "hybrid"], default="strict"
    )
    ap.add_argument("--k", type=int, default=3, help="k-NN manifold neighbor count")
    ap.add_argument(
        "--failures-only",
        action="store_true",
        help="score naturalness only on failed (reward<1) interactions (~1/3 cost)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        suf = "_failonly" if args.failures_only else ""
        args.out = str(FINAL.parent / f"sim2real_naturalness_{args.mode}{suf}.json")

    if args.mode != "hybrid":
        M.set_mode(args.mode)
    rows, _ = load_human()
    print(f"[human] extracting features (mode={args.mode})...")
    Hx_raw = feat(human_turn_lists(rows), args.mode)
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"
    print(
        f"== sim2real #2 naturalness (mode={args.mode}, k={args.k}, space={pca_name}) =="
    )
    print(
        f"human ref: {len(Hx_raw)} interactions, 19 features; cross-domain -> relative ranking\n"
    )

    diff = {}
    if FINAL.exists():
        for r in json.load(open(FINAL))["rows"]:
            diff[r["pid"]] = r

    out = []
    for pid, who in CELLS:
        path = CERT / pid / "telecom" / "results.json"
        if not path.exists():
            print(f"  {who:<26} no sims")
            continue
        tl = (
            failure_sim_turn_lists(str(path))
            if args.failures_only
            else sim_turn_lists(str(path))
        )
        print(
            f"[{who}] {len(tl)} interactions ({'failures-only' if args.failures_only else 'all'}), mode={args.mode}..."
        )
        Sx_raw = feat(tl, args.mode)
        sp = all_spaces(Hx_raw, Sx_raw, mu, sd, pca)
        Hp, Sp = sp[pca_name]
        P, Rc, F1 = precision_recall(Hp, Sp, args.k)
        bm = beh_mean(Sx_raw, pooled)
        dd = d1_d4(Sx_raw, pooled)
        fr = diff.get(pid, {})
        out.append(
            {
                "pid": pid,
                "who": who,
                "n_interactions": int(Sx_raw.shape[0]),
                "P_geom_naturalness": round(float(P), 4),
                "R_geom_coverage": round(float(Rc), 4),
                "F1": round(float(F1), 4),
                "behMean": round(float(bm), 2),
                "D1": round(dd["D1"], 1),
                "D2": round(dd["D2"], 1),
                "D3": round(dd["D3"], 1),
                "D4": round(dd["D4"], 1),
                "difficulty": fr.get("difficulty"),
                "severe": fr.get("severe"),
                "valid": fr.get("valid"),
            }
        )

    json.dump(out, open(args.out, "w"), indent=2)

    # ---- table ----
    print(
        f"{'who':<26}{'P(nat)':<8}{'R(cov)':<8}{'F1':<7}{'behMean':<9}"
        f"{'D1':<6}{'D2':<6}{'D3':<6}{'D4':<6}{'diff':<8}{'valid'}"
    )
    print("-" * 104)
    for r in sorted(out, key=lambda r: -(r["difficulty"] or 0)):
        v = "Y" if r["valid"] else ("GATED" if r["valid"] is False else "?")
        print(
            f"{r['who']:<26}{r['P_geom_naturalness']:<8}{r['R_geom_coverage']:<8}{r['F1']:<7}"
            f"{r['behMean']:<9}{r['D1']:<6}{r['D2']:<6}{r['D3']:<6}{r['D4']:<6}"
            f"{str(r['difficulty']):<8}{v}"
        )

    valid = [r for r in out if r["valid"] and r["difficulty"] is not None]
    if valid:
        valid.sort(key=lambda r: -r["difficulty"])
        b = valid[0]
        print(
            f"\nbest valid & hardest: {b['who']}  diff={b['difficulty']}  "
            f"P(nat)={b['P_geom_naturalness']}  R(cov)={b['R_geom_coverage']}"
        )
        nat_rank = sorted(valid, key=lambda r: -r["P_geom_naturalness"])
        print(
            "most natural (valid, by P_geom): "
            + ", ".join(f"{r['who']}={r['P_geom_naturalness']}" for r in nat_rank[:4])
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
