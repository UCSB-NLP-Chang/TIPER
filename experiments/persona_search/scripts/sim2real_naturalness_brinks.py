#!/usr/bin/env python3
"""Sim2Real naturalness for the Brinks fullset eval personas.

Compares 4 personas (2 seeds + 2 optimized) and no_persona baseline
against real Brinks human conversations.

Usage:
  .venv/bin/python experiments/persona_search/scripts/sim2real_naturalness_brinks.py [--mode strict]
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
sys.path.insert(0, str(REPO))

import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402
from sim2real_full_table import DIMS_NOIDS, dice  # noqa: E402

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
    sim_turn_lists,
    standardizer,
)

BRINKS_HUMAN = Path(
    os.path.join(
        os.environ.get("PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"),
        "virtual_agent",
        "user_simulator_dataset_v2.json",
    )
)

NO_PERSONA = (
    REPO
    / "data/simulations/gepa_certify/progcert_no_persona_fullbase228/telecom/results.json"
)

FULLSET_DIR = REPO / "data/simulations/persona_opt/fullset_brinks"

CELLS = [
    ("fullset_brinks_03_seed", "brinks_03 seed"),
    ("fullset_brinks_03_optimized", "brinks_03 optimized"),
    ("fullset_brinks_04_seed", "brinks_04 seed"),
    ("fullset_brinks_04_optimized", "brinks_04 optimized"),
]


def brinks_human_turn_lists() -> list[list[str]]:
    """Load Brinks human conversations -> list of (list of user turn strings)."""
    data = json.load(open(BRINKS_HUMAN))
    out = []
    for conv in data:
        turns = []
        for m in conv.get("messages", []):
            if m.get("speaker") != "VISITOR":
                continue
            text = str(m.get("text", "")).strip()
            if text and not M.is_meta(text):
                turns.append(text)
        if turns:
            out.append(turns)
    return out


def d1_d4(Sx_raw, pooled):
    cell = {k: float(np.mean(Sx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    return {
        dim: st.mean(dice(cell[k], pooled[k]) for k in ks)
        for dim, ks in DIMS_NOIDS.items()
    }


def score_one(label, results_path, Hx_raw, pooled, mu, sd, pca, pca_name, k):
    tl = sim_turn_lists(str(results_path))
    n_sims = len(json.load(open(results_path)).get("simulations", []))
    rewards = [
        (s.get("reward_info") or {}).get("reward")
        for s in json.load(open(results_path)).get("simulations", [])
        if (s.get("reward_info") or {}).get("reward") is not None
    ]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    difficulty = 1 - avg_reward if avg_reward is not None else None

    Sx_raw = feature_matrix(tl)
    sp = all_spaces(Hx_raw, Sx_raw, mu, sd, pca)
    Hp, Sp = sp[pca_name]
    P, Rc, F1 = precision_recall(Hp, Sp, k)
    bm = beh_mean(Sx_raw, pooled)
    dd = d1_d4(Sx_raw, pooled)
    return {
        "label": label,
        "n_interactions": int(Sx_raw.shape[0]),
        "n_sims": n_sims,
        "avg_reward": round(avg_reward, 4) if avg_reward is not None else None,
        "difficulty": round(difficulty, 4) if difficulty is not None else None,
        "P_geom_naturalness": round(float(P), 4),
        "R_geom_coverage": round(float(Rc), 4),
        "F1": round(float(F1), 4),
        "behMean": round(float(bm), 2),
        "D1": round(dd["D1"], 1),
        "D2": round(dd["D2"], 1),
        "D3": round(dd["D3"], 1),
        "D4": round(dd["D4"], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["faithful", "strict"], default="strict")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(REPO / "data/persona_opt/pgepa_runs/brinks_naturalness.json"),
    )
    args = ap.parse_args()

    M.set_mode(args.mode)

    print(f"[human] loading Brinks conversations from {BRINKS_HUMAN}...")
    h_turns = brinks_human_turn_lists()
    print(
        f"[human] {len(h_turns)} conversations -> extracting features (mode={args.mode})..."
    )
    Hx_raw = feature_matrix(h_turns)
    pooled = {k: float(np.nanmean(Hx_raw[:, i])) for i, k in enumerate(M.METRIC_ORDER)}
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"
    print(f"[human] {len(Hx_raw)} valid interactions, {pca_name}")
    print()

    results = []

    if NO_PERSONA.exists():
        print("[no_persona] scoring...")
        r = score_one(
            "no_persona", NO_PERSONA, Hx_raw, pooled, mu, sd, pca, pca_name, args.k
        )
        results.append(r)

    for pid, label in CELLS:
        path = FULLSET_DIR / pid / "telecom" / "results.json"
        if not path.exists():
            print(f"[{label}] results not found at {path}")
            continue
        print(f"[{label}] scoring...")
        r = score_one(label, path, Hx_raw, pooled, mu, sd, pca, pca_name, args.k)
        results.append(r)

    print()
    hdr = f"{'label':<24}{'P(nat)':<8}{'R(cov)':<8}{'F1':<7}{'behMean':<9}{'D1':<6}{'D2':<6}{'D3':<6}{'D4':<6}{'diff':<8}{'n'}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        d = r["difficulty"]
        print(
            f"{r['label']:<24}{r['P_geom_naturalness']:<8}{r['R_geom_coverage']:<8}{r['F1']:<7}"
            f"{r['behMean']:<9}{r['D1']:<6}{r['D2']:<6}{r['D3']:<6}{r['D4']:<6}"
            f"{d if d is not None else '?':<8}{r['n_interactions']}"
        )

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
