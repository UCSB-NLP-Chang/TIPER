#!/usr/bin/env python3
"""Cross-domain naturalness JSD: sim (seed/opt) vs real customer data.

For each domain, extracts 19-dim Sim2Real behavioral metrics from:
  1. Real customer corpus (VISITOR turns)
  2. Fullset simulation results (user turns)

Then computes per-metric JSD (8-bin quantile, alpha=0.5 smoothing)
and reports mean JSD + human-vs-human noise floor.

Usage:
    python experiments/persona_search/scripts/jsd_cross_domain.py
    python experiments/persona_search/scripts/jsd_cross_domain.py --domain telecom
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(REPO))
import sys

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
from sim2real_metrics import METRIC_ORDER, metrics_for_interaction  # noqa: E402

JSD_BINS = 8
JSD_ALPHA = 0.5
FLOOR_SPLITS = 50

CORPORA = {
    "telecom": {
        "path": os.path.join(
            os.environ.get(
                "PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"
            ),
            "virtual_agent",
            "user_simulator_dataset_v2.json",
        ),
        "customer_speaker": "VISITOR",
        "min_turns": 2,
    },
    "airline": {
        "path": os.path.join(
            os.environ.get(
                "PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"
            ),
            "virtual_agent",
            "llm_clean_alaska_air.json",
        ),
        "customer_speaker": "VISITOR",
        "min_turns": 2,
    },
    "retail": {
        "path": os.path.join(
            os.environ.get(
                "PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"
            ),
            "virtual_agent",
            "persona_opt_standalone",
            "data_cleaning_outputs",
            "guitar-center",
            "llm_clean_guitar_center.json",
        ),
        "customer_speaker": "VISITOR",
        "min_turns": 2,
    },
    "banking": {
        "path": os.path.join(
            os.environ.get(
                "PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"
            ),
            "virtual_agent",
            "persona_opt_standalone",
            "data_cleaning_outputs",
            "cardworks",
            "llm_clean_cardworks.json",
        ),
        "customer_speaker": "VISITOR",
        "min_turns": 2,
    },
}

FULLSET_CELLS = {
    "telecom": [
        (
            "brinks_00",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom/brinks_00_{cond}/results.json"
            ),
        ),
        (
            "brinks_02",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom/brinks_02_{cond}/results.json"
            ),
        ),
        (
            "brinks_03",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom/brinks_03_{cond}/results.json"
            ),
        ),
        (
            "brinks_04",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom/brinks_04_{cond}/results.json"
            ),
        ),
        (
            "p0003",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom_p0003/p0003_{cond}/results.json"
            ),
        ),
        (
            "no_persona",
            str(
                DATA_ROOT
                / "data/simulations/fullset/telecom/shared_no_persona/results.json"
            ),
        ),
    ],
    "airline": [
        (
            "alaska_01_v5",
            str(
                DATA_ROOT / "data/simulations/cert_alaska01_v5/{cond_cert}/results.json"
            ),
        ),
        (
            "alaska_03_v5",
            str(
                DATA_ROOT / "data/simulations/cert_alaska03_v5/{cond_cert}/results.json"
            ),
        ),
        (
            "alaska_00",
            str(
                DATA_ROOT
                / "data/simulations/fullset/airline/alaska_00_{cond}/results.json"
            ),
        ),
        (
            "alaska_01",
            str(
                DATA_ROOT
                / "data/simulations/fullset/airline/alaska_01_{cond}/results.json"
            ),
        ),
        (
            "no_persona",
            str(
                DATA_ROOT
                / "data/simulations/fullset/airline/shared_no_persona/results.json"
            ),
        ),
    ],
    "retail": [
        (
            "gc_02",
            str(
                DATA_ROOT / "data/simulations/fullset/retail/gc_02_{cond}/results.json"
            ),
        ),
        (
            "gc_03",
            str(
                DATA_ROOT / "data/simulations/fullset/retail/gc_03_{cond}/results.json"
            ),
        ),
        (
            "gc_04",
            str(
                DATA_ROOT / "data/simulations/fullset/retail/gc_04_{cond}/results.json"
            ),
        ),
        (
            "no_persona",
            str(
                DATA_ROOT
                / "data/simulations/fullset/retail/shared_no_persona/results.json"
            ),
        ),
    ],
    "banking": [
        (
            "cw_03",
            str(
                DATA_ROOT / "data/simulations/fullset/banking/cw_03_{cond}/results.json"
            ),
        ),
        (
            "cw_04",
            str(
                DATA_ROOT / "data/simulations/fullset/banking/cw_04_{cond}/results.json"
            ),
        ),
        (
            "no_persona",
            str(
                DATA_ROOT
                / "data/simulations/fullset/banking/shared_no_persona/results.json"
            ),
        ),
    ],
}


def extract_human_vecs(
    corpus_path: str, customer_speaker: str, min_turns: int
) -> list[dict]:
    convs = json.loads(Path(corpus_path).read_text())
    vecs = []
    for conv in convs:
        turns = []
        for m in conv.get("messages", []):
            if m.get("speaker") == customer_speaker:
                t = m.get("text", "").strip()
                if t:
                    turns.append(t)
        if len(turns) >= min_turns:
            v = metrics_for_interaction(turns)
            if v is not None:
                vecs.append(v)
    return vecs


def extract_sim_vecs(results_path: str) -> list[dict]:
    try:
        d = json.loads(Path(results_path).read_text())
    except Exception:
        return []
    vecs = []
    for s in d.get("simulations", []):
        if s.get("termination_reason") == "infra_error" or not s.get("messages"):
            continue
        ut = [
            m["content"]
            for m in s["messages"]
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if len(ut) >= 2:
            v = metrics_for_interaction(ut)
            if v is not None:
                vecs.append(v)
    return vecs


def compute_jsd(sim_vecs: list[dict], human_vecs: list[dict]) -> float:
    if not sim_vecs:
        return 1.0
    jsd_vals = []
    for metric in METRIC_ORDER:
        pooled = np.array(
            [v[metric] for v in sim_vecs] + [v[metric] for v in human_vecs], dtype=float
        )
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, JSD_BINS + 1)))
        if edges.size < 2:
            continue
        vals_s = np.array([v[metric] for v in sim_vecs], dtype=float)
        vals_h = np.array([v[metric] for v in human_vecs], dtype=float)
        counts_s, _ = np.histogram(vals_s, bins=edges)
        counts_h, _ = np.histogram(vals_h, bins=edges)
        p_s = counts_s.astype(float) + JSD_ALPHA
        p_s /= p_s.sum()
        p_h = counts_h.astype(float) + JSD_ALPHA
        p_h /= p_h.sum()
        d = jensenshannon(p_s, p_h, base=2)
        if not np.isnan(d):
            jsd_vals.append(d)
    return float(np.mean(jsd_vals)) if jsd_vals else 1.0


def compute_floor(human_vecs: list[dict], n: int) -> float:
    if n >= len(human_vecs):
        n = len(human_vecs) - 1
    rng = np.random.default_rng(42)
    jsds = []
    for _ in range(FLOOR_SPLITS):
        idx = np.arange(len(human_vecs))
        rng.shuffle(idx)
        a = [human_vecs[k] for k in idx[:n]]
        b = [human_vecs[k] for k in idx[n:]]
        jsds.append(compute_jsd(a, b))
    return float(np.mean(jsds))


def resolve_path(template: str, cond: str) -> str:
    cond_cert = "seed" if cond == "seed" else "optim"
    return str(REPO / template.format(cond=cond, cond_cert=cond_cert))


def run_domain(domain: str, human_cache: dict) -> None:
    cfg = CORPORA[domain]
    print(f"\n{'=' * 80}")
    print(f"  {domain.upper()}")
    print(f"{'=' * 80}")

    if domain in human_cache:
        human_vecs = human_cache[domain]
    else:
        print(f"  Loading human corpus: {os.path.basename(cfg['path'])}...")
        human_vecs = extract_human_vecs(
            cfg["path"], cfg["customer_speaker"], cfg["min_turns"]
        )
        human_cache[domain] = human_vecs
    print(f"  Human conversations: {len(human_vecs)}")

    cells = FULLSET_CELLS[domain]
    print(
        f"\n  {'Island':<18} {'Cond':<8} {'N sims':>8} {'JSD':>8} {'Floor':>8} {'Ratio':>8}"
    )
    print(f"  {'-' * 66}")

    for island, template in cells:
        if island == "no_persona":
            path = str(REPO / template)
            if not os.path.exists(path):
                continue
            sim_vecs = extract_sim_vecs(path)
            if not sim_vecs:
                continue
            jsd = compute_jsd(sim_vecs, human_vecs)
            floor = compute_floor(human_vecs, len(sim_vecs))
            ratio = jsd / floor if floor > 0 else float("inf")
            print(
                f"  {island:<18} {'base':<8} {len(sim_vecs):>8} {jsd:>8.4f} {floor:>8.4f} {ratio:>7.2f}x"
            )
        else:
            for cond in ["seed", "opt"]:
                path = resolve_path(template, cond)
                if not os.path.exists(path):
                    continue
                sim_vecs = extract_sim_vecs(path)
                if not sim_vecs:
                    continue
                jsd = compute_jsd(sim_vecs, human_vecs)
                floor = compute_floor(human_vecs, len(sim_vecs))
                ratio = jsd / floor if floor > 0 else float("inf")
                print(
                    f"  {island:<18} {cond:<8} {len(sim_vecs):>8} {jsd:>8.4f} {floor:>8.4f} {ratio:>7.2f}x"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", help="Run for a specific domain only")
    args = ap.parse_args()

    domains = (
        [args.domain] if args.domain else ["telecom", "airline", "retail", "banking"]
    )
    human_cache = {}
    for domain in domains:
        if domain not in CORPORA:
            print(f"Unknown domain: {domain}")
            continue
        run_domain(domain, human_cache)


if __name__ == "__main__":
    main()
