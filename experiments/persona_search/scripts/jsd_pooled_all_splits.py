#!/usr/bin/env python3
"""Pool ALL rollout data per persona (train + val + test + cert) and compute
JSD vs human. Maximizes n per persona -> lowers the noise floor -> cleaner
signal than any single split.

For each island (ev2v4_brinks_XX / ev2v4_p0003) we map every sim dir to the
persona idx it evaluated:
  g00_seed[_train]            -> idx0 (seed)
  g{N}_k00[_full_diff]        -> the child proposed at gen N (candidate gen==N)
  g{N}_parent_{fill,trace}    -> generations[gen==N].parent
plus cert dirs:
  ev2v4_cert/seed_XX          -> seed
  ev2v4_cert/bXX_{best,top1}  -> best
  ev2v4_cert/p0003_{seed,best}-> p0003 seed/best

Then pool user-turn vectors across all those results.json, compute JSD vs the
full Brinks-v2 human reference, and compare to the noise floor at that pooled n.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
REPO = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT", str(REPO_ROOT.parent / "tau2bench-data")
    )
)
sys.path.insert(0, str(REPO_ROOT / "experiments/persona_search/scripts"))
from naturalness_reference import (  # noqa: E402
    compute_jsd_vs_human,
    extract_sim_vecs,
    load_human_vecs,
)

STATE_ROOT = REPO / "data/persona_opt/pgepa_runs/ev2_3islands_v4"
SIM_ROOT = REPO / "data/simulations/gepa_opt"
CERT_ROOT = REPO / "data/simulations/ev2v4_cert"
ISLANDS = ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"]


def gen_logs(state):
    return {g["gen"]: g for g in state.get("generations", [])}


def child_idx_for_gen(state, gen):
    """Candidate idx of the child proposed at generation `gen` (gen>=1)."""
    for c in state["candidates"]:
        if c.get("gen") == gen:
            return c.get("idx")
    return None


def map_dir_to_idx(name: str, state) -> int | None:
    """Map a sim dir basename -> persona idx it evaluated."""
    logs = gen_logs(state)
    parts = name.split("_")
    # g00_seed / g00_seed_train
    if "_g00_seed" in name:
        return 0
    # g{N}_k00[_full_diff] -> child of gen N
    if "_k00" in name:
        for p in parts:
            if p.startswith("g") and p[1:].isdigit():
                gen = int(p[1:])
                return child_idx_for_gen(state, gen)
    # g{N}_parent_{fill,trace} -> parent idx at gen N
    if "_parent_fill" in name or "_parent_trace" in name:
        for p in parts:
            if p.startswith("g") and p[1:].isdigit():
                gen = int(p[1:])
                return logs.get(gen, {}).get("parent")
    return None


def vecs_from(results_path):
    try:
        d = json.load(open(results_path))
    except Exception:
        return []
    return extract_sim_vecs(d)


def is_val_or_test_dir(name: str) -> bool:
    """Keep only val (full) + test dirs; drop train minibatch / parent re-eval."""
    if "_train" in name or "_parent_fill" in name or "_parent_trace" in name:
        return False
    # _k00 alone = train minibatch (5 tasks); _k00_full_diff = full val eval.
    if "_k00" in name and "_full_diff" not in name:
        return False
    # g00_seed = val (keep); g00_seed_train already excluded above.
    return True


def pool_persona(island: str, target_idx: int, state) -> list[dict]:
    """User-turn vectors for persona `target_idx` across val + test only
    (excludes train minibatch and parent re-eval dirs for symmetry)."""
    base = f"ev2v4_{island}"
    vecs: list[dict] = []
    # GEPA run sim dirs (val only: g00_seed, gNN_k00_full_diff)
    for d in glob.glob(str(SIM_ROOT / f"{base}_*")):
        name = os.path.basename(d)
        if not is_val_or_test_dir(name):
            continue
        idx = map_dir_to_idx(name, state)
        if idx != target_idx:
            continue
        for rp in glob.glob(str(Path(d) / "eval_*/telecom/results.json")):
            vecs.extend(vecs_from(rp))
    # cert test
    cert_map = {
        "brinks_00": (["seed_00"], ["b00_top1", "b00_top2"]),
        "brinks_02": ([], ["b02_top1"]),
        "brinks_03": ([], ["b03_top1", "b03_top2"]),
        "brinks_04": (["seed_04"], ["b04_best", "b04_top1"]),
        "p0003": (["p0003_seed"], ["p0003_best", "p0003_idx8"]),
    }
    seeds, bests = cert_map.get(island, ([], []))
    is_seed = target_idx == 0
    for cn in seeds if is_seed else bests:
        rp = CERT_ROOT / cn / "results.json"
        if rp.exists():
            vecs.extend(vecs_from(rp))
    return vecs


def floor_at(n, human, splits=50):
    if n >= len(human):
        n = len(human) - 1
    rng = np.random.default_rng(42)
    jsds = []
    for _ in range(splits):
        idx = np.arange(len(human))
        rng.shuffle(idx)
        a = [human[k] for k in idx[:n]]
        b = [human[k] for k in idx[n:]]
        jsds.append(compute_jsd_vs_human(a, b))
    return float(np.mean(jsds))


def main() -> int:
    human = load_human_vecs()
    print(f"Human ref: {len(human)} Brinks-v2 conversations")
    print()
    print(
        f"{'island':<10}{'persona':<10}{'n_pooled':>9}{'JSD':>8}{'floor':>8}{'ratio':>8}  verdict"
    )
    print("-" * 70)
    for isl in ISLANDS:
        sp = STATE_ROOT / f"ev2v4_{isl}" / "pgepa_state.json"
        if not sp.exists():
            print(f"{isl:<10}(no state)")
            continue
        state = json.load(open(sp))
        best_idx = state.get("best_idx")
        cands = {c.get("idx"): c for c in state["candidates"]}
        for label, idx in [("seed", 0), ("best", best_idx)]:
            if idx is None or idx not in cands:
                continue
            v = pool_persona(isl, idx, state)
            if not v:
                print(f"{isl:<10}{label:<10}{'(no data)':>9}")
                continue
            j = compute_jsd_vs_human(v, human)
            f = floor_at(len(v), human)
            r = j / f
            tag = "AT floor" if r < 1.3 else ("REAL gap" if r > 2 else "marginal")
            print(f"{isl:<10}{label:<10}{len(v):>9}{j:>8.4f}{f:>8.4f}{r:>7.2f}x  {tag}")
        # seed->best delta
        vs = pool_persona(isl, 0, state)
        vb = pool_persona(isl, best_idx, state) if best_idx is not None else []
        if vs and vb:
            js, jb = compute_jsd_vs_human(vs, human), compute_jsd_vs_human(vb, human)
            f = floor_at(min(len(vs), len(vb)), human)
            d = jb - js
            below = "Δ<floor(噪声内)" if abs(d) < f else "Δ>floor(真实变化)"
            print(f"{'':<10}Δbest-seed{'':>9}{d:>+8.4f}{f:>8.4f}        {below}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
