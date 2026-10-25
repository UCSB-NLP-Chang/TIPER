#!/usr/bin/env python3
"""JSD lower bound: split real human (Brinks v2) data into 5 even groups and
compute pairwise JSD between them.

Both sides are genuine human samples, so the resulting JSD is the *irreducible
sampling-noise floor* — the JSD you get when there is no sim/human gap at all.
Any sim-vs-human JSD can only be read as "close to human" if it sits near this
lower bound; values well above it indicate a real behavioral gap.

Reuses naturalness_reference.compute_jsd_vs_human so the metric is identical to
the one used to score personas (19-metric mean, quantile binning, Laplace
alpha=0.5, base=2).

Usage:
  python experiments/persona_search/scripts/jsd_human_lowerbound.py
  python experiments/persona_search/scripts/jsd_human_lowerbound.py --splits 20 --groups 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from naturalness_reference import compute_jsd_vs_human, load_human_vecs  # noqa: E402

METRIC_ORDER_PATH = REPO / "sim2real_metrics.py"


def split_and_pairwise(human_vecs, groups: int, rng) -> list[float]:
    idx = np.arange(len(human_vecs))
    rng.shuffle(idx)
    chunks = np.array_split(idx, groups)
    jsds = []
    for i in range(groups):
        for j in range(i + 1, groups):
            a = [human_vecs[k] for k in chunks[i]]
            b = [human_vecs[k] for k in chunks[j]]
            jsds.append(compute_jsd_vs_human(a, b))
    return jsds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument(
        "--splits",
        type=int,
        default=20,
        help="random splits to average over for a stable estimate",
    )
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    human = load_human_vecs()
    print(f"Loaded {len(human)} human conversations (Brinks v2)")
    per_group = len(human) // a.groups
    print(
        f"Splitting into {a.groups} groups x {a.splits} random splits "
        f"(~{per_group} convs/group, {a.groups * (a.groups - 1) // 2} pairs/split)"
    )

    rng = np.random.default_rng(a.seed)
    all_jsds: list[float] = []
    for s in range(a.splits):
        jsds = split_and_pairwise(human, a.groups, rng)
        all_jsds.extend(jsds)
        print(
            f"  split {s + 1:2d}/{a.splits}: "
            f"mean={np.mean(jsds):.4f} min={np.min(jsds):.4f} "
            f"max={np.max(jsds):.4f}"
        )

    arr = np.array(all_jsds)
    print()
    print("=" * 60)
    print(
        f"JSD LOWER BOUND (human-vs-human, {a.groups} groups, "
        f"{a.splits} splits, {len(arr)} pairs total)"
    )
    print("=" * 60)
    print(f"  mean   = {arr.mean():.4f}")
    print(f"  median = {np.median(arr):.4f}")
    print(f"  std    = {arr.std():.4f}")
    print(f"  min    = {arr.min():.4f}")
    print(f"  max    = {arr.max():.4f}")
    print(f"  p5     = {np.percentile(arr, 5):.4f}")
    print(f"  p95    = {np.percentile(arr, 95):.4f}")
    print()
    print("Reference (from GEPA_NATURALNESS_JSD.md, tau2 19-metric JSD):")
    print("  no_persona   0.1502")
    print("  mean seed    0.1373")
    print("  mean best    0.1372")
    print("Interpretation: sim JSD near the lower bound => indistinguishable")
    print("from human at this sample size; well above => real gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
