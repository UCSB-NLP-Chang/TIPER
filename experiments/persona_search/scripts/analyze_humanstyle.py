#!/usr/bin/env python3
"""Compare humanstyle-augmented vs original persona behavioral metrics.

Computes turn-taking stats (Wds/trn, Short%, turn count) and sim2real Dice
for humanstyle_exp/ vs cert_alaska_v5/ data on the airline val set.

Usage:
  python experiments/persona_search/scripts/analyze_humanstyle.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
SIM = REPO / "data/simulations"
HUMAN_FILE = (
    Path(os.environ.get("PERSONA_DATA_DIR", "/Users/yutao.xie/Downloads"))
    / "tau_bench_tasks_unified.json"
)

SPLIT_FILE = REPO / "experiments/persona_search/configs/airline_alaska_split_v5.json"
SPLIT = json.loads(SPLIT_FILE.read_text())
VAL_IDS = set(SPLIT["val"]["airline"])


def load_sims(path: Path, val_only: bool = True) -> list[dict]:
    r = json.load(open(path))
    sims = r["simulations"]
    if val_only:
        sims = [s for s in sims if str(s["task_id"]) in VAL_IDS]
    return sims


def user_turn_stats(sims: list[dict]) -> dict:
    """Compute turn-taking statistics for user messages across simulations."""
    all_words = []
    turn_counts = []
    for s in sims:
        msgs = s.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        content_list = []
        for m in user_msgs:
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            content_list.append(c)
        wc = [len(c.split()) for c in content_list]
        all_words.extend(wc)
        turn_counts.append(len(user_msgs))

    if not all_words:
        return {}

    short = sum(1 for w in all_words if w <= 5)
    le15 = sum(1 for w in all_words if w <= 15)
    le30 = sum(1 for w in all_words if w <= 30)
    n = len(all_words)

    return {
        "n_sims": len(sims),
        "avg_turns": sum(turn_counts) / len(turn_counts),
        "avg_wds": sum(all_words) / n,
        "median_wds": sorted(all_words)[n // 2],
        "short_pct": short / n * 100,
        "le15_pct": le15 / n * 100,
        "le30_pct": le30 / n * 100,
        "avg_reward": sum(s.get("reward_info", {}).get("reward", 0) for s in sims)
        / len(sims),
    }


def human_airline_stats() -> dict:
    """Compute stats from real human airline data."""
    human = json.load(open(HUMAN_FILE))
    airline = {k: v for k, v in human.items() if k.startswith("airline")}

    all_words = []
    turn_counts = []
    for sid, sess in airline.items():
        conv = sess.get("conversation", [])
        user_msgs = [m for m in conv if m.get("role") == "user"]
        turn_counts.append(len(user_msgs))
        for m in user_msgs:
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            all_words.append(len(c.split()))

    short = sum(1 for w in all_words if w <= 5)
    le15 = sum(1 for w in all_words if w <= 15)
    n = len(all_words)

    return {
        "n_sessions": len(airline),
        "avg_turns": sum(turn_counts) / len(turn_counts),
        "avg_wds": sum(all_words) / n,
        "median_wds": sorted(all_words)[n // 2],
        "short_pct": short / n * 100,
        "le15_pct": le15 / n * 100,
    }


def main():
    # Define comparison pairs: (label, path)
    pairs = []

    # Human reference
    print("=" * 90)
    print(
        f"{'Condition':<30} {'Sims':>5} {'Turns':>6} {'Wds/t':>6} {'Med':>4} {'Short%':>7} {'≤15w%':>6} {'Reward':>7}"
    )
    print("-" * 90)

    h = human_airline_stats()
    print(
        f"{'HUMAN (airline)':<30} {h['n_sessions']:>5} {h['avg_turns']:>6.1f} {h['avg_wds']:>6.1f} {h['median_wds']:>4} {h['short_pct']:>6.1f}% {h['le15_pct']:>5.1f}%    {'N/A':>5}"
    )

    print("-" * 90)

    # Original persona runs (from cert v5 and fullset)
    sources = [
        ("orig:no_persona", SIM / "fullset/airline/shared_no_persona/results.json"),
        ("orig:alaska_01_seed", SIM / "cert_alaska01_v5/seed/results.json"),
        ("orig:alaska_01_opt", SIM / "cert_alaska01_v5/optim/results.json"),
        ("orig:alaska_03_seed", SIM / "cert_alaska03_v5/seed/results.json"),
        ("orig:alaska_03_opt", SIM / "cert_alaska03_v5/optim/results.json"),
    ]

    # Humanstyle v1 runs
    hs_sources = [
        ("v1:alaska_01_seed", SIM / "humanstyle_exp/alaska_01_seed_hs/results.json"),
        ("v1:alaska_01_best", SIM / "humanstyle_exp/alaska_01_best_hs/results.json"),
        ("v1:alaska_03_seed", SIM / "humanstyle_exp/alaska_03_seed_hs/results.json"),
        ("v1:alaska_03_best", SIM / "humanstyle_exp/alaska_03_best_hs/results.json"),
    ]

    # Humanstyle v2 runs (style-only)
    v2_sources = [
        ("v2:alaska_01_seed", SIM / "humanstyle_exp/alaska_01_seed_hsv2/results.json"),
        ("v2:alaska_01_best", SIM / "humanstyle_exp/alaska_01_best_hsv2/results.json"),
        ("v2:alaska_03_seed", SIM / "humanstyle_exp/alaska_03_seed_hsv2/results.json"),
        ("v2:alaska_03_best", SIM / "humanstyle_exp/alaska_03_best_hsv2/results.json"),
    ]

    for label, path in sources + hs_sources + v2_sources:
        if not path.exists():
            print(f"{label:<30} {'(not found)':>5}")
            continue
        sims = load_sims(path, val_only=True)
        if not sims:
            print(f"{label:<30} {'(no val sims)':>5}")
            continue
        st = user_turn_stats(sims)
        print(
            f"{label:<30} {st['n_sims']:>5} {st['avg_turns']:>6.1f} {st['avg_wds']:>6.1f} {st['median_wds']:>4} {st['short_pct']:>6.1f}% {st['le15_pct']:>5.1f}% {st['avg_reward']:>7.3f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
