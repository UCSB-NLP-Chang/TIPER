#!/usr/bin/env python3
"""
Stage 0.5: Expand decision_points into decision_pairs.

Each decision_point × eligible_skill becomes one pair.
The pair records whether the model triggered that skill at this turn.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/decision_points.jsonl"
OUT_FILE = DATA_ROOT / "data/skill_eval/decision_pairs.jsonl"


def main():
    pairs = []
    stats = defaultdict(int)

    with open(IN_FILE) as f:
        for line in f:
            dp = json.loads(line)
            dp_id = dp["dp_id"]
            dynamic_skills = dp["dynamic_skills"]
            triggered = set(dp.get("model_triggered_skills") or [])

            for skill in dynamic_skills:
                pair_id = f"{dp_id}__{skill}"
                model_triggered = skill in triggered

                pair = {
                    "pair_id": pair_id,
                    "dp_id": dp_id,
                    "skill": skill,
                    "model_triggered": model_triggered,
                    "sim_dir": dp["sim_dir"],
                    "task_id": dp["task_id"],
                    "turn_idx": dp["turn_idx"],
                    "msg_idx": dp["msg_idx"],
                    "n_user_turns": dp["n_user_turns"],
                    "agent_message": dp["agent_message"],
                    "preset": dp["preset"],
                    "preset_group": dp["preset_group"],
                    "domain": dp["domain"],
                    "capability": dp["capability"],
                    "emotion": dp["emotion"],
                    "reward": dp["reward"],
                }
                pairs.append(pair)
                stats["total_pairs"] += 1
                stats[f"skill_{skill}"] += 1
                if model_triggered:
                    stats["model_triggered"] += 1
                    stats[f"triggered_{skill}"] += 1

    with open(OUT_FILE, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Stage 0.5: Build Decision Pairs")
    print(f"{'=' * 50}")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Model-triggered pairs: {stats['model_triggered']}")
    if stats["total_pairs"] > 0:
        print(
            f"  trigger rate: {stats['model_triggered'] / stats['total_pairs'] * 100:.2f}%"
        )
    print()

    print("Pairs by skill:")
    for k, v in sorted(stats.items()):
        if k.startswith("skill_"):
            skill = k.removeprefix("skill_")
            t = stats.get(f"triggered_{skill}", 0)
            rate = t / v * 100 if v else 0
            print(f"  {skill}: {v} pairs, {t} triggered ({rate:.2f}%)")

    print(f"\nOutput: {OUT_FILE}")
    print(f"File size: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
