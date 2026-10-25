#!/usr/bin/env python3
"""
Stage 1C: Merge screen results and print statistics.

For now, this just reads rule_screened_pairs.jsonl (Stage 1A output)
and produces screened_pairs.jsonl. If Stage 1B (GPT screening) was run,
it would merge those results too.
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

RULE_FILE = DATA_ROOT / "data/skill_eval/rule_screened_pairs.jsonl"
GPT_FILE = DATA_ROOT / "data/skill_eval/gpt_screened_pairs.jsonl"
OUT_FILE = DATA_ROOT / "data/skill_eval/screened_pairs.jsonl"


def main():
    pairs = []
    with open(RULE_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    # If GPT screening was done, merge those results
    if GPT_FILE.exists():
        gpt_pairs = {}
        with open(GPT_FILE) as f:
            for line in f:
                p = json.loads(line)
                gpt_pairs[p["pair_id"]] = p

        for i, pair in enumerate(pairs):
            if pair["pair_id"] in gpt_pairs:
                gpt = gpt_pairs[pair["pair_id"]]
                if gpt["screen_score"] > pair["screen_score"]:
                    pairs[i] = gpt

        print(f"Merged {len(gpt_pairs)} GPT-screened pairs")

    # Classify
    stats = defaultdict(int)
    for pair in pairs:
        score = pair["screen_score"]
        source = pair["screen_source"]
        skill = pair["skill"]

        if score >= 0.3:
            pair["screen_class"] = "candidate"
            stats["candidate"] += 1
            stats[f"candidate_{skill}"] += 1
        else:
            pair["screen_class"] = "negative"
            stats["negative"] += 1
            stats[f"negative_{skill}"] += 1

        stats[f"source_{source}"] += 1

    with open(OUT_FILE, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Stage 1C: Merge Screen Results")
    print(f"{'=' * 50}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Candidates: {stats['candidate']}")
    print(f"Negatives: {stats['negative']}")
    print(
        f"\nBy source: {dict((k.removeprefix('source_'), v) for k, v in stats.items() if k.startswith('source_'))}"
    )

    print(f"\nOutput: {OUT_FILE}")


if __name__ == "__main__":
    main()
