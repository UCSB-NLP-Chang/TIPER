#!/usr/bin/env python3
"""
Stage 1A: Apply deterministic rule-based screeners to all decision pairs.

Outputs a screen_score (0.0-1.0) for each pair. Pairs with score >= 0.3
are candidates for Stage 2 labeling. model_triggered pairs are auto-candidates.
Boundary pairs (±1 turn from a trigger) are also auto-candidates.

Conversation history is loaded on demand from results.json files.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_utils import clear_cache, get_conversation_up_to
from rule_screeners import get_screener

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/decision_pairs.jsonl"
OUT_FILE = DATA_ROOT / "data/skill_eval/rule_screened_pairs.jsonl"

SKILLS_NEEDING_HISTORY = {"frustration-escalation", "transfer-pushback"}


def find_trigger_boundary_pairs(pairs: list[dict]) -> set[str]:
    triggered_keys: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        if p["model_triggered"]:
            sim_task = f"{p['sim_dir']}__{p['task_id']}__{p['skill']}"
            triggered_keys[sim_task].append(p["turn_idx"])

    boundary_ids = set()
    for p in pairs:
        sim_task = f"{p['sim_dir']}__{p['task_id']}__{p['skill']}"
        if sim_task in triggered_keys:
            for trig_turn in triggered_keys[sim_task]:
                if abs(p["turn_idx"] - trig_turn) == 1 and not p["model_triggered"]:
                    boundary_ids.add(p["pair_id"])
    return boundary_ids


def main():
    pairs = []
    with open(IN_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Stage 1A: Rule-Based Screening")
    print(f"{'=' * 50}")
    print(f"Input pairs: {len(pairs)}")

    boundary_ids = find_trigger_boundary_pairs(pairs)
    print(f"Boundary pairs (±1 turn from trigger): {len(boundary_ids)}")

    screener_cache: dict[str, object] = {}
    stats = defaultdict(int)
    score_bins = defaultdict(lambda: defaultdict(int))

    current_sim_dir = None
    results = []
    batch_count = 0

    for i, pair in enumerate(pairs):
        skill = pair["skill"]
        pair_id = pair["pair_id"]
        sim_dir = pair["sim_dir"]

        if sim_dir != current_sim_dir:
            if batch_count > 0 and batch_count % 20 == 0:
                clear_cache()
            current_sim_dir = sim_dir
            batch_count += 1

        if skill not in screener_cache:
            screener_cache[skill] = get_screener(skill)
        screener = screener_cache[skill]

        output_fields = {k: v for k, v in pair.items() if k != "model_triggered"}

        if pair["model_triggered"]:
            result = {
                **output_fields,
                "screen_score": 1.0,
                "screen_source": "auto_trigger",
                "screen_evidence": ["model_triggered=true"],
                "matched_rules": ["auto_trigger"],
            }
            stats["auto_trigger"] += 1
        elif pair_id in boundary_ids:
            result = {
                **output_fields,
                "screen_score": 0.8,
                "screen_source": "auto_boundary",
                "screen_evidence": ["±1 turn from model trigger"],
                "matched_rules": ["auto_boundary"],
            }
            stats["auto_boundary"] += 1
        else:
            conversation_history = []
            if skill in SKILLS_NEEDING_HISTORY:
                conversation_history = get_conversation_up_to(
                    pair["sim_dir"], pair["task_id"], pair["msg_idx"]
                )

            sr = screener.screen(
                agent_message=pair["agent_message"],
                conversation_history=conversation_history,
                capability=pair["capability"],
                emotion=pair["emotion"],
                domain=pair["domain"],
                turn_idx=pair["turn_idx"],
                n_user_turns=pair["n_user_turns"],
            )
            result = {
                **output_fields,
                "screen_source": "rule",
                **sr.to_dict(),
            }
            stats["rule_screened"] += 1

        score = result["screen_score"]
        if score >= 0.3:
            stats["candidate"] += 1
            stats[f"candidate_{skill}"] += 1
        else:
            stats["negative"] += 1
            stats[f"negative_{skill}"] += 1

        bin_label = f"{score:.1f}" if score < 1.0 else "1.0"
        score_bins[skill][bin_label] += 1
        score_bins["_all"][bin_label] += 1

        results.append(result)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(pairs)}...")

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Auto-trigger: {stats['auto_trigger']}")
    print(f"  Auto-boundary: {stats['auto_boundary']}")
    print(f"  Rule-screened: {stats['rule_screened']}")
    print(f"  Candidates (score >= 0.3): {stats['candidate']}")
    print(f"  Negatives (score < 0.3): {stats['negative']}")
    if results:
        print(f"  Candidate rate: {stats['candidate'] / len(results) * 100:.1f}%")

    print(f"\nPer-skill candidates:")
    for k, v in sorted(stats.items()):
        if k.startswith("candidate_"):
            skill = k.removeprefix("candidate_")
            neg = stats.get(f"negative_{skill}", 0)
            total = v + neg
            print(f"  {skill}: {v}/{total} ({v / total * 100:.1f}%)")

    print(f"\nScore distribution (all skills):")
    for bin_label in sorted(score_bins["_all"].keys()):
        count = score_bins["_all"][bin_label]
        pct = count / len(results) * 100 if results else 0
        bar = "█" * max(1, int(pct))
        print(f"  {bin_label}: {count:>6} ({pct:>5.1f}%) {bar}")

    print(f"\nOutput: {OUT_FILE}")
    print(f"File size: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
