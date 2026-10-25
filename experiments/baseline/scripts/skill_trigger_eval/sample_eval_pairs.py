#!/usr/bin/env python3
"""
Stage 1.5: Budget-controlled sampling + blind audit.

Total budget: ≤5,000 pairs for Stage 2 (GPT-5.5 labeling).

Budget allocation strategy (balanced for precision + recall):
- auto_trigger: ALL model-triggered pairs (critical for precision)
- fn_candidates: rule-flagged (score>=0.7) but model-untriggered (critical for recall)
- boundary: ±1 turn from triggers (timing error detection)
- blind_audit: random from negatives (population estimation)
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/screened_pairs.jsonl"
OUT_FILE = DATA_ROOT / "data/skill_eval/sampled_pairs.jsonl"

TOTAL_BUDGET = 5000
BLIND_PER_SKILL = 200
HIGH_SCORE_THRESHOLD = 0.7
BLIND_SCORE_THRESHOLD = 0.1

CANONICAL_SKILLS = [
    "frustration-escalation",
    "jargon-hit",
    "resolution",
    "transfer-pushback",
]


def canonicalize_skill(skill: str) -> str:
    if skill == "resolution-banking":
        return "resolution"
    return skill


def main():
    seed = 42
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed = int(sys.argv[idx + 1])
    rng = random.Random(seed)

    blind_only = "--blind-only" in sys.argv

    pairs = []
    with open(IN_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    # Load model_triggered info from decision_pairs
    dp_file = DATA_ROOT / "data/skill_eval/decision_pairs.jsonl"
    model_triggered_set = set()
    with open(dp_file) as f:
        for line in f:
            p = json.loads(line)
            if p["model_triggered"]:
                model_triggered_set.add(p["pair_id"])

    # Categorize pairs
    auto_trigger = []  # model-triggered → precision measurement
    fn_candidates = []  # high score, NOT model-triggered → FN detection
    boundary = []  # ±1 turn from trigger, NOT model-triggered
    medium_candidates = []  # score in [0.3, 0.7), NOT model-triggered
    negatives_by_skill: dict[str, list] = defaultdict(list)

    for pair in pairs:
        score = pair["screen_score"]
        source = pair.get("screen_source", "")
        canon_skill = canonicalize_skill(pair["skill"])
        is_model_triggered = pair["pair_id"] in model_triggered_set

        if is_model_triggered or source == "auto_trigger":
            auto_trigger.append(pair)
        elif source == "auto_boundary":
            boundary.append(pair)
        elif score >= HIGH_SCORE_THRESHOLD:
            fn_candidates.append(pair)
        elif score >= 0.3:
            medium_candidates.append(pair)
        else:
            negatives_by_skill[canon_skill].append(pair)

    print(f"Pool sizes:")
    print(f"  auto_trigger: {len(auto_trigger)}")
    print(f"  fn_candidates (score>=0.7, not triggered): {len(fn_candidates)}")
    print(f"  boundary (±1 turn, not triggered): {len(boundary)}")
    print(f"  medium_candidates (0.3-0.7): {len(medium_candidates)}")
    print(f"  negatives: {sum(len(v) for v in negatives_by_skill.values())}")

    # Blind audit sampling
    blind_audit = []
    for canon_skill in CANONICAL_SKILLS:
        pool = [
            p
            for p in negatives_by_skill[canon_skill]
            if p["screen_score"] < BLIND_SCORE_THRESHOLD
        ]
        n_sample = min(BLIND_PER_SKILL, len(pool))
        if pool:
            sampled = rng.sample(pool, n_sample)
            inclusion_prob = n_sample / len(pool)
            for p in sampled:
                p["sample_source"] = "blind_audit"
                p["inclusion_probability"] = round(inclusion_prob, 6)
                p["budget_tier"] = "blind"
                p["canonical_skill"] = canon_skill
                blind_audit.append(p)
        print(f"  Blind {canon_skill}: {n_sample}/{len(pool)}")

    if blind_only:
        with open(OUT_FILE, "w") as f:
            for p in blind_audit:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nBlind-only: {len(blind_audit)} pairs → {OUT_FILE}")
        return

    # Candidate budget after blind
    candidate_budget = TOTAL_BUDGET - len(blind_audit)
    selected = []
    remaining = candidate_budget

    def _add(pool, tier, budget):
        """Add up to `budget` pairs from pool, return actual count added."""
        nonlocal remaining
        n = min(len(pool), budget, remaining)
        if n <= 0:
            return 0
        if n < len(pool):
            chosen = rng.sample(pool, n)
            incl_prob = n / len(pool)
        else:
            chosen = pool
            incl_prob = 1.0
        for p in chosen:
            p["sample_source"] = "candidate"
            p["inclusion_probability"] = round(incl_prob, 6)
            p["budget_tier"] = tier
            p["canonical_skill"] = canonicalize_skill(p["skill"])
            selected.append(p)
        remaining -= n
        return n

    # Priority 1: ALL model-triggered pairs (essential for precision)
    n1 = _add(auto_trigger, "auto_trigger", len(auto_trigger))

    # Priority 2: FN candidates — stratified by skill to ensure coverage
    fn_by_skill = defaultdict(list)
    for p in fn_candidates:
        fn_by_skill[canonicalize_skill(p["skill"])].append(p)

    fn_budget = min(remaining, max(800, remaining // 2))
    per_skill_fn = fn_budget // len(CANONICAL_SKILLS)
    n2 = 0
    for skill in CANONICAL_SKILLS:
        pool = fn_by_skill[skill]
        n2 += _add(pool, "fn_candidate", per_skill_fn)

    # Priority 3: Boundary pairs (timing error detection)
    bnd_budget = min(remaining, 400)
    n3 = _add(boundary, "boundary", bnd_budget)

    # Priority 4: Medium score candidates (fill remaining)
    n4 = _add(medium_candidates, "medium_score", remaining)

    all_sampled = selected + blind_audit

    with open(OUT_FILE, "w") as f:
        for p in all_sampled:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nStage 1.5: Budget-Controlled Sampling")
    print(f"{'=' * 50}")
    print(f"Budget: {TOTAL_BUDGET}")
    print(f"Total sampled: {len(all_sampled)}")
    print(f"  auto_trigger: {n1}")
    print(f"  fn_candidates: {n2}")
    print(f"  boundary: {n3}")
    print(f"  medium_score: {n4}")
    print(f"  blind_audit: {len(blind_audit)}")
    print(f"  remaining unused: {remaining}")

    # Per-skill breakdown
    skill_counts = defaultdict(lambda: defaultdict(int))
    for p in all_sampled:
        skill_counts[p["canonical_skill"]][p["budget_tier"]] += 1
        skill_counts[p["canonical_skill"]]["total"] += 1

    print(f"\nPer-skill breakdown:")
    for skill in CANONICAL_SKILLS:
        c = skill_counts[skill]
        print(
            f"  {skill}: {c['total']} total "
            f"(trig={c.get('auto_trigger', 0)}, "
            f"fn={c.get('fn_candidate', 0)}, "
            f"bnd={c.get('boundary', 0)}, "
            f"med={c.get('medium_score', 0)}, "
            f"blind={c.get('blind', 0)})"
        )

    print(f"\nOutput: {OUT_FILE}")
    print(f"File size: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
