#!/usr/bin/env python3
"""
Label pairs for precision & recall evaluation.

Precision: 500 pairs from auto_trigger (model triggered), uniform per skill
Recall: 500 pairs from fn_candidate (model didn't trigger, rule says should), uniform per skill
Model: gpt-5.4 with reasoning_effort="high"
"""

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from load_utils import get_conversation_up_to

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/sampled_pairs.jsonl"
OUT_FILE = DATA_ROOT / "data/skill_eval/precision_recall_labeled.jsonl"
RUBRIC_DIR = Path("scripts/skill_trigger_eval/rubrics")

MODEL = "gpt-5.4"
BATCH_SIZE = 50  # Each pair = 5 API calls (maj@5), so 50 pairs = 250 concurrent calls
MAX_RETRIES = 3
SEED = 42

PRECISION_TOTAL = 500
RECALL_TOTAL = 500

RUBRIC_MAPPING = {
    "jargon-hit": "jargon_hit.yaml",
    "frustration-escalation": "frustration_escalation.yaml",
    "transfer-pushback": "transfer_pushback.yaml",
    "resolution": "resolution.yaml",
    "resolution-banking": "resolution.yaml",
}

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


def load_rubric(skill: str) -> str:
    fname = RUBRIC_MAPPING.get(skill)
    if not fname:
        return f"No rubric available for {skill}"
    path = RUBRIC_DIR / fname
    with open(path) as f:
        content = yaml.safe_load(f)
    return yaml.dump(content, default_flow_style=False, allow_unicode=True)


def format_conversation(history: list[dict], max_turns: int = 200) -> str:
    if len(history) > max_turns:
        history = history[-max_turns:]
        prefix = f"[... {len(history) - max_turns} earlier messages truncated ...]\n\n"
    else:
        prefix = ""

    lines = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if content:
            lines.append(f"[{role}]: {content}")
        tc = msg.get("tool_calls")
        if tc:
            for t in tc:
                lines.append(
                    f"[{role} TOOL CALL]: {t.get('name', '')}({json.dumps(t.get('arguments', {}))})"
                )
    return prefix + "\n".join(lines)


def build_prompt(pair: dict, rubric_text: str, conversation_text: str) -> str:
    return f"""You are evaluating whether a customer simulator should trigger a reaction skill at this point in the conversation.

## Evaluation Rubric for /{pair["skill"]}
{rubric_text}

## User Profile
- Capability: {pair["capability"]}
- Emotional baseline: {pair["emotion"]}
- Domain: {pair["domain"]}

## Conversation History
{conversation_text}

## Agent's Latest Message (the message the user is about to respond to)
{pair["agent_message"]}

## Instructions
Evaluate step by step:
1. Is the trigger CONDITION met? Does the agent's message and conversation context satisfy the rubric's positive conditions without hitting any negative conditions?
2. Is the trigger PERSONA-APPLICABLE? Given the user's capability and emotional baseline, would triggering this skill be in-character for this user?
3. Should the skill trigger at THIS specific point? (condition_met AND persona_applicable AND timing is appropriate — not too early, not if the condition was already triggered and handled)

Respond ONLY with a JSON object (no markdown fencing):
{{
  "condition_met": true or false,
  "condition_evidence": "quote or description of evidence",
  "persona_applicable": true or false,
  "persona_reason": "why this is or isn't appropriate for this user",
  "should_trigger": true or false,
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "edge_case": true or false
}}"""


def parse_label(response_text: str) -> Optional[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


async def call_gpt(client, prompt: str, retry: int = 0) -> Optional[dict]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        label = parse_label(text)
        if label is None and retry < MAX_RETRIES:
            return await call_gpt(client, prompt, retry + 1)
        return label
    except Exception as e:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2**retry)
            return await call_gpt(client, prompt, retry + 1)
        print(f"  ERROR after {MAX_RETRIES} retries: {e}")
        return None


async def majority_label(client, prompt: str, n: int = 5) -> dict:
    """Run N labels and take majority vote."""
    tasks = [call_gpt(client, prompt) for _ in range(n)]
    labels = await asyncio.gather(*tasks)

    valid = [l for l in labels if l is not None]
    if not valid:
        return {"should_trigger": False, "confidence": "FAILED", "n_votes": 0}

    yes_count = sum(1 for l in valid if l.get("should_trigger"))
    majority = yes_count > len(valid) / 2

    best = None
    for l in valid:
        if l.get("should_trigger") == majority:
            best = l
            break
    if best is None:
        best = valid[0]

    best["majority_vote"] = f"{yes_count}/{len(valid)}"
    best["n_votes"] = len(valid)
    best["is_majority"] = True
    return best


async def label_batch_maj5(
    client, batch: list[tuple[dict, str, str]]
) -> list[tuple[dict, dict]]:
    """Label a batch with majority@5 voting."""
    tasks = []
    for pair, rubric, conv in batch:
        prompt = build_prompt(pair, rubric, conv)
        tasks.append(majority_label(client, prompt, n=5))
    labels = await asyncio.gather(*tasks)
    return list(zip([p for p, _, _ in batch], labels))


def sample_pairs(all_pairs: list[dict]) -> list[dict]:
    """Sample precision (auto_trigger) and recall (fn_candidate) pairs, uniform per skill."""
    rng = random.Random(SEED)

    precision_pool = defaultdict(list)
    recall_pool = defaultdict(list)

    for p in all_pairs:
        skill = canonicalize_skill(p["skill"])
        if p["budget_tier"] == "auto_trigger":
            precision_pool[skill].append(p)
        elif p["budget_tier"] == "fn_candidate":
            recall_pool[skill].append(p)

    sampled = []

    # Precision: 500 total, uniform per skill (jargon-hit capped at available)
    print("Precision sampling:")
    jargon_available = len(precision_pool.get("jargon-hit", []))
    other_skills = [s for s in CANONICAL_SKILLS if s != "jargon-hit"]
    per_other = (PRECISION_TOTAL - jargon_available) // len(other_skills)

    for skill in CANONICAL_SKILLS:
        pool = precision_pool[skill]
        if skill == "jargon-hit":
            target = jargon_available
        else:
            target = min(per_other, len(pool))

        selected = rng.sample(pool, target) if len(pool) > target else pool
        for p in selected:
            p["eval_set"] = "precision"
            p["model_triggered"] = True
        sampled.extend(selected)
        print(f"  {skill:<25} {len(selected):>4} / {len(pool)}")

    # Recall: 500 total, uniform per skill
    print("Recall sampling:")
    per_skill_recall = RECALL_TOTAL // len(CANONICAL_SKILLS)

    for skill in CANONICAL_SKILLS:
        pool = recall_pool[skill]
        target = min(per_skill_recall, len(pool))
        selected = rng.sample(pool, target) if len(pool) > target else pool
        for p in selected:
            p["eval_set"] = "recall"
            p["model_triggered"] = False
        sampled.extend(selected)
        print(f"  {skill:<25} {len(selected):>4} / {len(pool)}")

    rng.shuffle(sampled)
    return sampled


async def run_labeling():
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # Load all sampled pairs
    all_pairs = []
    with open(IN_FILE) as f:
        for line in f:
            all_pairs.append(json.loads(line))

    # Sample
    pairs = sample_pairs(all_pairs)

    # Resume: skip already-labeled pairs
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                p = json.loads(line)
                done_ids.add(p["pair_id"])
        print(f"Resuming: {len(done_ids)} pairs already labeled")

    pairs = [p for p in pairs if p["pair_id"] not in done_ids]

    total = len(pairs)
    labeled_count = 0
    failed_count = 0
    start_time = time.time()

    print(
        f"\nLabeling {total} remaining pairs with {MODEL} (reasoning_effort=high, maj@5)"
    )
    print(f"API calls: ~{total * 5}")
    print(f"{'=' * 60}")

    rubric_cache = {}
    out_f = open(OUT_FILE, "a")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_pairs = pairs[batch_start:batch_end]

        prepared = []
        for pair in batch_pairs:
            skill = pair["skill"]
            if skill not in rubric_cache:
                rubric_cache[skill] = load_rubric(skill)
            rubric = rubric_cache[skill]

            conv_history = get_conversation_up_to(
                pair["sim_dir"], pair["task_id"], pair["msg_idx"]
            )
            conv_text = format_conversation(conv_history)
            prepared.append((pair, rubric, conv_text))

        results = await label_batch_maj5(client, prepared)

        for pair, label in results:
            if label.get("confidence") == "FAILED" and label.get("n_votes", 0) == 0:
                pair["label"] = label
                pair["label_method"] = "failed"
                failed_count += 1
            else:
                pair["label"] = label
                pair["label_method"] = "majority@5"
                labeled_count += 1

            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out_f.flush()

        elapsed = time.time() - start_time
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (total - batch_end) / rate if rate > 0 else 0
        print(
            f"  [{batch_end}/{total}] labeled={labeled_count} failed={failed_count} "
            f"| {rate:.1f}/s ETA {eta / 60:.0f}m"
        )

    out_f.close()

    # Print summary
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Labeled: {labeled_count}")
    print(f"  Failed: {failed_count}")

    # Quick metrics
    results_by_set = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    )
    with open(OUT_FILE) as f:
        for line in f:
            p = json.loads(line)
            eval_set = p.get("eval_set", "unknown")
            skill = canonicalize_skill(p["skill"])
            model = p.get("model_triggered", False)
            silver = (p.get("label") or {}).get("should_trigger", False)

            if model and silver:
                results_by_set[eval_set][skill]["tp"] += 1
            elif model and not silver:
                results_by_set[eval_set][skill]["fp"] += 1
            elif not model and silver:
                results_by_set[eval_set][skill]["fn"] += 1
            else:
                results_by_set[eval_set][skill]["tn"] += 1

    print(f"\n{'=' * 70}")
    print("PRECISION SET (model triggered → was it correct?)")
    print(f"{'Skill':<25} {'TP':>4} {'FP':>4} {'Precision':>10}")
    print("-" * 50)
    for skill in CANONICAL_SKILLS:
        m = results_by_set["precision"][skill]
        tp, fp = m["tp"], m["fp"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  {skill:<23} {tp:>4} {fp:>4} {prec:>10.1%}")

    print(f"\nRECALL SET (model didn't trigger → should it have?)")
    print(f"{'Skill':<25} {'Should':>7} {'Correct Skip':>13} {'Miss Rate':>10}")
    print("-" * 60)
    for skill in CANONICAL_SKILLS:
        m = results_by_set["recall"][skill]
        fn, tn = m["fn"], m["tn"]
        total_recall = fn + tn
        miss_rate = fn / total_recall if total_recall > 0 else 0
        print(f"  {skill:<23} {fn:>7} {tn:>13} {miss_rate:>10.1%}")

    print(f"\nOutput: {OUT_FILE}")


def main():
    asyncio.run(run_labeling())


if __name__ == "__main__":
    main()
