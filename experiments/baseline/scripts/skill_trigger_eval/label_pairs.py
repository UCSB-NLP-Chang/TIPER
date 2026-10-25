#!/usr/bin/env python3
"""
Stage 2: GPT-5.5 Silver Labeling.

Labels sampled pairs with three-layer decomposition:
  condition_met × persona_applicable → should_trigger

Uses per-skill rubrics. Does NOT show model_triggered to the judge.
Default: single label. Upgrades to Majority@5 for ambiguous/conflict cases.
"""

import asyncio
import json
import os
import sys
import time
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
OUT_FILE = DATA_ROOT / "data/skill_eval/labeled_pairs.jsonl"
RUBRIC_DIR = Path("scripts/skill_trigger_eval/rubrics")

MODEL = "gpt-5.4"
BATCH_SIZE = 20
MAX_RETRIES = 3
MAX_CONVERSATION_TURNS = 200

RUBRIC_MAPPING = {
    "jargon-hit": "jargon_hit.yaml",
    "frustration-escalation": "frustration_escalation.yaml",
    "transfer-pushback": "transfer_pushback.yaml",
    "resolution": "resolution.yaml",
    "resolution-banking": "resolution.yaml",
}


def load_rubric(skill: str) -> str:
    fname = RUBRIC_MAPPING.get(skill)
    if not fname:
        return f"No rubric available for {skill}"
    path = RUBRIC_DIR / fname
    with open(path) as f:
        content = yaml.safe_load(f)
    return yaml.dump(content, default_flow_style=False, allow_unicode=True)


def format_conversation(history: list[dict], max_turns: int = 200) -> str:
    """Format conversation history for the prompt, truncating if needed."""
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
    """Parse JSON from GPT response, handling potential markdown fencing."""
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
    """Call GPT-5.5 reasoning mode and parse the label."""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_completion_tokens=1000,
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


async def label_batch(
    client, batch: list[tuple[dict, str, str]]
) -> list[tuple[dict, Optional[dict]]]:
    """Label a batch of pairs concurrently."""
    tasks = []
    for pair, rubric, conv in batch:
        prompt = build_prompt(pair, rubric, conv)
        tasks.append(call_gpt(client, prompt))

    labels = await asyncio.gather(*tasks)
    return list(zip([p for p, _, _ in batch], labels))


def needs_majority(label: Optional[dict], pair: dict, model_triggered_set: set) -> bool:
    if label is None:
        return True
    if label.get("confidence") == "LOW":
        return True
    if label.get("edge_case"):
        return True
    is_triggered = pair["pair_id"] in model_triggered_set
    should = label.get("should_trigger", False)
    if is_triggered != should:
        return True
    if pair.get("budget_tier") == "boundary":
        return True
    return False


async def majority_label(
    client, pair: dict, rubric: str, conv: str, n: int = 5
) -> dict:
    """Run N labels and take majority vote."""
    prompt = build_prompt(pair, rubric, conv)
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

    # Load pairs
    pairs = []
    with open(IN_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    # Infer model_triggered from budget_tier (avoids loading 870MB decision_pairs)
    model_triggered_set = set()
    for p in pairs:
        if p.get("budget_tier") == "auto_trigger":
            model_triggered_set.add(p["pair_id"])

    # Pilot mode
    pilot_n = None
    if "--pilot" in sys.argv:
        idx = sys.argv.index("--pilot")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].startswith("--n="):
            pilot_n = int(sys.argv[idx + 1].split("=")[1])
        else:
            pilot_n = 400
    if "--n" in sys.argv:
        idx = sys.argv.index("--n")
        pilot_n = int(sys.argv[idx + 1])

    if pilot_n:
        pairs = pairs[:pilot_n]
        print(f"Pilot mode: {pilot_n} pairs")

    # Load rubrics
    rubric_cache = {}

    total = len(pairs)
    single_count = 0
    maj_count = 0
    failed = 0
    start_time = time.time()

    print(f"Stage 2: GPT-5.5 Silver Labeling", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Total pairs to label: {total}", flush=True)
    print(flush=True)

    out_f = open(OUT_FILE, "w")

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_pairs = pairs[batch_start:batch_end]

        # Prepare batch with rubrics and conversation
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

        # Single-label pass
        results = await label_batch(client, prepared)

        for pair, label in results:
            if label is None:
                pair["label"] = {"should_trigger": False, "confidence": "FAILED"}
                pair["label_method"] = "failed"
                failed += 1
            elif needs_majority(label, pair, model_triggered_set):
                skill = pair["skill"]
                conv_history = get_conversation_up_to(
                    pair["sim_dir"], pair["task_id"], pair["msg_idx"]
                )
                conv_text = format_conversation(conv_history)
                maj_label = await majority_label(
                    client, pair, rubric_cache[skill], conv_text
                )
                pair["label"] = maj_label
                pair["label_method"] = "majority@5"
                maj_count += 1
            else:
                pair["label"] = label
                pair["label_method"] = "single"
                single_count += 1

            pair["model_triggered"] = pair["pair_id"] in model_triggered_set
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out_f.flush()

        elapsed = time.time() - start_time
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        eta = (total - batch_end) / rate if rate > 0 else 0
        print(
            f"  [{batch_end}/{total}] single={single_count} maj={maj_count} "
            f"fail={failed} | {rate:.1f}/s, ETA {eta / 60:.0f}m",
            flush=True,
        )

    out_f.close()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Single labels: {single_count}")
    print(f"  Majority@5: {maj_count}")
    print(f"  Failed: {failed}")
    print(f"  API calls: ~{single_count + maj_count * 5 + failed}")
    print(f"\nOutput: {OUT_FILE}")


def main():
    asyncio.run(run_labeling())


if __name__ == "__main__":
    main()
