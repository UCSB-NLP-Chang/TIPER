#!/bin/bash
# Label fn_candidate pool to estimate under-trigger rate

set -e

echo "=============================================="
echo "Stage 2b: Label fn_candidate pool (804 pairs)"
echo "=============================================="
echo ""
echo "Model: gpt-5.4"
echo "Pairs: fn_candidate only"
echo "Purpose: Estimate under-trigger (Recall)"
echo ""

# Extract fn_candidate pairs to temporary file
jq 'select(.budget_tier == "fn_candidate")' \
  data/skill_eval/sampled_pairs.jsonl \
  > data/skill_eval/fn_candidates_only.jsonl

echo "Extracted $(wc -l < data/skill_eval/fn_candidates_only.jsonl) fn_candidate pairs"
echo ""
echo "Starting labeling with gpt-5.4..."
echo "Estimated cost: ~$30-50 (804 pairs)"
echo "Estimated time: ~45-60 minutes"
echo ""

# Run labeling (modify IN_FILE temporarily)
PYTHONUNBUFFERED=1 uv run python3 << 'EOF'
import json
import asyncio
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path("scripts/skill_trigger_eval")))
from load_utils import get_conversation_up_to

IN_FILE = Path("data/skill_eval/fn_candidates_only.jsonl")
OUT_FILE = Path("data/skill_eval/fn_candidates_labeled.jsonl")
RUBRIC_DIR = Path("scripts/skill_trigger_eval/rubrics")

MODEL = "gpt-5.4"
BATCH_SIZE = 20
MAX_RETRIES = 3

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
                lines.append(f"[{role} TOOL CALL]: {t.get('name', '')}({json.dumps(t.get('arguments', {}))})")
    return prefix + "\n".join(lines)

def build_prompt(pair: dict, rubric_text: str, conversation_text: str) -> str:
    return f"""You are evaluating whether a customer simulator should trigger a reaction skill at this point in the conversation.

## Evaluation Rubric for /{pair['skill']}
{rubric_text}

## User Profile
- Capability: {pair['capability']}
- Emotional baseline: {pair['emotion']}
- Domain: {pair['domain']}

## Conversation History
{conversation_text}

## Agent's Latest Message (the message the user is about to respond to)
{pair['agent_message']}

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

def parse_label(response_text: str):
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

async def call_gpt(client, prompt: str, retry: int = 0):
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
            await asyncio.sleep(2 ** retry)
            return await call_gpt(client, prompt, retry + 1)
        print(f"  ERROR after {MAX_RETRIES} retries: {e}")
        return None

async def label_batch(client, batch):
    tasks = []
    for pair, rubric, conv in batch:
        prompt = build_prompt(pair, rubric, conv)
        tasks.append(call_gpt(client, prompt))
    labels = await asyncio.gather(*tasks)
    return list(zip([p for p, _, _ in batch], labels))

async def run_labeling():
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    pairs = []
    with open(IN_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    rubric_cache = {}
    total = len(pairs)
    labeled_count = 0
    failed_count = 0

    print(f"Labeling {total} fn_candidate pairs with {MODEL}")
    print()

    out_f = open(OUT_FILE, "w")

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

        results = await label_batch(client, prepared)

        for pair, label in results:
            if label is None:
                pair["label"] = {"should_trigger": False, "confidence": "FAILED"}
                pair["label_method"] = "failed"
                failed_count += 1
            else:
                pair["label"] = label
                pair["label_method"] = "single"
                labeled_count += 1

            pair["model_triggered"] = False  # All fn_candidates are not triggered
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out_f.flush()

        print(f"  [{batch_end}/{total}] labeled={labeled_count} failed={failed_count}")

    out_f.close()
    print(f"\nDone! Output: {OUT_FILE}")
    print(f"  Labeled: {labeled_count}")
    print(f"  Failed: {failed_count}")

asyncio.run(run_labeling())
EOF

echo ""
echo "Done!"
echo "Output: data/skill_eval/fn_candidates_labeled.jsonl"
