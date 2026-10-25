#!/usr/bin/env python3
"""
LLM-based failure attribution for telecom persona experiments.

Compares passing vs failing conversations for the same task and uses
GPT-5.4 reasoning to attribute failure root cause.

Usage:
    OPENAI_API_KEY=... uv run python scripts/llm_failure_attribution.py
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

from persona_failure_attribution import (
    GPT_PRESETS,
    SONNET_PRESETS,
    find_pattern1_failures,
    find_pattern2_failures,
    get_reward,
    load_preset_runs,
    parse_task,
)

MODEL = "gpt-5.4"
BATCH_SIZE = 10
MAX_RETRIES = 3
OUT_FILE = DATA_ROOT / "data/failure_attribution/llm_attributed.jsonl"

ATTRIBUTION_CATEGORIES = {
    "information_withholding": "User failed to provide critical diagnostic information (e.g., travel status, error messages, device state) that was available in the passing conversation. The agent never received the info needed to diagnose correctly.",
    "information_timing": "User eventually provided the info, but too late — agent had already gone down a wrong path or exhausted the conversation budget.",
    "agent_strategy": "Agent had sufficient information but chose an inefficient or incorrect diagnostic path (e.g., skipped critical tools, pursued wrong hypothesis, didn't explore all faults).",
    "agent_tool_misuse": "Agent misunderstood tool capabilities — e.g., claimed it doesn't have certain tools, refused to instruct user on device-side actions, or used tools with wrong parameters.",
    "persona_interference": "User's persona behavior (impatience, verbosity, withholding, etc.) actively disrupted the diagnostic flow — e.g., demanded transfer, refused to follow steps, derailed the conversation.",
    "conversation_exhaustion": "Ran out of conversation turns or user/agent gave up before resolving all faults, even though the diagnostic path was reasonable.",
    "task_complexity": "Task had too many simultaneous faults for the conversation budget; even with perfect strategy, resolution would be extremely difficult.",
}

PROMPT_TEMPLATE = """You are analyzing why a customer support simulation failed. You are given two conversations for the SAME task:
- **PASS**: A conversation where the task was resolved successfully (reward=1.0)
- **FAIL**: A conversation where the task failed (reward=0.0)

## Task Information
- **Task ID**: {task_id}
- **Issue type**: {issue_type}
- **Fault conditions**: {conditions}
- **Agent model**: {agent_name}
- **Comparison**: {comparison_desc}

## What needed to happen (required actions)
{required_actions}

## What actually happened in FAIL
{failed_actions}

## PASSING Conversation
{pass_conversation}

## FAILING Conversation
{fail_conversation}

## Attribution Categories
{categories}

## Instructions

Compare the PASS and FAIL conversations carefully. Identify the PRIMARY root cause of failure by analyzing:

1. **Information flow**: Did the user provide the same diagnostic information in both conversations? If not, what was missing/delayed?
2. **Agent strategy**: Did the agent follow a similar diagnostic path? Where did it diverge?
3. **User behavior**: Did the user's behavior (compliance, tone, requests) differ in ways that affected the outcome?
4. **Tool usage**: Did the agent use the same tools? Were any critical tools skipped or misused?
5. **Conversation dynamics**: Was the conversation cut short? Did either party give up?

Respond with a JSON object (no markdown fencing):
{{
  "primary_attribution": "<category_name>",
  "secondary_attribution": "<category_name or null>",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "evidence": {{
    "pass_key_moment": "Quote or describe the critical moment in the PASS conversation",
    "fail_key_moment": "Quote or describe where the FAIL conversation went wrong",
    "divergence_point": "Describe when/why the conversations diverged",
    "missing_info_or_action": "What specific info/action was missing in FAIL that was present in PASS"
  }},
  "brief_summary": "1-2 sentence explanation of why this task failed"
}}"""


def format_conversation(sim: dict, max_messages: int = 120) -> str:
    messages = sim.get("messages", [])
    if len(messages) > max_messages:
        messages = (
            messages[:10]
            + [
                {
                    "role": "system",
                    "content": f"[... {len(messages) - max_messages - 10} messages truncated ...]",
                }
            ]
            + messages[-max_messages + 10 :]
        )

    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part["text"])
                    elif part.get("type") == "tool_use":
                        texts.append(
                            f"[TOOL_CALL: {part.get('name', '?')}({json.dumps(part.get('input', {}))[:200]})]"
                        )
                    elif part.get("type") == "tool_result":
                        rc = part.get("content", "")
                        if isinstance(rc, list):
                            for r in rc:
                                if isinstance(r, dict) and r.get("type") == "text":
                                    texts.append(
                                        f"[TOOL_RESULT: {r.get('text', '')[:300]}]"
                                    )
                        elif isinstance(rc, str):
                            texts.append(f"[TOOL_RESULT: {rc[:300]}]")
            content = " ".join(texts)
        else:
            content = str(content)

        if content:
            lines.append(f"[{i}] {role}: {content[:500]}")

        tc = msg.get("tool_calls")
        if tc:
            for t in tc:
                args = json.dumps(t.get("arguments", {}))[:200]
                lines.append(f"[{i}] {role} TOOL_CALL: {t.get('name', '?')}({args})")

    return "\n".join(lines)


def format_actions(sim: dict) -> tuple[str, str]:
    ri = sim.get("reward_info", {})
    required = []
    failed = []

    for ac in ri.get("action_checks", []):
        action = ac["action"]
        args = json.dumps(action.get("arguments", {}))
        line = f"  {'✓' if ac['action_match'] else '✗'} {action['name']}({args}) — requestor: {action['requestor']}"
        required.append(line)
        if not ac["action_match"]:
            failed.append(
                f"  ✗ {action['name']}({args}) — requestor: {action['requestor']}"
            )

    for ea in ri.get("env_assertions", []):
        a = ea["env_assertion"]
        met = ea.get("met", True)
        args = json.dumps(a.get("arguments", {}))
        line = f"  {'✓' if met else '✗'} [ENV] {a['func_name']}({args})"
        required.append(line)
        if not met:
            failed.append(f"  ✗ [ENV] {a['func_name']}({args})")

    return "\n".join(required) if required else "  (none recorded)", "\n".join(
        failed
    ) if failed else "  All actions completed"


def build_cases() -> list[dict]:
    sonnet_data = {
        preset: load_preset_runs(dirs) for preset, dirs in SONNET_PRESETS.items()
    }
    gpt_data = {preset: load_preset_runs(dirs) for preset, dirs in GPT_PRESETS.items()}

    cases = []
    case_id = 0

    for agent_name, data, presets in [
        ("Sonnet 4.6", sonnet_data, SONNET_PRESETS),
        ("GPT 5.4", gpt_data, GPT_PRESETS),
    ]:
        # Pattern 1: original → baseline
        p1 = find_pattern1_failures(data["original"], data["cc_baseline"])
        for task_id, orig_avg, base_avg in p1:
            pass_sim = next(
                (s for s in data["original"][task_id] if get_reward(s) == 1.0),
                data["original"][task_id][0],
            )
            fail_sim = next(
                (s for s in data["cc_baseline"][task_id] if get_reward(s) == 0.0),
                data["cc_baseline"][task_id][0],
            )

            cases.append(
                {
                    "case_id": f"p1_{agent_name.lower().replace(' ', '')}_{case_id}",
                    "pattern": "pattern1",
                    "comparison": "original_user → cc_baseline",
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "pass_preset": "original_user",
                    "fail_preset": "cc_baseline",
                    "pass_reward": orig_avg,
                    "fail_reward": base_avg,
                    "pass_sim": pass_sim,
                    "fail_sim": fail_sim,
                }
            )
            case_id += 1

        # Pattern 2: baseline → persona
        for preset in ["hard_imp", "terse_coop", "verbose_with", "techsavvy_imp"]:
            p2 = find_pattern2_failures(data["cc_baseline"], data[preset], preset)
            for task_id, base_avg, preset_avg in p2:
                pass_sim = next(
                    (s for s in data["cc_baseline"][task_id] if get_reward(s) == 1.0),
                    data["cc_baseline"][task_id][0],
                )
                fail_sim = next(
                    (s for s in data[preset][task_id] if get_reward(s) == 0.0),
                    data[preset][task_id][0],
                )

                cases.append(
                    {
                        "case_id": f"p2_{preset}_{agent_name.lower().replace(' ', '')}_{case_id}",
                        "pattern": "pattern2",
                        "comparison": f"cc_baseline → {preset}",
                        "agent_name": agent_name,
                        "task_id": task_id,
                        "pass_preset": "cc_baseline",
                        "fail_preset": preset,
                        "pass_reward": base_avg,
                        "fail_reward": preset_avg,
                        "pass_sim": pass_sim,
                        "fail_sim": fail_sim,
                    }
                )
                case_id += 1

    return cases


def parse_response(text: str) -> Optional[dict]:
    text = text.strip()
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
            max_completion_tokens=3000,
        )
        text = response.choices[0].message.content or ""
        result = parse_response(text)
        if result is None and retry < MAX_RETRIES:
            return await call_gpt(client, prompt, retry + 1)
        return result
    except Exception as e:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2**retry)
            return await call_gpt(client, prompt, retry + 1)
        print(f"  ERROR after {MAX_RETRIES} retries: {e}")
        return None


async def label_batch(client, batch: list[dict]) -> list[tuple[dict, Optional[dict]]]:
    tasks = []
    for case in batch:
        issue, conditions, persona = parse_task(case["task_id"])
        required_actions, failed_actions = format_actions(case["fail_sim"])

        cats_text = "\n".join(
            f"- **{k}**: {v}" for k, v in ATTRIBUTION_CATEGORIES.items()
        )

        prompt = PROMPT_TEMPLATE.format(
            task_id=case["task_id"],
            issue_type=issue,
            conditions=", ".join(conditions),
            agent_name=case["agent_name"],
            comparison_desc=case["comparison"],
            required_actions=required_actions,
            failed_actions=failed_actions,
            pass_conversation=format_conversation(case["pass_sim"]),
            fail_conversation=format_conversation(case["fail_sim"]),
            categories=cats_text,
        )
        tasks.append(call_gpt(client, prompt))

    results = await asyncio.gather(*tasks)
    return list(zip(batch, results))


async def run():
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

    print("Building failure cases...")
    cases = build_cases()
    print(f"Total cases: {len(cases)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                p = json.loads(line)
                done_ids.add(p["case_id"])
        print(f"Resuming: {len(done_ids)} already labeled")

    remaining = [c for c in cases if c["case_id"] not in done_ids]
    total = len(remaining)

    if total == 0:
        print("All cases already labeled!")
        print_summary()
        return

    print(f"\nLabeling {total} cases with {MODEL} (reasoning_effort=high)")
    print(f"{'=' * 60}")

    start_time = time.time()
    labeled = 0
    failed = 0

    out_f = open(OUT_FILE, "a")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = remaining[batch_start:batch_end]

        results = await label_batch(client, batch)

        for case, result in results:
            output = {
                "case_id": case["case_id"],
                "pattern": case["pattern"],
                "comparison": case["comparison"],
                "agent_name": case["agent_name"],
                "task_id": case["task_id"],
                "pass_preset": case["pass_preset"],
                "fail_preset": case["fail_preset"],
                "pass_reward": case["pass_reward"],
                "fail_reward": case["fail_reward"],
                "pass_msg_count": len(case["pass_sim"].get("messages", [])),
                "fail_msg_count": len(case["fail_sim"].get("messages", [])),
            }

            if result is None:
                output["attribution"] = None
                output["label_status"] = "failed"
                failed += 1
            else:
                output["attribution"] = result
                output["label_status"] = "success"
                labeled += 1

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_f.flush()

        elapsed = time.time() - start_time
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (total - batch_end) / rate if rate > 0 else 0
        print(
            f"  [{batch_end}/{total}] labeled={labeled} failed={failed} | {rate:.2f}/s ETA {eta / 60:.0f}m"
        )

    out_f.close()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Labeled: {labeled}")
    print(f"  Failed: {failed}")

    print_summary()


def print_summary():
    if not OUT_FILE.exists():
        return

    results = []
    with open(OUT_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"\n{'=' * 70}")
    print("ATTRIBUTION SUMMARY")
    print(f"{'=' * 70}")

    # Overall
    primary_counts = Counter()
    by_pattern = defaultdict(Counter)
    by_agent = defaultdict(Counter)
    by_preset = defaultdict(Counter)

    for r in results:
        attr = r.get("attribution")
        if attr is None:
            continue
        primary = attr.get("primary_attribution", "unknown")
        primary_counts[primary] += 1
        by_pattern[r["pattern"]][primary] += 1
        by_agent[r["agent_name"]][primary] += 1
        by_preset[r.get("fail_preset", "?")][primary] += 1

    total = sum(primary_counts.values())
    print(f"\nOverall ({total} cases):")
    for attr, count in primary_counts.most_common():
        print(f"  {attr:<30} {count:>4} ({count / total:.1%})")

    print(f"\nBy pattern:")
    for pattern in sorted(by_pattern.keys()):
        counts = by_pattern[pattern]
        t = sum(counts.values())
        print(f"  {pattern} ({t} cases):")
        for attr, count in counts.most_common():
            print(f"    {attr:<28} {count:>4} ({count / t:.1%})")

    print(f"\nBy agent:")
    for agent in sorted(by_agent.keys()):
        counts = by_agent[agent]
        t = sum(counts.values())
        print(f"  {agent} ({t} cases):")
        for attr, count in counts.most_common():
            print(f"    {attr:<28} {count:>4} ({count / t:.1%})")

    print(f"\nBy fail preset:")
    for preset in sorted(by_preset.keys()):
        counts = by_preset[preset]
        t = sum(counts.values())
        print(f"  {preset} ({t} cases):")
        for attr, count in counts.most_common(3):
            print(f"    {attr:<28} {count:>4} ({count / t:.1%})")

    # Confidence distribution
    conf_counts = Counter()
    for r in results:
        attr = r.get("attribution")
        if attr:
            conf_counts[attr.get("confidence", "?")] += 1
    print(f"\nConfidence: {dict(conf_counts.most_common())}")

    print(f"\nOutput: {OUT_FILE}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
