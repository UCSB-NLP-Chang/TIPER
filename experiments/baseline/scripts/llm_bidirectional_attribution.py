#!/usr/bin/env python3
"""
LLM-based bidirectional attribution for telecom persona experiments.

Covers BOTH directions:
- Degradation: baseline pass → preset fail (why did it break?)
- Improvement: baseline fail → preset pass (why did it help?)

Usage:
    OPENAI_API_KEY=... uv run python scripts/llm_bidirectional_attribution.py
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter
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
    get_reward,
    is_failing,
    is_passing,
    load_preset_runs,
    parse_task,
)

MODEL = "gpt-5.4"
BATCH_SIZE = 10
MAX_RETRIES = 3
OUT_FILE = DATA_ROOT / "data/failure_attribution/bidirectional_attributed.jsonl"

DEGRADATION_CATEGORIES = {
    "information_withholding": "Persona caused user to withhold critical diagnostic info (e.g., travel status, error details) that baseline user would have provided.",
    "information_timing": "Persona caused user to delay providing critical info until too late in the conversation.",
    "agent_strategy": "Agent chose wrong diagnostic path — not caused by persona, agent had enough info but made bad decisions.",
    "agent_tool_misuse": "Agent misused tools (wrong params, wrong line ID, claimed tools unavailable).",
    "persona_overwhelm": "Persona's behavior (impatience, verbosity, demands) overwhelmed the agent, causing it to give up, transfer, or lose track.",
    "persona_derail": "Persona actively derailed diagnosis — refused steps, demanded transfer, changed topic, or created confusion.",
    "conversation_exhaustion": "Ran out of turns; persona's style consumed too many turns on non-diagnostic exchanges.",
}

IMPROVEMENT_CATEGORIES = {
    "information_volunteered": "Persona caused user to proactively volunteer critical info (e.g., travel status, error messages, device state) that baseline user didn't share.",
    "user_compliance": "Persona was more cooperative/compliant with troubleshooting steps, allowing agent to progress faster.",
    "agent_strategy_unlocked": "User's different behavior led agent to discover a better diagnostic path — e.g., extra info triggered the right hypothesis.",
    "conversation_efficiency": "Persona's style (terse, direct) reduced wasted turns, leaving more budget for actual diagnosis.",
    "user_technical_assist": "Tech-savvy persona performed diagnostic steps proactively or provided technical details that accelerated resolution.",
    "lucky_path": "Different conversation dynamics led agent down a path that happened to find the fix, not because of systematically better info or strategy.",
}

DEGRADE_PROMPT = """You are analyzing why a customer support simulation DEGRADED when switching from cc_baseline to a persona preset.

## Task Information
- **Task ID**: {task_id}
- **Issue type**: {issue_type}
- **Fault conditions**: {conditions}
- **Agent model**: {agent_name}
- **Comparison**: cc_baseline (PASS) → {preset} (FAIL)

## What needed to happen (required actions)
{required_actions}

## What failed in the preset conversation
{failed_actions}

## PASSING Conversation (cc_baseline)
{pass_conversation}

## FAILING Conversation ({preset})
{fail_conversation}

## Attribution Categories
{categories}

## Instructions

Compare the two conversations to identify the PRIMARY reason the preset conversation failed while baseline succeeded. Focus on:
1. Did the persona cause the user to behave differently in ways that blocked diagnosis?
2. Or did the agent independently make worse decisions?
3. Did the persona's style consume conversation budget on non-diagnostic exchanges?

Respond with JSON (no markdown fencing):
{{
  "primary_attribution": "<category_name>",
  "secondary_attribution": "<category_name or null>",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "evidence": {{
    "pass_key_moment": "What happened in baseline that led to success",
    "fail_key_moment": "What went wrong in the preset conversation",
    "persona_effect": "How specifically did the persona change user behavior vs baseline",
    "divergence_point": "When/why the conversations diverged"
  }},
  "brief_summary": "1-2 sentence explanation"
}}"""

IMPROVE_PROMPT = """You are analyzing why a customer support simulation IMPROVED when switching from cc_baseline to a persona preset.

## Task Information
- **Task ID**: {task_id}
- **Issue type**: {issue_type}
- **Fault conditions**: {conditions}
- **Agent model**: {agent_name}
- **Comparison**: cc_baseline (FAIL) → {preset} (PASS)

## What needed to happen (required actions)
{required_actions}

## FAILING Conversation (cc_baseline)
{fail_conversation}

## PASSING Conversation ({preset})
{pass_conversation}

## Attribution Categories
{categories}

## Instructions

Compare the two conversations to identify the PRIMARY reason the preset conversation succeeded while baseline failed. Focus on:
1. Did the persona cause the user to provide different/better information?
2. Did the persona's style lead the agent to a better diagnostic path?
3. Did conversation efficiency play a role?

Respond with JSON (no markdown fencing):
{{
  "primary_attribution": "<category_name>",
  "secondary_attribution": "<category_name or null>",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "evidence": {{
    "fail_key_moment": "What went wrong in baseline that blocked resolution",
    "pass_key_moment": "What the preset conversation did differently that led to success",
    "persona_effect": "How specifically did the persona change user behavior vs baseline",
    "divergence_point": "When/why the conversations diverged"
  }},
  "brief_summary": "1-2 sentence explanation"
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
    presets = ["hard_imp", "terse_coop", "verbose_with", "techsavvy_imp"]

    for agent_name, data in [("Sonnet 4.6", sonnet_data), ("GPT 5.4", gpt_data)]:
        baseline = data["cc_baseline"]

        for preset in presets:
            preset_runs = data[preset]
            common_tasks = set(baseline.keys()) & set(preset_runs.keys())

            for task_id in common_tasks:
                b_sims = baseline[task_id]
                p_sims = preset_runs[task_id]
                b_pass = is_passing(b_sims)
                b_fail = is_failing(b_sims)
                p_pass = is_passing(p_sims)
                p_fail = is_failing(p_sims)

                if b_pass and p_fail:
                    # Degradation
                    pass_sim = next(
                        (s for s in b_sims if get_reward(s) == 1.0), b_sims[0]
                    )
                    fail_sim = next(
                        (s for s in p_sims if get_reward(s) == 0.0), p_sims[0]
                    )
                    cases.append(
                        {
                            "case_id": f"degrade_{agent_name.lower().replace(' ', '')}_{preset}_{case_id}",
                            "direction": "degradation",
                            "agent_name": agent_name,
                            "preset": preset,
                            "task_id": task_id,
                            "pass_sim": pass_sim,
                            "fail_sim": fail_sim,
                        }
                    )
                    case_id += 1

                elif b_fail and p_pass:
                    # Improvement
                    fail_sim = next(
                        (s for s in b_sims if get_reward(s) == 0.0), b_sims[0]
                    )
                    pass_sim = next(
                        (s for s in p_sims if get_reward(s) == 1.0), p_sims[0]
                    )
                    cases.append(
                        {
                            "case_id": f"improve_{agent_name.lower().replace(' ', '')}_{preset}_{case_id}",
                            "direction": "improvement",
                            "agent_name": agent_name,
                            "preset": preset,
                            "task_id": task_id,
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
        required_actions_sim = (
            case["fail_sim"] if case["direction"] == "degradation" else case["pass_sim"]
        )
        required_actions, failed_actions = format_actions(required_actions_sim)

        if case["direction"] == "degradation":
            cats_text = "\n".join(
                f"- **{k}**: {v}" for k, v in DEGRADATION_CATEGORIES.items()
            )
            prompt = DEGRADE_PROMPT.format(
                task_id=case["task_id"],
                issue_type=issue,
                conditions=", ".join(conditions),
                agent_name=case["agent_name"],
                preset=case["preset"],
                required_actions=required_actions,
                failed_actions=failed_actions,
                pass_conversation=format_conversation(case["pass_sim"]),
                fail_conversation=format_conversation(case["fail_sim"]),
                categories=cats_text,
            )
        else:
            cats_text = "\n".join(
                f"- **{k}**: {v}" for k, v in IMPROVEMENT_CATEGORIES.items()
            )
            prompt = IMPROVE_PROMPT.format(
                task_id=case["task_id"],
                issue_type=issue,
                conditions=", ".join(conditions),
                agent_name=case["agent_name"],
                preset=case["preset"],
                required_actions=required_actions,
                fail_conversation=format_conversation(case["fail_sim"]),
                pass_conversation=format_conversation(case["pass_sim"]),
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

    print("Building cases (degradation + improvement)...")
    cases = build_cases()

    degrade_count = sum(1 for c in cases if c["direction"] == "degradation")
    improve_count = sum(1 for c in cases if c["direction"] == "improvement")
    print(
        f"Total: {len(cases)} ({degrade_count} degradation + {improve_count} improvement)"
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume
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
                "direction": case["direction"],
                "agent_name": case["agent_name"],
                "preset": case["preset"],
                "task_id": case["task_id"],
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

    print(f"\n{'=' * 80}")
    print("BIDIRECTIONAL ATTRIBUTION SUMMARY")
    print(f"{'=' * 80}")

    AGENTS = ["Sonnet 4.6", "GPT 5.4"]
    PRESETS = ["hard_imp", "terse_coop", "verbose_with", "techsavvy_imp"]

    for direction in ["degradation", "improvement"]:
        dir_results = [r for r in results if r["direction"] == direction]
        if not dir_results:
            continue

        label = (
            "DEGRADATION (baseline→preset broke it)"
            if direction == "degradation"
            else "IMPROVEMENT (baseline→preset fixed it)"
        )
        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"{'─' * 80}")

        for agent in AGENTS:
            agent_results = [r for r in dir_results if r["agent_name"] == agent]
            if not agent_results:
                continue

            print(f"\n  {agent}:")
            print(f"  {'Preset':<16} {'N':>3}  {'Top attribution':>40}  {'2nd':>30}")
            print(f"  {'─' * 16} {'─' * 3}  {'─' * 40}  {'─' * 30}")

            for preset in PRESETS:
                preset_results = [r for r in agent_results if r["preset"] == preset]
                if not preset_results:
                    continue

                counts = Counter()
                for r in preset_results:
                    attr = r.get("attribution") or {}
                    counts[attr.get("primary_attribution", "unknown")] += 1

                t = len(preset_results)
                top = counts.most_common(2)
                top1 = (
                    f"{top[0][0]} {top[0][1]}/{t} ({top[0][1] / t:.0%})"
                    if len(top) > 0
                    else "-"
                )
                top2 = (
                    f"{top[1][0]} {top[1][1]}/{t} ({top[1][1] / t:.0%})"
                    if len(top) > 1
                    else "-"
                )

                print(f"  {preset:<16} {t:>3}  {top1:>40}  {top2:>30}")

    # Overall matrix
    print(f"\n{'─' * 80}")
    print(f"  AGENT × PRESET MATRIX (primary attribution distribution)")
    print(f"{'─' * 80}")

    for agent in AGENTS:
        print(f"\n  ┌── {agent} ──┐")
        for direction in ["degradation", "improvement"]:
            dir_label = "↓ Degraded" if direction == "degradation" else "↑ Improved"
            dir_results = [
                r
                for r in results
                if r["agent_name"] == agent and r["direction"] == direction
            ]
            if not dir_results:
                continue

            print(f"  │ {dir_label}:")

            all_attrs = Counter()
            for r in dir_results:
                attr = r.get("attribution") or {}
                all_attrs[attr.get("primary_attribution", "unknown")] += 1

            t = sum(all_attrs.values())
            for a, c in all_attrs.most_common():
                bar = "█" * int(c / t * 30)
                print(f"  │   {a:<28} {c:>3}/{t} ({c / t:>5.1%})  {bar}")

        print(f"  └{'─' * 40}┘")

    print(f"\nOutput: {OUT_FILE}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
