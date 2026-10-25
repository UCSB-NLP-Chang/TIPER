#!/usr/bin/env python3
"""
Persona Failure Attribution Analysis

Analyzes persona experiment data to identify and attribute failure patterns:
1. Pattern 1: Tasks where original_user succeeds but cc_baseline fails
2. Pattern 2: Tasks where cc_baseline succeeds but persona presets fail

Usage:
    uv run python scripts/persona_failure_attribution.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BASE = Path(__file__).parent.parent / "data" / "simulations"

# Experiment directory configuration
SONNET_PRESETS = {
    "original": [
        "t1_sonnet_original_run1",
        "t1_sonnet_original_run2",
        "t1_sonnet_original_run3",
    ],
    "cc_baseline": [
        "t1_sonnet_cc_baseline_run1",
        "t1_sonnet_cc_baseline_run2",
        "t1_sonnet_cc_baseline_run3",
    ],
    "hard_imp": [
        "t1_sonnet_hard_imp_run1",
        "t1_sonnet_hard_imp_run2",
        "t1_sonnet_hard_imp_run3",
    ],
    "terse_coop": [
        "t1_sonnet_terse_coop_run1",
        "t1_sonnet_terse_coop_run2",
        "t1_sonnet_terse_coop_run3",
    ],
    "verbose_with": [
        "t1_sonnet_verbose_with_run1",
        "t1_sonnet_verbose_with_run2",
        "t1_sonnet_verbose_with_run3",
    ],
    "techsavvy_imp": [
        "t1_sonnet_techsavvy_imp_run1",
        "t1_sonnet_techsavvy_imp_run2",
        "t1_sonnet_techsavvy_imp_run3",
    ],
}

GPT_PRESETS = {
    "original": [
        "t1_crossmodel_original_run1",
        "t1_crossmodel_original_run2",
        "t1_crossmodel_original_run3",
    ],
    "cc_baseline": [
        "t1_crossmodel_cc_baseline_run1",
        "t1_crossmodel_cc_baseline_run2",
        "t1_crossmodel_cc_baseline_run3",
    ],
    "hard_imp": [
        "t1_crossmodel_hard_imp_run1",
        "t1_crossmodel_hard_imp_run2",
        "t1_crossmodel_hard_imp_run3",
    ],
    "terse_coop": [
        "t1_crossmodel_terse_coop_run1",
        "t1_crossmodel_terse_coop_run2",
        "t1_crossmodel_terse_coop_run3",
    ],
    "verbose_with": [
        "t1_crossmodel_verbose_with_run1",
        "t1_crossmodel_verbose_with_run2",
        "t1_crossmodel_verbose_with_run3",
    ],
    "techsavvy_imp": [
        "t1_crossmodel_techsavvy_imp_run1",
        "t1_crossmodel_techsavvy_imp_run2",
        "t1_crossmodel_techsavvy_imp_run3",
    ],
}

TASK_RE = re.compile(r"\[(\w+)\](.*)\[PERSONA:(\w+)\]")


def parse_task(task_id: str) -> Tuple[str, List[str], str]:
    """Parse task_id to extract issue type, conditions, and persona."""
    m = TASK_RE.match(task_id)
    if not m:
        return "unknown", [], "None"
    issue = m.group(1)
    conditions = [c for c in m.group(2).split("|") if c]
    persona = m.group(3)
    return issue, conditions, persona


def get_reward(sim: dict) -> float:
    """Extract reward from simulation."""
    ri = sim.get("reward_info")
    if not ri:
        return 0.0
    return ri.get("reward", 0.0)


def load_preset_runs(preset_dirs: List[str]) -> Dict[str, List[dict]]:
    """
    Load simulation results for a preset across 3 runs.

    Returns:
        Dict mapping task_id to list of [run1_sim, run2_sim, run3_sim]
    """
    task_runs = defaultdict(list)

    for run_dir in preset_dirs:
        results_path = BASE / run_dir / "results.json"
        if not results_path.exists():
            print(f"Warning: {results_path} not found, skipping")
            continue

        with open(results_path) as f:
            data = json.load(f)

        for sim in data["simulations"]:
            task_id = sim["task_id"]
            task_runs[task_id].append(sim)

    return dict(task_runs)


def is_passing(task_sims: List[dict]) -> bool:
    """Returns True if ≥2/3 runs pass (reward=1.0)."""
    passing_runs = sum(1 for sim in task_sims if get_reward(sim) == 1.0)
    return passing_runs >= 2


def is_failing(task_sims: List[dict]) -> bool:
    """Returns True if ≤1/3 runs pass."""
    passing_runs = sum(1 for sim in task_sims if get_reward(sim) == 1.0)
    return passing_runs <= 1


def get_average_reward(task_sims: List[dict]) -> float:
    """Calculate average reward across runs."""
    if not task_sims:
        return 0.0
    rewards = [get_reward(sim) for sim in task_sims]
    return sum(rewards) / len(rewards)


def find_pattern1_failures(
    original_runs: Dict[str, List[dict]], baseline_runs: Dict[str, List[dict]]
) -> List[Tuple[str, float, float]]:
    """
    Find tasks where original succeeds but cc_baseline fails.

    Returns:
        List of (task_id, original_avg_reward, baseline_avg_reward)
    """
    failures = []

    all_tasks = set(original_runs.keys()) & set(baseline_runs.keys())

    for task_id in all_tasks:
        orig_sims = original_runs[task_id]
        base_sims = baseline_runs[task_id]

        if is_passing(orig_sims) and is_failing(base_sims):
            orig_avg = get_average_reward(orig_sims)
            base_avg = get_average_reward(base_sims)
            failures.append((task_id, orig_avg, base_avg))

    return failures


def find_pattern2_failures(
    baseline_runs: Dict[str, List[dict]],
    preset_runs: Dict[str, List[dict]],
    preset_name: str,
) -> List[Tuple[str, float, float]]:
    """
    Find tasks where baseline succeeds but preset fails.

    Returns:
        List of (task_id, baseline_avg_reward, preset_avg_reward)
    """
    failures = []

    all_tasks = set(baseline_runs.keys()) & set(preset_runs.keys())

    for task_id in all_tasks:
        base_sims = baseline_runs[task_id]
        preset_sims = preset_runs[task_id]

        if is_passing(base_sims) and is_failing(preset_sims):
            base_avg = get_average_reward(base_sims)
            preset_avg = get_average_reward(preset_sims)
            failures.append((task_id, base_avg, preset_avg))

    return failures


def extract_conversation_features(sim: dict) -> dict:
    """
    Extract behavioral features from a simulation.

    Returns:
        Dict with message counts, tool calls, termination info, etc.
    """
    messages = sim.get("messages", [])

    # Count messages by role
    user_messages = [m for m in messages if m.get("role") == "user"]
    agent_messages = [m for m in messages if m.get("role") == "assistant"]

    # Extract tool calls
    agent_tool_calls = []
    for msg in agent_messages:
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            agent_tool_calls.append(tc.get("name", "unknown"))

    # Check for travel mentions (for roaming tasks)
    first_travel_mention_turn = None
    travel_keywords = [
        "france",
        "travel",
        "abroad",
        "country",
        "trip",
        "international",
        "visiting",
    ]

    for i, msg in enumerate(user_messages):
        content = msg.get("content", "")
        if content and any(kw in content.lower() for kw in travel_keywords):
            first_travel_mention_turn = i
            break

    # Termination info
    termination_reason = sim.get("termination_reason", "UNKNOWN")

    # Reward components
    reward_info = sim.get("reward_info", {})

    return {
        "message_count": len(messages),
        "user_message_count": len(user_messages),
        "agent_message_count": len(agent_messages),
        "agent_tool_calls": agent_tool_calls,
        "termination_reason": termination_reason,
        "first_travel_mention_turn": first_travel_mention_turn,
        "conversation_duration": sim.get("duration", 0.0),
        "reward_components": {
            "env_assertions": reward_info.get("env_assertions", []),
            "action_checks": reward_info.get("action_checks", []),
            "db_check": reward_info.get("db_check", {}),
        },
    }


def attribute_failure(passing_sim: dict, failing_sim: dict, task_id: str) -> str:
    """
    Attribute failure to a root cause category.

    Categories:
    - information_disclosure: User didn't provide critical info
    - agent_strategy: Agent had info but chose wrong path
    - persona_interference: User behavior disrupted diagnosis
    - agent_capability: Fundamental model limitation
    - task_complexity: Too hard for any configuration
    """
    pass_features = extract_conversation_features(passing_sim)
    fail_features = extract_conversation_features(failing_sim)

    task_lower = task_id.lower()

    # Check for information disclosure failures (roaming tasks)
    if "roaming" in task_lower or "abroad" in task_lower:
        if fail_features["first_travel_mention_turn"] is None:
            return "information_disclosure"
        elif (
            pass_features["first_travel_mention_turn"] is not None
            and fail_features["first_travel_mention_turn"] is not None
        ):
            if (
                pass_features["first_travel_mention_turn"] < 10
                and fail_features["first_travel_mention_turn"] > 20
            ):
                return "information_disclosure"

    # Check for persona interference (premature termination)
    if (
        fail_features["termination_reason"] == "USER_TRANSFER"
        and pass_features["termination_reason"] == "USER_STOP"
    ):
        if fail_features["message_count"] < pass_features["message_count"] * 0.7:
            return "persona_interference"

    # Check for agent strategy differences
    failing_tools = set(fail_features["agent_tool_calls"])
    passing_tools = set(pass_features["agent_tool_calls"])
    critical_missing_tools = passing_tools - failing_tools

    if len(critical_missing_tools) > 2:
        return "agent_strategy"

    # Check for agent capability issues (GPT device tool limitations)
    if "gpt" in task_lower or any(
        tool.startswith("toggle_") or tool.startswith("set_")
        for tool in critical_missing_tools
    ):
        return "agent_capability"

    # Default to task complexity if unclear
    issue, conditions, persona = parse_task(task_id)
    if len(conditions) >= 7:
        return "task_complexity"

    return "agent_strategy"  # fallback


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    """Main analysis execution."""
    print_header("Persona Failure Attribution Analysis")

    # Load all data
    print("Loading experiment data...")

    sonnet_data = {
        preset: load_preset_runs(dirs) for preset, dirs in SONNET_PRESETS.items()
    }
    gpt_data = {preset: load_preset_runs(dirs) for preset, dirs in GPT_PRESETS.items()}

    print(
        f"Loaded {len(SONNET_PRESETS)} Sonnet presets, {len(GPT_PRESETS)} GPT presets"
    )

    # Pattern 1 Analysis: Original → CC Baseline failures
    print_header("Pattern 1: Original → CC Baseline Failures")

    print("\n--- Sonnet Agent ---")
    sonnet_p1 = find_pattern1_failures(
        sonnet_data["original"], sonnet_data["cc_baseline"]
    )

    print(f"Found {len(sonnet_p1)} tasks where original succeeds but cc_baseline fails")

    if sonnet_p1:
        print("\nTop 10 failures (sorted by performance drop):")
        sonnet_p1_sorted = sorted(sonnet_p1, key=lambda x: x[1] - x[2], reverse=True)

        for task_id, orig_avg, base_avg in sonnet_p1_sorted[:10]:
            issue, conditions, persona = parse_task(task_id)
            print(
                f"  {issue} ({len(conditions)} faults): {orig_avg:.1%} → {base_avg:.1%} ({orig_avg - base_avg:+.1%})"
            )
            print(f"    {task_id[:80]}...")

    print("\n--- GPT Agent ---")
    gpt_p1 = find_pattern1_failures(gpt_data["original"], gpt_data["cc_baseline"])

    print(f"Found {len(gpt_p1)} tasks where original succeeds but cc_baseline fails")

    if gpt_p1:
        print("\nTop 10 failures (sorted by performance drop):")
        gpt_p1_sorted = sorted(gpt_p1, key=lambda x: x[1] - x[2], reverse=True)

        for task_id, orig_avg, base_avg in gpt_p1_sorted[:10]:
            issue, conditions, persona = parse_task(task_id)
            print(
                f"  {issue} ({len(conditions)} faults): {orig_avg:.1%} → {base_avg:.1%} ({orig_avg - base_avg:+.1%})"
            )
            print(f"    {task_id[:80]}...")

    # Pattern 2 Analysis: Baseline → Persona failures
    print_header("Pattern 2: Baseline → Persona Failures")

    persona_presets = ["hard_imp", "terse_coop", "verbose_with", "techsavvy_imp"]

    for preset in persona_presets:
        print(f"\n{'─' * 80}")
        print(f"  {preset.upper()}")
        print(f"{'─' * 80}")

        print("\n--- Sonnet Agent ---")
        sonnet_p2 = find_pattern2_failures(
            sonnet_data["cc_baseline"], sonnet_data[preset], preset
        )

        print(
            f"Found {len(sonnet_p2)} tasks where cc_baseline succeeds but {preset} fails"
        )

        if sonnet_p2:
            print(f"\nTop 5 failures:")
            sonnet_p2_sorted = sorted(
                sonnet_p2, key=lambda x: x[1] - x[2], reverse=True
            )

            for task_id, base_avg, preset_avg in sonnet_p2_sorted[:5]:
                issue, conditions, persona = parse_task(task_id)
                print(
                    f"  {issue} ({len(conditions)} faults): {base_avg:.1%} → {preset_avg:.1%} ({base_avg - preset_avg:+.1%})"
                )

        print("\n--- GPT Agent ---")
        gpt_p2 = find_pattern2_failures(
            gpt_data["cc_baseline"], gpt_data[preset], preset
        )

        print(
            f"Found {len(gpt_p2)} tasks where cc_baseline succeeds but {preset} fails"
        )

        if gpt_p2:
            print(f"\nTop 5 failures:")
            gpt_p2_sorted = sorted(gpt_p2, key=lambda x: x[1] - x[2], reverse=True)

            for task_id, base_avg, preset_avg in gpt_p2_sorted[:5]:
                issue, conditions, persona = parse_task(task_id)
                print(
                    f"  {issue} ({len(conditions)} faults): {base_avg:.1%} → {preset_avg:.1%} ({base_avg - preset_avg:+.1%})"
                )

    print("\n" + "=" * 80)
    print("Analysis complete. See above for failure patterns.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
