#!/usr/bin/env python3
"""Summarize banking preset test results."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_results(output_dir):
    """Analyze results from banking preset test."""
    output_path = Path(output_dir)

    results = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "total_messages": 0,
            "total_duration": 0,
            "termination_stop": 0,
            "termination_transfer": 0,
        }
    )

    # Iterate through preset directories
    for preset_dir in output_path.iterdir():
        if not preset_dir.is_dir():
            continue

        preset_name = preset_dir.name
        results_file = preset_dir / "results.json"

        if not results_file.exists():
            print(f"Warning: No results.json found for {preset_name}")
            continue

        with open(results_file) as f:
            data = json.load(f)

        for sim in data.get("simulations", []):
            results[preset_name]["total"] += 1

            # Reward
            reward = sim["reward_info"]["reward"]
            if reward == 1.0:
                results[preset_name]["success"] += 1
            else:
                results[preset_name]["failed"] += 1

            # Messages
            messages = sim.get("messages", [])
            results[preset_name]["total_messages"] += len(messages)

            # Duration
            results[preset_name]["total_duration"] += sim.get("duration", 0)

            # Termination
            last_user_msg = None
            for m in reversed(messages):
                if m["role"] == "user" and m.get("content"):
                    last_user_msg = m
                    break

            if last_user_msg:
                content = last_user_msg.get("content", "")
                if "###STOP###" in content:
                    results[preset_name]["termination_stop"] += 1
                elif "###TRANSFER###" in content:
                    results[preset_name]["termination_transfer"] += 1

    return results


def print_summary(results):
    """Print summary table."""
    print("\n# Banking Domain Preset Test Summary\n")
    print("**Tasks**: task_086, task_085, task_067, task_064, task_021 (5 tasks)")
    print("**Agent**: Sonnet 4.6 (temp=1.0)")
    print("**User**: ClaudeCodeUserSimulator (temp=1.0)\n")

    print("## Overall Results\n")
    print(
        "| Preset | Success | Failed | Success Rate | Avg Messages | Avg Duration | STOP | TRANSFER |"
    )
    print(
        "|--------|---------|--------|--------------|--------------|--------------|------|----------|"
    )

    # Sort by success rate
    sorted_presets = sorted(
        results.items(),
        key=lambda x: x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True,
    )

    for preset, stats in sorted_presets:
        total = stats["total"]
        if total == 0:
            continue

        success = stats["success"]
        success_rate = success / total * 100
        avg_msgs = stats["total_messages"] / total
        avg_duration = stats["total_duration"] / total
        stop_count = stats["termination_stop"]
        transfer_count = stats["termination_transfer"]

        print(
            f"| {preset:30} | {success:7} | {stats['failed']:6} | {success_rate:11.1f}% | "
            f"{avg_msgs:12.1f} | {avg_duration:12.1f}s | {stop_count:4} | {transfer_count:8} |"
        )

    print("\n## Key Observations\n")

    # Find best and worst
    if sorted_presets:
        best = sorted_presets[0]
        worst = sorted_presets[-1]

        print(
            f"- **Best performer**: {best[0]} ({best[1]['success']}/{best[1]['total']} = "
            f"{best[1]['success'] / best[1]['total'] * 100:.1f}%)"
        )
        print(
            f"- **Worst performer**: {worst[0]} ({worst[1]['success']}/{worst[1]['total']} = "
            f"{worst[1]['success'] / worst[1]['total'] * 100:.1f}%)"
        )

        # Compare to baseline
        baseline = results.get("cc_baseline")
        if baseline and baseline["total"] > 0:
            baseline_rate = baseline["success"] / baseline["total"]
            print(f"\n**Comparison to cc_baseline ({baseline_rate * 100:.1f}%):**\n")

            for preset, stats in sorted_presets:
                if preset == "cc_baseline" or stats["total"] == 0:
                    continue
                preset_rate = stats["success"] / stats["total"]
                delta = (preset_rate - baseline_rate) * 100
                print(f"- {preset}: {delta:+.1f}pp")


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_banking_test.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    results = analyze_results(output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
