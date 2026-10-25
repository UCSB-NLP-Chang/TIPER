#!/usr/bin/env python3
"""Categorize banking_knowledge tasks by intent and action patterns.

Unlike telecom tasks which have explicit category labels in task IDs,
banking tasks need to be categorized based on their action patterns and content.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

TASKS_DIR = DATA_ROOT / "data/tau2/domains/banking_knowledge/tasks"
OUTPUT_PATH = DATA_ROOT / "data/tau2/domains/banking_knowledge/task_categories.json"


def categorize_task(task: dict) -> str:
    """Categorize a task based on its actions and patterns.

    Categories:
    - credit_card_inquiry: Customer asking about/applying for credit cards
    - referral_program: Submit or query referrals
    - human_transfer: Agent must transfer to human agent
    - discoverable_agent_tool: Agent must find and use discoverable tools
    - discoverable_user_tool: User must be given discoverable tools
    - account_management: Email changes, transactions, account queries
    - mixed_complex: Multiple action types (2+ categories)
    """
    actions = task["evaluation_criteria"]["actions"]
    action_names = set(a["name"] for a in actions)
    user_tools = set(task.get("user_tools", []))

    # Count category signals
    categories = []

    # Credit card related
    if (
        "apply_for_credit_card" in action_names
        or "get_credit_card_accounts_by_user" in action_names
    ):
        categories.append("credit_card_inquiry")

    # Referral related
    if "submit_referral" in action_names or "get_referrals_by_user" in action_names:
        categories.append("referral_program")

    # Human transfer required
    if (
        "transfer_to_human_agents" in action_names
        or "request_human_agent_transfer" in action_names
    ):
        categories.append("human_transfer")

    # Discoverable agent tools (agent must unlock/call)
    if (
        "unlock_discoverable_agent_tool" in action_names
        or "call_discoverable_agent_tool" in action_names
    ):
        categories.append("discoverable_agent_tool")

    # Discoverable user tools (agent must give to user)
    if (
        "give_discoverable_user_tool" in action_names
        or "call_discoverable_user_tool" in action_names
    ):
        categories.append("discoverable_user_tool")

    # Account management
    if "change_user_email" in action_names or "submit_transaction" in action_names:
        categories.append("account_management")

    # Categorization logic
    if len(categories) == 0:
        # Verification only or unknown
        if "log_verification" in action_names and len(action_names) == 1:
            return "verification_only"
        return "unknown"
    elif len(categories) == 1:
        return categories[0]
    else:
        # Mixed category - prioritize by complexity
        if "discoverable_agent_tool" in categories and len(action_names) > 5:
            return "discoverable_agent_tool"  # Dominant pattern
        return "mixed_complex"


def main():
    task_categories = defaultdict(list)
    task_metadata = {}

    for task_file in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.load(open(task_file))
        task_id = task["id"]

        category = categorize_task(task)
        task_categories[category].append(task_id)

        # Store metadata
        actions = task["evaluation_criteria"]["actions"]
        action_names = [a["name"] for a in actions]
        task_metadata[task_id] = {
            "category": category,
            "action_count": len(actions),
            "unique_actions": sorted(set(action_names)),
            "user_tools": task.get("user_tools", []),
            "required_documents": len(task.get("required_documents", [])),
        }

    print("=" * 80)
    print("Banking Task Categorization")
    print("=" * 80)
    print()

    # Sort by count
    sorted_categories = sorted(task_categories.items(), key=lambda x: -len(x[1]))

    for category, tasks in sorted_categories:
        print(f"{category:30s}: {len(tasks):3d} tasks ({len(tasks) / 97 * 100:.1f}%)")

    print()
    print("=" * 80)
    print("Sample tasks per category:")
    print("=" * 80)

    for category, tasks in sorted_categories:
        print(f"\n{category} ({len(tasks)} tasks):")
        for task_id in tasks[:3]:
            meta = task_metadata[task_id]
            print(
                f"  {task_id}: {meta['action_count']} actions, {meta['required_documents']} docs"
            )
            print(f"    Actions: {', '.join(meta['unique_actions'][:3])}")

    # Save to JSON
    output = {
        "categories": {cat: tasks for cat, tasks in task_categories.items()},
        "task_metadata": task_metadata,
        "summary": {
            cat: {
                "count": len(tasks),
                "percentage": round(len(tasks) / 97 * 100, 1),
                "samples": tasks[:5],
            }
            for cat, tasks in sorted_categories
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 80)
    print(f"✅ Categorization saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
