#!/usr/bin/env python3
"""
Extract real conversation trajectories from tau2-bench telecom simulation data
for two specific GPT failure patterns, formatted as markdown.

Case C1: GPT Policy Following Failure — "I Don't Have Device Tools"
Case C2: GPT Wrong Line Lookup (L1001 vs L1002)
"""

import json
import textwrap
from pathlib import Path

BASE = str(Path(__file__).resolve().parents[2] / "artifacts" / "simulations")


def load_simulation(run_dir: str, task_id: str):
    """Load a specific simulation by task_id from a run directory."""
    path = f"{BASE}/{run_dir}/results.json"
    with open(path) as f:
        data = json.load(f)
    for s in data["simulations"]:
        if s["task_id"] == task_id:
            return s
    raise ValueError(f"Task {task_id!r} not found in {run_dir}")


def format_tool_result(content_str: str, max_len: int = 600) -> str:
    """Format a tool result, truncating very long ones but keeping key fields."""
    try:
        obj = json.loads(content_str)
        # For customer records, keep key fields
        if isinstance(obj, dict):
            formatted = json.dumps(obj, indent=2)
            if len(formatted) > max_len:
                # Keep the full thing but note it's long
                return formatted
            return formatted
        return content_str
    except (json.JSONDecodeError, TypeError):
        return content_str


def format_message(msg: dict, turn_counter: dict) -> str:
    """Format a single message as markdown."""
    role = msg["role"]
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])

    lines = []

    if role == "assistant":
        turn_counter["assistant"] = turn_counter.get("assistant", 0) + 1
        turn = turn_counter["assistant"]
        lines.append(f"**Agent** (turn {turn}):")

        if content:
            # Try to parse JSON message format
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "message" in parsed:
                    content = parsed["message"]
            except (json.JSONDecodeError, TypeError):
                pass
            lines.append(f"> {content}")
            lines.append("")

        if tool_calls:
            lines.append("[Tool calls]:")
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                args_str = (
                    ", ".join(f"{k}={v!r}" for k, v in args.items())
                    if isinstance(args, dict)
                    else str(args)
                )
                lines.append(f"- `{name}({args_str})`")
            lines.append("")

    elif role == "user":
        turn_counter["user"] = turn_counter.get("user", 0) + 1
        turn = turn_counter["user"]

        # Check if user made tool calls (device-side actions)
        if tool_calls:
            lines.append(f"**User** (turn {turn}):")
            if content:
                lines.append(f"> {content}")
                lines.append("")
            lines.append("[User device actions]:")
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                args_str = (
                    ", ".join(f"{k}={v!r}" for k, v in args.items())
                    if isinstance(args, dict)
                    else str(args)
                )
                lines.append(f"- `{name}({args_str})`")
            lines.append("")
        else:
            lines.append(f"**User** (turn {turn}):")
            lines.append(f"> {content}")
            lines.append("")

    elif role == "tool":
        # Tool result — attach to previous tool call
        formatted = format_tool_result(content)
        lines.append(f"**Tool Result**:")
        lines.append(f"```json")
        lines.append(formatted)
        lines.append(f"```")
        lines.append("")

    return "\n".join(lines)


def format_trajectory(sim: dict) -> str:
    """Format a full conversation trajectory."""
    parts = []
    turn_counter = {}
    for msg in sim["messages"]:
        formatted = format_message(msg, turn_counter)
        if formatted.strip():
            parts.append(formatted)
    return "\n".join(parts)


def main():
    output = []

    # =========================================================================
    # CASE C1: GPT Policy Following Failure — "I Don't Have Device Tools"
    # =========================================================================
    c1_task_id = "[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]"
    c1_gpt_run = "full_gpt54_techsavvy_imp_run1"
    c1_sonnet_run = "full_sonnet_techsavvy_imp_run1"

    c1_gpt = load_simulation(c1_gpt_run, c1_task_id)
    c1_sonnet = load_simulation(c1_sonnet_run, c1_task_id)

    c1_gpt_reward = c1_gpt["reward_info"]["reward"]
    c1_sonnet_reward = c1_sonnet["reward_info"]["reward"]
    c1_gpt_msgs = len(c1_gpt["messages"])
    c1_sonnet_msgs = len(c1_sonnet["messages"])

    # Extract fault list from task_id
    c1_faults_str = c1_task_id.split("]")[1].split("[")[0]
    c1_faults = c1_faults_str.split("|")

    output.append(
        '### Case C1: GPT Policy Following Failure — "I Don\'t Have Device Tools"'
    )
    output.append("")
    output.append(f"**Task**: `{c1_task_id}`")
    output.append(f"**Faults**: {', '.join(f'`{f}`' for f in c1_faults)}")
    output.append(f"**Persona**: Hard (techsavvy, impatient)")
    output.append(f"**GPT Result**: {c1_gpt_reward} (fail), {c1_gpt_msgs} messages")
    output.append(
        f"**Sonnet Result**: {c1_sonnet_reward} (pass), {c1_sonnet_msgs} messages"
    )
    output.append("")

    # GPT Trajectory
    output.append(f"#### GPT-4.5 Trajectory (techsavvy_imp / run1)")
    output.append("")
    output.append(format_trajectory(c1_gpt))
    output.append("")

    # Analysis
    output.append("#### Analysis")
    output.append("")
    output.append(
        textwrap.dedent("""\
    GPT-4.5 fails immediately after identifying the account. At message 8, instead of guiding the user
    through device-side troubleshooting (checking airplane mode, app permissions), it declares:

    > "I'm limited to account and carrier-side actions here. For MMS troubleshooting, I need to check
    > and guide settings on your phone, and I don't have access to your device diagnostics in this chat."

    This is a **policy following failure**. The agent's policy explicitly includes device troubleshooting
    tools (toggle_airplane_mode, check_app_permissions, grant_app_permission, etc.) and mandates a
    specific troubleshooting sequence. GPT-4.5 has the tools available but incorrectly believes it cannot
    use them — it hallucinates a capability limitation that doesn't exist.

    The required actions were:
    1. `toggle_airplane_mode()` — turn off airplane mode (user device action)
    2. `grant_app_permission(app_name="messaging", permission="storage")` — fix storage permission
    3. `grant_app_permission(app_name="messaging", permission="sms")` — fix SMS permission

    None of these were attempted. GPT transferred to a human agent after just 5 turns of actual conversation.
    """)
    )
    output.append("")

    # Sonnet Trajectory
    output.append(f"#### Sonnet Trajectory (same task, techsavvy_imp / run1)")
    output.append("")
    output.append(format_trajectory(c1_sonnet))
    output.append("")

    # Comparison
    output.append("#### Comparison")
    output.append("")
    output.append(
        textwrap.dedent("""\
    | Aspect | GPT-4.5 | Sonnet |
    |--------|---------|--------|
    | Device tools used | 0 | 7 (check_network_status, can_send_mms, toggle_airplane_mode, check_apn_settings, check_network_mode_preference, check_wifi_calling_status, check_app_permissions, grant_app_permission x2) |
    | Identified root causes | 0/2 | 2/2 (airplane mode ON, missing storage+sms permissions) |
    | Troubleshooting turns | 0 | 20+ |
    | Outcome | Transferred to human (fail) | MMS restored (pass) |
    | Failure mode | Hallucinated capability limitation | N/A — correctly followed policy |

    Sonnet systematically worked through the troubleshooting tree: check MMS → check network status →
    find airplane mode ON → toggle it → recheck → check APN → check network mode → check wifi calling →
    check app permissions → find missing storage+sms → grant both → confirm MMS works. This is exactly
    the procedure specified in the agent policy.
    """)
    )
    output.append("")

    # =========================================================================
    # CASE C2: GPT Wrong Line Lookup (L1001 vs L1002)
    # =========================================================================
    c2_task_id = "[mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]"
    c2_gpt_run = "full_gpt54_hard_impatient_run3"
    c2_sonnet_run = "full_sonnet_hard_impatient_run3"

    c2_gpt = load_simulation(c2_gpt_run, c2_task_id)
    c2_sonnet = load_simulation(c2_sonnet_run, c2_task_id)

    c2_gpt_reward = c2_gpt["reward_info"]["reward"]
    c2_sonnet_reward = c2_sonnet["reward_info"]["reward"]
    c2_gpt_msgs = len(c2_gpt["messages"])
    c2_sonnet_msgs = len(c2_sonnet["messages"])

    c2_faults_str = c2_task_id.split("]")[1].split("[")[0]
    c2_faults = c2_faults_str.split("|")

    output.append("---")
    output.append("")
    output.append("### Case C2: GPT Wrong Line Lookup — L1001 vs L1002")
    output.append("")
    output.append(f"**Task**: `{c2_task_id}`")
    output.append(f"**Faults**: {', '.join(f'`{f}`' for f in c2_faults)}")
    output.append(f"**Persona**: Easy (cooperative)")
    output.append(f"**GPT Result**: {c2_gpt_reward} (fail), {c2_gpt_msgs} messages")
    output.append(
        f"**Sonnet Result**: {c2_sonnet_reward} (pass), {c2_sonnet_msgs} messages"
    )
    output.append("")

    # GPT Trajectory
    output.append(f"#### GPT-4.5 Trajectory (hard_impatient / run3)")
    output.append("")
    output.append(format_trajectory(c2_gpt))
    output.append("")

    # Analysis
    output.append("#### Analysis")
    output.append("")
    output.append(
        textwrap.dedent("""\
    GPT-4.5 makes a critical error at message 8: when the user says their problem is on 555-123-2002,
    GPT queries `get_details_by_id(id="L1001")` and `get_data_usage(customer_id="C1001", line_id="L1001")`.

    The result shows **L1001 = 555-123-2001** (a different phone number!) with 3.2/5.0 GB used — well
    under the limit. GPT notices the mismatch ("the line details I pulled were for a different line")
    but then enters a **verification death spiral**: it asks for the customer ID, name, DOB — none of
    which help identify which line_id maps to 555-123-2002.

    GPT never queries L1002 or L1003 to find the correct line. It had 3 line IDs available
    (L1001, L1002, L1003) and only checked the first one.

    The actual problem was on **L1002 (555-123-2002)**: 15.1 GB used / 15.0 GB limit — data exceeded.
    The fix required:
    1. `toggle_roaming()` — enable data roaming on device (user is abroad in France)
    2. `refuel_data(customer_id="C1001", line_id="L1002", gb_amount=2)` — add data

    Neither action was taken because GPT never found the right line.
    """)
    )
    output.append("")

    # Sonnet Trajectory
    output.append(f"#### Sonnet Trajectory (same task, hard_impatient / run3)")
    output.append("")
    output.append(format_trajectory(c2_sonnet))
    output.append("")

    # Comparison
    output.append("#### Comparison")
    output.append("")
    output.append(
        textwrap.dedent("""\
    | Aspect | GPT-4.5 | Sonnet |
    |--------|---------|--------|
    | Line queried first | L1001 (wrong — 555-123-2001) | L1001 (wrong — 555-123-2001) |
    | Recovery after wrong line | Gave up, asked for re-verification | Immediately queried L1002 and L1003 |
    | Found correct line (L1002) | No | Yes |
    | Discovered data exceeded (15.1/15.0 GB) | No | Yes |
    | Enabled device roaming | No | Yes (toggle_roaming) |
    | Refueled data | No | Yes (2 GB for $4.00) |
    | Outcome | Transferred to human (fail) | Data restored at 275 Mbps (pass) |

    The key difference: when Sonnet queried L1001 and found it was 555-123-2001 (wrong number), it
    immediately queried L1002 and L1003 in the **same turn** (message 18). This is basic error recovery —
    if the first line doesn't match, check the others. GPT-4.5 instead entered a re-verification loop,
    asking the user to re-identify themselves, which the user (already identified via phone number)
    found confusing and frustrating.

    Sonnet then systematically resolved both faults:
    1. Noticed data roaming was OFF on the device → guided user to toggle_roaming()
    2. Found data usage exceeded (15.1/15.0 GB) → refueled 2 GB at $2/GB = $4.00
    Both were needed because the speed test still failed after just enabling roaming.
    """)
    )

    # Print everything
    print("\n".join(output))


if __name__ == "__main__":
    main()
