#!/usr/bin/env python3
"""
Analyze GPT-4.5 "I don't have device tools" failure pattern.

This script examines the conversation trajectories where GPT-4.5 (gpt-5.4) fails
on tasks involving break_app_storage_permission and/or data_usage_exceeded,
specifically the pattern where GPT claims "I don't have device tools" and transfers.

Key finding: The answer is (c) with a twist -- The tools ARE user-side only
(the agent correctly does not have device tools in its tool list), but the agent's
POLICY tells it to INSTRUCT the user to perform these actions. GPT fails because
it misunderstands its role: instead of guiding the user through device actions,
it gives up and transfers to a human agent.

Run:
    python analyze_device_tool_failures.py
"""

import json

from cresta.paths import SIMULATIONS_DIR

BASE = SIMULATIONS_DIR

DATASETS = {
    "GPT_terse_coop_run1": "full_gpt54_terse_coop_run1/results.json",
    "GPT_hard_impatient": "full_gpt54_hard_impatient/results.json",
    "GPT_techsavvy_imp_run1": "full_gpt54_techsavvy_imp_run1/results.json",
    "Sonnet_terse_coop_run1": "full_sonnet_terse_coop_run1/results.json",
}

TARGET_TASKS = [
    "[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]",
    "[mobile_data_issue]data_saver_mode_on|data_usage_exceeded[PERSONA:Easy]",
]


def load_results(path):
    fpath = BASE / path
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


def find_simulation(data, task_id):
    for s in data.get("simulations", []):
        if s.get("task_id") == task_id:
            return s
    return None


def extract_conversation_summary(sim):
    """Extract a concise conversation flow from a simulation."""
    msgs = sim.get("messages", [])
    lines = []
    for i, msg in enumerate(msgs):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            names = [tc.get("name") for tc in tool_calls]
            args_summary = []
            for tc in tool_calls:
                a = tc.get("arguments", {})
                args_summary.append(f"{tc.get('name')}({json.dumps(a)})")
            lines.append(f"  [{i}] AGENT TOOL: {', '.join(args_summary)}")
        elif role == "assistant" and content:
            lines.append(f"  [{i}] AGENT: {content[:200]}")
        elif role == "user" and tool_calls:
            names = [tc.get("name") for tc in tool_calls]
            lines.append(f"  [{i}] USER TOOL: {', '.join(names)}")
        elif role == "user" and content:
            lines.append(f"  [{i}] USER: {content[:150]}")
        elif role == "tool":
            lines.append(f"  [{i}] TOOL RESULT: {str(content)[:120]}")
    return "\n".join(lines)


def main():
    print("=" * 90)
    print("INVESTIGATION: GPT-4.5 'I don't have device tools' failure pattern")
    print("=" * 90)

    # =====================================================================
    # Section 1: Architecture -- Agent vs User tools
    # =====================================================================
    print("\n" + "=" * 90)
    print("SECTION 1: TOOL ARCHITECTURE")
    print("=" * 90)
    print("""
The tau2 telecom domain has TWO separate tool sets:

AGENT (assistant) tools (from tools.py) -- given to GPT/Sonnet via API:
  - get_customer_by_phone, get_customer_by_id, get_customer_by_name
  - get_details_by_id, get_data_usage, get_bills_for_customer
  - suspend_line, resume_line, enable_roaming, disable_roaming
  - refuel_data, send_payment_request
  - transfer_to_human_agents

USER tools (from user_tools.py) -- used by the user simulator:
  - check_status_bar, check_network_status, check_network_mode_preference
  - run_speed_test, check_sim_status, check_apn_settings
  - check_wifi_status, check_wifi_calling_status, check_vpn_status
  - check_data_restriction_status, check_installed_apps
  - check_app_status, check_app_permissions, can_send_mms
  - toggle_airplane_mode, toggle_data, toggle_roaming
  - toggle_data_saver_mode, toggle_wifi, toggle_wifi_calling
  - set_network_mode_preference, set_apn_settings, reset_apn_settings
  - reseat_sim_card, grant_app_permission, reboot_device
  - connect_vpn, disconnect_vpn, check_payment_request, make_payment

KEY: The agent NEVER has device tools in its API tool list.
     The POLICY tells the agent to INSTRUCT THE USER to perform device actions.
     The user simulator then executes those tools on its side.
""")

    # =====================================================================
    # Section 2: Comparative results
    # =====================================================================
    print("=" * 90)
    print("SECTION 2: COMPARATIVE RESULTS (GPT vs Sonnet)")
    print("=" * 90)

    for tid in TARGET_TASKS:
        print(f"\nTask: {tid}")
        for ds_name, path in DATASETS.items():
            data = load_results(path)
            if data is None:
                continue
            sim = find_simulation(data, tid)
            if sim is None:
                continue
            reward = sim.get("reward_info", {}).get("reward", "?")
            msgs = sim.get("messages", [])
            agent_tools_used = set()
            for msg in msgs:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        agent_tools_used.add(tc.get("name"))
            print(f"  {ds_name}: reward={reward}, tools={sorted(agent_tools_used)}")

    # =====================================================================
    # Section 3: Full conversation flow for key failures
    # =====================================================================
    print("\n" + "=" * 90)
    print("SECTION 3: CONVERSATION FLOWS FOR KEY FAILURES")
    print("=" * 90)

    # Show the simplest GPT failure case
    data = load_results("full_gpt54_terse_coop_run1/results.json")
    tid = "[mobile_data_issue]data_saver_mode_on|data_usage_exceeded[PERSONA:Easy]"
    sim = find_simulation(data, tid)
    print(f"\n--- GPT FAILURE: {tid} ---")
    print(f"Reward: {sim.get('reward_info', {}).get('reward')}")
    print(f"Expected: toggle_data_saver_mode (user) + refuel_data (agent)")
    print(f"\nConversation:")
    print(extract_conversation_summary(sim))

    # Sonnet success on same task
    sdata = load_results("full_sonnet_terse_coop_run1/results.json")
    ssim = find_simulation(sdata, tid)
    print(f"\n--- SONNET SUCCESS: {tid} ---")
    print(f"Reward: {ssim.get('reward_info', {}).get('reward')}")
    print(f"\nConversation:")
    print(extract_conversation_summary(ssim))

    # =====================================================================
    # Section 4: Aggregate failure statistics
    # =====================================================================
    print("\n" + "=" * 90)
    print("SECTION 4: AGGREGATE FAILURE STATISTICS")
    print("=" * 90)

    gpt_datasets = {k: v for k, v in DATASETS.items() if "gpt" in k.lower()}

    for ds_name, path in gpt_datasets.items():
        data = load_results(path)
        if data is None:
            continue
        sims = data.get("simulations", [])
        total = len(sims)
        failures = [s for s in sims if s.get("reward_info", {}).get("reward", 1) == 0.0]

        transfer_count = 0
        wrong_line_count = 0
        no_refuel_despite_needed = 0

        for s in failures:
            msgs = s.get("messages", [])
            did_transfer = False
            did_refuel = False
            checked_wrong_line = False

            for msg in msgs:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                        if name == "transfer_to_human_agents":
                            did_transfer = True
                        if name == "refuel_data":
                            did_refuel = True
                        if name in ("get_details_by_id", "get_data_usage"):
                            line_arg = args.get("line_id", args.get("id", ""))
                            if line_arg == "L1001":
                                checked_wrong_line = True

            if did_transfer:
                transfer_count += 1
            if checked_wrong_line:
                wrong_line_count += 1
            if "data_usage_exceeded" in s.get("task_id", "") and not did_refuel:
                no_refuel_despite_needed += 1

        print(f"\n{ds_name}:")
        print(f"  Total tasks: {total}")
        print(f"  Failures: {len(failures)}")
        print(f"  Failures with transfer: {transfer_count}")
        print(f"  Failures checking wrong line (L1001 not L1002): {wrong_line_count}")
        print(f"  data_usage_exceeded tasks without refuel: {no_refuel_despite_needed}")

    # =====================================================================
    # Section 5: Root cause analysis
    # =====================================================================
    print("\n" + "=" * 90)
    print("SECTION 5: ROOT CAUSE ANALYSIS")
    print("=" * 90)
    print("""
ANSWER: The failure is OPTION (C) with additional compounding issues.

1. ARCHITECTURE (c): Device tools are USER-SIDE ONLY.
   The agent CORRECTLY does not have check_data_restriction_status,
   toggle_data_saver_mode, grant_app_permission, etc. in its tool list.
   These are user-side tools.

   BUT: The agent's POLICY explicitly describes these tools under
   "What the user can do on their device" and instructs the agent to
   GUIDE the user through performing these actions. The agent should say
   "Please run check_data_restriction_status()" and the user will do it.

2. GPT's FAILURE MODE: When GPT encounters a situation where device-side
   troubleshooting is needed (after account-side checks are done), it:
   - Says "I do not have access to those" or similar
   - Immediately transfers to a human agent
   - Fails to instruct the user to perform device actions

3. COMPOUNDING BUG - WRONG LINE (L1001 vs L1002):
   In 40/47 failures (85%), GPT also checks the WRONG line. The user's
   phone is 555-123-2002 (line L1002), but GPT often checks L1001 first.
   When it gets details for L1001, the data usage looks fine (3.2/5.0 GB),
   so GPT never realizes data is exceeded on L1002 (15.1/15.0 GB) and
   never offers to refuel.

4. SONNET COMPARISON: Sonnet handles the same tasks correctly by:
   - Checking ALL lines (L1001 then L1002) to find the right one
   - Instructing the user to run device diagnostics
     (e.g., "Could you please run check_data_restriction_status()?")
   - Acting on results (telling user to toggle_data_saver_mode)
   - Recognizing data exceeded and offering refuel

   Sonnet gets reward=1.0 on both target tasks; GPT gets reward=0.0.

5. SPECIFIC EXAMPLE (data_saver_mode_on|data_usage_exceeded):
   - GPT successfully refuels data (good!)
   - User runs speed test: 44 Mbps (Good, not Excellent)
   - User says "still isn't where I need it"
   - GPT says: "I can only continue technical troubleshooting using the
     supported device diagnostic actions, and I do not have access to
     those from here. YOU ARE BEING TRANSFERRED."
   - Should have said: "Please run check_data_restriction_status()"
   - Data Saver was ON, causing speed reduction
   - Sonnet correctly identifies this and tells user to toggle it off

CONCLUSION: This is NOT a hallucination about tool availability (b).
The tools genuinely aren't in GPT's API tool list (c is correct).
The failure is that GPT doesn't understand the agent-instructs-user
paradigm described in its policy. Sonnet correctly reads the policy
section "What the user can do on their device" and instructs the user
to perform those actions. GPT treats "I don't have the tool" as a
dead end, when the policy says to TELL THE USER to use those tools.
""")


if __name__ == "__main__":
    main()
