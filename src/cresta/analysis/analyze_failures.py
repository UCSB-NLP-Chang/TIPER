#!/usr/bin/env python3
"""
Analyze tau2-bench telecom simulation failures:
1. Non-roaming failures (tasks without set_abroad/user_abroad that still fail)
2. Persona-dependent failures (tasks that pass with some personas but fail with others)
3. Extract conversation snippets for the most interesting cases
"""

import json
import os
from collections import defaultdict

from cresta.paths import SIMULATIONS_DIR

BASE = str(SIMULATIONS_DIR)

# Define all run directories grouped by agent+persona
SONNET_RUNS = {
    "sonnet_terse_coop": [
        "full_sonnet_terse_coop_run1",
        "full_sonnet_terse_coop_run2",
        "full_sonnet_terse_coop_run3",
    ],
    "sonnet_hard_impatient": [
        "full_sonnet_hard_impatient",
        "full_sonnet_hard_impatient_run2",
        "full_sonnet_hard_impatient_run3",
    ],
    "sonnet_verbose_withhold": [
        "full_sonnet_verbose_withhold_run1",
        "full_sonnet_verbose_withhold_run2",
        "full_sonnet_verbose_withhold_run3",
    ],
    "sonnet_techsavvy_imp": [
        "full_sonnet_techsavvy_imp_run1",
        "full_sonnet_techsavvy_imp_run2",
        "full_sonnet_techsavvy_imp_run3",
    ],
}

GPT_RUNS = {
    "gpt54_terse_coop": [
        "full_gpt54_terse_coop_run1",
        "full_gpt54_terse_coop_run2",
        "full_gpt54_terse_coop_run3",
    ],
    "gpt54_hard_impatient": [
        "full_gpt54_hard_impatient",
        "full_gpt54_hard_impatient_run2",
        "full_gpt54_hard_impatient_run3",
    ],
    "gpt54_verbose_withhold": [
        "full_gpt54_verbose_withhold_run1",
        "full_gpt54_verbose_withhold_run2",
        "full_gpt54_verbose_withhold_run3",
    ],
    "gpt54_techsavvy_imp": [
        "full_gpt54_techsavvy_imp_run1",
        "full_gpt54_techsavvy_imp_run2",
        "full_gpt54_techsavvy_imp_run3",
    ],
}

ALL_RUNS = {**SONNET_RUNS, **GPT_RUNS}


def load_results(dir_name: str) -> dict:
    path = os.path.join(BASE, dir_name, "results.json")
    with open(path) as f:
        return json.load(f)


def is_roaming_task(task_id: str) -> bool:
    """Check if task involves roaming based on task_id."""
    return "user_abroad" in task_id


def get_base_task_id(task_id: str) -> str:
    """Strip persona tag from task_id to get base task."""
    if "[PERSONA:" in task_id:
        return task_id[: task_id.index("[PERSONA:")]
    return task_id


def get_faults_from_task_id(task_id: str) -> list[str]:
    """Extract fault names from task_id."""
    # Format: [type]fault1|fault2|...[PERSONA:X]
    base = get_base_task_id(task_id)
    # Remove the [type] prefix
    if "]" in base:
        faults_str = base[base.index("]") + 1 :]
    else:
        faults_str = base
    return faults_str.split("|") if faults_str else []


def get_task_type(task_id: str) -> str:
    """Extract task type (service_issue, mobile_data_issue, mms_issue)."""
    if task_id.startswith("["):
        return task_id[1 : task_id.index("]")]
    return "unknown"


def extract_message_text(msg: dict) -> str:
    """Extract text content from a message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "tool_use":
                    texts.append(
                        f"[TOOL: {part.get('name', '?')}({json.dumps(part.get('input', {}))[:100]})]"
                    )
                elif part.get("type") == "tool_result":
                    result_content = part.get("content", "")
                    if isinstance(result_content, list):
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                texts.append(f"[RESULT: {rc.get('text', '')[:150]}]")
                    elif isinstance(result_content, str):
                        texts.append(f"[RESULT: {result_content[:150]}]")
        return " ".join(texts)
    return str(content)[:200]


def extract_conversation_snippet(
    sim: dict, max_first=3, max_last=4, max_critical=3
) -> str:
    """Extract key parts of a conversation for analysis."""
    messages = sim.get("messages", [])
    if not messages:
        return "  (no messages)"

    lines = []

    # First few messages (opening)
    lines.append("  --- OPENING ---")
    for i, m in enumerate(messages[: min(max_first, len(messages))]):
        role = m["role"].upper()
        text = extract_message_text(m)
        lines.append(f"  [{i}] {role}: {text[:250]}")

    # Look for critical failure points: messages mentioning payment, suspension, escalation, etc.
    critical_keywords = [
        "suspend",
        "overdue",
        "payment",
        "bill",
        "contract",
        "cancel",
        "expired",
        "locked",
        "PUK",
        "escalat",
        "unable",
        "cannot",
        "sorry",
        "unfortunately",
        "give up",
        "frustrated",
        "not help",
        "store",
        "visit",
        "in-person",
    ]

    critical_msgs = []
    for i, m in enumerate(messages):
        text = extract_message_text(m).lower()
        if any(kw.lower() in text for kw in critical_keywords):
            critical_msgs.append((i, m))

    if critical_msgs:
        lines.append("  --- CRITICAL POINTS ---")
        shown = 0
        for i, m in critical_msgs:
            if shown >= max_critical:
                lines.append(
                    f"  ... ({len(critical_msgs) - shown} more critical messages)"
                )
                break
            role = m["role"].upper()
            text = extract_message_text(m)
            lines.append(f"  [{i}] {role}: {text[:300]}")
            shown += 1

    # Last few messages
    if len(messages) > max_first:
        lines.append("  --- ENDING ---")
        start_idx = max(max_first, len(messages) - max_last)
        for i in range(start_idx, len(messages)):
            m = messages[i]
            role = m["role"].upper()
            text = extract_message_text(m)
            lines.append(f"  [{i}] {role}: {text[:300]}")

    lines.append(
        f"  --- OUTCOME: reward={sim['reward_info']['reward']}, "
        f"termination={sim.get('termination_reason', '?')}, "
        f"total_messages={len(messages)} ---"
    )

    return "\n".join(lines)


def get_reward_details(sim: dict) -> str:
    """Get details about why a reward was 0."""
    ri = sim.get("reward_info", {})
    details = []
    if "db_check" in ri:
        db = ri["db_check"]
        if not db.get("db_match", True):
            details.append(f"db_match=False (reward={db.get('db_reward')})")
    if "env_assertions" in ri:
        for ea in ri["env_assertions"]:
            if not ea.get("met", True):
                assertion = ea.get("env_assertion", {})
                details.append(
                    f"FAILED: {assertion.get('func_name', '?')}("
                    f"{json.dumps(assertion.get('arguments', {}))}) "
                    f"msg={assertion.get('message', '')}"
                )
    return "; ".join(details) if details else "no details"


def main():
    print("=" * 100)
    print("TAU2-BENCH TELECOM FAILURE ANALYSIS")
    print("=" * 100)

    # Load all data
    all_data = {}
    for persona_key, dirs in ALL_RUNS.items():
        for d in dirs:
            all_data[d] = load_results(d)

    # Build task registry: base_task_id -> {persona -> [results across runs]}
    # Structure: task_results[base_task_id][persona_key] = [(reward, termination, sim_ref, dir_name)]
    task_results = defaultdict(lambda: defaultdict(list))

    for persona_key, dirs in ALL_RUNS.items():
        for d in dirs:
            data = all_data[d]
            for i, sim in enumerate(data["simulations"]):
                task_id = sim["task_id"]
                base_id = get_base_task_id(task_id)
                reward = sim["reward_info"]["reward"]
                termination = sim.get("termination_reason", "unknown")
                task_results[base_id][persona_key].append(
                    {
                        "reward": reward,
                        "termination": termination,
                        "dir": d,
                        "sim_idx": i,
                        "task_id": task_id,
                        "num_messages": len(sim.get("messages", [])),
                    }
                )

    # =========================================================================
    # PART 1: Non-roaming failures
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 1: NON-ROAMING TASK FAILURES")
    print("=" * 100)

    non_roaming_failures = defaultdict(lambda: defaultdict(list))

    for base_id, persona_data in task_results.items():
        if is_roaming_task(base_id + "["):  # hack: base_id doesn't have persona tag
            continue
        # Check if any faults related to abroad are in the base_id
        if "user_abroad" in base_id or "set_abroad" in base_id:
            continue

        for persona_key, runs in persona_data.items():
            for run in runs:
                if run["reward"] == 0:
                    non_roaming_failures[base_id][persona_key].append(run)

    # Sort by number of total failures
    sorted_nr_failures = sorted(
        non_roaming_failures.items(),
        key=lambda x: sum(len(v) for v in x[1].values()),
        reverse=True,
    )

    print(
        f"\nTotal non-roaming tasks with at least one failure: {len(sorted_nr_failures)}"
    )
    print()

    # Categorize failures
    categories = {
        "overdue_bill": [],
        "contract_end": [],
        "lock_sim": [],
        "apn_settings": [],
        "multi_fault": [],
        "data_usage": [],
        "permissions": [],
        "other": [],
    }

    for base_id, persona_failures in sorted_nr_failures:
        faults = get_faults_from_task_id(base_id)
        task_type = get_task_type(base_id + "[")
        total_failures = sum(len(v) for v in persona_failures.values())
        total_runs = sum(len(v) for v in task_results[base_id].values())

        entry = {
            "base_id": base_id,
            "faults": faults,
            "task_type": task_type,
            "total_failures": total_failures,
            "total_runs": total_runs,
            "persona_failures": persona_failures,
        }

        # Categorize
        if "contract_end_suspension" in faults and "lock_sim_card_pin" in faults:
            categories["contract_end"].append(entry)
        elif "contract_end_suspension" in faults:
            categories["contract_end"].append(entry)
        elif "overdue_bill_suspension" in faults and "lock_sim_card_pin" in faults:
            categories["overdue_bill"].append(entry)
        elif "overdue_bill_suspension" in faults:
            categories["overdue_bill"].append(entry)
        elif "lock_sim_card_pin" in faults:
            categories["lock_sim"].append(entry)
        elif "break_apn_settings" in faults or "break_apn_mms_setting" in faults:
            categories["apn_settings"].append(entry)
        elif "data_usage_exceeded" in faults:
            categories["data_usage"].append(entry)
        elif (
            "break_app_both_permissions" in faults
            or "break_app_sms_permission" in faults
            or "break_app_storage_permission" in faults
        ):
            categories["permissions"].append(entry)
        else:
            categories["other"].append(entry)

    for cat_name, entries in categories.items():
        if not entries:
            continue
        print(f"\n--- Category: {cat_name.upper()} ({len(entries)} tasks) ---")
        for entry in sorted(entries, key=lambda x: x["total_failures"], reverse=True):
            print(f"\n  Task: {entry['base_id']}")
            print(f"  Type: {entry['task_type']}, Faults: {entry['faults']}")
            print(
                f"  Failures: {entry['total_failures']}/{entry['total_runs']} across all runs"
            )
            for pk, runs in sorted(entry["persona_failures"].items()):
                total_for_persona = len(task_results[entry["base_id"]][pk])
                fail_count = len(runs)
                terms = [r["termination"] for r in runs]
                print(
                    f"    {pk}: {fail_count}/{total_for_persona} failed, terminations: {terms}"
                )

    # =========================================================================
    # PART 2: Persona-dependent tasks
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 2: PERSONA-DEPENDENT TASKS (outcome changes with persona)")
    print("=" * 100)

    persona_dependent = []

    for base_id, persona_data in task_results.items():
        # Compute pass rate per persona
        persona_rates = {}
        for persona_key, runs in persona_data.items():
            if not runs:
                continue
            pass_rate = sum(1 for r in runs if r["reward"] == 1) / len(runs)
            persona_rates[persona_key] = {
                "pass_rate": pass_rate,
                "passes": sum(1 for r in runs if r["reward"] == 1),
                "fails": sum(1 for r in runs if r["reward"] == 0),
                "total": len(runs),
            }

        if not persona_rates:
            continue

        rates = [v["pass_rate"] for v in persona_rates.values()]
        if max(rates) - min(rates) >= 0.5:  # At least 50% difference
            # Separate by agent
            sonnet_rates = {
                k: v for k, v in persona_rates.items() if k.startswith("sonnet_")
            }
            gpt_rates = {
                k: v for k, v in persona_rates.items() if k.startswith("gpt54_")
            }

            # Check within-agent persona dependency
            s_range = (
                max(v["pass_rate"] for v in sonnet_rates.values())
                - min(v["pass_rate"] for v in sonnet_rates.values())
                if sonnet_rates
                else 0
            )
            g_range = (
                max(v["pass_rate"] for v in gpt_rates.values())
                - min(v["pass_rate"] for v in gpt_rates.values())
                if gpt_rates
                else 0
            )

            persona_dependent.append(
                {
                    "base_id": base_id,
                    "task_type": get_task_type(base_id + "["),
                    "faults": get_faults_from_task_id(base_id),
                    "is_roaming": is_roaming_task(base_id),
                    "persona_rates": persona_rates,
                    "sonnet_range": s_range,
                    "gpt_range": g_range,
                    "max_range": max(s_range, g_range),
                }
            )

    # Sort by max persona range (most persona-dependent first)
    persona_dependent.sort(key=lambda x: x["max_range"], reverse=True)

    # Filter to non-roaming for focus, but show roaming count
    nr_persona_dep = [p for p in persona_dependent if not p["is_roaming"]]
    r_persona_dep = [p for p in persona_dependent if p["is_roaming"]]

    print(
        f"\nTotal persona-dependent tasks (>=50% rate diff): {len(persona_dependent)}"
    )
    print(f"  Non-roaming: {len(nr_persona_dep)}")
    print(f"  Roaming: {len(r_persona_dep)}")

    print("\n--- NON-ROAMING Persona-Dependent Tasks ---")
    for entry in nr_persona_dep:
        print(f"\n  Task: {entry['base_id']}")
        print(f"  Type: {entry['task_type']}, Faults: {entry['faults']}")
        print(
            f"  Sonnet persona range: {entry['sonnet_range']:.2f}, GPT range: {entry['gpt_range']:.2f}"
        )
        for pk in sorted(entry["persona_rates"].keys()):
            pr = entry["persona_rates"][pk]
            print(
                f"    {pk}: {pr['passes']}/{pr['total']} pass ({pr['pass_rate']:.0%})"
            )

    print("\n--- ROAMING Persona-Dependent Tasks (top 10 by range) ---")
    for entry in r_persona_dep[:10]:
        print(f"\n  Task: {entry['base_id']}")
        print(f"  Type: {entry['task_type']}, Faults: {entry['faults']}")
        for pk in sorted(entry["persona_rates"].keys()):
            pr = entry["persona_rates"][pk]
            print(
                f"    {pk}: {pr['passes']}/{pr['total']} pass ({pr['pass_rate']:.0%})"
            )

    # =========================================================================
    # PART 3: Tasks that ALWAYS fail (across all personas and runs)
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 3: ALWAYS-FAILING NON-ROAMING TASKS")
    print("=" * 100)

    always_fail_nr = []
    for base_id, persona_data in task_results.items():
        if is_roaming_task(base_id):
            continue
        all_rewards = []
        for runs in persona_data.values():
            all_rewards.extend(r["reward"] for r in runs)
        if all_rewards and all(r == 0 for r in all_rewards):
            always_fail_nr.append(
                {
                    "base_id": base_id,
                    "faults": get_faults_from_task_id(base_id),
                    "task_type": get_task_type(base_id + "["),
                    "total_runs": len(all_rewards),
                }
            )

    print(f"\nNon-roaming tasks that ALWAYS fail: {len(always_fail_nr)}")
    for entry in always_fail_nr:
        print(f"  {entry['base_id']} ({entry['total_runs']} runs)")
        print(f"    Faults: {entry['faults']}")

    # =========================================================================
    # PART 4: Detailed conversation analysis for most interesting cases
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 4: DETAILED CONVERSATION ANALYSIS")
    print("=" * 100)

    # Select interesting cases:
    # 1. Non-roaming tasks with high persona dependency
    # 2. Always-failing non-roaming tasks
    # 3. Surprising failures (easy persona fails but hard passes)
    interesting_cases = []

    # From persona-dependent non-roaming
    for entry in nr_persona_dep[:6]:
        interesting_cases.append(
            {
                "reason": "PERSONA-DEPENDENT (non-roaming)",
                "base_id": entry["base_id"],
                "faults": entry["faults"],
                "task_type": entry["task_type"],
                "persona_rates": entry["persona_rates"],
            }
        )

    # From always-failing
    for entry in always_fail_nr[:3]:
        interesting_cases.append(
            {
                "reason": "ALWAYS FAILS (non-roaming)",
                "base_id": entry["base_id"],
                "faults": entry["faults"],
                "task_type": entry["task_type"],
            }
        )

    # De-duplicate
    seen = set()
    unique_cases = []
    for c in interesting_cases:
        if c["base_id"] not in seen:
            seen.add(c["base_id"])
            unique_cases.append(c)

    for case in unique_cases[:8]:
        print(f"\n{'=' * 80}")
        print(f"CASE: {case['base_id']}")
        print(f"Reason: {case['reason']}")
        print(f"Task Type: {case['task_type']}")
        print(f"Faults: {case['faults']}")

        if "persona_rates" in case:
            print("Pass rates:")
            for pk in sorted(case["persona_rates"].keys()):
                pr = case["persona_rates"][pk]
                print(f"  {pk}: {pr['passes']}/{pr['total']} ({pr['pass_rate']:.0%})")

        # Find a failing example and a passing example
        base_id = case["base_id"]
        fail_example = None
        pass_example = None

        for persona_key, dirs in ALL_RUNS.items():
            for d in dirs:
                data = all_data[d]
                for i, sim in enumerate(data["simulations"]):
                    if get_base_task_id(sim["task_id"]) == base_id:
                        if sim["reward_info"]["reward"] == 0 and fail_example is None:
                            fail_example = (d, persona_key, sim)
                        elif sim["reward_info"]["reward"] == 1 and pass_example is None:
                            pass_example = (d, persona_key, sim)
                    if fail_example and pass_example:
                        break

        if fail_example:
            d, pk, sim = fail_example
            print(f"\n  FAILING EXAMPLE ({pk}, dir={d}):")
            print(f"  Reward details: {get_reward_details(sim)}")
            print(extract_conversation_snippet(sim))

        if pass_example:
            d, pk, sim = pass_example
            print(f"\n  PASSING EXAMPLE ({pk}, dir={d}):")
            print(extract_conversation_snippet(sim))

    # =========================================================================
    # PART 5: Failure pattern summary with termination reasons
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 5: TERMINATION REASON ANALYSIS FOR NON-ROAMING FAILURES")
    print("=" * 100)

    termination_counts = defaultdict(int)
    termination_by_persona = defaultdict(lambda: defaultdict(int))
    termination_by_fault = defaultdict(lambda: defaultdict(int))

    for base_id, persona_data in task_results.items():
        if is_roaming_task(base_id):
            continue
        faults = get_faults_from_task_id(base_id)
        for persona_key, runs in persona_data.items():
            for run in runs:
                if run["reward"] == 0:
                    term = run["termination"]
                    termination_counts[term] += 1
                    termination_by_persona[persona_key][term] += 1
                    for f in faults:
                        termination_by_fault[f][term] += 1

    print("\nOverall termination reasons for non-roaming failures:")
    for term, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
        print(f"  {term}: {count}")

    print("\nTermination reasons by persona:")
    for pk in sorted(termination_by_persona.keys()):
        terms = termination_by_persona[pk]
        total = sum(terms.values())
        print(f"  {pk} (total failures: {total}):")
        for term, count in sorted(terms.items(), key=lambda x: -x[1]):
            print(f"    {term}: {count}")

    print("\nTermination reasons by fault type (top faults):")
    for fault in sorted(
        termination_by_fault.keys(),
        key=lambda f: sum(termination_by_fault[f].values()),
        reverse=True,
    )[:10]:
        terms = termination_by_fault[fault]
        total = sum(terms.values())
        print(f"  {fault} (total failures: {total}):")
        for term, count in sorted(terms.items(), key=lambda x: -x[1]):
            print(f"    {term}: {count}")

    # =========================================================================
    # PART 6: Message count analysis for failures
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 6: MESSAGE COUNT ANALYSIS (agent giving up early?)")
    print("=" * 100)

    short_failures = []  # Failures with very few messages
    for base_id, persona_data in task_results.items():
        if is_roaming_task(base_id):
            continue
        for persona_key, runs in persona_data.items():
            for run in runs:
                if run["reward"] == 0 and run["num_messages"] <= 10:
                    short_failures.append(
                        {
                            "base_id": base_id,
                            "persona": persona_key,
                            "num_messages": run["num_messages"],
                            "termination": run["termination"],
                            "dir": run["dir"],
                        }
                    )

    short_failures.sort(key=lambda x: x["num_messages"])

    print(
        f"\nNon-roaming failures with <=10 messages (potential early termination): {len(short_failures)}"
    )
    for sf in short_failures[:15]:
        print(
            f"  {sf['base_id']} | {sf['persona']} | {sf['num_messages']} msgs | {sf['termination']} | {sf['dir']}"
        )

    # =========================================================================
    # PART 7: Sonnet vs GPT comparison on non-roaming tasks
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 7: SONNET vs GPT COMPARISON (non-roaming)")
    print("=" * 100)

    sonnet_vs_gpt = defaultdict(
        lambda: {"sonnet_pass": 0, "sonnet_total": 0, "gpt_pass": 0, "gpt_total": 0}
    )

    for base_id, persona_data in task_results.items():
        if is_roaming_task(base_id):
            continue
        for persona_key, runs in persona_data.items():
            for run in runs:
                if persona_key.startswith("sonnet_"):
                    sonnet_vs_gpt[base_id]["sonnet_total"] += 1
                    if run["reward"] == 1:
                        sonnet_vs_gpt[base_id]["sonnet_pass"] += 1
                else:
                    sonnet_vs_gpt[base_id]["gpt_total"] += 1
                    if run["reward"] == 1:
                        sonnet_vs_gpt[base_id]["gpt_pass"] += 1

    # Find tasks where one agent does much better than the other
    agent_diff = []
    for base_id, stats in sonnet_vs_gpt.items():
        if stats["sonnet_total"] > 0 and stats["gpt_total"] > 0:
            s_rate = stats["sonnet_pass"] / stats["sonnet_total"]
            g_rate = stats["gpt_pass"] / stats["gpt_total"]
            if abs(s_rate - g_rate) >= 0.4:
                agent_diff.append(
                    {
                        "base_id": base_id,
                        "sonnet_rate": s_rate,
                        "gpt_rate": g_rate,
                        "diff": s_rate - g_rate,
                        "faults": get_faults_from_task_id(base_id),
                    }
                )

    agent_diff.sort(key=lambda x: abs(x["diff"]), reverse=True)

    print(
        f"\nTasks with >=40% pass rate difference between Sonnet and GPT: {len(agent_diff)}"
    )
    for ad in agent_diff[:15]:
        direction = "SONNET>GPT" if ad["diff"] > 0 else "GPT>SONNET"
        print(f"  {ad['base_id']}")
        print(
            f"    Sonnet: {ad['sonnet_rate']:.0%}, GPT: {ad['gpt_rate']:.0%} ({direction}, diff={abs(ad['diff']):.0%})"
        )
        print(f"    Faults: {ad['faults']}")

    # =========================================================================
    # PART 8: Specific interesting persona-dependent case deep-dives
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("PART 8: DEEP-DIVE INTO PERSONA-DEPENDENT FAILURE CONVERSATIONS")
    print("=" * 100)

    # For each non-roaming persona-dependent task, show a PASS vs FAIL conversation side-by-side
    for entry in nr_persona_dep[:5]:
        base_id = entry["base_id"]
        print(f"\n{'=' * 80}")
        print(f"TASK: {base_id}")
        print(f"Faults: {entry['faults']}")
        print("Pass rates:")
        for pk in sorted(entry["persona_rates"].keys()):
            pr = entry["persona_rates"][pk]
            print(f"  {pk}: {pr['passes']}/{pr['total']} ({pr['pass_rate']:.0%})")

        # Find best-persona pass and worst-persona fail
        best_persona = max(
            entry["persona_rates"].keys(),
            key=lambda k: entry["persona_rates"][k]["pass_rate"],
        )
        worst_persona = min(
            entry["persona_rates"].keys(),
            key=lambda k: entry["persona_rates"][k]["pass_rate"],
        )

        # Get a passing example from best persona
        pass_sim = None
        fail_sim = None
        pass_dir = None
        fail_dir = None

        for d in ALL_RUNS.get(best_persona, []):
            data = all_data[d]
            for i, sim in enumerate(data["simulations"]):
                if (
                    get_base_task_id(sim["task_id"]) == base_id
                    and sim["reward_info"]["reward"] == 1
                ):
                    pass_sim = sim
                    pass_dir = d
                    break
            if pass_sim:
                break

        for d in ALL_RUNS.get(worst_persona, []):
            data = all_data[d]
            for i, sim in enumerate(data["simulations"]):
                if (
                    get_base_task_id(sim["task_id"]) == base_id
                    and sim["reward_info"]["reward"] == 0
                ):
                    fail_sim = sim
                    fail_dir = d
                    break
            if fail_sim:
                break

        if pass_sim:
            print(f"\n  PASSING ({best_persona}, {pass_dir}):")
            print(extract_conversation_snippet(pass_sim))
        else:
            print(f"\n  No passing example found for {best_persona}")

        if fail_sim:
            print(f"\n  FAILING ({worst_persona}, {fail_dir}):")
            print(f"  Reward details: {get_reward_details(fail_sim)}")
            print(extract_conversation_snippet(fail_sim))
        else:
            print(f"\n  No failing example found for {worst_persona}")


if __name__ == "__main__":
    main()
