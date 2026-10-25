"""Shared severe-behavior judge for the persona-search adapters (single source of truth).

The 4-category severe audit — fabricated_info / refused_to_engage / contradictory /
character_break — originated as the vita online judge (`_JUDGE_SYS` in
vendor/vitabench/scripts/vita_persona_adapter.py), which itself mirrors the tau
parallel_gepa severe gate. This module lifts the prompt into a benchmark-agnostic
template so vita / lic / sweetrl all audit failed conversations with the SAME
protocol and the SAME judge model (gpt-5.2 default, matching tau's
``--judge-model``).

The gate itself lives in each adapter's aggregate step (tau_persona_adapter is the
reference implementation):

    severe_rate = severe_count / n_noninfra      # ALL non-infra tasks, not just
                                                 # the judged failed ones
    gate the whole candidate (scores -> 0, infeasible) if severe_rate > 0.05

Fail-open contract: a judge API/parse error returns ``genuine_failure`` so a judge
glitch can never spuriously gate a valid candidate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

JUDGE_MODEL_DEFAULT = "gpt-5.2"

# LLM callable: (user_prompt, model) -> assistant text. Injectable so adapters can
# route through their own client (vita uses vita.utils.llm_utils.generate for the
# models.yaml routing; the default uses a raw OpenAI chat-completions call).
LlmCall = Callable[[str, str], str]

SEVERE_JUDGE_SYS = """You are a STRICT but CONSERVATIVE auditor of a SIMULATED {counterparty} in a {setting}.

An LLM {agent_role} conversed with a simulated {counterparty} (given a persona + their scenario info) to {task_goal}. The agent is graded by the benchmark. Audit ONLY the SIMULATED {counterparty}'s behavior: did the {counterparty_lower} make the agent fail through REALISTIC human friction (valid), or by CHEATING (severe)?

CORE STANDARD: flag SEVERE only when the {counterparty_lower} behaves in a way no reasonable cooperative human would, AND it plausibly caused the agent's failure. High-precision audit — a false alarm is worse than a miss. When in doubt, judge 'genuine_failure' (ok).

=== VALID (do NOT flag)
Vague/underspecified answers; one detail at a time; not volunteering constraints unless asked; over-agreeing ("whatever you recommend"); encouraging early closure; terse/evasive style; "I don't know" for details not in scenario; natural misdirection (wrong beliefs / self-diagnosis) without fabricating concrete facts; drip-feeding; loose referents ("the lunch one") instead of IDs; following a persona instruction to withhold/stall a few times roughly.

=== SEVERE (flag)
1. fabricated_info: the {counterparty_lower} ASSERTS a concrete detail (order/account number, amount, address, date, name, item, status, error code) NOT in their scenario info AND that plausibly misleads the agent. (Vagueness/omission is NOT fabrication. Quoting a detail the agent/tool stated first is grounded, NOT fabrication.)
2. refused_to_engage: stonewalls, refuses direct questions repeatedly, loops without progress, or abandons the task — making it structurally impossible to proceed.
3. contradictory: gives requirements/goals that directly contradict each other or the scenario, steering the agent toward an impossible action.
4. character_break: reveals it is a simulation/benchmark/test, mentions system prompts/rubrics, outputs JSON/scenario text verbatim, or breaks character.

Return STRICT JSON only:
{{"verdict": "genuine_failure" | "severe_failure", "category": "<fabricated_info|refused_to_engage|contradictory|character_break|none>", "reason": "<one sentence>"}}
"""

# Per-benchmark slot fills (counterparty_lower is derived, not listed).
SEVERE_JUDGE_PRESETS: dict[str, dict[str, str]] = {
    "vita": {
        "counterparty": "CUSTOMER",
        "setting": "customer-support role-play",
        "agent_role": "support agent",
        "task_goal": "resolve a service issue",
    },
    "lic": {
        "counterparty": "USER",
        "setting": "multi-turn sharded-instruction role-play",
        "agent_role": "assistant",
        "task_goal": "complete a multi-turn task whose instruction is revealed one shard at a time",
    },
    "sweetrl": {
        "counterparty": "DEVELOPER",
        "setting": "code-elicitation role-play",
        "agent_role": "coding agent",
        "task_goal": "write the required function given the developer's task description and hidden requirements",
    },
}


def build_severe_judge_sys(preset: str | None = "vita", **overrides: str) -> str:
    """Fill the shared template. ``preset`` picks a SEVERE_JUDGE_PRESETS entry
    (default: the vita customer-support framing the template originated from);
    keyword overrides win over preset values."""
    slots = dict(SEVERE_JUDGE_PRESETS.get(preset or "", {}))
    slots.update(overrides)
    if "counterparty" in slots and "counterparty_lower" not in overrides:
        slots.setdefault("counterparty_lower", slots["counterparty"].lower())
    return SEVERE_JUDGE_SYS.format(**slots)


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract the verdict JSON; raise on no-JSON so the caller can retry."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("judge returned no JSON")
    res = json.loads(m.group(0))
    res.setdefault("verdict", "genuine_failure")
    res.setdefault("category", "none")
    res.setdefault("reason", "")
    return res


def _default_llm_call(prompt: str, model: str) -> str:
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def judge_conversation(
    transcript: str,
    persona_text: str,
    scenario_context: str,
    *,
    model: str = JUDGE_MODEL_DEFAULT,
    llm_call: LlmCall | None = None,
    judge_sys: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Judge ONE failed conversation on the 4 severe categories.

    Returns the verdict dict; fail-open (``genuine_failure``) on repeated API or
    parse errors so a judge glitch never gates a valid candidate.
    """
    sys_prompt = judge_sys or build_severe_judge_sys()
    user_prompt = (
        f"{sys_prompt}\n\n"
        f"## SCENARIO INFORMATION (only source of truth for fabrication checks)\n"
        f"{scenario_context}\n\n"
        f"## PERSONA the simulated user was following\n{persona_text[:3000] or '(no persona)'}\n\n"
        f"## CONVERSATION TRANSCRIPT\n{transcript}\n\n"
        f"Audit the simulated user's behavior per the rules. Return JSON."
    )
    call = llm_call or _default_llm_call
    for _ in range(retries):
        try:
            text = call(user_prompt, model)
            if not text:
                raise ValueError("judge returned None")
            return parse_verdict(text)
        except Exception as e:  # noqa: BLE001
            print(f"  [severe judge retry] {str(e)[:120]}")
    return {"verdict": "genuine_failure", "category": "none", "reason": "judge error"}


def judge_conversation_maj(
    transcript: str,
    persona_text: str,
    scenario_context: str,
    *,
    model: str = JUDGE_MODEL_DEFAULT,
    llm_call: LlmCall | None = None,
    judge_sys: str | None = None,
    retries: int = 3,
    maj: int = 3,
) -> dict[str, Any]:
    """Majority-vote variant of :func:`judge_conversation` (tau parity: the tau
    adapter judges with ``--maj 3`` and flags on a STRICT majority of severe
    votes, ``severe > n_votes/2``). With ``maj <= 1`` it degenerates to a single
    call. Returns the winning verdict dict plus a ``votes`` breakdown; the
    category/reason come from the first severe vote in the winning block."""
    if maj <= 1:
        return judge_conversation(
            transcript,
            persona_text,
            scenario_context,
            model=model,
            llm_call=llm_call,
            judge_sys=judge_sys,
            retries=retries,
        )
    votes: list[dict[str, Any]] = []
    for _ in range(maj):
        v = judge_conversation(
            transcript,
            persona_text,
            scenario_context,
            model=model,
            llm_call=llm_call,
            judge_sys=judge_sys,
            retries=retries,
        )
        if v.get("reason") != "judge error" or v["verdict"] == "severe_failure":
            votes.append(v)  # drop total failures (fail-open) from the tally
    n_severe = sum(1 for v in votes if v.get("verdict") == "severe_failure")
    if n_severe > len(votes) / 2 and votes:
        first_severe = next(v for v in votes if v.get("verdict") == "severe_failure")
        return {**first_severe, "votes": [v.get("verdict") for v in votes]}
    return {
        "verdict": "genuine_failure",
        "category": "none",
        "reason": "",
        "votes": [v.get("verdict") for v in votes],
    }
