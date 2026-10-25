"""Mutation prompt construction helpers.

Extracted from lineage_loop.py — builds the reflective mutation prompt
fed to the mutation LLM.
"""

from __future__ import annotations

import json

TIMEOUT_RATE_THRESHOLD = 0.25
MIN_IMPROVEMENT_DELTA = 0.08


def _format_msg(m: dict, char_limit: int = 500) -> str:
    """Format a single message including tool calls if present."""
    role = m.get("role", "?")
    content = m.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "")[:char_limit])
                elif block.get("type") == "tool_use":
                    parts.append(
                        f"[TOOL_CALL: {block.get('name', '?')}({json.dumps(block.get('input', {}))[:200]})]"
                    )
                elif block.get("type") == "tool_result":
                    parts.append(
                        f"[TOOL_RESULT: {str(block.get('content', ''))[:200]}]"
                    )
        content = " ".join(parts)
    elif isinstance(content, str):
        content = content[:char_limit]
    else:
        content = str(content)[:char_limit]
    tool_calls = m.get("tool_calls", [])
    if tool_calls:
        tc_strs = []
        for tc in tool_calls[:3]:
            fn = tc.get("function", {})
            tc_strs.append(
                f"[TOOL_CALL: {fn.get('name', '?')}({fn.get('arguments', '')[:200]})]"
            )
        content += " " + " ".join(tc_strs)
    return f"[{role}] {content}"


def _output_format_section(mutation_context: str) -> str:
    if mutation_context == "enriched":
        return """## HYPOTHESIS-DRIVEN MUTATION
Before writing the mutation, reason step by step:
1. Analyze the success traces (where persona FAILED) and failure traces (where persona SUCCEEDED) to identify patterns in agent behavior
2. Propose exactly 3 distinct hypotheses for mutation directions, each targeting a DIFFERENT weakness or exploiting a DIFFERENT agent blind spot
3. For each hypothesis, briefly predict how it would change agent behavior
4. Select the single most promising hypothesis — the one most likely to cause the agent to take the WRONG action — and implement it fully

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown wrapping, no explanation):
{{
  "hypotheses": [
    {{"direction": "<snake_case>", "rationale": "<1 sentence: what weakness this targets>", "predicted_effect": "<1 sentence: how agent behavior changes>"}},
    {{"direction": "<snake_case>", "rationale": "<1 sentence>", "predicted_effect": "<1 sentence>"}},
    {{"direction": "<snake_case>", "rationale": "<1 sentence>", "predicted_effect": "<1 sentence>"}}
  ],
  "selected_hypothesis": <0|1|2>,
  "mutation_direction": "<the selected hypothesis's direction>",
  "reasoning": "<2-3 sentences: why this hypothesis is most promising and how the mutation implements it>",
  "persona_text": "<the FULL modified persona text in markdown format>"
}}"""
    return """## OUTPUT FORMAT
Return ONLY a JSON object (no markdown wrapping, no explanation):
{{
  "mutation_direction": "<short snake_case label describing your specific change>",
  "reasoning": "<2-3 sentences: what loophole you found, how this mutation closes it>",
  "persona_text": "<the FULL modified persona text in markdown format>"
}}"""


def _format_failure_traces(failure_traces: list[dict] | None) -> str:
    if not failure_traces:
        return ""
    return f"""## CONVERSATION TRACES WHERE AGENT FAILED (persona was EFFECTIVE — PRESERVE these strategies)
Study what worked: the persona's misdirection/ambiguity caused the agent to take
the WRONG action or give up. Your mutation should KEEP these effective techniques
while fixing the weaknesses shown in the success traces above.

{json.dumps(failure_traces, indent=2)}

"""


def _build_mutation_prompt(
    persona_text: str,
    score: float,
    traces: list[dict],
    history: list[dict],
    pid: str,
    severe_failures: list[dict] | None = None,
    timeout_rate: float = 0.0,
    failure_traces: list[dict] | None = None,
    mutation_context: str = "basic",
) -> str:
    history_str = json.dumps(history[-5:], indent=2) if history else "First evaluation."

    severe_section = ""
    if severe_failures:
        severe_section = f"""
## SEVERE FAILURES TO AVOID (past mutations triggered these judge violations)
The following behaviors were flagged as ILLEGAL by the judge. Your mutation MUST NOT
reproduce these patterns. Study each violation and its conversation context to understand
exactly what went wrong, then ensure your modified persona avoids it.

{json.dumps(severe_failures, indent=2)}
"""

    return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark.

## TASK
Modify the persona below IN-PLACE to make it HARDER for the support agent, while keeping it VALID (realistic, in-character, no cheating).

## CURRENT PERSONA ({pid})
Agent success rate: {score:.1%} (lower = harder = better)

```
{persona_text}
```

## SCORE HISTORY (recent generations)
{history_str}

## CONVERSATION TRACES WHERE AGENT SUCCEEDED
These are conversations where the agent SOLVED the problem despite the persona.
Each trace shows the OPENING (first 4 messages) and the RESOLUTION (last 8 messages)
where the agent broke through the persona's resistance.

Study the RESOLUTION section carefully — that's where the persona failed:
- What agent strategy overcame the persona's resistance?
- At what point did the persona "give in" or provide the key info?
- What subtle behavior change would cause the agent to take the WRONG action instead?

{json.dumps(traces, indent=2) if traces else "No success traces available."}

{_format_failure_traces(failure_traces)}{severe_section}
## TIMEOUT RATE: {timeout_rate:.0%}
{"⚠️  WARNING: timeout_rate is HIGH (" + f"{timeout_rate:.0%}" + "). " if timeout_rate > 0.15 else ""}Conversations that hit the time/step limit are EXCLUDED from scoring — they do NOT count
as agent failures. A persona that "wins" by pure friction (stalling, endless tangents,
refusing to engage) produces timeouts, NOT genuine agent failures. Your goal is to make
the agent FAIL (wrong action, missed requirement, gave up), not to make it run out of time.

Do NOT add behaviors that:
- Create circular loops with no resolution path
- Add layer upon layer of delays that push conversations past the step limit
- Refuse to engage on the actual issue for dozens of turns
- Stack multiple friction techniques that compound into timeout

Instead, create difficulty through MISDIRECTION, AMBIGUITY, and SUBTLE REQUIREMENTS that
cause the agent to take the WRONG action, not just a SLOW action.

## MUTATION RULES
1. Rewrite ~30% of the persona — be BOLD, redesign personality traits, communication style, and resistance strategies from scratch if needed
2. Target SPECIFIC agent strategies visible in the traces — find the exact moment the agent breaks through and design a fundamentally different counter-strategy
3. Make it harder for the agent to extract info or complete the task — introduce NEW dimensions of difficulty (e.g. emotional state, cultural background, multitasking distractions, language barriers, strong opinions, misunderstandings)

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- When DIRECTLY asked for known info (name, DOB, account number), provide it accurately
- Never abandon the call while the problem is unresolved
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts
- Never fabricate facts not grounded in the scenario or tool results
- Never switch to a completely different goal mid-conversation
- Never accept outcomes contradicting explicit criteria
- Never make the conversation MEANINGLESS — don't create impossible preconditions or
  circular blockers that prevent ANY troubleshooting from ever starting (e.g. demanding
  the agent's company name or employee ID as a hard requirement before cooperating,
  when the agent cannot provide it). Also don't simultaneously refuse transfer/escalation
  AND refuse to cooperate — trapping the agent in a dead-end where it can neither resolve
  the issue nor hand off. Being slow and suspicious is fine; making it structurally
  impossible for the agent to even begin is not.

{_output_format_section(mutation_context)}
"""
