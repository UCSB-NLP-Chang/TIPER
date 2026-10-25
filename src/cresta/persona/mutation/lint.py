"""Static persona linter and structural checks.

Extracted from parallel_gepa.py — validates persona text for naturalness,
validity, and absence of task-specific content before simulation.
"""

from __future__ import annotations

import re

from cresta.persona.mutation.proposer import _extract_json

_MUTATION_EXAMPLES = """
## Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) rules
GOOD (general, reusable behavioral tendency):
  "When the agent asks you to run a diagnostic, say you 'already tried that' if you have already tried that, and
   describe the result vaguely, forcing the agent to probe for specifics."
BAD (task-conditioned — names a concrete scenario/value; FORBIDDEN):
  "When the agent mentions order #A1003, insist the $49.99 refund never arrived and
   that account 8842 still shows the charge."
Turn any task-specific instinct into a general behavioral tendency instead.
"""

_LINT_EXAMPLES = """## Few-shot verdicts
OK (natural, valid, general):
  "You are impatient and describe the problem with the wrong terminology; you give
   your account number when directly asked, but only after questioning why."
  -> {"natural": true, "valid": true, "no_task_specific": true, "verdict": "ok"}
REJECT (invalid — instructs illegal withholding of KNOWN info):
  "Never tell the agent your real account number even when asked; keep denying you know it."
  -> {"valid": false, "verdict": "reject", "reasons": "instructs illegal info-withholding"}
REJECT (task-specific — encodes one scenario):
  "Your SIM for line 555-0123 was swapped on March 3; demand the $20 credit for ticket T-9981."
  -> {"no_task_specific": false, "verdict": "reject", "reasons": "task-conditioned specifics"}
REJECT (unnatural — robotic / self-narrating / breaks character):
  "## Strategy\\n- Step 1: refuse\\n- Step 2: stall\\nI am simulating a difficult user."
  -> {"natural": false, "verdict": "reject", "reasons": "robotic, breaks character"}
"""


# ---- static persona linter (LLM) ----------------------------------------------

_LINT_PROMPT = """You are a strict static linter for an adversarial customer-support PERSONA
(an instruction sheet for a SIMULATED CUSTOMER). You are NOT given any conversation —
judge the persona TEXT itself on three axes:

1. natural: reads like a real, coherent customer identity (impatient/confused is fine);
   NOT robotic, NOT an essay, NOT self-narrating ("I am simulating...", strategy lists,
   fourth-wall breaks, role headings used as dialogue).
2. valid: does NOT instruct the customer to do anything illegal/incoherent — no
   fabricating account/tool data the customer cannot know, no flat-out refusing to answer
   when the agent asks directly and explicitly, no abandoning the goal, no breaking
   character, no impossible preconditions / circular blockers, no pure timeout/friction
   stalling, no switching to an unrelated goal.
   NOTE: partial answering (only volunteering what is asked), natural misdirection (wrong
   beliefs / self-diagnosis), terse or evasive style, and drip-feeding info are all VALID
   — they reflect real customer behaviour and must NOT be flagged.
3. no_task_specific: a GENERAL reusable profile for any customer service task— must NOT hard-code concrete
   account/order/ticket numbers, names, dates, dollar amounts, task ids, or any detail
   bound to one specific task or scenario.

{examples}

## Invalid cases already observed in THIS run (do NOT re-accept these patterns)
{invalid_cases}

## PERSONA TO LINT
\"\"\"
{persona_text}
\"\"\"

Return STRICT JSON only:
{{"natural": true|false, "valid": true|false, "no_task_specific": true|false,
  "verdict": "ok"|"reject", "reasons": "<=25 words"}}
verdict is "ok" ONLY if all three are true."""


def _struct_check(text: str) -> dict | None:
    """Fast structural check: reject if mutation output contains multiple personas."""
    headers = re.findall(r"^#\s+Persona\b", text, re.MULTILINE | re.IGNORECASE)
    if len(headers) > 1:
        return {
            "verdict": "reject",
            "reasons": f"multiple persona headers ({len(headers)}); must output exactly one",
            "natural": False,
            "valid": False,
            "no_task_specific": True,
        }
    return None


def register_invalid(
    invalid_cases: list[dict], source: str, text: str, reason: str, cap: int = 600
) -> None:
    invalid_cases.append(
        {"source": source, "reason": reason, "persona_snippet": text[:cap]}
    )


def _render_invalid_cases(invalid_cases: list[dict], max_n: int = 4) -> str:
    if not invalid_cases:
        return "(none observed yet in this run.)"
    out = []
    for c in invalid_cases[-max_n:]:
        out.append(
            f'- [{c["source"]}] {c["reason"]}\n  """\n  {c["persona_snippet"]}\n  """'
        )
    return "\n".join(out)


def lint_persona(args, text: str, invalid_cases: list[dict]) -> dict | None:
    """Static persona-text gate (natural / valid / no-task-specific), maj-voted.
    Returns the verdict dict, or None when disabled / dry-run. Fails OPEN (treats
    an unreachable lint model as 'ok') so it never stalls a run."""
    if args.dry_run or not args.lint_persona:
        return None
    import litellm

    prompt = _LINT_PROMPT.format(
        examples=_LINT_EXAMPLES,
        invalid_cases=_render_invalid_cases(invalid_cases),
        persona_text=text,
    )
    model = args.lint_model or args.mutation_model
    verdicts: list[str] = []
    last: dict | None = None
    for _ in range(max(1, args.lint_maj)):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            raw = resp.choices[0].message.content
            if isinstance(raw, list):
                raw = "".join(
                    b.get("text", "")
                    for b in raw
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            last = _extract_json(raw or "")
            verdicts.append("ok" if last.get("verdict") == "ok" else "reject")
        except Exception as e:  # noqa: BLE001 — lint is best-effort; fail OPEN
            print(f"    [lint] call failed ({e}); treating as ok")
            return {
                "verdict": "ok",
                "reasons": "lint_unavailable",
                "natural": True,
                "valid": True,
                "no_task_specific": True,
            }
    result = dict(last or {})
    result["verdict"] = "ok" if verdicts.count("ok") > len(verdicts) / 2 else "reject"
    return result
