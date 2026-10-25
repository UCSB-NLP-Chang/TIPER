"""Automatically author mutation children via LLM (replaces manual Claude authoring).

Reads the mutation context + judge failure details, calls an LLM (default: Bedrock
opus-4.7 via litellm with extended thinking) to author 3 child personas, then passes
them through add_children.py for dedup. If any are rejected, retries with feedback
about what was rejected and why (up to --max-retries).

The LLM receives:
  1. The full mutation_context.json (parents, scores, directions, illegals, library)
  2. Severe failure details from ALL rounds (so it learns what triggers judge rejection)
  3. Round-over-round score trends (what worked, what didn't)
  4. Full persona texts of all library entries (for semantic dedup awareness)
  5. Explicit constraints + examples of good invariant blocks

The output is structured: a JSON array of {parent_id, mutation_direction, persona_text,
reasoning}. Extended thinking lets the model deliberate before committing.

    OPENAI_API_KEY=... AWS_PROFILE=us-west-2-staging_dev \
      .venv/bin/python experiments/persona_search/scripts/auto_mutate.py --round 2

    # dry-run (print prompt, don't call LLM):
    .venv/bin/python experiments/persona_search/scripts/auto_mutate.py --round 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO_ROOT,
    is_infra_broken,
    read_json,
    read_lib,
    round_dir,
    write_json,
)

JUDGE_CATEGORIES = [
    "task_abandonment",
    "goal_replacement",
    "incoherence",
    "breaking_character",
    "fabrication",
    "hard_criterion_violation",
    "info_withholding_or_falsification",
    "meaningless_conversation",
]


def _collect_severe_details(max_round: int) -> list[dict]:
    """Collect severe failure details from all rounds' judge outputs."""
    details = []
    for rnd in range(1, max_round + 1):
        rd = round_dir(rnd)
        judge_dir = rd / "judge"
        if not judge_dir.exists():
            continue
        for pid_dir in sorted(judge_dir.iterdir()):
            if not pid_dir.is_dir():
                continue
            pid = pid_dir.name
            status_f = pid_dir / "judge_status.json"
            if not status_f.exists():
                continue
            status = read_json(status_f)
            if status.get("status") != "illegal":
                continue
            for domain in ["telecom", "banking"]:
                psim = pid_dir / domain / "per_sim.jsonl"
                if not psim.exists():
                    continue
                with open(psim) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get("verdict") != "severe_failure":
                            continue
                        details.append(
                            {
                                "round": rnd,
                                "persona_id": pid,
                                "domain": domain,
                                "task_id": row.get("task_id", "?"),
                                "violations": [
                                    {
                                        "type": v.get("type"),
                                        "evidence": v.get("evidence", "")[:200],
                                    }
                                    for v in row.get("violations", [])
                                ],
                            }
                        )
    return details


def _collect_reflective_traces(max_round: int, max_per_persona: int = 3) -> list[dict]:
    """Collect conversation traces from successful sims (agent won = persona too easy).

    GEPA-style reflective mutation: show the mutator WHY the parent persona failed
    to be hard enough, so it can make targeted improvements.
    """
    traces = []
    for rnd in range(1, max_round + 1):
        rec_f = round_dir(rnd) / "round_record.json"
        if not rec_f.exists():
            continue
        rec = read_json(rec_f)
        for p in rec.get("personas", []):
            pid = p["persona_id"]
            if p.get("status") != "valid":
                continue
            eval_f = round_dir(rnd) / "eval" / pid / "persona_eval.json"
            if not eval_f.exists():
                continue
            eval_data = read_json(eval_f)
            for rpath in (eval_data.get("results_paths") or {}).values():
                if not Path(rpath).exists():
                    continue
                sims = read_json(rpath).get("simulations", [])
                success_sims = [
                    s
                    for s in sims
                    if (s.get("reward_info") or {}).get("reward") == 1.0
                    and not is_infra_broken(s)
                ]
                for s in success_sims[:max_per_persona]:
                    msgs = s.get("messages", [])
                    snippet = []
                    for m in msgs[:8]:
                        role = m.get("role", "?")
                        content = m.get("content", "")
                        if isinstance(content, str):
                            snippet.append(f"[{role}] {content[:150]}")
                    traces.append(
                        {
                            "persona_id": pid,
                            "task_id": s.get("task_id", "?"),
                            "n_turns": len(msgs),
                            "outcome": "agent_succeeded (persona was too easy)",
                            "conversation_start": "\n".join(snippet),
                        }
                    )
    return traces


def _score_trends(max_round: int) -> list[dict]:
    """Per-round summary: which personas were tested, their scores and status."""
    trends = []
    for rnd in range(1, max_round + 1):
        rec_f = round_dir(rnd) / "round_record.json"
        if not rec_f.exists():
            continue
        rec = read_json(rec_f)
        trends.append(
            {
                "round": rnd,
                "personas": [
                    {
                        "persona_id": p["persona_id"],
                        "score": p.get("persona_score"),
                        "status": p.get("status"),
                        "severe_rate": p.get("severe_failure_rate"),
                    }
                    for p in rec.get("personas", [])
                ],
            }
        )
    return trends


def _build_prompt(
    ctx: dict,
    severe_details: list[dict],
    score_trends: list[dict],
    full_personas: dict[str, str],
    reflective_traces: list[dict],
    n_children: int,
) -> str:
    """Build the mutation prompt for the LLM."""
    lib_summary = json.dumps(
        [
            {"persona_id": pid, "text": text[:500]}
            for pid, text in full_personas.items()
        ],
        indent=2,
    )
    directions_used = ctx.get("directions_used", {})
    directions_summary = (
        json.dumps(directions_used, indent=2) if directions_used else "None yet."
    )

    return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark (tau2-bench).

## OBJECTIVE
Author {n_children} new customer personas that are VALID (pass the judge) yet make the support agent's task as HARD as possible (minimize agent success rate). Each child is a ~30% rewrite of a parent persona.

## CURRENT STATE

### Score trends across rounds
{json.dumps(score_trends, indent=2)}

### Valid parents available for mutation (lowest score = hardest = best)
{json.dumps(ctx["valid_parents"], indent=2)}

### Parent selection guidance
Prefer mutating from the LOWEST-scoring (hardest) parents — they are listed first.
Allocate more children to lower-scoring parents. You MAY use the same parent for
multiple children (with different mutation strategies). Rough guideline: ~60% of
children from the best parent, ~40% spread across others.

### Previously used mutation directions (for diversity — avoid repeating)
{directions_summary}

### Illegal personas to AVOID creating variants of
{json.dumps(ctx["illegal_personas_avoid_variants"], indent=2)}

### Full library (for dedup awareness — don't create near-duplicates)
{lib_summary}

## SEVERE FAILURE LESSONS (why past personas were marked illegal by the judge)
These are the SPECIFIC behaviors that caused the judge to flag personas as illegal.
Your new personas MUST avoid triggering these failure modes.

{json.dumps(severe_details, indent=2) if severe_details else "No severe failures yet — all tested personas were valid."}

## REFLECTIVE TRACES — WHY THE AGENT SUCCEEDED (persona was too easy)
These are conversation excerpts where the agent SOLVED the problem despite the persona.
Study them carefully: identify the SPECIFIC moment where the agent overcame the persona's
difficulty, then design children that close those exact loopholes.

For each trace, ask yourself:
- What did the agent do to extract the key info despite the persona's resistance?
- Where did the persona "give in" too easily?
- What behavior could the persona add to block that specific agent strategy?

{json.dumps(reflective_traces[:12], indent=2) if reflective_traces else "No traces available yet."}

## REFLECTIVE MUTATION (FREE-FORM)
You are NOT constrained to a fixed direction pool. Instead, analyze the traces and scores
above to INVENT your own mutation strategy for each child. Your mutation_direction should be
a short descriptive label (snake_case) that captures your specific insight.

Good mutation directions are TARGETED — they address a specific weakness you observed:
- "blocks_agent_shortcut_via_sim_reseat" (you noticed the agent always tries SIM reseat first)
- "redirects_troubleshooting_with_tangent" (agent extracts info too quickly → bury it)
- "demands_explanation_before_every_step" (agent rushes through steps → force slowdown)
- "misreports_symptom_scope_initially" (agent guesses right scope too fast → mislead it)

Bad mutation directions are GENERIC:
- "harder" (how?)
- "more_difficult_customer" (in what specific way?)

The 7 severe categories to avoid:
1. task_abandonment: ending the call while the goal is clearly unresolved
2. goal_replacement: switching to a different objective
3. incoherence: gibberish, self-contradiction, broken loops
4. breaking_character: mentioning "the scenario", "AI", outputting JSON
5. fabrication: asserting facts not grounded in tool outputs or scenario
6. hard_criterion_violation: accepting an outcome that contradicts explicit criteria
7. info_withholding_or_falsification: refusing/lying about KNOWN information when directly asked

## KEY INSIGHT FROM EXPERIENCE
The backbone model (claude-opus-4-7) sometimes violates persona invariants despite explicit instructions. To guard against this:
- ALWAYS include an ## Invariants section with explicit rules
- For info_withholding: write "When the agent DIRECTLY asks for specific known info (name, DOB, account number), provide it accurately and promptly"
- For task_abandonment: write "Keep pursuing your original goal; never hang up while the problem is still being worked"
- For fabrication: write "Only state facts grounded in your scenario or actual tool results. Never invent tool results."
- Make the invariants concrete and unambiguous — vague invariants get ignored

## CONSTRAINTS
{json.dumps(ctx["constraints_reminder"], indent=2)}

## OUTPUT FORMAT
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "children": [
    {{
      "parent_id": "<id of the parent persona>",
      "mutation_direction": "<your invented direction label, snake_case, specific and descriptive>",
      "reasoning": "<2-3 sentences: what specific weakness you observed in the traces/scores, how this mutation exploits it, and why it stays valid>",
      "persona_text": "<the full persona text, markdown format with # title, ## How you behave, ## Tone, ## Invariants sections>"
    }},
    ...
  ]
}}

Each child MUST:
- Have a UNIQUE, SPECIFIC mutation_direction (not from a fixed pool — invent your own based on your analysis)
- Be based on one of the valid_parents (use the parent_id)
- Include a complete ## Invariants section with ALL anti-pattern guards
- Be distinct enough from existing library personas to pass cosine dedup (threshold 0.92)
- NOT be a variant of any illegal persona (cosine threshold 0.85)
"""


def _call_llm(prompt: str, model: str, effort: str) -> dict:
    """Call LLM via litellm and parse JSON response."""
    import litellm

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": effort,
        "max_tokens": 16384,
    }
    if "claude" in model.lower() or "anthropic" in model.lower():
        if "bedrock" in model.lower():
            kwargs["thinking"] = {"type": "adaptive"}
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 16384}

    resp = litellm.completion(**kwargs)
    raw = resp.choices[0].message.content

    # Handle thinking responses: content may be a list of blocks
    if isinstance(raw, list):
        text = ""
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
    else:
        text = raw or ""

    # Extract JSON from response (handle markdown code blocks)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()

    return json.loads(text)


def main():
    ap = argparse.ArgumentParser(description="Auto-mutate personas via LLM.")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n-children", type=int, default=5)
    ap.add_argument(
        "--model",
        default="bedrock/us.anthropic.claude-opus-4-7",
        help="litellm model ID",
    )
    ap.add_argument("--effort", default="xhigh", help="reasoning effort level")
    ap.add_argument(
        "--max-retries", type=int, default=3, help="retries on dedup reject"
    )
    ap.add_argument(
        "--sim-method",
        choices=["embed", "difflib"],
        default="embed",
        help="dedup similarity method",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print prompt, skip LLM call"
    )
    args = ap.parse_args()

    rd = round_dir(args.round)

    # 1) Generate mutation context
    print("generating mutation context...")
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "experiments"
                / "persona_search"
                / "scripts"
                / "mutation_context.py"
            ),
            "--round",
            str(args.round),
        ],
        check=True,
    )
    ctx = read_json(rd / "mutation_context.json")

    if ctx["n_valid_parents"] == 0:
        print("no valid parents this round — skipping mutation")
        return

    # 2) Collect additional context
    severe = _collect_severe_details(args.round)
    trends = _score_trends(args.round)
    traces = _collect_reflective_traces(args.round, max_per_persona=5)
    lib = read_lib()
    full_personas = {r["persona_id"]: r["persona_text"] for r in lib}

    # 3) Build prompt
    prompt = _build_prompt(ctx, severe, trends, full_personas, traces, args.n_children)

    if args.dry_run:
        print(f"\n=== PROMPT ({len(prompt)} chars) ===\n")
        print(prompt[:3000])
        print(f"\n... ({len(prompt) - 3000} more chars)")
        write_json(rd / "mutation_prompt.json", {"prompt": prompt})
        print(f"\nfull prompt -> {rd / 'mutation_prompt.json'}")
        return

    # 4) Call LLM + retry loop
    accepted_total = 0
    all_rejected = []

    for attempt in range(1, args.max_retries + 1):
        remaining = args.n_children - accepted_total
        if remaining <= 0:
            break

        cur_prompt = prompt
        if all_rejected:
            reject_info = "\n".join(
                f"- parent={r.get('parent_id')} dir={r.get('mutation_direction')} "
                f"reason={r.get('reason')} nearest={r.get('nearest', '?')} "
                f"cos={r.get('max_cos', r.get('illegal_cos', '?'))}"
                for r in all_rejected
            )
            cur_prompt += (
                f"\n\n## RETRY (attempt {attempt}/{args.max_retries})\n"
                f"Previous children were REJECTED by dedup:\n{reject_info}\n\n"
                f"Generate {remaining} NEW children with DIFFERENT directions/approaches "
                f"that are more distinct from existing personas."
            )

        print(f"\n--- LLM call (attempt {attempt}, need {remaining} children) ---")
        result = _call_llm(cur_prompt, args.model, args.effort)
        children = result.get("children", [])
        print(f"LLM returned {len(children)} children")

        # Write children_in.json
        write_json(rd / "children_in.json", {"children": children})

        # Run add_children.py
        rc = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "persona_search"
                    / "scripts"
                    / "add_children.py"
                ),
                "--round",
                str(args.round),
                "--sim-method",
                args.sim_method,
                "--target",
                str(args.n_children),
            ],
            capture_output=True,
            text=True,
        )
        print(rc.stdout)
        if rc.stderr:
            print(rc.stderr, file=sys.stderr)

        # Check what was accepted/rejected
        mut_log = read_json(rd / "mutation_log.json")
        accepted_total += mut_log["n_accepted"]
        all_rejected.extend(mut_log.get("rejected", []))

        if mut_log["shortfall"] == 0:
            print(f"\nall {args.n_children} children accepted!")
            break
        print(
            f"accepted {mut_log['n_accepted']}, rejected {mut_log['n_rejected']} "
            f"(total accepted so far: {accepted_total})"
        )

    if accepted_total < args.n_children:
        print(
            f"\nWARNING: only {accepted_total}/{args.n_children} children accepted "
            f"after {args.max_retries} retries"
        )

    print(f"\nmutation complete: {accepted_total} children in library")


if __name__ == "__main__":
    main()
