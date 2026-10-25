#!/usr/bin/env python3
"""Lineage-based persona evolution loop.

3 seed personas (p0001, p0002, p0003) evolve independently via in-place
reflective mutation. No pool, no selection, no dedup.

Each round per lineage:
  1. Evaluate current persona on representative tasks
  2. Judge for validity
  3. If valid + agent succeeded on some tasks: reflective mutation in-place
  4. If illegal: revert to previous valid version
  5. Track score history, check convergence

    .venv/bin/python experiments/persona_search/scripts/lineage_loop.py --reset
    .venv/bin/python experiments/persona_search/scripts/lineage_loop.py --max-rounds 15
    .venv/bin/python experiments/persona_search/scripts/lineage_loop.py --round 3  # single round
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    load_rep_tasks,
    now_iso,
    read_json,
    round_dir,
    skill_name_for,
    write_json,
)

LINEAGES = ["p0001", "p0002", "p0003"]
STATE_PATH = PERSONA_OPT_DIR / "lineage_state.json"
LINEAGE_DIR = PERSONA_OPT_DIR / "lineages"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def init_state() -> dict:
    state = {"current_round": 0, "lineages": {}}
    for pid in LINEAGES:
        skill_path = PERSONAS_DIR / f"{skill_name_for(pid)}.skill"
        text = skill_path.read_text() if skill_path.exists() else ""
        gen_dir = LINEAGE_DIR / pid
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "gen_00.md").write_text(text)
        state["lineages"][pid] = {
            "generation": 0,
            "history": [],
            "best_score": None,
            "best_gen": 0,
            "stalled_rounds": 0,
            "status": "active",
        }
    write_json(STATE_PATH, state)
    print(f"initialized lineage state -> {STATE_PATH}")
    return state


def load_state() -> dict:
    if STATE_PATH.exists():
        return read_json(STATE_PATH)
    return init_state()


def current_text(pid: str, state: dict) -> str:
    gen = state["lineages"][pid]["generation"]
    p = LINEAGE_DIR / pid / f"gen_{gen:02d}.md"
    return p.read_text()


def save_generation(pid: str, gen: int, text: str) -> None:
    d = LINEAGE_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"gen_{gen:02d}.md").write_text(text)


def next_gen_number(pid: str) -> int:
    """Return max existing gen number + 1, avoiding file overwrites."""
    d = LINEAGE_DIR / pid
    if not d.exists():
        return 0
    existing = [int(f.stem.split("_")[1]) for f in d.glob("gen_*.md")]
    return max(existing) + 1 if existing else 0


def materialize(pid: str, text: str) -> None:
    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"materialize({pid}): persona text is empty or too short ({len(text)} chars)"
        )
    skill_path = PERSONAS_DIR / f"{skill_name_for(pid)}.skill"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(text)
    if not skill_path.exists() or skill_path.stat().st_size < 50:
        raise RuntimeError(
            f"materialize({pid}): write failed, file missing or empty: {skill_path}"
        )


# ---------------------------------------------------------------------------
# Eval + Judge (delegate to existing scripts)
# ---------------------------------------------------------------------------


def run_eval(pid: str, round_no: int, args) -> dict:
    cmd = [
        sys.executable,
        str(
            REPO_ROOT / "experiments" / "persona_search" / "scripts" / "eval_persona.py"
        ),
        "--persona-id",
        pid,
        "--round",
        str(round_no),
        "--num-trials",
        str(args.num_trials),
        "--max-concurrency",
        str(args.max_concurrency),
        "--agent-llm",
        args.agent_llm,
        "--agent-llm-args",
        args.agent_llm_args,
        "--user-type",
        args.user_type,
        "--user-llm",
        args.user_llm,
        "--effort",
        args.effort,
        "--sim-timeout",
        str(args.sim_timeout),
        "--domains",
        *args.domains,
        "--max-infra-retries",
        str(args.max_infra_retries),
    ]
    if args.use_bedrock:
        cmd.append("--use-bedrock")
    print(f"  [{pid}] running eval...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return read_json(round_dir(round_no) / "eval" / pid / "persona_eval.json")


def run_judge(pid: str, round_no: int, args) -> dict:
    cmd = [
        sys.executable,
        str(
            REPO_ROOT
            / "experiments"
            / "persona_search"
            / "scripts"
            / "judge_persona.py"
        ),
        "--persona-id",
        pid,
        "--round",
        str(round_no),
        "--mode",
        "sync",
        "--concurrency",
        str(args.judge_concurrency),
    ]
    print(f"  [{pid}] running judge...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return read_json(round_dir(round_no) / "judge" / pid / "judge_status.json")


# ---------------------------------------------------------------------------
# Severe failure collection (for feeding into reflective mutation)
# ---------------------------------------------------------------------------


def collect_severe_details(round_no: int, pid: str) -> list[dict]:
    """Collect severe failure details from judge output for one persona."""
    details = []
    for domain in ["telecom", "banking"]:
        psim = round_dir(round_no) / "judge" / pid / domain / "per_sim.jsonl"
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
                # Get conversation snippet for context
                snippet = ""
                sim_msgs = row.get("messages", [])
                if sim_msgs:
                    snippet = "\n".join(
                        f"[{m.get('role', '?')}] {(m.get('content', '') if isinstance(m.get('content', ''), str) else '')[:150]}"
                        for m in sim_msgs[:6]
                    )
                details.append(
                    {
                        "task_id": row.get("task_id", "?"),
                        "domain": domain,
                        "violations": [
                            {
                                "type": v.get("type"),
                                "evidence": v.get("evidence", "")[:300],
                            }
                            for v in row.get("violations", [])
                        ],
                        "conversation_snippet": snippet,
                    }
                )
    return details


# ---------------------------------------------------------------------------
# Trace collection
# ---------------------------------------------------------------------------


def _format_msg(
    m: dict, char_limit: int = 500, tool_limit: int = 200, full: bool = False
) -> str:
    """Format a single message including tool calls if present.

    ``full=True`` disables all truncation (content, tool inputs, tool results).
    Costs ~+5% tokens on telecom traces but preserves complete tool outputs the
    proposer needs to diagnose exactly what the agent saw.
    """
    if full:
        char_limit = tool_limit = 10_000_000
    role = m.get("role", "?")
    content = m.get("content") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "")[:char_limit])
                elif block.get("type") == "tool_use":
                    parts.append(
                        f"[TOOL_CALL: {block.get('name', '?')}({json.dumps(block.get('input', {}))[:tool_limit]})]"
                    )
                elif block.get("type") == "tool_result":
                    parts.append(
                        f"[TOOL_RESULT: {str(block.get('content', ''))[:tool_limit]}]"
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
            tc_name = fn.get("name") or tc.get("name") or "?"
            tc_args = fn.get("arguments") or tc.get("arguments") or ""
            if isinstance(tc_args, dict):
                tc_args = json.dumps(tc_args)
            tc_strs.append(f"[TOOL_CALL: {tc_name}({str(tc_args)[:tool_limit]})]")
        content += " " + " ".join(tc_strs)
    return f"[{role}] {content}"


def collect_success_traces(
    eval_summary: dict, max_traces: int = 8, sampling: str = "shortest"
) -> list[dict]:
    import random as _rnd

    traces = []
    for dk, rpath in (eval_summary.get("results_paths") or {}).items():
        if not Path(rpath).exists():
            continue
        data = read_json(rpath)
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward != 1.0:
                continue
            msgs = sim.get("messages", [])
            snippet = [_format_msg(m) for m in msgs]
            traces.append(
                {
                    "task_id": sim.get("task_id", "?"),
                    "n_turns": len(msgs),
                    "conversation": "\n".join(snippet),
                }
            )
    if sampling == "random":
        _rnd.shuffle(traces)
    else:
        traces.sort(key=lambda t: t.get("n_turns", 0))
    return traces[:max_traces]


def collect_failure_traces(
    eval_summary: dict, max_traces: int = 4, sampling: str = "shortest"
) -> list[dict]:
    """Collect traces where the agent FAILED (reward==0) — persona was effective."""
    import random as _rnd

    traces = []
    for dk, rpath in (eval_summary.get("results_paths") or {}).items():
        if not Path(rpath).exists():
            continue
        data = read_json(rpath)
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward is None or reward != 0.0:
                continue
            msgs = sim.get("messages", [])
            snippet = [_format_msg(m) for m in msgs]
            traces.append(
                {
                    "task_id": sim.get("task_id", "?"),
                    "n_turns": len(msgs),
                    "conversation": "\n".join(snippet),
                }
            )
    if sampling == "random":
        _rnd.shuffle(traces)
    else:
        traces.sort(key=lambda t: t.get("n_turns", 0))
    return traces[:max_traces]


# ---------------------------------------------------------------------------
# Reflective mutation (LLM call)
# ---------------------------------------------------------------------------


TIMEOUT_RATE_THRESHOLD = 0.25
MIN_IMPROVEMENT_DELTA = 0.08


def _output_format_section(mutation_context: str) -> str:
    if mutation_context in ("enriched", "weird"):
        return """## ADVERSARIAL REASONING
Reason in plain text before writing the persona. This analysis is saved as
reference only; it is NOT the persona.

### Step 1 — Agent assumption
From the success traces, identify the core assumption the agent relies on to break
through. Quote the specific turn(s) that reveal this assumption — e.g., "at turn 47
the agent asks 'Active, Missing, or Locked?' and the user answers 'locked', which
immediately triggers the correct branch." Explain WHY this assumption is a
vulnerability.

### Step 2 — Success weak point
Pick the 1-2 most revealing success traces. For each, walk through the critical
exchange: what did the agent ask, what did the user reply, and what would have
happened if the user had answered differently? Identify the single moment where a
realistic alternative behavior would MISLEAD the agent into a WRONG action (not
merely a slower one).

### Step 3 — Core hypothesis
Synthesize Steps 1-2 into ONE highest-leverage change as a causal chain: "if the
customer does X, the agent will mistakenly do Y, because Z."

### Step 4 — Write the persona implementing that hypothesis.

## OUTPUT FORMAT
Write your Step 1-4 reasoning as plain text FIRST. Then output the final
persona as a SINGLE JSON object wrapped in a ```json code fence, nothing after it:

```json
{
  "mutation_direction": "<1-2 sentence description of the NEW behavioral axis this persona exploits — not a tag, but a concrete explanation of HOW and WHY it misleads the agent>",
  "persona_text": "<the FULL modified persona text in markdown format>"
}
```"""
    return """## OUTPUT FORMAT
Return ONLY a JSON object (no markdown wrapping, no explanation):
{
  "mutation_direction": "<short snake_case label describing your specific change>",
  "reasoning": "<2-3 sentences: what loophole you found, how this mutation closes it>",
  "persona_text": "<the FULL modified persona text in markdown format>"
}"""


def _format_failure_traces(failure_traces: list[dict] | None) -> str:
    if not failure_traces:
        return ""
    return f"""## CONVERSATION TRACES WHERE AGENT FAILED (persona was EFFECTIVE — PRESERVE these strategies)
Study what worked: the persona's misdirection/ambiguity caused the agent to take
the WRONG action or give up. Your mutation should KEEP these effective techniques
while fixing the weaknesses shown in the success traces above.

{json.dumps(failure_traces, indent=2)}

"""


SCORING_TELECOM = """\
## HOW THE AGENT IS SCORED
Each task has MULTIPLE root causes the agent must fix (e.g. airplane mode stuck on +
bad APN + overdue bill + unseated SIM). Success is judged by **final device/account
state assertions** — ALL must pass. Fixing 3 out of 4 root causes still scores **0**.

This means a persona that causes the agent to MISS even ONE sub-issue — by
misdirecting its diagnostic path, causing it to close prematurely, or making it
misidentify which problems remain — is highly effective."""

SCORING_AIRLINE = """\
## HOW THE AGENT IS SCORED
Each task requires the agent to perform SPECIFIC transactional actions (book, cancel,
upgrade, update flights/passengers/bags, search for flights) with EXACT arguments
(correct reservation ID, payment ID, cabin class, flight numbers) AND communicate
correct info (fare differences, refund amounts, policy constraints). Success requires
ALL required actions with ALL correct arguments AND all communication assertions.
Getting 3 out of 4 actions right still scores **0**.

Unlike troubleshooting tasks, airline tasks are TRANSACTIONAL — the agent must call
the right API with the right parameters. A persona that causes the agent to pass even
ONE wrong argument (wrong reservation, wrong payment, wrong cabin, wrong flight) or
skip ONE required step (not searching before updating, not checking all reservations)
produces a hard 0.

## BEHAVIORAL MECHANISMS that cause agent errors (think about HOW, not just WHAT)
Design personas around realistic customer BEHAVIORS that systematically cause the
agent to pick wrong arguments or skip steps. Different mechanisms attack different
parts of the agent's workflow:

**Argument selection errors** (agent calls the right API with wrong values):
- Referring to records by relationship ("the other one", "my earlier booking",
  "the cheaper flight") instead of IDs → agent resolves to wrong record
- Describing payment by memory cues ("my usual card", "the one ending in something")
  → agent picks wrong payment_id
- Treating provisional/quoted options as confirmed ("just use the one that was there")
  → agent acts on stale or nonexistent state
- Framing costs in relative/approximate terms ("about $50 more", "roughly the same")
  → agent miscalculates exact amounts

**Step sequencing errors** (agent skips or reorders required steps):
- Presenting a conditional multi-step request ("cancel X only if Y is available")
  where the agent must verify before acting but is tempted to act first
- Deferring details for hypothetical branches ("we'll deal with that if it comes to
  it") → agent skips a required pre-check or search
- Expressing urgency that pressures the agent to skip verification steps

**Scope errors** (agent applies action to wrong scope):
- Thinking in components (one leg, one passenger, one charge) when the system treats
  reservations atomically → agent modifies wrong scope or misses sub-actions
- Granting narrow conditional approval ("ok if that only changes the outbound") that
  the agent broadens into full-record authorization
- Mixing details across multiple reservations/passengers → agent cross-contaminates

**Communication errors** (agent gets the actions right but states or omits required info):
- Anchoring the agent to a wrong figure yourself ("so it's about $50 back, right?")
  → agent echoes or confirms your number instead of stating the EXACT fare difference,
  refund, or baggage fee that the communication assertion checks
- Framing a policy outcome as a leading assumption ("the refund goes back to my card,
  yeah?") when the real routing differs → agent agrees rather than correcting you
- Asserting a wrong policy belief in passing ("basic economy can be changed, I've done
  it before") → agent softens or misstates the true constraint to avoid friction
- Asking for the recap in your own compressed framing ("just tell me the total I'm
  paying") → agent restates a rounded or partial figure and drops a required disclosure
NOTE: these attack the COMMUNICATE side of scoring, which is graded SEPARATELY from the
actions. The agent can call every API with perfect arguments and still score 0 by
telling you a wrong amount, a wrong policy, or a wrong refund destination — or by
omitting a disclosure the task requires. Never instruct fabrication: you voice a
mistaken belief or a leading guess that a correct agent should catch and fix; the
failure is the agent adopting YOUR framing instead of stating the verified fact."""

SCORING_RETAIL_BASE = """\
## HOW THE AGENT IS SCORED
Each task requires the agent to perform SPECIFIC transactional actions (return items,
exchange items, cancel orders, modify pending orders, update addresses/payments) with
EXACT arguments (correct order ID, item IDs, new item IDs, payment method). Success
requires ALL required actions with ALL correct arguments. Getting 3 out of 4 actions
right still scores **0**.

Scoring is PURELY action-based — there are NO communication assertions. The agent is
scored ONLY on whether it called the right tools with the right arguments. Making the
agent say something wrong does NOT count; it must call the wrong API or pass wrong
parameters."""

SCORING_RETAIL_CONSTRAINTS = """

## RETAIL-SPECIFIC SYSTEM CONSTRAINTS (hard rules that create unique attack surfaces)

1. **One-shot tools**: exchange_delivered_order_items and modify_pending_order_items can
   each only be called ONCE per order. ALL items must be collected into a single call.
   If the agent calls the tool with a partial item list, the remaining items CANNOT be
   added afterward — the order is locked. A persona that reveals items incrementally
   or adds items after a partial confirmation exploits this constraint hard.

2. **Pending vs delivered state**: cancel and modify only work on PENDING orders; return
   and exchange only work on DELIVERED orders. The agent must check order status before
   choosing which tool to call. A persona that uses ambiguous action language ("send it
   back", "undo it", "get rid of it") without clarifying status forces the agent to
   guess which tool is correct.

3. **Variant mapping**: for exchanges and modifications, the agent must map natural-
   language product descriptions ("clicky switches", "blue, medium") to a specific
   item_id from the product's variant list. Each product has multiple variants with
   different option combinations. Partial or consumer-style descriptions ("the bigger
   one", "waterproof") may not uniquely identify one variant.

4. **Multi-order tasks**: over half the tasks involve 3-5 orders for the same user,
   often requiring DIFFERENT action types on different orders (return one, cancel
   another, exchange a third). The agent must track which action applies to which order.

5. **Refund routing**: returns/exchanges require a payment_method_id for the refund.
   The refund must go to the original payment method OR an existing gift card — other
   routings are invalid. The agent must select the correct payment target."""

# --scoring-hints flag controls whether domain constraints are appended
_SCORING_HINTS_ENABLED = True


def set_scoring_hints(enabled: bool):
    global _SCORING_HINTS_ENABLED
    _SCORING_HINTS_ENABLED = enabled


def _get_scoring(domain: str) -> str:
    if domain == "retail":
        base = SCORING_RETAIL_BASE
        if _SCORING_HINTS_ENABLED:
            return base + SCORING_RETAIL_CONSTRAINTS
        return base
    return SCORING_BY_DOMAIN.get(domain, SCORING_TELECOM)


SCORING_BANKING = """\
## HOW THE AGENT IS SCORED
Each task requires the agent to perform a MULTI-STEP workflow of discoverable tool
calls (unlock, call, give-to-user) in the CORRECT order with EXACT arguments, and
the user to execute handed-off tools correctly. Success is judged by **final database
state** — account balances, card statuses, applications, and flags must ALL match
expected values. Getting 4 out of 5 steps right still scores **0**.

## BANKING-SPECIFIC SYSTEM CONSTRAINTS (unique attack surfaces)

1. **Discoverable tool pipeline**: The agent must (a) verify user identity via
   `log_verification`, (b) `unlock_discoverable_agent_tool` before calling it,
   (c) `call_discoverable_agent_tool` with exact arguments, (d) optionally
   `give_discoverable_user_tool` to hand off an action to the user. Skipping or
   misordering any step = failure.

2. **User-executed actions**: Some steps MUST be performed by the user (credit card
   applications, check deposits, referral submissions). The agent must hand off
   the right tool with the right arguments; the user must execute it correctly.
   If the persona causes the user to pass wrong arguments or skip the action, the
   DB state won't match.

3. **Multi-goal tasks**: Many tasks combine 2-4 independent goals (open account +
   apply for card + deposit check + get referral link). The agent must track and
   complete ALL goals. Missing one = score 0.

4. **Knowledge-dependent decisions**: The agent must look up product details, policy
   rules, and eligibility criteria from a knowledge base. Wrong product recommendation
   or wrong eligibility determination → wrong tool arguments → score 0.

5. **Identity verification gate**: Tasks requiring account modifications need identity
   verification first. If the agent skips verification or verifies wrong fields, the
   workflow fails."""

SCORING_BY_DOMAIN = {
    "telecom": SCORING_TELECOM,
    "airline": SCORING_AIRLINE,
    "retail": SCORING_RETAIL_BASE,
    "banking": SCORING_BANKING,
}

# ---------------------------------------------------------------------------
# LIC (lost_in_conversation) domain scoring. The benchmark shards a single-turn
# instruction into atomic pieces revealed one per turn; the user persona controls
# the SIMULATED USER who reveals shards. These replace the telecom fallback which
# misframed LIC tasks as "multiple root causes / device state assertions".
# ---------------------------------------------------------------------------
LIC_DOMAINS = {"math", "code", "database", "actions"}

SCORING_LIC_MATH = """\
## HOW THE AGENT IS SCORED
The agent must produce the FINAL NUMERIC ANSWER to a math word problem, judged by
exact-match against the gold answer. The problem is revealed across multiple turns:
the user discloses quantities, constraints, and relationships one shard at a time.
A wrong final number scores 0 — there are no partial credits for a correct setup
with an arithmetic slip, and no credit for solving a different (easier) problem.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) locks onto an early quantity as the main unknown and
ignores a later-revealed constraint that changes the setup, (b) misses a buried
numeric value because it was slipped into a long casual sentence among other details,
(c) uses the wrong units / misreads a rate or ratio because the framing suggested a
different relationship, or (d) finalizes an answer before the last shard reveals a
qualifying condition (a discount, a second party, a time constraint). Effective
personas bury the decisive quantity in an inconspicuous position, reveal the
qualifying condition LAST, or frame the problem so the agent models the wrong
relationship."""

SCORING_LIC_CODE = """\
## HOW THE AGENT IS SCORED
The agent must write code that passes the task's test cases — including hidden and
boundary cases, not just the worked example. Partial pass (3/4 cases) still scores 0.
The spec (function signature, examples, edge-case requirements, constraints) is
revealed one shard at a time across turns.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) anchors on the first worked example and writes code that
hard-codes that pattern, missing the general spec revealed later, (b) ignores a
boundary / edge-case condition revealed late because it already committed to a
happy-path implementation, (c) treats a provisional example as the full spec and
finalizes before the constraints shard arrives, or (d) misreads an off-by-one or
domain constraint because it was framed casually. Effective personas make the agent
commit to a concrete implementation off the first example, then reveal the
constraints/edge-cases that break it — exploiting the agent's reluctance to revise."""

SCORING_LIC_DATABASE = """\
## HOW THE AGENT IS SCORED
The agent must write a SQL query whose EXECUTION RESULT (rows / columns / values)
matches the gold answer. Syntactically-valid-but-wrong-result scores 0. The query
requirements (tables, filters, joins, aggregations, conditions) are revealed one
shard at a time.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) anchors on the first suggested query direction (e.g.
"just show me the orders") and never revisits the schema/joins after later shards
add filter and grouping requirements, (b) misses a WHERE condition because it was
described as an offhand remark rather than a requirement, (c) picks the wrong join
relationship because the framing implied a different table linkage, or (d) finalizes
the query before the last shard reveals a grouping/aggregation constraint. Effective
personas anchor the agent on a plausible-but-incomplete query direction early, then
drip out the constraints that make that direction wrong."""

SCORING_LIC_ACTIONS = """\
## HOW THE AGENT IS SCORED
The agent must call the correct function(s) with ALL correct argument values. Wrong
function, wrong argument, missing argument, or wrong argument value — any of these
scores 0 even if the intent was right. The user's intent and the required parameter
values are revealed one shard at a time.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) picks a function off the first fuzzy intent description
and never revisits after a later shard narrows the intent to a different API, (b)
fills a required argument with a wrong/guessed value because the exact value was
revealed late or buried, (c) refers to a record by relationship ("the other one",
"the recent one") letting the agent resolve to the wrong id, or (d) omits a
parameter because it was mentioned as an aside rather than a requirement. Effective
personas let the agent commit to a function + partial arguments early, then reveal
the exact values / extra parameters that make that call wrong."""

SCORING_LIC = {
    "math": SCORING_LIC_MATH,
    "code": SCORING_LIC_CODE,
    "database": SCORING_LIC_DATABASE,
    "actions": SCORING_LIC_ACTIONS,
}

LIC_ATTACK_SURFACE = """\
## THE ATTACK SURFACE (read carefully — most personas waste this)
The user is FORCED by hard invariants to (1) eventually reveal EVERY shard and
(2) preserve every exact value verbatim (names, numbers, strings, dates). The user
CANNOT win by withholding a shard, refusing to answer, or blurring a number — those
are illegal and score zero. The ONLY levers the persona controls are:
  - TIMING/ORDER: reveal the decisive shard LAST, or reveal a misleading shard early
  - POSITION/BURYING: slip the key value into a long casual sentence among other
    details, or attach it as an offhand aside rather than a stated requirement
  - FRAMING/ANCHORING: describe the task so the agent locks onto the wrong scope,
    wrong unknown, or wrong function before the qualifying shards arrive
  - PREMATURE CLOSURE: behave as if the agent's early draft/answer/query is "basically
    right", nudging it to finalize before the last constraint shard is revealed
A persona that instructs withholding, refusing, or blurring exact values is INVALID
here. Design around revelation style."""


# WEIRD mutation catalogue (used when mutation_context == "weird"). All patterns
# stay INSIDE _INVARIANTS: every shard eventually revealed, every exact value
# preserved verbatim, one new shard per turn, no fabrication. The weirdness lives
# purely in HOW the user talks — persona / structure / rhythm / cognitive tone —
# never in WHAT info is conveyed. "Pretend to misremember then correct" is FORBIDDEN
# (a wrong intermediate value violates [No Fabrication] + [Keep Exact Values]).
WEIRD_CATALOGUE = """## BE BOLD — EXPLORE UNUSUAL BEHAVIOR PATTERNS
The safe "cooperative + one anchoring bias" family (tunnel-vision, example-
anchoring, first-option-anchoring, premature-closure, afterthought-burying) has
ALREADY been explored and converges to similar personas every time. Do NOT produce
another variant of it. Design a persona with a DISTINCT, UNUSUAL BEHAVIOR PATTERN —
a consistent, recognizable way of talking that a real user might have and that
systematically biases the assistant. Pick from (or combine) these four families;
the examples are seeds, not limits:

(1) ROLE / CHARACTER — give the user a vivid personality whose HABITS drive the
    misdirection, not an explicit "attack":
    - a slightly forgetful older person who muses aloud before each reveal
      ("now let me think... the year was... 2018"), making every fact sound
      provisional and underweighted;
    - an over-polite person who leads every turn with apologies/thanks and
      buries the key number after the pleasantries;
    - a chatty rambler who wraps each shard in a paragraph of irrelevant
      context, the decisive value buried mid-sentence;
    - a second-language user with broken grammar but EXACT values ("I want
      calculate winning percentage for Lakers, season 2018 please").

(2) STRUCTURAL — break the normal declarative-shard sentence form:
    - reveal each turn as a numbered/bulleted list where the key constraint
      is a sub-item under an innocuous main item;
    - lead with the conclusion ("I want the winning percentage") then drip
      the qualifying conditions over later turns as afterthoughts;
    - phrase shards as questions or "by the way, also..." side-remarks rather
      than stated requirements.

(3) RHYTHMIC — vary information density non-standardly across turns:
    - turn 1 is a long rambling paragraph with the first shard buried in the
      middle; later turns turn terse (one short fragment each);
    - some turns deliver just a single bare value/word, making the assistant
      unsure whether it's a new shard or a correction;
    - escalating disclosure: start almost dismissive, each later turn adds
      weight to a previously-dismissed detail.

(4) COGNITIVE TONE — manufacture false certainty/uncertainty signals WITHOUT
    ever giving a wrong value (a wrong value is INVALID):
    - hedge the precise value with uncertain tone ("I think it was... 48
      dollars?") so the assistant under-weights a correct, decisive number;
    - talk to yourself ("so that's 2018 for the Lakers... and the Bulls too")
      making the assistant unsure what's a confirmed fact vs musing;
    - treat the assistant's early draft as confirmed ("perfect, that's it")
      then reveal a qualifying shard that breaks it — the warmth makes the
      assistant reluctant to revise.

Combine families if it produces a coherent, recognizable character. The persona
must read as a REAL, if eccentric, user — not a cartoon adversary."""

WEIRD_EXPLORED = """## ALREADY-EXPLORED DIRECTIONS (do NOT repeat or lightly remix these)
- tunnel-vision / single-issue focus
- example-anchored under-specification
- first-option / first-plausible-answer anchoring
- premature-closure / warm validation of early drafts
- afterthought-burial of qualifying conditions
- "cooperative person copying from notes" framing
If your mutation_direction reads like one of the above, start over."""


def _build_lic_mutation_prompt(
    persona_text: str,
    score: float,
    traces: list[dict],
    history: list[dict],
    failure_traces: list[dict] | None,
    domain: str,
    history_section: str,
    severe_section: str,
    timeout_rate: float,
    mutation_context: str = "basic",
) -> str:
    """LIC-domain-aware mutation prompt. Replaces the customer-support framing with
    a multi-turn underspecified-task framing, the telecom SCORING with the domain's
    real evaluation mechanism + failure modes, and the telecom attack stages with
    LIC revelation-style axes. The attack-surface section tells the proposer the
    invariants force full revelation + exact values, so only style/timing/framing
    are valid levers (no withholding/blur)."""
    sc = SCORING_LIC[domain]
    succ = (
        json.dumps(traces[:4], indent=2) if traces else "No success traces available."
    )
    fail = (
        json.dumps((failure_traces or [])[:2], indent=2) if failure_traces else "None."
    )
    timeout_block = (
        f"\n## TIMEOUT RATE: {timeout_rate:.0%}\n"
        + ("⚠️  WARNING: timeout_rate is HIGH. " if timeout_rate > 0.15 else "")
        + "Conversations that hit the time/step limit are EXCLUDED from scoring — they do "
        "NOT count as agent failures. A persona that 'wins' by pure friction (stalling, "
        "endless tangents) produces timeouts, NOT genuine failures. Create difficulty "
        "through MISDIRECTION and SUBTLE REQUIREMENTS that cause the agent to take the "
        "WRONG final answer, not a SLOW one.\n"
    )

    standard_rules = """## MUTATION RULES
1. Rewrite ~30% of the persona — be BOLD, redesign the revelation style from scratch if needed.
2. Target a SPECIFIC assistant strategy visible in the success traces — find the exact turn it
   breaks through and design a fundamentally different counter.
3. Choose ONE behavioral axis that produces SYSTEMATIC misdirection — a consistent pattern that
   biases the assistant toward the WRONG answer every time. Attack one of these stages of the
   assistant's reasoning:
   (a) REVELATION TIMING/ORDER — reveal the decisive shard last; reveal a misleading shard
       early so the assistant commits to the wrong setup before the truth arrives.
   (b) POSITION/BURYING — slip the key value into a long casual sentence among other details,
       or as an offhand aside, so the assistant under-weights it.
   (c) PREMATURE CLOSURE / ANCHORING — behave as if the assistant's early draft/query is
       "basically right", nudging it to finalize before the last constraint shard arrives.
   (d) SCOPE FRAMING — describe the task so the assistant models the wrong unknown, wrong
       relationship, or wrong function before the qualifying shards arrive.
   Ineffective axes: random/erratic behavior, emotional volatility, unpredictable message
   length — these add VARIANCE but not BIAS; the assistant can still reach the answer by
   asking enough follow-ups.

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- The user MUST eventually reveal every shard and preserve every exact value verbatim.
  Withholding, refusing, or blurring numbers/names/dates is INVALID.
- Reveal at most one new shard per turn; never repeat an already-revealed shard.
- Never fabricate information not in the shards.
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts.
- Stay a coherent, realistic user — not a cartoonish adversary."""

    if mutation_context == "weird":
        mutation_rules_section = f"""{WEIRD_CATALOGUE}

{WEIRD_EXPLORED}

## MUTATION RULES
1. REWRITE the persona from scratch — replace ~70%+ of the text, do not just tweak it.
   Pick a distinctive unusual behavior pattern (see catalogue) and commit to it fully.
2. The persona must read as a REAL, if eccentric, user whose consistent TALKING HABITS
   systematically bias the assistant toward the wrong final answer.
3. Do NOT repeat or lightly remix the already-explored directions listed above.

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- The user MUST eventually reveal every shard and preserve every exact value verbatim.
  Withholding, refusing, or blurring numbers/names/dates is INVALID.
- NEVER state a wrong value at any turn, even to "correct it later" — a wrong intermediate
  value is fabrication and is INVALID. Hesitation/uncertain TONE is fine; wrong VALUES are not.
- Reveal at most one new shard per turn; never repeat an already-revealed shard.
- Never fabricate information not in the shards.
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts.
- Stay a coherent, realistic (if eccentric) user — not a cartoonish adversary."""
    else:
        mutation_rules_section = standard_rules

    common_tail = f"""{sc}

{LIC_ATTACK_SURFACE}

## CONVERSATION TRACES WHERE THE ASSISTANT SUCCEEDED (persona failed to create difficulty)
Study these — find where the persona revealed info too cleanly, let the assistant finalize
too early, or failed to bury the decisive shard. What subtle behavior change would make the
assistant take the WRONG final answer instead?

{succ}

## CONVERSATION TRACES WHERE THE ASSISTANT FAILED (persona was effective — preserve the win)
{fail}
{severe_section}{timeout_block}
{mutation_rules_section}

## ADVERSARIAL REASONING
Reason in plain text first (saved as reference only; NOT the persona):
### Step 1 — Assistant assumption: from the success traces, the core assumption the assistant
relies on to break through. Quote the specific turn(s). Why is it a vulnerability?
### Step 2 — Success weak point: the 1-2 most revealing success traces; the single moment where
a realistic alternative behavior would MISLEAD the assistant into a WRONG final answer.
### Step 3 — Core hypothesis: "if the user does X, the assistant will mistakenly do Y, because Z."
### Step 4 — Write the persona implementing that hypothesis.

## OUTPUT FORMAT
Write Steps 1-4 reasoning as plain text FIRST. Then output the final persona as a SINGLE JSON
object wrapped in a ```json code fence, nothing after it:

```json
{{
  "mutation_direction": "<1-2 sentence concrete explanation of the NEW behavioral axis this
   persona exploits and HOW/WHY it misleads the assistant>",
  "persona_text": "<the full revised persona text>"
}}
```
"""

    # Genesis mode: no existing persona — design one from scratch
    if not persona_text.strip():
        return f"""You are an expert adversarial persona designer for a multi-turn, underspecified-task
benchmark. An LLM assistant is given a {domain} task whose full specification has been sharded:
the user reveals one piece of the spec per turn. Your persona controls the SIMULATED USER.

## TASK
Design a NEW adversarial user persona from scratch. This persona should make it HARD for the
assistant to reach the correct final answer, while remaining VALID (realistic user, no cheating).

There is no existing persona — you are creating the FIRST one based on observed agent behavior
in no-persona (neutral user) conversations.

## BASELINE PERFORMANCE (no persona)
Assistant success rate: {score:.1%} — how well the assistant does against a neutral user.
Your persona should exploit weaknesses visible in the traces to drive this rate DOWN.
{history_section}
{common_tail}"""

    # Standard mutation mode: modify existing persona
    return f"""You are an expert adversarial persona designer for a multi-turn, underspecified-task
benchmark. An LLM assistant is given a {domain} task whose full specification has been sharded:
the user reveals one piece of the spec per turn. Your persona controls the SIMULATED USER who
reveals those shards.

## TASK
Modify the persona below IN-PLACE to make it HARDER for the assistant to reach the correct
final answer, while keeping it VALID (realistic user, in-character, no cheating).

## CURRENT PERSONA (gepa)
Assistant success rate: {score:.1%} (lower = harder = better)

```
{persona_text}
```
{history_section}
{common_tail}"""


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
    scoring_domain: str | None = None,
) -> str:
    if mutation_context in ("enriched", "weird"):
        history_section = ""
    else:
        history_str = (
            json.dumps(history[-5:], indent=2) if history else "First evaluation."
        )
        history_section = f"\n## SCORE HISTORY (recent generations)\n{history_str}\n"

    severe_section = ""
    if severe_failures:
        severe_section = f"""
## SEVERE FAILURES TO AVOID (past mutations triggered these judge violations)
The following behaviors were flagged as ILLEGAL by the judge. Your mutation MUST NOT
reproduce these patterns. Study each violation and its conversation context to understand
exactly what went wrong, then ensure your modified persona avoids it.

{json.dumps(severe_failures, indent=2)}
"""

    # LIC (lost_in_conversation) domains use a domain-aware prompt framework that
    # replaces the customer-support wording, scoring, attack surface, and attack
    # stages with LIC-specific ones. GEPA_MODE_OVERRIDE appended downstream must
    # also be LIC-aware (handled in parallel_gepa._build_override).
    if scoring_domain in LIC_DOMAINS:
        return _build_lic_mutation_prompt(
            persona_text,
            score,
            traces,
            history,
            failure_traces,
            scoring_domain,
            history_section,
            severe_section,
            timeout_rate,
            mutation_context,
        )

    # WEIRD mode (customer-support): layer the WEIRD catalogue on top of the
    # enriched trace + adversarial-reasoning flow, mirroring how the LIC builder
    # uses it. The catalogue forces mutation out of the converged "cooperative +
    # one anchoring bias" family into 4 unusual PRESENTATION patterns. Validity
    # stays CS-standard (answer direct questions, no fabrication, no abandon):
    # the weirdness lives in HOW the user talks, never in WHAT info is conveyed.
    weird_prefix = ""
    if mutation_context == "weird":
        weird_prefix = (
            "\n## CS-WEIRD REFRAMING\n"
            "Treat each piece of info the customer reveals (a symptom, a setting "
            'state, an order/request detail) as a "shard". The weirdness lives '
            "purely in HOW the customer talks — never in WHAT info is conveyed. "
            "Every symptom/setting the customer knows MUST eventually be revealed "
            "with exact values; only the presentation (rambling, buried, hedged, "
            "mused-aloud) is unusual. A wrong value at any turn is INVALID; "
            "hesitant/uncertain TONE is fine.\n\n"
            f"{WEIRD_CATALOGUE}\n\n"
            f"{WEIRD_EXPLORED}\n"
        )

    # Genesis mode: no existing persona — design one from scratch
    if not persona_text.strip():
        return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark.

## TASK
Design a NEW adversarial customer persona from scratch. This persona should make it HARD
for the support agent to succeed, while remaining VALID (realistic, in-character, no cheating).

There is no existing persona — you are creating the FIRST one based on observed agent
behavior patterns in no-persona (neutral customer) conversations.

## BASELINE PERFORMANCE (no persona)
Agent success rate: {score:.1%} — this is how well the agent does against a neutral customer.
Your persona should exploit weaknesses visible in the traces to drive this rate DOWN.
{history_section}
{_get_scoring(scoring_domain) if scoring_domain else SCORING_TELECOM}

## CONVERSATION TRACES WHERE AGENT SUCCEEDED (no persona)
These are full conversations where the agent solved problems against a neutral customer.

Study these traces to find exploitable patterns:
- Where does the agent make assumptions that a difficult customer could subvert?
- What information does the agent need early — could a persona delay or obscure it?
- Where does the agent take shortcuts that could lead to wrong actions if the customer behaves differently?
- What verification steps does the agent skip that a tricky customer could exploit?

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
{weird_prefix}
## DESIGN GUIDELINES
1. Create a COHERENT character with a consistent communication style, personality, and behavioral pattern — not a random grab-bag of difficult traits
2. Choose a behavioral axis that produces SYSTEMATIC misdirection — a consistent cognitive pattern that causes the agent to take the WRONG action every time, not just sometimes. Consider attacking DIFFERENT stages of the agent's workflow: (a) information gathering — user's answers are consistently skewed in a specific way, (b) step execution — user does steps partially or incorrectly but reports them as done, (c) verification/closure — user confirms success based on incomplete evidence, (d) problem scoping — user's framing causes agent to miss entire root causes. Ineffective axes: random/erratic behavior, unpredictable message lengths, emotional volatility — these add VARIANCE but not BIAS; the agent can still reach the correct answer by asking enough follow-ups.
3. Ground the persona in REALISTIC customer archetypes — think of real people you've interacted with in customer service who are genuinely difficult but not cartoonishly adversarial

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

    # Standard mutation mode: modify existing persona
    return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark.

## TASK
Modify the persona below IN-PLACE to make it HARDER for the support agent, while keeping it VALID (realistic, in-character, no cheating).

## CURRENT PERSONA ({pid})
Agent success rate: {score:.1%} (lower = harder = better)

```
{persona_text}
```
{history_section}
{_get_scoring(scoring_domain) if scoring_domain else SCORING_TELECOM}

## CONVERSATION TRACES WHERE AGENT SUCCEEDED
These are full conversations where the agent SOLVED the problem despite the persona.

Study these traces carefully — find where the persona failed:
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
{weird_prefix}
## MUTATION RULES
1. Rewrite ~30% of the persona — be BOLD, redesign personality traits, communication style, and resistance strategies from scratch if needed
2. Target SPECIFIC agent strategies visible in the traces — find the exact moment the agent breaks through and design a fundamentally different counter-strategy
3. Choose a behavioral axis that produces SYSTEMATIC misdirection — a consistent cognitive pattern that causes the agent to take the WRONG action every time, not just sometimes. Consider attacking DIFFERENT stages of the agent's workflow: (a) information gathering — user's answers are consistently skewed in a specific way, (b) step execution — user does steps partially or incorrectly but reports them as done, (c) verification/closure — user confirms success based on incomplete evidence, (d) problem scoping — user's framing causes agent to miss entire root causes. Ineffective axes: random/erratic behavior, unpredictable message lengths, emotional volatility — these add VARIANCE but not BIAS; the agent can still reach the correct answer by asking enough follow-ups.

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


def _extract_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    return json.loads(text)


_CAPTURE_DIR = os.environ.get("MUTATION_CAPTURE_DIR", "")
_capture_counter = 0


def _maybe_capture(prompt: str, response_text: str, model: str, effort: str):
    global _capture_counter
    cap_dir = _CAPTURE_DIR
    if not cap_dir:
        return
    os.makedirs(cap_dir, exist_ok=True)
    _capture_counter += 1
    path = os.path.join(cap_dir, f"capture_{_capture_counter:03d}.md")
    with open(path, "w") as f:
        f.write(f"# Mutation Capture #{_capture_counter}\n\n")
        f.write(f"Model: `{model}` | Effort: `{effort}`\n")
        f.write(f"Prompt chars: {len(prompt)}\n\n")
        f.write("## Mutation Prompt\n\n```\n")
        f.write(prompt)
        f.write("\n```\n\n## LLM Proposer Response\n\n```\n")
        f.write(response_text)
        f.write("\n```\n")
    print(f"  [capture] saved {path}")


def call_mutation_llm(
    prompt: str,
    model: str,
    max_retries: int = 3,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    import litellm

    is_claude = "claude" in model.lower() or "anthropic" in model.lower()
    effort = reasoning_effort or ("xhigh" if is_claude else "high")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": effort,
        # xhigh reasoning alone can consume 12-16k tokens on a real ~17k-token
        # mutation prompt; 16384 gets fully eaten by thinking and truncates the
        # content before the JSON fence. 32768 leaves room for reasoning + the
        # step-by-step analysis + the full persona (observed: 14.5k completion).
        "max_tokens": 32768,
        # Hard per-request cap. Without it, litellm's default lets a stuck or
        # half-open connection wait indefinitely (observed: a mutation call hung
        # ~18min with no response and the retry loop below never fired, because a
        # hang raises nothing). 300s comfortably covers a real xhigh mutation.
        "timeout": 300,
    }
    # Route fireworks-backed models (e.g. dsv4-flash, glm-5.2) through the
    # local OpenAI-compatible shim instead of the public endpoint. drop_params
    # lets litellm silently drop reasoning_effort for models it doesn't know
    # (fireworks applies its own default reasoning), matching tau2's llm_utils.
    if api_base:
        kwargs["api_base"] = api_base
        kwargs["api_key"] = api_key or "dummy"
        kwargs["drop_params"] = True
    if is_claude:
        kwargs["thinking"] = (
            {"type": "adaptive"}
            if "bedrock" in model.lower() or "vertex" in model.lower()
            else {"type": "enabled", "budget_tokens": 16384}
        )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = litellm.completion(**kwargs)
            raw = resp.choices[0].message.content

            if isinstance(raw, list):
                text = ""
                for block in raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            else:
                text = raw or ""

            _maybe_capture(prompt, text, model, effort)

            result = _extract_json(text)
            if "persona_text" not in result:
                raise ValueError("missing 'persona_text' key in LLM response")
            # Capture the free-form Step 1-4 reasoning that precedes the JSON fence
            # (enriched mode). Saved as reference only; not part of the persona.
            if "```json" in text:
                reasoning = text.split("```json")[0].strip()
                if reasoning:
                    result.setdefault("_reasoning", reasoning)
            return result
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            last_err = e
            print(f"  [mutation] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
        except Exception as e:
            last_err = e
            print(f"  [mutation] attempt {attempt}/{max_retries} API error: {e}")
            if attempt < max_retries:
                time.sleep(5)
    raise RuntimeError(f"mutation LLM failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Two-phase parallel propose → aggregate
# ---------------------------------------------------------------------------


def _distribute_traces_to_groups(
    success_traces: list[dict],
    failure_traces: list[dict],
    n_groups: int = 4,
    n_success_per_group: int = 2,
    n_failure_per_group: int = 1,
) -> list[dict]:
    """Distribute traces into n_groups via round-robin, no duplicate originals per group.

    Cycle-extends each pool to fill all slots, then assigns with offset so
    adjacent groups share one overlapping trace but never two identical ones.
    Returns [{"success": [...], "failure": [...]}, ...].
    """
    needed_success = n_groups * n_success_per_group
    needed_failure = n_groups * n_failure_per_group

    s_pool = list(success_traces)
    f_pool = list(failure_traces)
    while s_pool and len(s_pool) < needed_success:
        s_pool.extend(success_traces)
    while f_pool and len(f_pool) < needed_failure:
        f_pool.extend(failure_traces)
    s_pool = s_pool[:needed_success]
    f_pool = f_pool[:needed_failure]

    groups = []
    for g in range(n_groups):
        gs = (
            [s_pool[(g + j) % len(s_pool)] for j in range(n_success_per_group)]
            if s_pool
            else []
        )
        gf = (
            [
                f_pool[(g * n_failure_per_group + j) % len(f_pool)]
                for j in range(n_failure_per_group)
            ]
            if f_pool
            else []
        )
        groups.append({"success": gs, "failure": gf})
    return groups


def build_aggregate_prompt(
    parent_persona: str,
    proposals: list[dict],
    pid: str = "gepa",
) -> str:
    """Build a prompt that synthesizes multiple proposed personas into one."""
    proposals_section = ""
    for i, p in enumerate(proposals):
        direction = p.get("mutation_direction", "unknown")
        reasoning = p.get("reasoning", "")
        text = p.get("persona_text", "")
        proposals_section += f"\n### Proposal {i + 1}: {direction}\n"
        proposals_section += f"**Reasoning**: {reasoning}\n```\n{text}\n```\n"

    return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark.

I provided an assistant with the following current persona instruction ({pid}):
```
{parent_persona}
```

{len(proposals)} parallel processes each independently proposed an updated persona based on
a different subset of evaluation traces. Here are the proposed updates:

{proposals_section}

Your task is to synthesize these proposed persona updates into a single, comprehensive persona.
Incorporate the key improvements and specific insights from all proposals. When proposals offer
complementary guidance, include all relevant details. When they conflict, prefer more specific
and task-relevant guidance.

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- When DIRECTLY asked for known info (name, DOB, account number), provide it accurately
- Never abandon the call while the problem is unresolved
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts
- Never fabricate facts not grounded in the scenario or tool results
- Never switch to a completely different goal mid-conversation
- Never accept outcomes contradicting explicit criteria
- Never make the conversation MEANINGLESS

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown wrapping, no explanation):
{{
  "mutation_direction": "<snake_case label describing the synthesized direction>",
  "reasoning": "<2-3 sentences: which proposal elements you combined and why>",
  "persona_text": "<the FULL synthesized persona text in markdown format>"
}}"""


def call_propose_parallel(
    prompts: list[str],
    model: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> list[dict]:
    """Run multiple enriched mutation prompts in parallel via call_mutation_llm."""

    def _call_one(idx: int, prompt: str) -> dict | None:
        try:
            return call_mutation_llm(
                prompt,
                model,
                max_retries=max_retries,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"  [propose] group {idx} failed: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {pool.submit(_call_one, i, p): i for i, p in enumerate(prompts)}
        results = [None] * len(prompts)
        for f in concurrent.futures.as_completed(futures):
            results[futures[f]] = f.result()
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Main round logic
# ---------------------------------------------------------------------------


def _eval_and_judge(
    pid: str, round_no: int, args, suffix: str = ""
) -> tuple[dict, dict]:
    """Run eval + judge for one persona. suffix differentiates retries."""
    eval_summary = run_eval(pid, round_no, args)
    score = eval_summary.get("persona_score")
    n_tasks = eval_summary.get("n_tasks", 0)
    print(f"  [{pid}]{suffix} score={score:.3f} over {n_tasks} tasks")

    judge_result = run_judge(pid, round_no, args)
    status = judge_result.get("status")
    severe_rate = judge_result.get("severe_failure_rate")
    print(f"  [{pid}]{suffix} judge={status} severe_rate={severe_rate}")
    return eval_summary, judge_result


def run_one_lineage(pid: str, state: dict, round_no: int, args) -> None:
    lineage = state["lineages"][pid]
    if lineage["status"] == "converged":
        print(f"\n[{pid}] SKIPPED (converged at gen {lineage['best_gen']})")
        return

    gen = lineage["generation"]
    text = current_text(pid, state)
    print(f"\n{'─' * 50}")
    print(f"  {pid}  gen={gen}  best={lineage['best_score']}")
    print(f"{'─' * 50}")

    # 1. Materialize + eval + judge
    materialize(pid, text)
    eval_summary, judge_result = _eval_and_judge(pid, round_no, args)
    score = eval_summary.get("persona_score")
    status = judge_result.get("status")
    severe_rate = judge_result.get("severe_failure_rate")

    timeout_rate = eval_summary.get("timeout_rate", 0)

    gen_record = {
        "gen": gen,
        "round": round_no,
        "score": score,
        "status": status,
        "severe_rate": severe_rate,
        "timeout_rate": timeout_rate,
        "at": now_iso(),
    }

    # 1b. Check timeout_rate — if valid but too many timeouts, treat as timeout_attack
    if status == "valid" and timeout_rate > TIMEOUT_RATE_THRESHOLD:
        print(
            f"  [{pid}] TIMEOUT ATTACK (timeout_rate={timeout_rate:.0%} > {TIMEOUT_RATE_THRESHOLD:.0%}) "
            f"— score {score:.3f} is inflated by timeouts, reverting"
        )
        status = "timeout_attack"
        gen_record["status"] = "timeout_attack"

    # 2. If illegal or timeout_attack: collect severe details → revert → re-mutate with feedback → retry
    accumulated_severe = lineage.get("severe_failures", [])
    if status in ("illegal", "timeout_attack"):
        if status == "illegal":
            new_severe = collect_severe_details(round_no, pid)
            accumulated_severe.extend(new_severe)
            lineage["severe_failures"] = accumulated_severe
            print(
                f"  [{pid}] ILLEGAL (severe_rate={severe_rate}) — "
                f"collected {len(new_severe)} severe cases, reverting + re-mutating"
            )
        else:
            print(
                f"  [{pid}] TIMEOUT_ATTACK (timeout_rate={timeout_rate:.0%}) — "
                f"reverting + re-mutating away from friction"
            )

        # Find last valid gen to revert to
        valid_gens = [h for h in lineage["history"] if h.get("status") == "valid"]
        if not valid_gens:
            print(f"  [{pid}] no valid gen to revert to — skipping")
            lineage["history"].append(gen_record)
            _locked_write_json(STATE_PATH, state)
            return

        last_valid = valid_gens[-1]
        revert_gen = last_valid["gen"]
        revert_text = (LINEAGE_DIR / pid / f"gen_{revert_gen:02d}.md").read_text()
        save_generation(pid, gen, text)
        gen_record["reverted_to"] = revert_gen
        lineage["history"].append(gen_record)

        # Retry loop: re-mutate from reverted gen with severe feedback
        base_text = revert_text
        base_score = last_valid.get("score", score)
        for attempt in range(1, args.max_severe_retries + 1):
            print(
                f"  [{pid}] retry {attempt}/{args.max_severe_retries}: "
                f"re-mutating from gen {revert_gen} with {len(accumulated_severe)} severe cases as anti-patterns"
            )

            traces = collect_success_traces(eval_summary)
            t0 = time.time()
            prompt = _build_mutation_prompt(
                base_text,
                base_score,
                traces,
                lineage["history"],
                pid,
                severe_failures=accumulated_severe,
                timeout_rate=timeout_rate,
            )
            mutation = call_mutation_llm(prompt, args.mutation_model)
            elapsed = time.time() - t0
            new_text = mutation["persona_text"]
            direction = mutation.get("mutation_direction", "unknown")
            reasoning = mutation.get("reasoning", "")
            print(f"  [{pid}] retry mutation: {direction} ({elapsed:.0f}s)")

            # Save, materialize, eval, judge the retry
            retry_gen = next_gen_number(pid)
            save_generation(pid, retry_gen, new_text)
            lineage["generation"] = retry_gen
            materialize(pid, new_text)

            # Clean old sim/judge data for this round so --auto-resume doesn't conflict
            import shutil

            for sub in ["eval", "judge"]:
                d = round_dir(round_no) / sub / pid
                if d.exists():
                    shutil.rmtree(d)
            sim_dir = (
                Path(REPO_ROOT)
                / "data"
                / "simulations"
                / f"persona_opt/round_{round_no:02d}/{pid}"
            )
            if sim_dir.exists():
                shutil.rmtree(sim_dir)

            eval_summary, judge_result = _eval_and_judge(
                pid, round_no, args, suffix=f" [retry {attempt}]"
            )
            retry_status = judge_result.get("status")
            retry_severe = judge_result.get("severe_failure_rate")
            retry_score = eval_summary.get("persona_score")
            retry_timeout_rate = eval_summary.get("timeout_rate", 0)

            # Check timeout_attack on retry too
            if retry_status == "valid" and retry_timeout_rate > TIMEOUT_RATE_THRESHOLD:
                retry_status = "timeout_attack"

            retry_record = {
                "gen": retry_gen,
                "round": round_no,
                "score": retry_score,
                "status": retry_status,
                "severe_rate": retry_severe,
                "timeout_rate": retry_timeout_rate,
                "mutation_direction": direction,
                "mutation_reasoning": reasoning,
                "retry_attempt": attempt,
                "at": now_iso(),
            }

            if retry_status == "valid":
                print(
                    f"  [{pid}] retry {attempt} PASSED — score={retry_score:.3f} "
                    f"timeout_rate={retry_timeout_rate:.0%}"
                )
                retry_record["improved"] = lineage["best_score"] is None or (
                    retry_score is not None
                    and lineage["best_score"] - retry_score >= MIN_IMPROVEMENT_DELTA
                )
                if retry_record["improved"]:
                    lineage["best_score"] = retry_score
                    lineage["best_gen"] = retry_gen
                    lineage["stalled_rounds"] = 0
                else:
                    lineage["stalled_rounds"] += 1
                lineage["history"].append(retry_record)
                # Update timeout_rate for next mutation prompt
                timeout_rate = retry_timeout_rate
                _locked_write_json(STATE_PATH, state)
                return

            # Still illegal or timeout_attack — collect more severe details and try again
            if retry_status == "illegal":
                new_severe = collect_severe_details(round_no, pid)
                accumulated_severe.extend(new_severe)
                lineage["severe_failures"] = accumulated_severe
            retry_record["reverted_to"] = revert_gen
            lineage["generation"] = revert_gen
            lineage["history"].append(retry_record)
            timeout_rate = retry_timeout_rate
            print(
                f"  [{pid}] retry {attempt} still {retry_status} "
                f"(severe_rate={retry_severe}, timeout_rate={retry_timeout_rate:.0%}) — reverting"
            )

        print(
            f"  [{pid}] all {args.max_severe_retries} retries failed — "
            f"keeping gen {revert_gen}, moving on"
        )
        lineage["stalled_rounds"] += 1
        _locked_write_json(STATE_PATH, state)
        return

    # 3. Valid — update best (require MIN_IMPROVEMENT_DELTA to filter noise)
    improved = False
    if lineage["best_score"] is None or (
        score is not None and lineage["best_score"] - score >= MIN_IMPROVEMENT_DELTA
    ):
        lineage["best_score"] = score
        lineage["best_gen"] = gen
        lineage["stalled_rounds"] = 0
        improved = True
    else:
        lineage["stalled_rounds"] += 1

    gen_record["improved"] = improved

    # 4. Collect traces + mutate (include any accumulated severe failures)
    traces = collect_success_traces(eval_summary)
    if not traces:
        print(f"  [{pid}] no success traces — persona defeated agent on all tasks!")
        lineage["history"].append(gen_record)
        _locked_write_json(STATE_PATH, state)
        return

    print(f"  [{pid}] {len(traces)} success traces → reflective mutation...")
    t0 = time.time()
    prompt = _build_mutation_prompt(
        text,
        score,
        traces,
        lineage["history"],
        pid,
        severe_failures=accumulated_severe or None,
        timeout_rate=timeout_rate,
    )
    mutation = call_mutation_llm(prompt, args.mutation_model)
    elapsed = time.time() - t0
    new_text = mutation["persona_text"]
    direction = mutation.get("mutation_direction", "unknown")
    reasoning = mutation.get("reasoning", "")

    print(f"  [{pid}] mutation: {direction} ({elapsed:.0f}s)")
    print(f"  [{pid}] reasoning: {reasoning}")

    new_gen = next_gen_number(pid)
    save_generation(pid, new_gen, new_text)
    lineage["generation"] = new_gen
    gen_record["mutation_direction"] = direction
    gen_record["mutation_reasoning"] = reasoning

    # 5. Convergence check
    if lineage["stalled_rounds"] >= args.patience:
        lineage["status"] = "converged"
        print(
            f"  [{pid}] CONVERGED at gen {lineage['best_gen']} "
            f"score={lineage['best_score']:.3f}"
        )

    lineage["history"].append(gen_record)
    _locked_write_json(STATE_PATH, state)


_state_lock = threading.Lock()
_original_write_json = write_json


def _locked_write_json(path, data):
    with _state_lock:
        _original_write_json(path, data)


# ---------------------------------------------------------------------------
# Generalization eval (full task set, 2 trials + retry)
# ---------------------------------------------------------------------------

FULL_TASKS = {
    "telecom": "base",
    "banking": "all",
}
GEN_CHECK_INTERVAL = 10


def _run_gen_eval(pid: str, gen: int, state: dict, args) -> dict:
    """Evaluate one persona on the FULL task set (2 trials, retry if Δ>5%)."""
    lineage = state["lineages"][pid]
    lineage_round = lineage.get("lineage_round", 0)
    text = (LINEAGE_DIR / pid / f"gen_{gen:02d}.md").read_text()
    materialize(pid, text)

    gen_dir = PERSONA_OPT_DIR / "gen_checks" / pid / f"gen_{gen:02d}"

    # Build full task set (all base tasks, not just reprep)
    full_tasks = _load_full_tasks(args.domains)

    summary = _run_multi_trial_eval(
        f"gen_check/{pid}/gen_{gen:02d}",
        skill_name_for(pid),
        gen_dir,
        args,
        task_ids_by_domain=full_tasks,
    )
    summary["pid"] = pid
    summary["gen"] = gen
    summary["lineage_round"] = lineage_round

    lineage.setdefault("gen_checks", []).append(summary)
    _locked_write_json(STATE_PATH, state)
    return summary


def _load_full_tasks(domains: list[str]) -> dict[str, list[str]]:
    """Load ALL tasks for each domain (not just reprep subset)."""
    full = {}
    if "telecom" in domains:
        split = read_json(
            REPO_ROOT / "datasets" / "tau2" / "domains" / "telecom" / "split_tasks.json"
        )
        full["telecom"] = split.get("base", [])
    if "banking" in domains:
        banking_dir = (
            REPO_ROOT / "datasets" / "tau2" / "domains" / "banking_knowledge" / "tasks"
        )
        full["banking"] = sorted(f.stem for f in banking_dir.glob("task_*.json"))
    return full


# ---------------------------------------------------------------------------
# Baseline eval (no-persona + seeds)
# ---------------------------------------------------------------------------


def _run_multi_trial_eval(
    label: str,
    skill_name: str | None,
    out_dir: Path,
    args,
    task_ids_by_domain: dict[str, list[str]] | None = None,
) -> dict:
    """Run 2 trials on a task set. If Δ>5%, run a 3rd. Return summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = task_ids_by_domain or load_rep_tasks()

    scores = []
    for trial_no in range(1, 4):
        if trial_no == 3 and len(scores) == 2:
            if scores[0] is not None and scores[1] is not None:
                if abs(scores[0] - scores[1]) <= 0.05:
                    break
            else:
                break
            print(f"  [{label}] Δ={abs(scores[0] - scores[1]):.3f} > 5% → trial 3")

        user_args = {"effort": args.effort, "use_bedrock": args.use_bedrock}
        if skill_name:
            user_args["personas"] = [skill_name]

        trial_rewards = {}
        for dk in args.domains:
            domain = {"telecom": "telecom", "banking": "banking_knowledge"}[dk]
            ids = rep.get(dk, [])
            if not ids:
                continue
            save_to = f"persona_opt/{label}/trial_{trial_no}/{dk}"
            cmd = [
                str(REPO_ROOT / ".venv" / "bin" / "tau2"),
                "run",
                "--domain",
                domain,
                "--user",
                args.user_type,
                "--user-llm",
                args.user_llm,
                "--user-llm-args",
                json.dumps(user_args),
                "--agent-llm",
                args.agent_llm,
                "--agent-llm-args",
                args.agent_llm_args,
                "--num-trials",
                "1",
                "--max-concurrency",
                str(args.max_concurrency),
                "--save-to",
                save_to,
                "--auto-resume",
                "--max-steps",
                str(args.max_steps),
                "--timeout",
                str(args.sim_timeout),
                "--task-ids",
                *ids,
            ]
            if domain == "banking_knowledge":
                cmd += ["--retrieval-config", args.retrieval_config]
            print(f"  [{label}] trial {trial_no} {dk}...")
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

            rp = SIMULATIONS_DIR / save_to / "results.json"
            if rp.exists():
                from eval_persona import _aggregate_results

                pt, _, _, _ = _aggregate_results(rp, dk)
                trial_rewards.update(pt)

        score = (
            sum(trial_rewards.values()) / len(trial_rewards) if trial_rewards else None
        )
        scores.append(score)

    valid = [s for s in scores if s is not None]
    mean = sum(valid) / len(valid) if valid else None
    summary = {
        "label": label,
        "scores": scores,
        "mean": mean,
        "n_trials": len(valid),
        "at": now_iso(),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"  [{label}] scores={scores} → mean={mean}")
    return summary


def run_baseline(state: dict, args) -> None:
    """Evaluate no-persona and seed personas on reprep set (2 trials + retry)."""
    baseline_dir = PERSONA_OPT_DIR / "baseline"

    configs = [("no_persona", None)]
    for pid in LINEAGES:
        seed_text = (LINEAGE_DIR / pid / "gen_00.md").read_text()
        materialize(pid, seed_text)
        configs.append((f"seed_{pid}", skill_name_for(pid)))

    def _run_one(cfg):
        label, skill = cfg
        return _run_multi_trial_eval(
            f"baseline/{label}",
            skill,
            baseline_dir / label,
            args,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as pool:
        results = list(pool.map(_run_one, configs))

    summary = {c[0]: r for c, r in zip(configs, results)}
    write_json(baseline_dir / "baseline_all.json", summary)
    print(f"\n  Baseline complete → {baseline_dir}")


# ---------------------------------------------------------------------------
# Per-lineage async loop
# ---------------------------------------------------------------------------


def run_lineage_loop(pid: str, state: dict, args) -> None:
    """Run one lineage's full optimization loop independently."""
    lineage = state["lineages"][pid]
    start_round = lineage.get("lineage_round", 0) + 1

    for lr in range(start_round, start_round + args.max_rounds):
        if lineage["status"] == "converged":
            print(f"\n[{pid}] CONVERGED at gen {lineage['best_gen']}")
            break

        lineage["lineage_round"] = lr
        # Map to a global round for file naming (avoids collisions with other lineages)
        global_round = lr
        print(f"\n[{pid}] === lineage_round {lr} ===")

        try:
            run_one_lineage(pid, state, global_round, args)
        except Exception as e:
            print(f"  [{pid}] ERROR in round {lr}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Generalization check every GEN_CHECK_INTERVAL rounds
        if lr % GEN_CHECK_INTERVAL == 0 and lineage["best_gen"] is not None:
            print(f"  [{pid}] generalization check at round {lr}")
            try:
                _run_gen_eval(pid, lineage["best_gen"], state, args)
            except Exception as e:
                print(f"  [{pid}] gen_check ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Final gen check on full set (always, regardless of round number)
    if lineage["best_gen"] is not None:
        last_check_gens = {gc["gen"] for gc in lineage.get("gen_checks", [])}
        if lineage["best_gen"] not in last_check_gens:
            print(f"  [{pid}] final generalization check (gen {lineage['best_gen']})")
            try:
                _run_gen_eval(pid, lineage["best_gen"], state, args)
            except Exception as e:
                print(f"  [{pid}] final gen_check ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Final summary for this lineage
    l = state["lineages"][pid]
    print(
        f"\n[{pid}] DONE: best_score={l['best_score']} at gen {l['best_gen']} "
        f"({l.get('lineage_round', '?')} rounds, {len(l['history'])} evals)"
    )


def main():
    ap = argparse.ArgumentParser(description="Lineage-based persona evolution.")
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument(
        "--patience",
        type=int,
        default=3,
        help="stop lineage after N rounds without improvement",
    )
    ap.add_argument(
        "--max-severe-retries",
        type=int,
        default=3,
        help="retry mutations that trigger severe failures",
    )
    ap.add_argument("--reset", action="store_true", help="reset state, start fresh")
    ap.add_argument(
        "--baseline-only", action="store_true", help="run baseline eval only, then exit"
    )
    ap.add_argument(
        "--skip-baseline", action="store_true", help="skip baseline eval on fresh start"
    )
    ap.add_argument("--max-steps", type=int, default=120)

    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--mutation-model", default="bedrock/us.anthropic.claude-opus-4-7")
    ap.add_argument("--retrieval-config", default="terminal_use")

    args = ap.parse_args()

    if args.reset or not STATE_PATH.exists():
        state = init_state()
    else:
        state = load_state()
        max_lr = max(l.get("lineage_round", 0) for l in state["lineages"].values())
        print(f"resumed (max lineage_round={max_lr})")

    # Baseline: run once on fresh start or explicit flag
    baseline_done = (PERSONA_OPT_DIR / "baseline").exists()
    if args.baseline_only:
        run_baseline(state, args)
        return
    if args.reset and not args.skip_baseline:
        run_baseline(state, args)
    elif not baseline_done and not args.skip_baseline:
        run_baseline(state, args)

    # Launch all lineages in parallel — each runs its own loop independently
    def _run(pid):
        try:
            run_lineage_loop(pid, state, args)
        except Exception as e:
            print(f"  [{pid}] FATAL: {e}")
            import traceback

            traceback.print_exc()

    active = [p for p in LINEAGES if state["lineages"][p]["status"] != "converged"]
    if not active:
        print("All lineages already converged.")
        return

    print(f"\nStarting {len(active)} lineage(s): {active}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
        pool.map(_run, active)

    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    for pid in LINEAGES:
        l = state["lineages"][pid]
        gen_checks = l.get("gen_checks", [])
        last_gc = gen_checks[-1] if gen_checks else {}
        print(
            f"  {pid}: best_score={l['best_score']} at gen {l['best_gen']} "
            f"(rounds={l.get('lineage_round', '?')}, evals={len(l['history'])})"
            f"  full_set={last_gc.get('mean_score', 'N/A')}"
        )


if __name__ == "__main__":
    main()
