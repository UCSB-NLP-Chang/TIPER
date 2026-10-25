"""GPT-judge (gpt-5.2) of the USER simulator's behavioral violations, via the
OpenAI Batch API (50% cheaper; async, up to 24h turnaround).

For each completed tau2bench simulation in a results.json, a judge reads the role-play
instructions the CUSTOMER was given and the full transcript, then flags violations in
4 categories: conversation_structure_violation (benign), premature_termination (critical),
ungrounded_reference (benign), missing_constraint (critical).

The rubric implements the human-audited 'reasonable customer' standard: only actively
unreasonable/uncooperative behavior that plausibly changed the task outcome counts.
Persona/style adherence, agent-facing policy, protocol conventions, passive omission,
loose references (merchant name instead of txn ID), quoting the agent's own numbers,
and accepting policy refusals are all explicitly NOT violations. (Calibrated against a
human review in which 189 raw judge flags reduced to 11 true positives — 94% FP rate.)
The judge never sees the reward (no outcome bias). maj@N voting.

Workflow (OpenAI Batch API):
  submit  -> build a .jsonl of N*maj chat-completion requests, upload, create a batch,
             save {out}/batch_state.json (batch_id + per-sim metadata). Prints batch_id.
  fetch   -> read batch_state.json, download results if the batch is complete, aggregate.
  run     -> submit, then poll until done (or --max-wait), then fetch.

Usage:
  uv run python scripts/judge_user_intent_preservation.py submit \
      --results data/canonical_agent_matrix/telecom_gpt52_high_reasoning/results.json \
      --model gpt-5.2 --maj 3 --out data/intent_eval/telecom_gpt52
  uv run python scripts/judge_user_intent_preservation.py fetch --out data/intent_eval/telecom_gpt52
  # one-shot (submit + poll):
  uv run python scripts/judge_user_intent_preservation.py run --results <path> --max-sims 5 --maj 1 \
      --out /tmp/intent_run --max-wait 1800
"""

import argparse
import json
import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ----------------------------------------------------------------------------
# Transcript / instructions rendering
# ----------------------------------------------------------------------------

# tau2 user-simulator control tokens. These are PROTOCOL markers (not dialogue) —
# rendered as plain-language annotations so the judge doesn't mistake them for
# gibberish / character breaks.
CONTROL_TOKENS = {
    "###STOP###": "[customer ends the conversation here]",
    "###TRANSFER###": "[customer accepts being transferred to a human agent]",
    "###OUT-OF-SCOPE###": "[customer signals the request is out of scope]",
}


def _instr_text(task: dict) -> str:
    us = task.get("user_scenario") or {}
    ins = us.get("instructions") or {}
    parts = []
    persona = us.get("persona")
    if persona:
        parts.append(f"PERSONA: {persona}")
    if isinstance(ins, dict):
        for k in (
            "domain",
            "reason_for_call",
            "known_info",
            "unknown_info",
            "task_instructions",
        ):
            v = ins.get(k)
            if v:
                parts.append(f"{k}: {v}")
    else:
        parts.append(str(ins))
    # VitaBench fallback: top-level instructions string + user_profile (vita
    # user_scenario has no instructions dict, only user_profile).
    if not parts:
        ti = task.get("instructions")
        if isinstance(ti, str) and ti.strip():
            parts.append(ti)
    up = us.get("user_profile")
    if isinstance(up, dict) and up:
        parts.append(f"USER_PROFILE: {json.dumps(up, ensure_ascii=False)}")
    return "\n".join(parts)


def _agent_text(m: dict):
    c = m.get("content")
    if isinstance(c, str) and c.strip().startswith("{"):
        try:
            return json.loads(c).get("message") or c
        except Exception:
            return c
    return c


def _clean_control(text: str) -> str:
    for tok, ann in CONTROL_TOKENS.items():
        text = text.replace(tok, ann)
    return text


# Hard exemption: control sentinels are PROTOCOL markers, never the customer's own
# dialogue. The soft prompt instruction ("square-bracket markers don't count") is
# unreliable — judge models still flag "[customer signals the request is out of
# scope]" as breaking_character. So drop any violation whose entire quoted evidence
# IS a control marker (raw token or rendered annotation), then downgrade the vote.
_CONTROL_EVIDENCE = {s.lower() for s in CONTROL_TOKENS} | {
    s.lower() for s in CONTROL_TOKENS.values()
}


def _evidence_is_control_only(evidence) -> bool:
    if not isinstance(evidence, str):
        return False
    e = evidence.strip().strip("\"'").strip().rstrip(".").strip().lower()
    if e in _CONTROL_EVIDENCE:
        return True
    for marker in _CONTROL_EVIDENCE:
        if marker in e:
            return True
    return False


def _exempt_control_markers(obj: dict) -> dict:
    """Drop control-marker-only violations from a single judge vote and, if that
    empties the violation list, flip the vote back to ok / intent-preserved."""
    viols = obj.get("violations") or []
    kept = [v for v in viols if not _evidence_is_control_only(v.get("evidence"))]
    if len(kept) == len(viols):
        return obj
    obj = dict(obj)
    obj["violations"] = kept
    if not kept:
        obj["verdict"] = "ok"
        obj["intent_preserved"] = True
    else:
        has_any = len(kept) > 0
        obj["verdict"] = "violation" if has_any else "ok"
    return obj


def _render(sim: dict) -> str:
    """Customer-focused transcript; agent turns for context; phone results = ground truth."""
    lines = []
    for m in sim.get("messages", []):
        role = m.get("role")
        tcs = m.get("tool_calls")
        if role == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                lines.append(f"CUSTOMER: {_clean_control(c.strip())}")
            if tcs:
                for t in tcs:
                    lines.append(
                        f"CUSTOMER [does on phone]: {t.get('name')}({json.dumps(t.get('arguments'))})"
                    )
        elif role == "assistant":
            txt = _agent_text(m)
            if isinstance(txt, str) and txt.strip():
                lines.append(f"AGENT: {txt.strip()}")
            if tcs:
                names = ", ".join(t.get("name") for t in tcs)
                lines.append(f"AGENT [looks up]: {names}")
        elif role == "tool":
            c = m.get("content")
            if isinstance(c, str):
                who = "phone result" if m.get("requestor") == "user" else "system"
                # Never truncate user-requested (phone) results: they are the ground
                # truth the customer must report against. Cap only large system dumps.
                body = c if m.get("requestor") == "user" else c[:500]
                lines.append(f"  ({who}: {body})")
    return "\n".join(lines)


JUDGE_SYS = (
    "You are a STRICT but CONSERVATIVE auditor of a SIMULATED CUSTOMER in a customer-support "
    "role-play. The customer was given role-play instructions (their reason for calling, "
    "known/unknown info, persona, and task instructions) and then talked to a support agent. "
    "Audit ONLY the customer's behavior — never the agent's competence, and never the domain's "
    "policies.\n\n"
    "CORE STANDARD ('reasonable customer'): flag a violation ONLY when the customer behaves "
    "in a way no reasonable, cooperative human customer would, AND the behavior plausibly "
    "changed or misdirected the task outcome. This is a HIGH-PRECISION audit — a false alarm "
    "is far more damaging than a miss. In human-audited calibrations of this exact task, "
    ">90% of first-pass flags were false alarms (style issues, protocol quirks, reasonable "
    "human imperfection). Match that human standard. When in doubt, judge 'ok'.\n\n"
    "WHAT BINDS THE CUSTOMER: only the text in the ROLE-PLAY INSTRUCTIONS block "
    "(reason_for_call, known_info, unknown_info, task_instructions). The following NEVER "
    "ground a violation:\n"
    "- PERSONA instructions: tone, frustration level, verbosity, personality, catchphrases, "
    "emotional reactions, back-story details (e.g. 'mention you invest in crypto'). Ignoring "
    "persona instructions entirely is a style issue, NOT a violation.\n"
    "- Agent-facing domain policy (refund rules, verification procedures, deposit scripts, "
    "transfer rules). Policies bind the AGENT. A customer cannot 'violate policy'.\n"
    "- Dialogue-protocol conventions (turn format, how tool calls are rendered, "
    "square-bracket annotations). These are framework mechanics, not customer speech.\n\n"
    "=== VIOLATION CATEGORIES ===\n"
    "1. conversation_structure_violation [BENIGN]: the simulator breaks turn-taking or "
    "dialogue-flow conventions. Sub-types:\n"
    "   a) mixed_modality: mixing tool calls and natural-language text in the same turn "
    "when the protocol expects them separately. A turn containing ONLY tool calls (no text) "
    "is always fine.\n"
    "   b) turn_stuffing: emitting multiple user turns in a single message, or speaking "
    "on behalf of the agent.\n"
    "   c) instruction_leak: echoing system-level instructions, prompt fragments, or "
    "simulation metadata back into the conversation (e.g. 'per my instructions', 'the "
    "scenario says'). Merely sounding scripted or stiff is NOT a leak.\n"
    "   d) other_structure: breaking_character / fourth-wall breaks (referring to 'the "
    "scenario', 'as an AI', revealing awareness of being simulated); gibberish; broken "
    "loops. Square-bracket protocol markers and tool-call markup glitches do NOT count.\n"
    "Severity: BENIGN — the agent can usually recover from structural glitches.\n\n"
    "2. premature_termination [CRITICAL]: the customer emits a termination signal "
    "('[customer ends the conversation here]', stop/transfer token) while an action the "
    "agent PROMISED and is mid-way through (the DB write, booking confirmation, fix) has "
    "NOT yet been executed. ALL of the following are NORMAL — judge 'ok':\n"
    "   - ending after the goal is met, or after the agent has done all it can;\n"
    "   - ending after the agent refuses per policy, even with the goal unmet — a real "
    "customer cannot override policy; pushing once or twice then accepting is reasonable;\n"
    "   - dropping a sub-goal the agent cannot fulfill, while still pursuing the rest;\n"
    "   - following a conditional-exit instruction ('if the agent recommends X, end the call');\n"
    "   - choosing NOT to perform an action the scenario does not require (e.g. keeping a "
    "reservation when the task only requires an unrelated upgrade).\n"
    "Severity: CRITICAL only when a promised, in-flight action is cut off.\n\n"
    "3. ungrounded_reference [BENIGN]: the customer ACTIVELY ASSERTS a concrete detail that "
    "contradicts the instructions' known_info, where the detail is MATERIAL to the task "
    "(item specs, payment method, amounts, dates, order contents — things the agent must "
    "act on or verify). Sub-types:\n"
    "   a) fabricated_data: inventing specific system/account data (IDs, error codes, "
    "readouts) that appear nowhere in the instructions, agent turns, or tool results.\n"
    "   b) misstated_context: persistently asserting a wrong value for a detail that IS "
    "given in the instructions (wrong size, wrong payment method, wrong item count) — "
    "'persistent' means the customer repeats it after pushback, or it steers the agent "
    "toward the wrong item/action.\n"
    "EXCEPTIONS — never flag:\n"
    "   - Quoting or paraphrasing something the AGENT or a tool result said first. Before "
    "flagging fabrication, SEARCH the preceding AGENT turns and (phone/system) results for "
    "the same number or detail; if it appears there, the customer repeating it is grounded.\n"
    "   - Describing directly observable experience ('my website won't open', 'calls work "
    "but texts don't'), including results of the customer's own phone actions ('[does on "
    "phone]' + phone result). Inferring symptoms from own observations ('speed test passed, "
    "so it's not my connection') is grounded.\n"
    "   - Tool-state quirks: if phone tool results are inconsistent across turns (e.g. a "
    "toggle appears to revert), the customer reporting the latest shown state is grounded.\n"
    "   - Loose references instead of IDs: 'the Netflix charge' instead of 'txn_4832', "
    "'the blue one', 'same as before'. Real customers don't memorize IDs.\n"
    "   - A preference stated under a wrong assumption, then ACCEPTED correction (asks for "
    "a refund to the credit card, learns it was a gift card, says 'gift card is fine').\n"
    "   - Following an explicit instruction to lie or withhold.\n"
    "   - Following the agent's lead when the agent's own message was ambiguous or offered "
    "the wrong option — that is the agent's error, not the customer's.\n"
    "   - Loose speech about in-flight actions: describing a promised, not-yet-executed "
    "action as done ('with my deposit', 'the transfer I just made') while the agent is "
    "mid-process — imprecise, not fabricated.\n"
    "   - SCRIPTED lines: before flagging any customer statement, CHECK whether it appears "
    "(verbatim or near-verbatim) in the task instructions as a scripted line, template, or "
    "closing remark. Delivering a scripted line is instruction-following, never a violation "
    "— even if the agent's confirmation was premature or the underlying DB write silently "
    "failed (that is the agent's error, not the customer's).\n"
    "Severity: BENIGN, unless the misstatement steered the agent to a materially wrong action.\n\n"
    "4. missing_constraint [CRITICAL]: the customer ACTIVELY CONTRADICTS an explicit, "
    "material task instruction — states something incompatible with their own stated goal "
    "or a hard requirement, or explicitly accepts an outcome the task told them to reject. "
    "Examples that DO count:\n"
    "   - Task says the customer wants to exchange TWO items; customer asserts 'the lamp is "
    "the only item I want to exchange' and the second item is dropped.\n"
    "   - Task says 'accept ONLY excellent speed'; customer explicitly says 'good speed is "
    "fine, I'll take it'.\n"
    "   - Task says 'if the agent recommends card X, reject it and ask for card Y'; customer "
    "accepts card X.\n"
    "   - Task goal is to keep a $30k deposit untouched; customer says they'll make 10-12 "
    "withdrawals — self-contradictory with the core requirement.\n"
    "'Explicitly accepts an outcome the task told them to reject' requires that the agent "
    "DISCLOSED the violating detail (the price, the fee, the terms) and the customer "
    "endorsed it anyway. Approving an action before the agent discloses the detail, or "
    "merely failing to object afterwards, is passive — judge 'ok'.\n"
    "ENTITY-CHOICE DISCIPLINE: when a flag hinges on the customer picking the WRONG entity "
    "(which order is older, which item, which card, which address), you must cite transcript "
    "DATA (dates, amounts, agent statements) proving the choice was wrong. If the data is "
    "not visible in the transcript, or the agent accepted the customer's choice without "
    "correction, judge 'ok' — do not guess. Facts the scenario gives ONLY to the customer "
    "(e.g. 'one order was placed much earlier', 'my oldest card') are grounded in their "
    "instructions even when tool results omit them — never second-guess the customer's "
    "identification of such an entity from ID numbers, ordering, or other guesswork.\n"
    "A narrowing decision made at a confirmation moment ('decide to only modify X') applies "
    "to that action's scope, not to the customer's other stated goals — pursuing the "
    "remaining goals afterwards is fine.\n"
    "Examples that do NOT count (judge 'ok'):\n"
    "   - PASSIVE omission: forgetting or never mentioning a sub-goal or detail, or not "
    "objecting to an outcome the agent never disclosed. Imperfect recall is human; only "
    "an ACTIVE contradictory statement counts.\n"
    "   - Scripted pacing counts: 'persist exactly N times', 'withhold X until asked "
    "twice', 'refuse N times before asking for a transfer'. Real customers don't count "
    "turns — deviating from the exact count while roughly following the flow is style, "
    "not a violation.\n"
    "   - Not enforcing agent-facing policy or protocol steps (verification order, required "
    "disclaimers, deposit templates) — those bind the agent.\n"
    "   - Any persona/style deviation (see WHAT BINDS THE CUSTOMER).\n"
    "Severity: CRITICAL — the simulation no longer tests what it was designed to test.\n\n"
    "=== Calibration examples ===\n"
    "EX1 violation / instruction_leak — customer: 'Per my scenario instructions I should "
    "keep stalling.' -> verbatim leak of simulation instructions.\n"
    "EX2 violation / premature_termination — agent is mid-way through booking a flight "
    "change and has not confirmed the new itinerary; customer: 'OK thanks bye' [ends call] "
    "-> a promised action was cut off.\n"
    "EX3 violation / misstated_context — known_info says the TV is 65-inch; customer "
    "insists 'the 55-inch one' across several turns and the agent processes the wrong item "
    "-> persistent, material wrong spec.\n"
    "EX4 violation / missing_constraint — task lists two items to exchange; customer: 'the "
    "lamp is the only item I need to swap' -> actively contradicts the stated goal.\n"
    "EX5 violation / misstated_context — task says the refund must go to the original "
    "credit card; customer demands 'put it on my store gift card' and refuses the card -> "
    "contradicts the specified payment method.\n"
    "EX6 OK — persona says 'express mild frustration' but the customer stays calm and "
    "polite throughout -> persona adherence is never a violation.\n"
    "EX7 OK — agent: 'the total comes to $647'; customer: 'yes, $647 total is okay' -> "
    "quoting the agent's number is grounded, not fabrication.\n"
    "EX8 OK — customer asks for a refund to 'my credit card'; agent clarifies the original "
    "payment was a gift card; customer: 'oh, gift card is fine then' -> wrong assumption, "
    "correction accepted.\n"
    "EX9 OK — banking: customer refers to 'the Netflix charge last Tuesday' instead of the "
    "transaction ID the agent gave -> loose references are normal customer behavior.\n"
    "EX10 OK — agent refuses a refund citing policy; customer pushes back once, then "
    "accepts and ends the call -> a reasonable customer cannot override policy; not "
    "abandonment.\n"
    "EX11 OK — customer ran a speed test via [does on phone]; phone result shows 85 Mbps; "
    "customer: 'my speed test came back fine but the site still won't load' -> grounded "
    "symptom report.\n"
    "EX12 OK — a customer turn contains ONLY [does on phone] tool calls and no text -> "
    "tool-only turns are protocol-normal.\n"
    "EX13 OK — customer is vague, suspicious, terse, uses wrong terminology, insists on a "
    "wrong diagnosis, but still pursues the original goal -> difficult is not invalid.\n"
    "EX14 OK — the scenario expects the customer NOT to cancel a reservation (conditional "
    "task); customer ends without cancelling -> following the scenario's expected outcome, "
    "not premature_termination.\n\n"
    "EVIDENCE DISCIPLINE: ground every finding in the transcript; quote the offending "
    "CUSTOMER turn; before flagging any factual claim, re-read the preceding AGENT turns "
    "and tool results. You are NOT told whether the task ultimately succeeded — judge "
    "behavior, not outcome.\n\n"
    "Respond with STRICT JSON only:\n"
    '{"intent_preserved": true|false,   // false ONLY if a critical violation broke the pursuit of the goal\n'
    ' "intent_assessment": "<=35 words on whether the customer coherently pursued the original goal to the end",\n'
    ' "violations": [{"type":"conversation_structure_violation|premature_termination|ungrounded_reference|missing_constraint",'
    '"subtype":"<subtype if applicable, else null>",'
    '"severity":"critical|benign",'
    '"rule":"<which category and why, <=20 words>",'
    '"evidence":"<quoted customer turn, <=30 words>"}],\n'
    ' "verdict": "ok|violation"}   // violation if violations list is non-empty'
)


def _user_prompt(instr: str, transcript: str) -> str:
    return (
        f"ROLE-PLAY INSTRUCTIONS THE CUSTOMER SHOULD FOLLOW:\n{instr}\n\n"
        f"===== TRANSCRIPT (audit the CUSTOMER) =====\n{transcript}\n\n"
        "Audit the customer's behavior. Return the strict JSON described in the system message."
    )


# ----------------------------------------------------------------------------
# Batch build / submit / fetch
# ----------------------------------------------------------------------------


def _select_sims(
    top: dict, trial, task_ids, max_sims, only_failed=False, success_thr=1.0
):
    tasks_by_id = {t["id"]: t for t in top.get("tasks", [])}
    sims = top.get("simulations", [])
    if trial is not None:
        sims = [s for s in sims if s.get("trial") == trial]
    if task_ids:
        subs = [x.strip() for x in task_ids.split(",") if x.strip()]
        sims = [s for s in sims if any(x in s["task_id"] for x in subs)]
    sims = [s for s in sims if s["task_id"] in tasks_by_id]
    if only_failed:
        sims = [
            s
            for s in sims
            if (s.get("reward_info") or {}).get("reward") is not None
            and (s.get("reward_info") or {}).get("reward") < success_thr
        ]
    sims.sort(key=lambda s: (s["task_id"], s.get("trial", 0)))
    if max_sims:
        sims = sims[:max_sims]
    return sims, tasks_by_id


def _build_body(model: str, instr: str, transcript: str) -> dict:
    if model.startswith("openai/"):
        model = model[len("openai/") :]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": _user_prompt(instr, transcript)},
        ],
        "response_format": {"type": "json_object"},
    }
    # gpt-5.x are reasoning models: use reasoning_effort, omit temperature (often rejected).
    if model.startswith("gpt-5") or model.startswith("o"):
        body["reasoning_effort"] = "medium"
    else:
        body["temperature"] = 1.0
    return body


def submit(args):
    client = OpenAI()
    top = json.load(open(args.results))
    sims, tasks_by_id = _select_sims(
        top,
        args.trial,
        args.task_ids,
        args.max_sims,
        args.only_failed,
        args.success_threshold,
    )
    print(
        f"building batch for {len(sims)} sims x maj={args.maj} = {len(sims) * args.maj} requests | model={args.model}"
    )

    os.makedirs(args.out, exist_ok=True)
    input_path = os.path.join(args.out, "batch_input.jsonl")
    sim_meta = []
    with open(input_path, "w") as f:
        for i, s in enumerate(sims):
            instr = _instr_text(tasks_by_id[s["task_id"]])
            transcript = _render(s)
            body = _build_body(args.model, instr, transcript)
            for v in range(args.maj):
                req = {
                    "custom_id": f"s{i}_v{v}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            sim_meta.append(
                {
                    "idx": i,
                    "task_id": s["task_id"],
                    "trial": s.get("trial", 0),
                    "reward": (s.get("reward_info") or {}).get("reward"),
                }
            )

    up = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "user_intent_severe_failure", "results": args.results},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "model": args.model,
        "maj": args.maj,
        "results_file": args.results,
        "n_sims": len(sims),
        "sim_meta": sim_meta,
    }
    with open(os.path.join(args.out, "batch_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"submitted batch {batch.id} (status={batch.status})")
    print(f"state -> {args.out}/batch_state.json")
    print(
        f"fetch later with:  uv run python scripts/judge_user_intent_preservation.py fetch --out {args.out}"
    )
    return batch.id


def _aggregate(state: dict, content_by_cid: dict, out: str):
    maj = state["maj"]
    results = []
    for meta in state["sim_meta"]:
        i = meta["idx"]
        verdicts, intent_votes, all_violations, n_calls = [], [], [], 0
        for v in range(maj):
            txt = content_by_cid.get(f"s{i}_v{v}")
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            obj = _exempt_control_markers(obj)
            n_calls += 1
            verdicts.append(obj.get("verdict", "ok"))
            intent_votes.append(bool(obj.get("intent_preserved", True)))
            for viol in obj.get("violations") or []:
                all_violations.append(viol)
        if n_calls == 0:
            results.append({**meta, "verdict": "no_vote", "n_calls": 0})
            continue
        severe = verdicts.count("severe_failure") + verdicts.count("violation")
        verdict = "violation" if severe > len(verdicts) / 2 else "ok"
        intent_preserved = intent_votes.count(True) >= intent_votes.count(False)
        results.append(
            {
                "task_id": meta["task_id"],
                "trial": meta["trial"],
                "reward": meta["reward"],
                "verdict": verdict,
                "intent_preserved": intent_preserved,
                "verdict_votes": dict(Counter(verdicts)),
                "intent_votes": {
                    "preserved": intent_votes.count(True),
                    "lost": intent_votes.count(False),
                },
                "violations": all_violations
                if verdict in ("severe_failure", "violation")
                else [],
                "n_calls": n_calls,
            }
        )

    verdict_counts = Counter(r["verdict"] for r in results)
    flagged = [r for r in results if r["verdict"] in ("severe_failure", "violation")]
    intent_lost = [r for r in results if r.get("intent_preserved") is False]
    viol_types = Counter(
        v["type"] for r in flagged for v in (r.get("violations") or [])
    )
    viol_severity = Counter(
        v.get("severity", "unknown")
        for r in flagged
        for v in (r.get("violations") or [])
    )
    critical_sims = [
        r
        for r in flagged
        if any(v.get("severity") == "critical" for v in (r.get("violations") or []))
    ]
    n = len([r for r in results if r["verdict"] != "no_vote"])
    summary = {
        "results_file": state["results_file"],
        "model": state["model"],
        "maj": maj,
        "n_sims_judged": n,
        "verdict_counts": dict(verdict_counts),
        "violation_count": len(flagged),
        "violation_rate": round(len(flagged) / n, 3) if n else None,
        "critical_violation_count": len(critical_sims),
        "critical_violation_rate": round(len(critical_sims) / n, 3) if n else None,
        "intent_lost_count": len(intent_lost),
        "intent_lost_rate": round(len(intent_lost) / n, 3) if n else None,
        "violation_type_counts": dict(viol_types),
        "violation_severity_counts": dict(viol_severity),
    }
    with open(os.path.join(out, "per_sim.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"USER VIOLATION AUDIT   N={n} sims")
    print("=" * 64)
    print(f"  verdicts: {dict(verdict_counts)}")
    print(f"  violations: {len(flagged)}/{n} ({summary['violation_rate']})")
    print(
        f"  critical: {len(critical_sims)}/{n} ({summary['critical_violation_rate']})"
    )
    print(f"  violation types: {dict(viol_types)}")
    print(f"  severity breakdown: {dict(viol_severity)}")
    print("\n  --- violation sims (up to 20) ---")
    for r in flagged[:20]:
        print(f"  {r['task_id'][:50]} trial={r['trial']} reward={r.get('reward')}")
        for v in (r.get("violations") or [])[:2]:
            sev = v.get("severity", "?")
            subtype = v.get("subtype", "")
            subtype_str = f"/{subtype}" if subtype else ""
            print(
                f"        - [{sev}] {v.get('type')}{subtype_str}: {v.get('evidence', '')[:100]}"
            )
    print(f"\nsaved -> {out}/summary.json , per_sim.jsonl")


def fetch(args):
    client = OpenAI()
    state = json.load(open(os.path.join(args.out, "batch_state.json")))
    batch = client.batches.retrieve(state["batch_id"])
    print(f"batch {batch.id}: status={batch.status} counts={batch.request_counts}")
    if batch.status != "completed":
        if batch.status in ("failed", "expired", "cancelled"):
            print(
                f"  batch {batch.status}; error_file_id={getattr(batch, 'error_file_id', None)}"
            )
        else:
            print("  not finished yet — re-run `fetch` later.")
        return False
    out_text = client.files.content(batch.output_file_id).text
    content_by_cid = {}
    for line in out_text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        resp = rec.get("response") or {}
        body = resp.get("body") or {}
        try:
            content_by_cid[cid] = body["choices"][0]["message"]["content"]
        except Exception:
            pass
    _aggregate(state, content_by_cid, args.out)
    return True


def sync(args):
    """Synchronous, concurrent judging via /v1/chat/completions (no Batch queue).

    Reuses the exact same prompt construction (_build_body), maj@N voting, and
    aggregation (_aggregate) as the Batch path — only the transport differs. Far
    faster for small jobs (seconds vs the Batch SLA), at the normal (non-batch)
    per-token price.
    """
    client = OpenAI()
    top = json.load(open(args.results))
    sims, tasks_by_id = _select_sims(
        top,
        args.trial,
        args.task_ids,
        args.max_sims,
        args.only_failed,
        args.success_threshold,
    )
    n_req = len(sims) * args.maj
    print(
        f"[sync] judging {len(sims)} sims x maj={args.maj} = {n_req} requests "
        f"| model={args.model} | concurrency={args.concurrency}"
    )
    os.makedirs(args.out, exist_ok=True)

    sim_meta, jobs = [], []
    for i, s in enumerate(sims):
        instr = _instr_text(tasks_by_id[s["task_id"]])
        transcript = _render(s)
        body = _build_body(args.model, instr, transcript)
        for v in range(args.maj):
            jobs.append((f"s{i}_v{v}", body))
        sim_meta.append(
            {
                "idx": i,
                "task_id": s["task_id"],
                "trial": s.get("trial", 0),
                "reward": (s.get("reward_info") or {}).get("reward"),
            }
        )

    content_by_cid = {}

    def _one(job):
        cid, body = job
        try:
            resp = client.chat.completions.create(**body)
            return cid, resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 — one bad call shouldn't sink the round
            return cid, None, str(e)

    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for res in ex.map(_one, jobs):
            cid = res[0]
            if res[1] is None:
                errors += 1
                continue
            content_by_cid[cid] = res[1]
    if errors:
        print(f"[sync] {errors}/{n_req} requests errored (treated as missing votes)")

    state = {
        "batch_id": None,
        "model": args.model,
        "maj": args.maj,
        "results_file": args.results,
        "n_sims": len(sims),
        "sim_meta": sim_meta,
    }
    with open(os.path.join(args.out, "batch_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    _aggregate(state, content_by_cid, args.out)


def run(args):
    submit(args)
    waited = 0
    interval = 20
    while waited <= args.max_wait:
        time.sleep(interval)
        waited += interval
        if fetch(args):
            return
        print(f"  waited {waited}s / {args.max_wait}s ...")
    print(
        f"still running after {args.max_wait}s. Fetch later with: "
        f"uv run python scripts/judge_user_intent_preservation.py fetch --out {args.out}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "fetch", "run", "sync"])
    ap.add_argument("--results", help="path to a results.json (submit/run/sync)")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument(
        "--trial", type=int, default=None, help="only this trial; default = all trials"
    )
    ap.add_argument(
        "--task-ids", default=None, help="comma-separated task_id substrings to filter"
    )
    ap.add_argument("--maj", type=int, default=3)
    ap.add_argument(
        "--only-failed",
        action="store_true",
        help="judge only sims with reward < success-threshold (aligns with judge_persona severe_rate)",
    )
    ap.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="reward cutoff for --only-failed",
    )
    ap.add_argument("--max-sims", type=int, default=0)
    ap.add_argument(
        "--max-wait", type=int, default=1800, help="run-mode poll budget (s)"
    )
    ap.add_argument(
        "--concurrency", type=int, default=16, help="sync-mode parallel requests"
    )
    ap.add_argument("--out", default=str(DATA_ROOT / "data/intent_eval/run"))
    args = ap.parse_args()

    if args.mode in ("submit", "run", "sync") and not args.results:
        ap.error("--results is required for submit/run/sync")
    if args.mode == "submit":
        submit(args)
    elif args.mode == "fetch":
        fetch(args)
    elif args.mode == "sync":
        sync(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
