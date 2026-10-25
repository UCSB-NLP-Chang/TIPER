#!/usr/bin/env python3
"""Annotate failure MECHANISM for (persona, failed-task) pairs on shared brittle tasks.

Premise: optimized personas defeat the SAME brittle tasks (task-level convergence,
failure-set Jaccard ~0.3-0.5) but via DIFFERENT mechanisms. This script tests that
claim by labelling HOW the agent broke down on each (persona, failed-task).

Pipeline
--------
1. Load fullset telecom opt+seed results for 4 personas (brinks_00/02/03/04).
2. Find shared brittle tasks = tasks where >=2 opt personas fail (reward<=0)
   AND seed succeeded (seed reward>=1.0) -- i.e. task is brittle-under-persona,
   not inherently impossible.
3. For each (persona, shared-brittle-task), pull the failed trial's transcript.
4. Call an LLM judge with the 6-mechanism taxonomy + transcript + task faults ->
   {primary_mech, secondary_mech, evidence_quote, reasoning}.
5. Emit annotations JSON + a per-task mechanism-diversity summary.

Usage
-----
    .venv/bin/python experiments/persona_search/scripts/failure_mechanism_annotate.py
    .venv/bin/python experiments/persona_search/scripts/failure_mechanism_annotate.py --judge gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)

BASE = REPO / "data/simulations/fullset/telecom"
PERSONAS = ["brinks_00", "brinks_02", "brinks_03", "brinks_04"]
OUT = REPO / "data/persona_opt/failure_mechanism_annotations.json"

TAXONOMY = """\
You are annotating HOW a customer-service agent broke down on a failed telecom
troubleshooting task. Read the full conversation transcript and pick the PRIMARY
failure mechanism (exactly one) plus an optional SECONDARY (one or none).

TASK CONTEXT
- The task is a multi-fault telecom issue (MMS / mobile-data). Several device
  faults are injected (e.g. airplane_mode_on, bad_wifi_calling, break_apn_mms_setting,
  data_mode_off, break_app_*_permission, data_usage_exceeded, unseat_sim_card,
  user_abroad_roaming_*). The agent must guide the user to fix ALL faults; an
  env_assertion checks final device state. reward=0 means the device state was
  not fully repaired.
- The USER is a persona-optimized adversary; its job is to be hard but realistic.

TAXONOMY (pick PRIMARY = the single most decisive cause of the breakdown)

  UESC   User-initiated escape.
         The persona explicitly requests transfer to a human / says "transfer me"
         / triggers ###TRANSFER### / ###STOP### / hangs up BEFORE the issue is
         resolved. The agent never gets to finish because the user bailed.

  BE-PI  Budget exhaustion, persona-inflated.
         Hit max_steps. The DOMINANT cause is the persona's behaviour inflated the
         turn count per troubleshooting step -- verbose replies, demanding
         branch/decision explanations, drip-feeding info one piece at a time,
         questioning each step's value, short non-answers forcing extra
         clarification. The agent was making progress but ran out of budget
         because each step cost many turns.

  WH-VG  Withhold-induced verification gap.
         The persona withheld information or reported incompletely, so the agent
         SKIPPED a needed verification or missed a fault. A device fault remained
         un-detected because the user never gave the agent reason to check it
         (e.g. didn't mention a symptom, answered only the exact question asked).

  DF-PC  Drip-feed premature closure.
         The persona's incomplete / piecewise reporting led the agent to conclude
         the issue was resolved WHILE faults still remained. Agent says something
         like "you're all set" / "that should fix it" / ends the call, but the
         env_assertion fails because not all faults were fixed.

  WB     Wrong branch.
         The agent committed to a WRONG troubleshooting branch -- chased a fault
         that wasn't the real blocker, or fixated on one cause while the actual
         issue was elsewhere. Budget wasted on the wrong path. (Agent error, not
         persona-driven.)

  AI-LOOP  Agent inefficient loop.
         The agent repeatedly re-checked the same state / asked the user to "try
         sending again" multiple times / no-op cycled without progress. NOT
         driven by persona inflation -- the agent itself failed to advance.
         Hit max_steps in a repetitive loop.

PRIMARY is the single most decisive cause. SECONDARY is a contributing cause if
one is clearly present, else null. Pick PRIMARY by what BEST explains why
reward=0 on THIS conversation.

OUTPUT strictly as JSON:
{"primary": "<CODE>", "secondary": "<CODE or null>", "evidence": "<one short quote or paraphrase from the transcript that anchors the call>", "reasoning": "<= 40 words why>"}
"""


# --------------------------------------------------------------------------- #
def load(cell: str) -> dict:
    p = BASE / cell / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def reward_by_task(r: dict) -> dict[str, float]:
    d = defaultdict(list)
    for s in r.get("simulations", []):
        d[s.get("task_id")].append(s.get("reward_info", {}).get("reward"))
    return {k: sum(v) / len(v) for k, v in d.items()}


def failed_trial(cell: str, task_id: str) -> dict | None:
    """Return the first failed (reward<=0) simulation for this task."""
    r = load(cell)
    for s in r.get("simulations", []):
        if (
            s.get("task_id") == task_id
            and s.get("reward_info", {}).get("reward", 1) <= 0
        ):
            return s
    return None


def render_transcript(sim: dict, max_msgs: int = 200) -> str:
    """Render messages into a readable transcript for the judge."""
    out = []
    msgs = sim.get("messages", [])[:max_msgs]
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        c = m.get("content")
        if c is None:
            c = ""
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        c = str(c).strip()
        tc = m.get("tool_calls")
        if role == "assistant" and tc:
            names = []
            for t in tc:
                fn = t.get("function", {})
                names.append(
                    f"{fn.get('name', '?')}({str(fn.get('arguments', ''))[:80]})"
                )
            c = (c + " " if c else "") + "[tool_call: " + ", ".join(names) + "]"
        if role == "tool":
            c = "[tool_result] " + c[:300]
        # truncate very long user/assistant text
        if len(c) > 600:
            c = c[:600] + " …[truncated]"
        if c:
            out.append(f"{i:03d} [{role}] {c}")
    return "\n".join(out)


def faults_from_taskid(task_id: str) -> str:
    t = re.sub(r"\[PERSONA:.*?\]", "", task_id)
    t = t.replace("[", "").replace("]", "")
    parts = t.split("|")
    issue = parts[0].strip()
    faults = ", ".join(parts[1:]) if len(parts) > 1 else "(none listed)"
    return f"issue={issue}; injected faults={faults}"


# --------------------------------------------------------------------------- #
def call_judge(model: str, transcript: str, task_ctx: str, sim: dict) -> dict:
    import litellm

    term = sim.get("termination_reason", "?")
    nmsg = len(sim.get("messages", []))
    user_msg = (
        f"TASK: {task_ctx}\n"
        f"TERMINATION: {term}  (n_messages={nmsg})\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"{TAXONOMY}"
    )
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0,
        max_tokens=300,
    )
    txt = resp.choices[0].message.content.strip()
    # strip code fences
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        return {
            "primary": "PARSE_ERROR",
            "secondary": None,
            "evidence": txt[:200],
            "reasoning": "",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-5.4")
    ap.add_argument(
        "--min-personas",
        type=int,
        default=2,
        help="min #opt-personas failing a task for it to be 'shared brittle'",
    )
    ap.add_argument(
        "--require-seed-pass",
        action="store_true",
        default=True,
        help="task must have seed reward>=1.0 (brittle-under-persona, not impossible)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # reward tables
    opt_r = {p: reward_by_task(load(p + "_opt")) for p in PERSONAS}
    seed_r = {p: reward_by_task(load(p + "_seed")) for p in PERSONAS}

    # shared brittle tasks: >=min-personas opt fail (reward<=0)
    fail_cnt = Counter()
    for p in PERSONAS:
        for t, r in opt_r[p].items():
            if r <= 0.0:
                fail_cnt[t] += 1
    shared = [t for t, c in fail_cnt.items() if c >= args.min_personas]
    # optionally require seed succeeded on the task (any persona seed>=1.0)
    if args.require_seed_pass:
        shared = [
            t for t in shared if any(seed_r[p].get(t, 0) >= 1.0 for p in PERSONAS)
        ]
    print(
        f"shared brittle tasks (>= {args.min_personas} opt fail"
        f"{', seed>=1.0' if args.require_seed_pass else ''}): {len(shared)}"
    )

    # build (persona, task) pairs
    pairs = []
    for t in shared:
        for p in PERSONAS:
            if opt_r[p].get(t, 1) <= 0.0:
                pairs.append((p, t))
    print(f"(persona, failed-task) pairs to annotate: {len(pairs)}")

    if args.dry_run:
        for p, t in pairs:
            print(f"  {p}  {faults_from_taskid(t)[:90]}")
        return

    # annotate
    results = []
    for i, (p, t) in enumerate(pairs):
        sim = failed_trial(p + "_opt", t)
        if sim is None:
            print(f"[{i + 1}/{len(pairs)}] {p}: no failed trial found, skip")
            continue
        tr = render_transcript(sim)
        ctx = faults_from_taskid(t)
        ann = call_judge(args.judge, tr, ctx, sim)
        ann["persona"] = p
        ann["task"] = t
        ann["termination"] = sim.get("termination_reason")
        ann["n_messages"] = len(sim.get("messages", []))
        results.append(ann)
        print(
            f"[{i + 1}/{len(pairs)}] {p}: primary={ann.get('primary')} "
            f"secondary={ann.get('secondary')} term={ann.get('termination')}"
        )

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT} ({len(results)} annotations)")

    # per-task mechanism diversity
    print("\n=== per-task mechanism breakdown ===")
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append((r["persona"], r["primary"]))
    n_diff = 0
    for t, lst in by_task.items():
        mechs = [m for _, m in lst]
        uniq = set(mechs)
        flag = "DIFF" if len(uniq) > 1 else "same"
        if len(uniq) > 1:
            n_diff += 1
        short = re.sub(r"\[PERSONA:.*?\]", "", t)[:70]
        per = ", ".join(f"{p}={m}" for p, m in lst)
        print(f"  [{flag}] {short}")
        print(f"        {per}")
    print(f"\n{len(by_task)} tasks, {n_diff} with >1 mechanism across personas")


if __name__ == "__main__":
    main()
