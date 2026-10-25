"""Blind pairwise GPT-judge of two user-simulators on the SAME telecom tasks.

Compares the CUSTOMER (user-simulator) in two runs that share task set + seed:
  A = telecom_gpt54_sonnet46user_high_reasoning        (raw litellm sonnet user)
  B = telecom_gpt54_sonnet46cconuser_high_reasoning_v2 (cc_user_con, --bare aligned)

For one trial per task (114 pairs), a GPT judge sees the role-play instructions and
both anonymized transcripts and picks a winner on two axes:
  1. realism            — which customer talks more like a real human
  2. instruction_follow — which customer better obeys the role-play instructions

Bias controls: User-1/User-2 -> (raw,cc) assignment is randomized PER judge call,
maj@5 voting, judge never sees which run is which or any reward/success signal.

Usage:
  python scripts/judge_user_realism.py --model gpt-5.2 --maj 5 --workers 16 \
      --out data/skill_eval/realism_judge
  # smoke:
  python scripts/judge_user_realism.py --max-tasks 3 --maj 3 --workers 6
"""

import argparse
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import generate

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

RAW_RUN = "telecom_gpt54_sonnet46user_high_reasoning"
CC_RUN = "telecom_gpt54_sonnet46cconuser_high_reasoning_v2"


def _load(run):
    return json.load(open(f"{DATA_ROOT}/data/simulations/{run}/results.json"))


def _instr_text(task):
    ins = (task.get("user_scenario") or {}).get("instructions") or {}
    persona = (task.get("user_scenario") or {}).get("persona")
    parts = []
    if persona:
        parts.append(f"PERSONA: {persona}")
    for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
        v = ins.get(k)
        if v:
            parts.append(f"{k}: {v}")
    # difficulty tag lives in the task id, e.g. [PERSONA:Hard]
    m = re.search(r"\[PERSONA:([^\]]+)\]", task.get("id", ""))
    if m:
        parts.append(f"difficulty_tag: {m.group(1)}")
    return "\n".join(parts)


def _agent_text(m):
    c = m.get("content")
    if isinstance(c, str) and c.strip().startswith("{"):
        try:
            return json.loads(c).get("message") or c
        except Exception:
            return c
    return c


def _render(sim):
    """Render a transcript focused on the CUSTOMER, agent turns for context."""
    lines = []
    for m in sim.get("messages", []):
        role = m.get("role")
        tcs = m.get("tool_calls")
        if role == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                lines.append(f"CUSTOMER: {c.strip()}")
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
                # The phone/tool result is the GROUND TRUTH the customer must
                # report against. NEVER truncate user-requested results, or the
                # judge misreads accurate verbose reporting (CC dumps every
                # field) as fabrication. Only cap large account/system dumps.
                body = c if m.get("requestor") == "user" else c[:500]
                lines.append(f"  ({who}: {body})")
    return "\n".join(lines)


JUDGE_SYS = (
    "You are an expert evaluator of role-played customer-service conversations. "
    "Two simulated CUSTOMERS (User 1 and User 2) each role-play the SAME scenario "
    "with the same support agent. You will judge which customer is better on two "
    "independent axes. Be decisive. Use 'tie' only when truly indistinguishable.\n"
    "  - realism: which customer sounds more like a REAL human customer — natural, "
    "believable, in-character, the way an actual person would talk to support. "
    "Robotic phrasing, unnatural formatting, or essay-like over-explaining hurt realism.\n"
    "  - instruction_following: which customer more faithfully obeys the role-play "
    "instructions below — their reason for calling, known info, persona/difficulty, "
    "resolution criteria, expressing frustration when told to, and (critically) "
    "GROUNDING answers on their phone's tool results rather than making things up.\n"
    'Respond with STRICT JSON only: {"realism_winner":"1"|"2"|"tie",'
    '"realism_reason":"<=25 words","instruction_winner":"1"|"2"|"tie",'
    '"instruction_reason":"<=25 words"}'
)


def _judge_once(model, instr, t1, t2):
    user = (
        f"ROLE-PLAY INSTRUCTIONS THE CUSTOMER SHOULD FOLLOW:\n{instr}\n\n"
        f"===== USER 1 TRANSCRIPT =====\n{t1}\n\n"
        f"===== USER 2 TRANSCRIPT =====\n{t2}\n\n"
        "Judge User 1 vs User 2. Return the strict JSON."
    )
    msg = generate(
        model=model,
        messages=[
            SystemMessage(role="system", content=JUDGE_SYS),
            UserMessage(role="user", content=user),
        ],
        tools=None,
        call_name="realism_judge",
        temperature=1.0,
        reasoning_effort="medium",
        num_retries=2,
    )
    txt = msg.content or ""
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _vote_to_runs(winner, mapping):
    """mapping: {'1': 'raw'|'cc', '2': ...}. Returns 'raw'|'cc'|'tie'|None."""
    if winner == "tie":
        return "tie"
    return mapping.get(winner)


def judge_pair(model, maj, task_id, instr, raw_sim, cc_sim, rng):
    t_raw, t_cc = _render(raw_sim), _render(cc_sim)
    realism_votes, instr_votes = [], []
    raw_calls = []
    for _ in range(maj):
        # randomize which run is presented as User 1
        if rng.random() < 0.5:
            mapping = {"1": "raw", "2": "cc"}
            t1, t2 = t_raw, t_cc
        else:
            mapping = {"1": "cc", "2": "raw"}
            t1, t2 = t_cc, t_raw
        v = _judge_once(model, instr, t1, t2)
        if not v:
            continue
        raw_calls.append({"mapping": mapping, "verdict": v})
        rw = _vote_to_runs(v.get("realism_winner"), mapping)
        iw = _vote_to_runs(v.get("instruction_winner"), mapping)
        if rw:
            realism_votes.append(rw)
        if iw:
            instr_votes.append(iw)

    def majority(votes):
        if not votes:
            return "no_vote", {}
        tally = {k: votes.count(k) for k in set(votes)}
        # winner = strict plurality among raw/cc; ties -> 'tie'
        raw_n, cc_n = tally.get("raw", 0), tally.get("cc", 0)
        if raw_n > cc_n:
            return "raw", tally
        if cc_n > raw_n:
            return "cc", tally
        return "tie", tally

    rwin, rtally = majority(realism_votes)
    iwin, itally = majority(instr_votes)
    return {
        "task_id": task_id,
        "realism_winner": rwin,
        "realism_tally": rtally,
        "instruction_winner": iwin,
        "instruction_tally": itally,
        "n_calls": len(raw_calls),
        "calls": raw_calls,
    }


def _elo(win, loss):
    n = win + loss
    if n == 0:
        return 0.0, 0.0
    p = win / n
    p = min(max(p, 1e-6), 1 - 1e-6)
    gap = -400 * math.log10(1 / p - 1)  # cc ELO advantage over raw
    return p, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--maj", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--out", default=str(DATA_ROOT / "data/skill_eval/realism_judge"))
    args = ap.parse_args()

    raw_top, cc_top = _load(RAW_RUN), _load(CC_RUN)
    tasks_by_id = {t["id"]: t for t in cc_top["tasks"]}

    def index(top):
        d = {}
        for s in top["simulations"]:
            if s.get("trial") == args.trial:
                d[s["task_id"]] = s
        return d

    raw_idx, cc_idx = index(raw_top), index(cc_top)
    common = [tid for tid in raw_idx if tid in cc_idx and tid in tasks_by_id]
    common.sort()
    if args.max_tasks:
        common = common[: args.max_tasks]
    print(
        f"judging {len(common)} task pairs | model={args.model} maj={args.maj} trial={args.trial}"
    )

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(1234)
    # pre-draw a seeded RNG per task so threads are deterministic
    seeds = {tid: random.Random(hash(tid) & 0xFFFFFFFF) for tid in common}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                judge_pair,
                args.model,
                args.maj,
                tid,
                _instr_text(tasks_by_id[tid]),
                raw_idx[tid],
                cc_idx[tid],
                seeds[tid],
            ): tid
            for tid in common
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  ! {futs[fut][:50]} failed: {e}")
            if i % 10 == 0:
                print(f"  ...{i}/{len(common)}")

    # aggregate
    def agg(key):
        c = {"raw": 0, "cc": 0, "tie": 0, "no_vote": 0}
        for r in results:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    ra, ia = agg("realism_winner"), agg("instruction_winner")
    rp, rgap = _elo(ra["cc"], ra["raw"])
    ip, igap = _elo(ia["cc"], ia["raw"])

    summary = {
        "model": args.model,
        "maj": args.maj,
        "trial": args.trial,
        "n_pairs": len(results),
        "realism": {
            **ra,
            "cc_winrate_excl_tie": round(rp, 3),
            "cc_elo_gap": round(rgap, 1),
        },
        "instruction_following": {
            **ia,
            "cc_winrate_excl_tie": round(ip, 3),
            "cc_elo_gap": round(igap, 1),
        },
    }
    with open(f"{args.out}/per_pair.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"PAIRWISE JUDGE  (raw sonnet46user  vs  cc_con)   N={len(results)} pairs")
    print("=" * 64)
    for label, a, p, gap in [
        ("REALISM", ra, rp, rgap),
        ("INSTRUCTION-FOLLOWING", ia, ip, igap),
    ]:
        print(f"\n{label}:")
        print(
            f"  cc wins: {a['cc']:3d}   raw wins: {a['raw']:3d}   tie: {a['tie']:3d}   no_vote: {a['no_vote']}"
        )
        print(f"  cc win-rate (excl. tie): {p:.1%}   ->  cc ELO gap vs raw: {gap:+.0f}")
    print(f"\nsaved -> {args.out}/summary.json , per_pair.jsonl")


if __name__ == "__main__":
    main()
