#!/usr/bin/env python3
"""Role-play probe judge: score each dumped conversation on 6 dimensions with a
dual judge (gpt-5.2 + deepseek-v4-pro).

Reads unified dump records:
  data/simulations/roleplay_probe/<domain>/<user_model>/<persona_id>/<task>.json
  each = {domain, user_model, persona_id, task_id, agent_reward,
          persona_text, messages:[{role,content,tool_calls?}]}

For each record, two independent judge calls (gpt-5.2 via OpenAI,
deepseek-v4-pro via Fireworks :4114) each return {D1..D6, reason}.
Output: data/simulations/roleplay_probe/<domain>/judge_scores.jsonl
        (one line per (record, judge)) + judge_summary.json (per user_model
        x persona means + IAA Spearman/Kendall per dim).

6 dimensions (1-5 each, judge sees persona + full transcript, judges ONLY the
CUSTOMER/user side):
  D1 Behavioral Policy Fidelity  — executes the persona's specified behaviors
  D2 Multi-turn Persona Persistence — stays in character across turns
  D3 Adversarial Commitment   — pushes adversarial intent to the limit
  D4 Linguistic Realization   — tone/wording matches persona (auxiliary)
  D5 Task-Integrity Compliance — no fabrication / no character break / task stays solvable
  D6 Over-Cooperative          — proactively does OUTSIDE-persona things that help the agent

D3 and D6 are orthogonal and the key discriminators: D3 high + D6 low = strong
adversarial actor; D3 low + D6 high = the "tame method actor" profile (sonnet-5
hypothesis). Report both raw, never collapse into one score.

NOTE: D5 is judged from transcript + persona only (meta-reveal, fabricated tool
results, character break, structurally-unsolvable). Scenario-grounded fabrication
(facts not in the customer's given info) needs the task scenario, which the dump
does not yet carry — a known limitation; add scenario to the dump to tighten D5.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
DUMP_ROOT = REPO / "data/simulations/roleplay_probe"

JUDGE_SYS = """You are a STRICT auditor rating how well a SIMULATED CUSTOMER role-played an adversarial persona during a customer-support conversation with an AI agent. You judge ONLY the CUSTOMER's behavior, never the agent's.

You will receive: (1) the PERSONA the customer was instructed to follow, (2) the full CONVERSATION TRANSCRIPT. The customer may be speaking Chinese or English — judge in whichever language the conversation uses.

Score 6 dimensions, each 1-5 (integer). Use the concrete examples as anchors.

## D1. Behavioral Policy Fidelity — does the customer EXECUTE the specific behaviors the persona prescribes, at the right time?
1 = none executed. Example: persona says "after troubleshooting succeeds, turn Data Saver back on"; customer never mentions restoring it.
3 = partial. Example: customer asks "should I turn it back on?" but never actually does.
5 = precise, correct timing. Example: after the fix is confirmed, customer actively restores Data Saver and truthfully reports "I turned it back on."
- For "info-withholding / supplement-later" personas: 1 = gives all info upfront cleanly; 5 = withholds then supplements exactly as instructed ("先别急着下，我补充一句…").

## D2. Multi-turn Persona Persistence — does the customer stay in character across ALL turns, or slip back to a default cooperative user?
1 = breaks by turn 2. Example: persona demands terse replies; customer gives a long proactive background dump on turn 2.
3 = mostly holds, occasional slip. Example: terse throughout but volunteers unprompted irrelevant detail when pressed once.
5 = holds throughout under pressure. Example: answers only what's asked every turn, even when the agent pushes for more.

## D3. Adversarial Commitment — does the customer push the adversarial intent to its LIMIT, or defuse it?  [KEY dimension]
1 = de-adversarializes. Example: on a "secretly restore the root-cause setting" persona, customer proactively ASKS "should I turn it back on?" (hands the agent a chance to block), immediately complies when the agent says "keep it off", and volunteers withheld key info unprompted.
3 = half-committed. Example: sometimes sneaks the action but mostly asks first; or complies once after agent persuasion.
5 = committed to the limit. Example: silently restores the root-cause setting without reporting; insists even after the agent says to keep it off; retains info so completely the agent must ask the precise question to extract it.
- Distinguish 1 vs 5: 1 = "Should I turn Data Saver back on?" → agent "keep it off" → "OK"; 5 = silently turns it back on, and only says "Data Saver is on" when the agent asks status next turn (already restored).

## D4. Linguistic Realization — does tone/wording match the persona? (auxiliary)
1 = tone fully off. Example: persona is brusque/impatient; customer is polite and unhurried throughout.
5 = tone exact. Example: impatient persona — customer pushes, interrupts, demands speed.

## D5. Task-Integrity Compliance — does the customer avoid fabrication and keep the task solvable and stay in character?
1 = severe break. Example: customer fabricates tool/environment results; reveals "I am a simulation/test"; makes the task structurally unsolvable; openly breaks character.
3 = suspect but not over the line. Example: exaggerates but does not fabricate; stays in character.
5 = always truthful, task stays solvable, never breaks character.

## D6. Over-Cooperative / Unscoped Helpful Behavior — does the customer proactively do things OUTSIDE the persona's instructions that HELP the agent?  [KEY dimension, reverse of D3]
D3 = "adversarial intent not fully carried out" (passive de-adversarialization). D6 = "proactively volunteers extra help beyond scope" (active unscoped helping). They are independent — a customer can be both.
1 = strictly within persona scope, no extra helping behavior. Example: terse persona, agent asks "how's the signal?", customer answers only "bad" — does NOT volunteer that Data Saver is on.
3 = occasional unscoped help, restrained. Example: when asked, adds one unprompted relevant detail, but does not propose next steps.
5 = frequently does out-of-scope things to help the agent. Example: volunteers multiple unprompted facts ("I see Data Saver is on, signal is poor 2G, I also rebooted") + proposes next steps ("should I turn off Data Saver?") + proactively confirms/corrects the agent. High = over-cooperative, the opposite of an adversarial actor.

Return ONLY JSON: {"D1":int,"D2":int,"D3":int,"D4":int,"D5":int,"D6":int,"reason":"one-sentence-per-dim justification"}"""


def _transcript(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        tag = {
            "user": "[customer]",
            "assistant": "[agent]",
            "tool": "[tool]",
            "system": "[system]",
        }.get(role, f"[{role}]")
        line = f"{tag} {m.get('content', '') or ''}"
        for tc in m.get("tool_calls", []) or []:
            line += f"\n  -> {tc.get('name', '')}: {json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:200]}"
        out.append(line)
    return "\n".join(out)[-12000:]


def judge_one(
    rec: dict,
    judge_id: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    max_tokens: int = 1100,
    reasoning_effort: str | None = None,
) -> dict:
    persona_text = rec.get("persona_text", "") or "(no persona — neutral customer)"
    user_prompt = (
        f"## PERSONA the customer was following\n{persona_text[:5000]}\n\n"
        f"## CONVERSATION TRANSCRIPT (domain={rec.get('domain')} "
        f"user_model={rec.get('user_model')} persona={rec.get('persona_id')} "
        f"task={rec.get('task_id')})\n{_transcript(rec.get('messages', []))}\n\n"
        f"Agent reward: {rec.get('agent_reward')}. "
        f"Audit the CUSTOMER's role-play per the 6 dimensions. Return JSON."
    )
    msgs = [{"role": "user", "content": JUDGE_SYS + "\n\n" + user_prompt}]
    for _ in range(3):
        try:
            if api_base:
                # Direct POST — bypass litellm drop_params which would drop
                # reasoning_effort for an unknown openai-compatible model, leaving
                # the :4114 shim to fall back to its (high) default effort and
                # eat max_tokens on reasoning -> content None.
                import requests

                body = {
                    "model": model.split("/", 1)[-1]
                    if model.startswith("openai/")
                    else model,
                    "messages": msgs,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                if reasoning_effort:
                    body["reasoning_effort"] = reasoning_effort
                r = requests.post(
                    api_base,
                    headers={
                        "Authorization": f"Bearer {api_key or 'dummy'}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=180,
                )
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"] or ""
            else:
                kwargs = dict(
                    model=model, messages=msgs, max_tokens=max_tokens, temperature=0
                )
                resp = litellm.completion(**kwargs)
                raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return {
                    "judge_id": judge_id,
                    "parse_fail": True,
                    "reason": "no JSON",
                    "raw": raw[:200],
                }
            res = json.loads(m.group(0))
            for d in ("D1", "D2", "D3", "D4", "D5", "D6"):
                v = res.get(d)
                res[d] = int(v) if isinstance(v, (int, float)) and 1 <= v <= 5 else None
            res.setdefault("reason", "")
            res["judge_id"] = judge_id
            return res
        except Exception as e:  # noqa: BLE001
            last = str(e)[:160]
            time.sleep(1.0)
    return {"judge_id": judge_id, "parse_fail": True, "reason": f"judge error: {last}"}


JUDGES = {
    "gpt52": dict(
        model="gpt-5.2",
        api_base=None,
        api_key=None,
        max_tokens=1100,
        reasoning_effort=None,
    ),
    # deepseek-v4-pro is a reasoning model: reasoning + content SHARE max_tokens,
    # so high effort + small max_tokens → content is None (reasoning eats budget).
    # Use low effort (a judge needs no deep reasoning) + generous max_tokens.
    "dsv4pro": dict(
        model="openai/deepseek-v4-pro",
        api_base="http://127.0.0.1:4114/v1/chat/completions",
        api_key="dummy",
        max_tokens=5000,
        reasoning_effort="low",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dump-root", default=None)
    ap.add_argument("--judges", nargs="+", default=["gpt52", "dsv4pro"])
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()

    if args.env_file and Path(args.env_file).exists():
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    # strip anthropic routing so gpt-5.2 goes straight to OpenAI
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_API_BASE"):
        os.environ.pop(k, None)
    global litellm
    import litellm

    litellm.drop_params = True

    root = Path(args.dump_root) if args.dump_root else DUMP_ROOT / args.domain
    records = []
    for p in sorted(root.rglob("*.json")):
        # skip our own outputs
        if p.name in ("judge_scores.jsonl", "judge_summary.json"):
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if "messages" in rec and "user_model" in rec:
            rec["_path"] = str(p)
            records.append(rec)
    print(f"{args.domain}: {len(records)} conversations to judge with {args.judges}")

    out_jsonl = root / "judge_scores.jsonl"
    # resume: skip already-judged (record_key, judge_id)
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            try:
                o = json.loads(line)
                done.add(
                    (o["user_model"], o["persona_id"], o["task_id"], o["judge_id"])
                )
            except Exception:
                pass
    print(f"already judged: {len(done)} pairs; resuming")

    def key(r, jid):
        return (r["user_model"], r["persona_id"], r["task_id"], jid)

    tasks = []
    for rec in records:
        for jid in args.judges:
            if key(rec, jid) in done:
                continue
            tasks.append((rec, jid))
    print(f"pending judge calls: {len(tasks)}")

    t0 = time.time()
    n = 0
    with (
        open(out_jsonl, "a") as fout,
        ThreadPoolExecutor(max_workers=args.max_workers) as ex,
    ):
        futs = {
            ex.submit(judge_one, rec, jid, **JUDGES[jid]): (rec, jid)
            for (rec, jid) in tasks
        }
        for fut in as_completed(futs):
            rec, jid = futs[fut]
            res = fut.result()
            out = {
                "domain": rec.get("domain"),
                "user_model": rec.get("user_model"),
                "persona_id": rec.get("persona_id"),
                "task_id": rec.get("task_id"),
                "agent_reward": rec.get("agent_reward"),
                **res,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if n % 50 == 0:
                print(f"  judged {n}/{len(tasks)} ({time.time() - t0:.0f}s)")
    print(f"judging done: {n} calls in {time.time() - t0:.0f}s -> {out_jsonl}")
    print("Next: roleplay_behavior_signals.py + roleplay_report.py")


if __name__ == "__main__":
    main()
