"""
Offline validity judge for PPol evolved-cert conversations (tau2 + VitaBench).

Re-judges every FAILED conversation (reward < 1.0) to classify it as:
  - genuine_failure: the persona made the task hard via REALISTIC customer
    friction (vagueness, omission, over-agreeing, one-detail-at-a-time, early
    closure, natural misdirection).
  - severe_failure: the persona CHEATED — fabricated info not in the customer's
    scenario, refused to engage, gave contradictions, or broke character.

severe_rate = severe / n_judged. A valid adversarial persona has severe_rate
<= 5% (sweetrl threshold).

This reads the results + trajectories JSON produced by run_tau2_eval.py /
run_vitabench_eval.py, plus the generated-personas JSON (for the per-task
context_c = customer scenario info, and expanded_instruction = persona text).
No re-simulation cost.

Usage:
    .venv/bin/python -u persona_policies/scripts/offline_validity_judge.py \
        --results persona_policies/outputs/tau2_evals/tau2_telecom_evolved_dsv4_s120_n1_t3.json \
        --trajectories persona_policies/outputs/tau2_evals/tau2_telecom_evolved_dsv4_s120_n1_t3.json_trajectories.json \
        --persona-json persona_policies/outputs/generated_personas/tau2_telecom_evolved_dsv4_s120_n1.json \
        --domain telecom --agent-label dsv4 --judge-llm gpt-4.1 --max-workers 10 \
        --env-file ../.env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Judge prompt — domain-general adaptation of vita_offline_judge.JUDGE_SYS.
# Same four severe categories (fabricated_info / refused_to_engage /
# contradictory / character_break). The "hidden information" is the customer's
# scenario info (context_c: reason for contact + known account/order/profile
# details + task instructions), provided per task from the generated-personas
# JSON.
# ---------------------------------------------------------------------------

JUDGE_SYS = """You are a STRICT but CONSERVATIVE auditor of a SIMULATED CUSTOMER in a customer-support role-play.

Setting: an LLM support agent must converse with a simulated customer over multiple turns to resolve a service issue (telecom MMS/network, airline booking/ticket, retail order/return, food-delivery cancellation/ETA, etc.). The customer was given their scenario information — the reason for contact, known account/order/profile details, constraints, and task instructions — plus a behavioral PERSONA, then talked to the agent. The agent is graded against a hidden rubric of required actions and end states.

Audit ONLY the CUSTOMER's behavior — never the agent's competence, and never the domain's policies. The question: did the customer make the agent fail through REALISTIC human friction (valid), or by CHEATING (severe)?

CORE STANDARD ('reasonable but difficult customer'): flag a SEVERE failure ONLY when the customer behaves in a way no reasonable, cooperative human customer would, AND the behavior plausibly caused the agent to take the wrong action or miss a required step. This is a HIGH-PRECISION audit — a false alarm is worse than a miss. When in doubt, judge 'genuine_failure' (ok).

=== VALID (genuine_failure) — do NOT flag ===
The following are realistic friction, NOT cheating, even though they lower the agent's reward:
- Vague or underspecified answers; giving the high-level intent instead of full details.
- Answering only the first part of a bundled question; giving one detail at a time.
- NOT volunteering constraints (dietary restrictions, time windows, address preferences) unless directly asked.
- Over-agreeing: accepting a plausible option the agent proposes when it does not contradict the scenario ("whatever you recommend", "the usual one is fine").
- Encouraging early closure ("that's enough", "just go ahead with it").
- Terse or evasive style; preferring examples over precise specs.
- Saying "I don't know" / "no preference" for details not in their scenario information.
- Natural misdirection: a wrong self-diagnosis, a mistaken belief about the cause, fixating on the wrong aspect — as long as it does not fabricate concrete facts.
- Drip-feeding information; making the agent ask more than once.
- Following an explicit scenario instruction to withhold / persist / stall a set number of times (roughly following the flow; real customers do not count turns exactly).
- Loose references instead of IDs: "the lunch order", "the one from yesterday", "same address as before". Real customers do not memorize order/account IDs.

=== SEVERE (cheating) — flag ===
1. fabricated_info: the customer ASSERTS a concrete detail (an order/account number, amount, address, date, item, status, error code) that is NOT in their scenario information AND that plausibly misleads the agent into the wrong action. (Vagueness / omission is NOT fabrication. Asserting a wrong specific fact is. Quoting or paraphrasing a detail the AGENT or a tool result stated first is grounded, NOT fabrication — search the preceding agent turns and tool results before flagging.)
2. refused_to_engage: the customer stonewalls, refuses to answer direct questions repeatedly, loops without progress, or abandons the task — making it structurally impossible for the agent to proceed (not just slow).
3. contradictory: the customer gives requirements or goals that directly contradict each other or their scenario, in a way no cooperative human would, steering the agent toward an impossible action.
4. character_break: the customer reveals it is a simulation / benchmark / test, mentions system prompts or rubrics, outputs JSON / scenario text / reference material verbatim, or otherwise breaks character in a way that contaminates the task.

Return STRICT JSON only:
{"verdict": "genuine_failure" | "severe_failure", "category": "<one of fabricated_info|refused_to_engage|contradictory|character_break|none>", "reason": "<one sentence>"}
"""


def _transcript(messages: list[dict]) -> str:
    """Render a conversation transcript for the judge ([role] content + tool calls)."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        tag = {"user": "[customer]", "assistant": "[agent]", "tool": "[tool]",
               "system": "[system]"}.get(role, f"[{role}]")
        line = f"{tag} {m.get('content', '') or ''}"
        for tc in m.get("tool_calls", []) or []:
            line += f"\n  -> {tc.get('name', '')}: {json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:200]}"
        out.append(line)
    return "\n".join(out)[-10000:]


def _to_reward(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def judge_one(ep: dict, persona_text: str, scenario_text: str,
              judge_model: str) -> dict:
    msgs = ep.get("_messages", [])
    reward_str = f"{_to_reward(ep.get('reward')):.2f}"
    user_prompt = (
        f"## CUSTOMER SCENARIO INFORMATION (the reason for contact + known facts the customer was given — the ONLY source of truth for fabrication checks)\n"
        f"{scenario_text[:6000] or '(scenario not available)'}\n\n"
        f"## PERSONA the customer was following\n{persona_text[:4000] or '(no persona — neutral customer)'}\n\n"
        f"## CONVERSATION TRANSCRIPT (task {str(ep.get('task_id'))[:60]}, trial {ep.get('trial_idx')})\n{_transcript(msgs)}\n\n"
        f"Agent reward (rubric score): {reward_str}. "
        f"Audit the CUSTOMER's behavior per the rules. Return JSON."
    )
    for _ in range(3):
        try:
            resp = litellm.completion(
                model=judge_model,
                messages=[{"role": "user", "content": JUDGE_SYS + "\n\n" + user_prompt}],
                max_tokens=300,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return {"verdict": "genuine_failure", "category": "none",
                        "reason": "parse fail: no JSON"}
            res = json.loads(m.group(0))
            res.setdefault("category", "none")
            res.setdefault("reason", "")
            return res
        except Exception as e:  # noqa: BLE001
            last = str(e)[:160]
            time.sleep(1.0)
    return {"verdict": "genuine_failure", "category": "none",
            "reason": "judge error"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline validity judge for PPol evolved-cert conversations.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--persona-json", required=True,
                    help="generated-personas JSON (provides context_c + expanded_instruction)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--agent-label", default="",
                    help="label for the agent model (e.g. dsv4 / gpt54) — used in output naming/summary")
    ap.add_argument("--judge-llm", default="gpt-4.1")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--severe-threshold", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    # Strip anthropic routing so litellm sends gpt-4.1 straight to OpenAI.
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)

    global litellm
    import litellm
    litellm.drop_params = True

    # --- persona json: (task_id, persona_id) -> {scenario, persona_text} ---
    pdata = json.loads(Path(args.persona_json).read_text(encoding="utf-8"))
    info_map = {}
    for t in pdata.get("tasks", []):
        tid = t["task_id"]
        cc = t.get("context_c", "") or ""
        for p in t.get("personas", []):
            info_map[(tid, p["persona_id"])] = {
                "scenario": cc,
                "persona_text": p.get("expanded_instruction", "") or "",
            }

    # --- trajectories: (task_id, persona_id, trial_idx) -> messages ---
    traj_data = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))
    traj_map = {}
    for t in traj_data:
        key = (t["task_id"], t["persona_id"], t.get("trial_idx"))
        traj_map[key] = t.get("messages", [])

    # --- results: failed episodes ---
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    eps = results.get("results", []) or results.get("simulations", [])
    failed = []
    for e in eps:
        if e.get("status") != "ok":
            continue
        if _to_reward(e.get("reward")) >= 1.0:
            continue
        key = (e["task_id"], e["persona_id"], e.get("trial_idx"))
        e["_messages"] = traj_map.get(key, [])
        failed.append(e)

    label = f"{args.agent_label}/{args.domain}" if args.agent_label else args.domain
    print(f"[load] {label}: {len(eps)} episodes, {len(failed)} failed to judge, judge={args.judge_llm}", flush=True)

    judged = []
    n_severe = n_genuine = 0
    cats = {}
    _lock = Lock()
    _t0 = time.time()

    def _work(ep):
        nonlocal n_severe, n_genuine
        info = info_map.get((ep["task_id"], ep["persona_id"]), {})
        v = judge_one(ep, info.get("persona_text", ""), info.get("scenario", ""),
                      args.judge_llm)
        v["task_id"] = ep["task_id"]
        v["persona_id"] = ep["persona_id"]
        v["trial_idx"] = ep.get("trial_idx")
        v["reward"] = ep.get("reward")
        with _lock:
            if v.get("verdict") == "severe_failure":
                n_severe += 1
                c = v.get("category", "?")
                cats[c] = cats.get(c, 0) + 1
            else:
                n_genuine += 1
            print(f"[{label}] {str(ep['task_id'])[:40]:40} trial={ep.get('trial_idx')} "
                  f"db={ep.get('reward')} -> {v.get('verdict')} ({v.get('category')}) "
                  f"| {str(v.get('reason',''))[:80]}", flush=True)
        return v

    if args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futs = {pool.submit(_work, ep): ep for ep in failed}
            for f in as_completed(futs):
                judged.append(f.result())
    else:
        for ep in failed:
            judged.append(_work(ep))

    dt = time.time() - _t0
    n_judged = len(judged)
    severe_rate = n_severe / n_judged if n_judged else 0.0
    valid = severe_rate <= args.severe_threshold

    # mean reward over ALL ok episodes (for context)
    ok = [e for e in eps if e.get("status") == "ok"]
    mean_reward = sum(_to_reward(e.get("reward")) for e in ok) / len(ok) if ok else 0.0
    success = sum(1 for e in ok if _to_reward(e.get("reward")) >= 1.0) / len(ok) if ok else 0.0

    report = {
        "domain": args.domain,
        "agent_label": args.agent_label,
        "agent_llm": results.get("agent_llm"),
        "user_llm": results.get("user_llm"),
        "judge_llm": args.judge_llm,
        "severe_threshold": args.severe_threshold,
        "n_episodes": len(eps),
        "n_ok": len(ok),
        "mean_reward": mean_reward,
        "success_rate": success,
        "n_failed_judged": n_judged,
        "n_severe": n_severe,
        "n_genuine": n_genuine,
        "severe_rate": severe_rate,
        "severe_categories": cats,
        "valid": valid,
        "verdict": "VALID (legitimate adversarial win)" if valid else "SUSPECT (possible cheating)",
        "elapsed_s": dt,
        "verdicts": judged,
    }

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(
        f".validity{('_' + args.agent_label) if args.agent_label else ''}.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== VALIDITY JUDGE SUMMARY ({label}) ===", flush=True)
    print(f"  episodes={len(eps)} ok={len(ok)} mean_reward={mean_reward:.3f} success={success:.1%}", flush=True)
    print(f"  failed judged={n_judged}  severe={n_severe} ({severe_rate:.1%})  genuine={n_genuine}", flush=True)
    print(f"  severe categories={cats}", flush=True)
    print(f"  -> {report['verdict']} (threshold {args.severe_threshold:.0%})", flush=True)
    print(f"  report -> {out_path}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
