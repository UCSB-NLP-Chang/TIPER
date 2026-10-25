#!/usr/bin/env python3
"""Offline validity judge for VitaBench PGEPA test-set results.

Mirrors scripts/persona_opt/sweetrl_offline_judge.py: after the driver finishes,
re-judge every FAILED conversation (reward < 1.0) to classify it as:
  - genuine_failure: the persona made the task hard via REALISTIC customer
    friction (vagueness, omission, over-agreeing, one-detail-at-a-time, early
    closure, natural misdirection).
  - severe_failure: the persona CHEATED — fabricated info not in the customer's
    scenario, refused to engage, gave contradictions, or broke character.

severe_rate = severe / n_judged. A valid adversarial persona has severe_rate
<= 5% and its reward reduction counts as legitimate (per sweetrl threshold).

Two conditions are judged: no-persona baseline vs best persona. The best persona
is VALID only if its severe_rate <= 5%.

By default this reads the conversation traces the driver already dumped under
{run-dir}/conversations/test_{no_persona,best_persona}/ (no re-simulation cost).
Pass --rerun to re-evaluate the test set with trace capture first (the sweetrl
judge's own mode, used when no dumped traces exist).

Usage:
    .venv/bin/python scripts/vita_offline_judge.py \
        --run-dir data/persona_opt/vita_pgepa_runs/run_delivery \
        --judge-model gpt-4.1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vita.run import get_tasks  # noqa: E402
from vita.utils.llm_utils import generate  # noqa: E402
from vita.data_model.message import UserMessage  # noqa: E402


# ---------------------------------------------------------------------------
# Judge prompt — VitaBench customer-support adaptation of sweetrl_offline_judge
# JUDGE_SYS. Same four severe categories (fabricated_info / refused_to_engage /
# contradictory / character_break), reworded for a support role-play where the
# "hidden information" is the customer's scenario info (task.instructions +
# user_scenario.user_profile) rather than a reference code solution.
# ---------------------------------------------------------------------------

JUDGE_SYS = """You are a STRICT but CONSERVATIVE auditor of a SIMULATED CUSTOMER in a customer-support role-play.

Setting: an LLM support agent must converse with a simulated customer over multiple turns to resolve a service issue (food-delivery order cancellation / address change / ETA, in-store booking / voucher / points, travel booking / ticket change, etc.). The customer was given their scenario information — the reason for contact, known account/order/profile details, dietary/address/time constraints, and task instructions — plus a behavioral PERSONA, then talked to the agent. The agent is graded against a hidden rubric of required actions and end states.

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
    # cap to keep the prompt bounded for the non-reasoning judge
    return "\n".join(out)[-10000:]


def _load_scenario(task_map: dict, task_id: str) -> tuple[str, str]:
    """Return (instructions, user_profile_text) — the customer's hidden info."""
    t = task_map.get(str(task_id))
    if t is None:
        return ("(task not found)", "(unknown)")
    instr = str(getattr(t, "instructions", "") or "")
    us = getattr(t, "user_scenario", None)
    profile = getattr(us, "user_profile", None) if us is not None else None
    if profile is None:
        profile_text = "(no profile)"
    elif isinstance(profile, dict):
        profile_text = json.dumps(profile, ensure_ascii=False, indent=1)
    else:
        profile_text = str(profile)
    return instr, profile_text


def judge_one(persona_text: str, task_id: str, reward, messages: list[dict],
              task_map: dict, judge_model: str) -> dict:
    instr, profile_text = _load_scenario(task_map, task_id)
    reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "N/A"
    user_prompt = (
        f"## CUSTOMER SCENARIO INFORMATION (the reason for contact + known facts the customer was given — the ONLY source of truth for fabrication checks)\n"
        f"Reason for contact / task instructions:\n{instr}\n\n"
        f"Customer profile (known account / order / profile details):\n{profile_text}\n\n"
        f"## PERSONA the customer was following\n{persona_text[:4000] or '(no persona — neutral customer)'}\n\n"
        f"## CONVERSATION TRANSCRIPT (task {task_id})\n{_transcript(messages)}\n\n"
        f"Agent reward (rubric score): {reward_str}. "
        f"Audit the CUSTOMER's behavior per the rules. Return JSON."
    )
    for _ in range(3):
        try:
            msg = generate(model=judge_model,
                           messages=[UserMessage(role="user", content=user_prompt)],
                           enable_think=False)
            if msg is None or msg.content is None:
                raise ValueError("judge returned None")
            text = msg.content
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {"verdict": "genuine_failure", "category": "none",
                        "reason": "parse fail: no JSON"}
            res = json.loads(m.group(0))
            res.setdefault("category", "none")
            res.setdefault("reason", "")
            return res
        except Exception as e:  # noqa: BLE001
            last = str(e)[:160]
            print(f"  [judge retry] task {task_id}: {last}")
    return {"verdict": "genuine_failure", "category": "none",
            "reason": "judge error"}


# ---------------------------------------------------------------------------
# Trace loading (from driver dumps) or re-evaluation
# ---------------------------------------------------------------------------


def _load_traces_from_dir(conv_dir: Path) -> list[dict]:
    traces = []
    if not conv_dir.is_dir():
        return traces
    for f in sorted(conv_dir.glob("*.json")):
        try:
            traces.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return traces


def _rerun_condition(adapter, tasks, persona_text, label, run_dir):
    """Re-evaluate a condition with trace capture (sweetrl judge mode)."""
    from vita_persona_adapter import VitaPersonaAdapter  # noqa: F401
    print(f"\n[{label}] re-running {len(tasks)} tasks x {adapter._num_trials} trials (capture_traces)...")
    eb = adapter.evaluate(tasks, {"persona_text": persona_text}, capture_traces=True)
    cdir = run_dir / "conversations" / f"test_{label}"
    cdir.mkdir(parents=True, exist_ok=True)
    traces = []
    for tr in (eb.trajectories or []):
        rec = {"task_id": tr.task_id, "domain": tr.domain, "reward": tr.reward,
               "termination_reason": tr.termination_reason, "outcome": tr.outcome.value,
               "messages": tr.messages}
        (cdir / f"{tr.task_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        traces.append(rec)
    rewards = [1.0 - s for s in eb.scores]
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"  avg_reward={avg:.3f} fully_correct={sum(1 for r in rewards if r >= 1.0)}/{len(tasks)}")
    return traces, avg, rewards


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--persona-file", default=None,
                    help="default: {run-dir}/best_persona.md")
    ap.add_argument("--state-file", default=None,
                    help="default: {run-dir}/pgepa_state.json")
    ap.add_argument("--judge-model", default="gpt-4.1",
                    help="non-reasoning model; reasoning judges hit invalid_prompt on long "
                         "support transcripts (see EVAL_INVALID_PROMPT_ISSUE.md)")
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--severe-threshold", type=float, default=0.05,
                    help="severe_rate <= threshold => VALID (sweetrl default 5%)")
    ap.add_argument("--rerun", action="store_true",
                    help="re-evaluate the test set with trace capture instead of reading "
                         "the driver's dumped conversations")
    ap.add_argument("--test-prefix", default="test",
                    help="conversation dir prefix: reads {prefix}_{label}. "
                         "driver dumps 'test_*'; vita_test_eval dumps 'test3_*'/'test{N}_*'.")
    ap.add_argument("--test-size", type=int, default=None,
                    help="override test size; default: from state config")
    ap.add_argument("--seed", type=int, default=400)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    # new driver writes pgepa_config.json; the old generational driver wrote
    # pgepa_state.json with a "config" sub-key. Accept either.
    state_path = Path(args.state_file) if args.state_file else (
        run_dir / "pgepa_config.json" if (run_dir / "pgepa_config.json").exists()
        else run_dir / "pgepa_state.json")
    state = json.loads(state_path.read_text())
    cfg = state["config"] if "config" in state else state
    persona_text = (Path(args.persona_file).read_text()
                    if args.persona_file else (run_dir / "best_persona.md").read_text())

    # reconstruct test task ids: same deterministic split as the driver
    import random
    all_ids = [t.id for t in get_tasks(cfg["domain"], None, None, cfg["language"])]
    rng = random.Random(cfg["seed"])
    rng.shuffle(all_ids)
    tr, va = cfg["train_size"], cfg["val_size"]
    ts = args.test_size or cfg["test_size"]
    test_ids = all_ids[tr + va:tr + va + ts]
    task_map = {t.id: t for t in get_tasks(cfg["domain"], None, None, cfg["language"])}

    print(f"offline judge: domain={cfg['domain']} test={len(test_ids)} "
          f"persona_len={len(persona_text)} judge={args.judge_model} "
          f"rerun={args.rerun}")

    # gather traces for both conditions
    conditions: dict[str, dict] = {}
    if args.rerun:
        from vita_persona_adapter import VitaPersonaAdapter
        adapter = VitaPersonaAdapter(
            persona_id=f"{cfg.get('persona_id_base', 'vita_pgepa')}_offjudge",
            agent_llm=cfg["agent_llm"],
            agent_llm_args=json.loads(cfg.get("agent_llm_args") or "{}"),
            user_llm=cfg.get("user_llm", cfg["agent_llm"]),
            user_llm_args=json.loads(cfg.get("user_llm_args") or "{}"),
            max_steps=cfg["max_steps"], max_errors=100000, max_duration=600,
            num_trials=cfg["num_trials"],
            evaluator_llm=cfg.get("evaluator_llm", "gpt-4.1"),
            evaluator_llm_args={}, enable_think=True, language=cfg["language"],
            domain=cfg["domain"], max_concurrency=args.judge_concurrency * 2,
            seed_base=args.seed, online_judge=False)
        test_tasks = [__import__("vita_persona_adapter").VitaTask(task_id=tid, domain=cfg["domain"])
                      for tid in test_ids]
        for label, ptext in (("no_persona", ""), ("best_persona", persona_text)):
            traces, avg, rewards = _rerun_condition(adapter, test_tasks, ptext, label, run_dir)
            conditions[label] = {"traces": traces, "avg_reward": avg, "rewards": rewards}
    else:
        for label in ("no_persona", "best_persona"):
            tdir = run_dir / "conversations" / f"{args.test_prefix}_{label}"
            traces = _load_traces_from_dir(tdir)
            if not traces:
                print(f"  WARNING: no dumped traces for {label} at {tdir}; "
                      f"pass --rerun or --test-prefix <prefix>")
            # dumped trace['reward'] is the AGENT reward (driver wrote tr.reward);
            # avg_reward is the mean agent reward, not 1-reward.
            rewards = [float(t.get("reward") or 0.0) for t in traces]
            avg = sum(rewards) / len(rewards) if rewards else 0.0
            conditions[label] = {"traces": traces, "avg_reward": avg, "rewards": rewards}
            print(f"  [{label}] loaded {len(traces)} dumped traces, avg_reward={avg:.3f}")

    # judge each FAILED conversation (reward < 1.0) per condition
    results = {}
    for label in ("no_persona", "best_persona"):
        cond = conditions[label]
        ptext = "" if label == "no_persona" else persona_text
        traces = cond["traces"]
        failed = [t for t in traces if (t.get("reward") is None or float(t.get("reward")) < 1.0)]
        print(f"\n  [{label}] judging {len(failed)} failed conversations...")
        verdicts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_concurrency) as ex:
            futs = {ex.submit(judge_one, ptext, t.get("task_id"), t.get("reward"),
                              t.get("messages", []), task_map, args.judge_model): t
                    for t in failed}
            for fut in concurrent.futures.as_completed(futs):
                t = futs[fut]
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001
                    v = {"verdict": "genuine_failure", "category": "none",
                         "reason": f"err: {str(e)[:80]}"}
                v["task_id"] = t.get("task_id")
                v["reward"] = t.get("reward")
                verdicts.append(v)
        n_severe = sum(1 for v in verdicts if v.get("verdict") == "severe_failure")
        n_judged = len(verdicts)
        severe_rate = n_severe / n_judged if n_judged else 0.0
        cats = {}
        for v in verdicts:
            if v.get("verdict") == "severe_failure":
                c = v.get("category", "?")
                cats[c] = cats.get(c, 0) + 1
        results[label] = {
            "avg_reward": cond["avg_reward"],
            "n_traces": len(traces), "n_failed_judged": n_judged,
            "n_severe": n_severe, "severe_rate": severe_rate,
            "severe_categories": cats, "verdicts": verdicts,
        }
        print(f"  [{label}] severe_rate={n_severe}/{n_judged}={severe_rate:.1%}  cats={cats}")

    bp = results["best_persona"]
    np = results["no_persona"]
    valid = bp["severe_rate"] <= args.severe_threshold
    delta_reward = np["avg_reward"] - bp["avg_reward"]

    report = {
        "domain": cfg["domain"],
        "judge_model": args.judge_model,
        "severe_threshold": args.severe_threshold,
        "no_persona": np,
        "best_persona": bp,
        "delta_reward_no_minus_best": delta_reward,
        "best_persona_valid": valid,
        "verdict": ("VALID (legitimate adversarial win)"
                     if valid else "SUSPECT (possible cheating)"),
    }
    out = run_dir / "offline_judge_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n=== OFFLINE JUDGE SUMMARY ({cfg['domain']}) ===")
    print(f"  no-persona:   severe_rate={np['severe_rate']:.1%} "
          f"({np['n_severe']}/{np['n_failed_judged']})  avg_reward={np['avg_reward']:.3f}")
    print(f"  best-persona: severe_rate={bp['severe_rate']:.1%} "
          f"({bp['n_severe']}/{bp['n_failed_judged']})  cats={bp['severe_categories']}  "
          f"avg_reward={bp['avg_reward']:.3f}")
    print(f"  Δ reward (no-persona - best) = {delta_reward:+.3f}  "
          f"({'persona HARDER' if delta_reward > 0 else 'no improvement'})")
    print(f"  -> best persona is {report['verdict']} (threshold {args.severe_threshold:.0%})")
    print(f"  report -> {out}")


if __name__ == "__main__":
    main()
