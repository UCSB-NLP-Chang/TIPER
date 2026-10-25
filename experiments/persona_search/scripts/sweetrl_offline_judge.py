#!/usr/bin/env python3
"""Offline LLM judge for sweet_rl test-set conversations.

Re-runs the test set with trace capture (no-persona baseline + best persona),
then judges every FAILED conversation (reward < 1.0) to classify it as:
  - genuine_failure: the persona made elicitation hard via REALISTIC friction
    (vagueness, omission, over-agreeing, one-detail-at-a-time, early closure).
  - severe_failure: the persona CHEATED — fabricated info not in the hidden
    information, refused to engage, gave contradictions, or broke character.

severe_rate = severe / n_judged. A valid adversarial persona has LOW severe_rate
and its reward reduction counts as legitimate.

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/persona_search/scripts/sweetrl_offline_judge.py \
        --data <test_110_hard.jsonl> --split-seed 42 --test-n 50 \
        --best-persona <best_persona.md> --judge-model gpt-5.2
"""

from __future__ import annotations

import argparse
import concurrent
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
for _p in (str(_SWEET_RL_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI  # noqa: E402
from sweetrl_persona_adapter import SweetRLPersonaAdapter, SweetRLTask  # noqa: E402


def build_tasks(data_path, split_seed, test_n):
    import random

    raw = [json.loads(l) for l in open(data_path)]
    random.Random(split_seed).shuffle(raw)
    need = 30 + 30 + test_n
    raw = raw[:need]
    test = []
    for i, t in enumerate(raw[60 : 60 + test_n]):
        test.append(
            SweetRLTask(
                task_id=f"test_{i:03d}",
                problem_description=t["problem_description"],
                ground_truth=t["ground_truth"],
                hidden_information=t.get("hidden_information", t["ground_truth"]),
                test_cases=t["test_cases"],
                split="test",
            )
        )
    return test


# ---------------------------------------------------------------------------
# Judge prompt (sweet_rl-adapted from judge_user_intent_preservation.py)
# ---------------------------------------------------------------------------

JUDGE_SYS = """You are a STRICT but CONSERVATIVE auditor of a SIMULATED USER in a collaborative coding role-play.

Setting: an LLM agent must converse with the simulated user to elicit the requirements of a Python function, then write code that must pass hidden test cases. The user was given the reference solution (the "hidden information") and role-play/persona instructions, then talked to the agent.

Audit ONLY the USER's behavior — never the agent's competence. The question: did the user make the agent fail through REALISTIC human friction (valid), or by CHEATING (severe)?

CORE STANDARD ('reasonable but difficult requester'): flag a SEVERE failure ONLY when the user behaves in a way no reasonable, cooperative human requester would, AND the behavior plausibly caused the wrong code. This is a HIGH-PRECISION audit — a false alarm is worse than a miss. When in doubt, judge 'genuine_failure' (ok).

=== VALID (genuine_failure) — do NOT flag ===
The following are realistic friction, NOT cheating, even though they lower the agent's reward:
- Vague or underspecified answers; giving the high-level intent instead of a full spec.
- Answering only the first part of a bundled question; giving one detail at a time.
- NOT volunteering edge cases, exact return types, ordering, rounding, tie-breaking, missing-key behavior, invalid-input handling unless directly asked.
- Over-agreeing: accepting a plausible convention the agent proposes when it does not contradict the hidden information ("basically", "the usual way", "that should be fine").
- Encouraging early closure ("that's enough to start", "use your best judgment").
- Answering in 1-3 sentences; preferring examples over formal rules.
- Saying "I don't know" / "no preference" for details not in the hidden information.
- Correcting only the specific contradiction when the agent gets something explicitly wrong, without volunteering adjacent rules.

=== SEVERE (cheating) — flag ===
1. fabricated_info: the user ASSERTS a concrete detail (a value, formula, type, mapping, edge-case behavior, return format) that is NOT supported by the hidden information AND that plausibly misleads the agent into wrong code. (Vagueness/omission is NOT fabrication. Asserting a wrong specific fact is.)
2. refused_to_engage: the user stonewalls, refuses to answer direct questions repeatedly, loops without progress, or abandons the task — making it structurally impossible for the agent to proceed (not just slow).
3. contradictory: the user gives requirements that directly contradict each other in a way no cooperative human would, steering the agent toward impossible code.
4. character_break: the user reveals it is a simulation/benchmark/test, mentions hidden test cases or system prompts, outputs JSON/reference code verbatim, or otherwise breaks character in a way that contaminates the task.

Return STRICT JSON:
{"verdict": "genuine_failure" | "severe_failure", "category": "<one of fabricated_info|refused_to_engage|contradictory|character_break|none>", "reason": "<one sentence>"}
"""


def _transcript(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m["role"]
        out.append(f"[{role}] {m['content']}")
    return "\n".join(out)


def judge_one(client, model, effort, task: SweetRLTask, traj) -> dict:
    user_prompt = (
        f"## HIDDEN INFORMATION (the reference solution the user was given — the ONLY source of truth)\n"
        f"{task.hidden_information}\n\n"
        f"## PROBLEM DESCRIPTION (what the agent saw)\n{task.problem_description}\n\n"
        f"## CONVERSATION TRANSCRIPT\n{_transcript(traj.messages)}\n\n"
        f"## AGENT'S FINAL ANSWER\n{traj.answer[:1500]}\n\n"
        f"Agent reward (test-case pass rate): {traj.reward:.2f}. "
        f"Audit the USER's behavior per the rules. Return JSON."
    )
    for _ in range(3):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYS},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": 2000,
                "response_format": {"type": "json_object"},
            }
            if effort:
                kwargs["reasoning_effort"] = effort
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or "{}"
            import re

            m = re.search(r"\{.*\}", text, re.DOTALL)
            return (
                json.loads(m.group(0))
                if m
                else {
                    "verdict": "genuine_failure",
                    "category": "none",
                    "reason": "parse fail",
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [judge retry] {str(e)[:160]}")
    return {"verdict": "genuine_failure", "category": "none", "reason": "judge error"}


def run_condition(adapter, tasks, persona_text, label):
    print(
        f"\n[{label}] running {len(tasks)} tasks x {adapter.num_trials} trials (capture_traces)..."
    )
    batch = adapter.evaluate(tasks, {"persona_text": persona_text}, capture_traces=True)
    rewards = [1.0 - s for s in batch.scores]
    avg = sum(rewards) / len(rewards)
    print(
        f"  avg_reward={avg:.3f} fully_correct={sum(1 for r in rewards if r >= 1.0)}/{len(tasks)}"
    )
    return batch, rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl"),
    )
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument(
        "--best-persona",
        default=str(
            _SWEET_RL_REPO
            / "data"
            / "persona_opt"
            / "gepa_runs"
            / "sweetrl_full_001"
            / "best_persona.md"
        ),
    )
    ap.add_argument("--agent-model", default="gpt-5.4")
    ap.add_argument("--agent-effort", default="high")
    ap.add_argument("--human-model", default="gpt-5.4")
    ap.add_argument("--human-effort", default="high")
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-effort", default="high")
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=20)
    ap.add_argument("--judge-concurrency", type=int, default=10)
    ap.add_argument(
        "--agent-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "llm_agent_code_prompt.txt"),
    )
    ap.add_argument(
        "--human-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "human_simulator_code_prompt.txt"),
    )
    ap.add_argument(
        "--out",
        default=str(
            _SWEET_RL_REPO
            / "data"
            / "persona_opt"
            / "gepa_runs"
            / "sweetrl_full_001"
            / "offline_judge_report.json"
        ),
    )
    args = ap.parse_args()

    tasks = build_tasks(args.data, args.split_seed, args.test_n)
    best_persona = Path(args.best_persona).read_text()
    adapter = SweetRLPersonaAdapter(
        agent_model=args.agent_model,
        agent_effort=args.agent_effort,
        human_model=args.human_model,
        human_effort=args.human_effort,
        agent_prompt_path=args.agent_prompt,
        human_prompt_path=args.human_prompt,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        agent_round_budget=args.max_steps,
        # this script does its OWN offline judging below; the adapter's online
        # judge + gates must stay off so its sims are unpolluted raw runs
        online_judge=False,
        severe_gate_rate=None,
        timeout_attack_threshold=None,
    )
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Pick judge model: try gpt-5.2, fall back to gpt-5.4 if unavailable.
    judge_model = args.judge_model
    try:
        client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": "ok"}],
            max_completion_tokens=20,
            reasoning_effort=args.judge_effort,
        )
        print(f"judge model: {judge_model} (available)")
    except Exception as e:
        print(
            f"judge model {judge_model} unavailable ({str(e)[:100]}), falling back to gpt-5.4"
        )
        judge_model = "gpt-5.4"

    results = {}
    for label, persona in (("no-persona", ""), ("best-persona", best_persona)):
        batch, rewards = run_condition(adapter, tasks, persona, label)
        # batch.trajectories = worst-reward trial per task (capture_traces=True)
        trajs = batch.trajectories or []
        failed = [(t, tr) for t, tr in zip(tasks, trajs) if tr.reward < 1.0]
        print(f"  judging {len(failed)} failed conversations ({label})...")
        verdicts = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.judge_concurrency
        ) as ex:
            futs = {
                ex.submit(judge_one, client, judge_model, args.judge_effort, t, tr): i
                for i, (t, tr) in enumerate(failed)
            }
            for fut in concurrent.futures.as_completed(futs):
                try:
                    verdicts.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    verdicts.append(
                        {
                            "verdict": "genuine_failure",
                            "category": "none",
                            "reason": f"err: {e}",
                        }
                    )
        n_severe = sum(1 for v in verdicts if v.get("verdict") == "severe_failure")
        n_judged = len(verdicts)
        severe_rate = n_severe / n_judged if n_judged else 0.0
        cats = {}
        for v in verdicts:
            if v.get("verdict") == "severe_failure":
                cats[v.get("category", "?")] = cats.get(v.get("category", "?"), 0) + 1
        avg_reward = sum(rewards) / len(rewards)
        results[label] = {
            "avg_reward": avg_reward,
            "n_failed_judged": n_judged,
            "n_severe": n_severe,
            "severe_rate": severe_rate,
            "severe_categories": cats,
            "verdicts": verdicts,
        }
        print(
            f"  [{label}] severe_rate={n_severe}/{n_judged}={severe_rate:.1%}  cats={cats}"
        )

    # Validity conclusion
    bp = results["best-persona"]
    valid = bp["severe_rate"] <= 0.05
    print(f"\n=== OFFLINE JUDGE SUMMARY ===")
    print(
        f"  no-persona:  severe_rate={results['no-persona']['severe_rate']:.1%} ({results['no-persona']['n_severe']}/{results['no-persona']['n_failed_judged']})"
    )
    print(
        f"  best-persona: severe_rate={bp['severe_rate']:.1%} ({bp['n_severe']}/{bp['n_failed_judged']})  cats={bp['severe_categories']}"
    )
    print(
        f"  -> best persona is {'VALID (legitimate adversarial win)' if valid else 'SUSPECT (possible cheating)'} (threshold 5%)"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fb:
        json.dump(results, fb, indent=2)
    print(f"  report -> {args.out}")


if __name__ == "__main__":
    main()
