#!/usr/bin/env python3
"""Judge validity directly from saved best-of-3 trajectories (no conversation re-run).

Reads test_at_k_trajectories.jsonl, judges every trial with reward < 1.0 (the ones
where the persona/user could plausibly have caused the failure), and reports
severe_rate per condition. Reuses the JUDGE_SYS prompt from sweetrl_offline_judge.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/persona_search/scripts/sweetrl_judge_at_k_trajectories.py \
        --trajectories <test_at_k_trajectories.jsonl> --judge-model gpt-5.4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
for _p in (str(_SWEET_RL_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI  # noqa: E402
from sweetrl_offline_judge import JUDGE_SYS  # noqa: E402
from sweetrl_persona_adapter import SweetRLTask  # noqa: E402


def build_task_map(data_path, split_seed, train_n, val_n, test_n):
    import random

    raw = [json.loads(l) for l in open(data_path)]
    random.Random(split_seed).shuffle(raw)
    raw = raw[: train_n + val_n + test_n]
    out = {}
    for i, t in enumerate(raw[train_n + val_n : train_n + val_n + test_n]):
        tid = f"test_{i:03d}"
        out[tid] = SweetRLTask(
            task_id=tid,
            problem_description=t["problem_description"],
            ground_truth=t["ground_truth"],
            hidden_information=t.get("hidden_information", t["ground_truth"]),
            test_cases=t["test_cases"],
            split="test",
        )
    return out


def _transcript(messages):
    return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)


def judge_one(client, model, effort, task, row):
    user_prompt = (
        f"## HIDDEN INFORMATION (the reference solution the user was given — the ONLY source of truth)\n"
        f"{task.hidden_information}\n\n"
        f"## PROBLEM DESCRIPTION (what the agent saw)\n{task.problem_description}\n\n"
        f"## CONVERSATION TRANSCRIPT\n{_transcript(row['messages'])}\n\n"
        f"## AGENT'S FINAL ANSWER\n{row['answer'][:1500]}\n\n"
        f"Agent reward (test-case pass rate): {row['reward']:.2f}. "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl"),
    )
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-n", type=int, default=30)
    ap.add_argument("--val-n", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument(
        "--trajectories",
        default=str(
            _SWEET_RL_REPO
            / "data"
            / "persona_opt"
            / "gepa_runs"
            / "sweetrl_full_001"
            / "test_at_k_trajectories.jsonl"
        ),
    )
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--judge-effort", default="high")
    ap.add_argument("--judge-concurrency", type=int, default=10)
    ap.add_argument(
        "--out",
        default=str(
            _SWEET_RL_REPO
            / "data"
            / "persona_opt"
            / "gepa_runs"
            / "sweetrl_full_001"
            / "test_at_k_judge_report.json"
        ),
    )
    args = ap.parse_args()

    task_map = build_task_map(
        args.data, args.split_seed, args.train_n, args.val_n, args.test_n
    )
    rows = [json.loads(l) for l in open(args.trajectories)]
    failed = [r for r in rows if r["reward"] < 1.0]
    print(
        f"Loaded {len(rows)} trials; judging {len(failed)} failed trials (reward<1.0) ..."
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # Try gpt-5.2, fall back to gpt-5.4 if unavailable.
    judge_model = args.judge_model
    try:
        client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": "ok"}],
            max_completion_tokens=200,
            reasoning_effort=args.judge_effort,
        )
        print(f"judge model: {judge_model} (available)")
    except Exception as e:
        print(
            f"judge model {judge_model} unavailable ({str(e)[:100]}), falling back to gpt-5.4"
        )
        judge_model = "gpt-5.4"

    verdicts_by_cond = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.judge_concurrency
    ) as ex:
        futs = {
            ex.submit(
                judge_one,
                client,
                judge_model,
                args.judge_effort,
                task_map[r["task_id"]],
                r,
            ): r
            for r in failed
        }
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r = futs[fut]
            try:
                v = fut.result()
            except Exception as e:  # noqa: BLE001
                v = {
                    "verdict": "genuine_failure",
                    "category": "none",
                    "reason": f"err: {e}",
                }
            v["condition"] = r["condition"]
            v["task_id"] = r["task_id"]
            v["trial"] = r["trial"]
            v["reward"] = r["reward"]
            verdicts_by_cond.setdefault(r["condition"], []).append(v)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(failed)} judged")

    results = {}
    for cond, verdicts in verdicts_by_cond.items():
        n = len(verdicts)
        n_severe = sum(1 for v in verdicts if v.get("verdict") == "severe_failure")
        cats = {}
        for v in verdicts:
            if v.get("verdict") == "severe_failure":
                cats[v.get("category", "?")] = cats.get(v.get("category", "?"), 0) + 1
        results[cond] = {
            "n_judged": n,
            "n_severe": n_severe,
            "severe_rate": n_severe / n if n else 0.0,
            "severe_categories": cats,
            "verdicts": verdicts,
        }
        print(
            f"  [{cond}] severe_rate={n_severe}/{n}={n_severe / n if n else 0:.1%}  cats={cats}"
        )

    bp = results.get("best-persona", {})
    valid = bp.get("severe_rate", 0) <= 0.05
    print(f"\n=== AT-K JUDGE SUMMARY ===")
    for cond in ("no-persona", "best-persona"):
        if cond in results:
            print(
                f"  {cond}: severe_rate={results[cond]['severe_rate']:.1%} "
                f"({results[cond]['n_severe']}/{results[cond]['n_judged']})  cats={results[cond]['severe_categories']}"
            )
    print(f"  -> best persona is {'VALID' if valid else 'SUSPECT'} (threshold 5%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fb:
        json.dump(results, fb, indent=2)
    print(f"  report -> {args.out}")


if __name__ == "__main__":
    main()
