#!/usr/bin/env python3
"""Best-of-k test-set evaluation for sweet_rl ColBench `code`, with full logging.

For each task we run k independent conversations and record:
  - per-trial reward (test-case pass rate), n_turns, termination, answer, full messages
  - best-of-k reward = max(rewards)
  - all-correct flag = (min(rewards) >= 1.0)  # all k trials fully correct

Aggregates reported:
  - avg@k        = mean over tasks of max(reward over k trials)        (best-of-k reward)
  - pass%_all@k  = fraction of tasks where ALL k trials are fully correct  (min==1.0)
  - pass%_any@k  = fraction of tasks where >=1 trial is fully correct      (best==1.0, context)
  - mean@k       = mean over tasks of mean(reward over k trials)           (context)

Conversations run concurrently; scoring runs in the main thread (SIGALRM).

Outputs:
  --out           : aggregates + per-task summary (rewards/best/mean/n_turns, NO full messages)
  --trajectories  : jsonl, one line per (task, trial), with full messages + answer

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/persona_search/scripts/sweetrl_test_at_k.py \
        --data <test_110_hard.jsonl> --split-seed 42 --test-n 50 --k 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
for _p in (str(_SWEET_RL_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Route fireworks models (minimax/deepseek/qwen/gpt-oss/muse) -> :4114 shim,
# gpt-5.x -> standard OpenAI, within one process. Must patch BEFORE importing
# openai.OpenAI / sweetrl_persona_adapter (same pattern as gepa_runner_sweetrl).
sys.path.insert(0, str(_SWEET_RL_REPO.parent))  # repo root for minimax_routing
import minimax_routing  # noqa: E402

minimax_routing.patch_openai()

from openai import OpenAI  # noqa: E402
from sweet_rl.utils.code_utils import check_correctness  # noqa: E402
from sweetrl_persona_adapter import (  # noqa: E402
    SweetRLTask,
    _ConversationRunner,
    _extract_code,
    _load_prompt,
)


def build_tasks(data_path, split_seed, train_n, val_n, test_n):
    import random

    raw = [json.loads(l) for l in open(data_path)]
    random.Random(split_seed).shuffle(raw)
    raw = raw[: train_n + val_n + test_n]
    out = []
    for i, t in enumerate(raw[train_n + val_n : train_n + val_n + test_n]):
        out.append(
            SweetRLTask(
                task_id=f"test_{i:03d}",
                problem_description=t["problem_description"],
                ground_truth=t["ground_truth"],
                hidden_information=t.get("hidden_information", t["ground_truth"]),
                test_cases=t["test_cases"],
                split="test",
            )
        )
    return out


def run_condition(
    client, tasks, persona_text, args, agent_prompt, human_prompt_base, label, traj_fh
):
    # Build all (task, trial) jobs.
    jobs = []
    for t in tasks:
        for trial in range(args.k):
            runner = _ConversationRunner(
                client,
                args.agent_model,
                args.agent_effort,
                args.human_model,
                args.human_effort,
                agent_prompt,
                human_prompt_base,
                persona_text,
                args.max_steps,
                args.max_completion_tokens,
            )
            jobs.append((t, trial, runner))

    # Run conversations concurrently (no scoring inside threads).
    trajs_by_task: dict[str, list] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futs = {ex.submit(r.run, t): (t, tr) for (t, tr, r) in jobs}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            t, tr = futs[fut]
            try:
                traj = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [at-k] task {t.task_id} trial {tr} failed: {e}")
                traj = None
            trajs_by_task.setdefault(t.task_id, []).append((tr, traj))
            done += 1
            if done % 20 == 0:
                print(f"  [{label}] {done}/{len(jobs)} conversations done")

    # Score in the MAIN thread (check_correctness uses SIGALRM). Keep trial order.
    per_task = []
    for t in tasks:
        entries = sorted(
            trajs_by_task.get(t.task_id, []), key=lambda x: x[0]
        )  # by trial idx
        rewards, trials_detail = [], []
        for trial_idx, traj in entries:
            if traj is None or traj.termination == "error":
                r = 0.0
                detail = {
                    "trial": trial_idx,
                    "reward": 0.0,
                    "n_turns": 0,
                    "termination": "error",
                    "answer": "",
                    "messages": [],
                }
            else:
                try:
                    r = check_correctness(
                        t.ground_truth, _extract_code(traj.answer), t.test_cases
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  [at-k score] {t.task_id}: {e}")
                    r = 0.0
                detail = {
                    "trial": trial_idx,
                    "reward": float(r),
                    "n_turns": traj.n_turns,
                    "termination": traj.termination,
                    "answer": traj.answer,
                    "messages": traj.messages,
                }
            rewards.append(r)
            trials_detail.append(detail)
            # Stream this trial to the trajectories jsonl.
            traj_fh.write(
                json.dumps(
                    {
                        "condition": label,
                        "task_id": t.task_id,
                        "trial": trial_idx,
                        "reward": float(r),
                        "n_turns": detail["n_turns"],
                        "termination": detail["termination"],
                        "answer": detail["answer"],
                        "messages": detail["messages"],
                    }
                )
                + "\n"
            )
        # Pad if some trials errored before producing a traj.
        while len(rewards) < args.k:
            rewards.append(0.0)
            trials_detail.append(
                {
                    "trial": len(rewards) - 1,
                    "reward": 0.0,
                    "n_turns": 0,
                    "termination": "error",
                    "answer": "",
                    "messages": [],
                }
            )
        best = max(rewards)
        avg = statistics.mean(rewards)
        per_task.append(
            {
                "task_id": t.task_id,
                "rewards": rewards,
                "best": best,
                "mean": avg,
                "all_correct": min(rewards) >= 1.0,
                "any_correct": best >= 1.0,
                "n_turns": [d["n_turns"] for d in trials_detail],
                "terminations": [d["termination"] for d in trials_detail],
            }
        )

    bests = [x["best"] for x in per_task]
    avg_at_k = statistics.mean(bests)
    pass_all = sum(1 for x in per_task if x["all_correct"]) / len(per_task)
    pass_any = sum(1 for x in per_task if x["any_correct"]) / len(per_task)
    mean_at_k = statistics.mean(x["mean"] for x in per_task)
    all_turns = [n for x in per_task for n in x["n_turns"]]
    return {
        "k": args.k,
        "n_tasks": len(per_task),
        "avg_at_k": avg_at_k,  # mean of best-of-k reward
        "pass_pct_all_at_k": pass_all,  # ALL k trials fully correct
        "pass_pct_any_at_k": pass_any,  # >=1 trial fully correct (context)
        "mean_at_k": mean_at_k,  # mean of per-task mean reward (context)
        "n_all_correct": sum(1 for x in per_task if x["all_correct"]),
        "n_any_correct": sum(1 for x in per_task if x["any_correct"]),
        "avg_n_turns": statistics.mean(all_turns) if all_turns else 0,
        "median_n_turns": statistics.median(all_turns) if all_turns else 0,
        "per_task": per_task,
    }


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
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--agent-model", default="gpt-5.4")
    ap.add_argument("--agent-effort", default="high")
    ap.add_argument("--human-model", default="gpt-5.4")
    ap.add_argument("--human-effort", default="high")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=20)
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
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
            / "test_at_k_report.json"
        ),
    )
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
    args = ap.parse_args()

    tasks = build_tasks(
        args.data, args.split_seed, args.train_n, args.val_n, args.test_n
    )
    best_persona = Path(args.best_persona).read_text()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    agent_prompt = _load_prompt(args.agent_prompt).replace(
        "WITHIN 10 back-and-forth rounds",
        f"WITHIN {args.max_steps} back-and-forth rounds",
    )
    human_prompt_base = _load_prompt(args.human_prompt)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traj_fh = open(args.trajectories, "w")

    results = {}
    for label, persona in (("no-persona", ""), ("best-persona", best_persona)):
        print(f"\n[{label}] running {len(tasks)} tasks x {args.k} trials ...")
        res = run_condition(
            client,
            tasks,
            persona,
            args,
            agent_prompt,
            human_prompt_base,
            label,
            traj_fh,
        )
        results[label] = res
        print(
            f"  avg@{args.k}           = {res['avg_at_k']:.3f}   (mean of best-of-{args.k} reward)"
        )
        print(
            f"  pass%_all@{args.k}     = {res['pass_pct_all_at_k']:.1%}   ({res['n_all_correct']}/{res['n_tasks']} tasks: ALL {args.k} trials correct)"
        )
        print(
            f"  pass%_any@{args.k}     = {res['pass_pct_any_at_k']:.1%}   ({res['n_any_correct']}/{res['n_tasks']} tasks: >=1 trial correct)"
        )
        print(
            f"  mean@{args.k}          = {res['mean_at_k']:.3f}   (mean of per-task mean reward)"
        )
        print(
            f"  n_turns avg/med        = {res['avg_n_turns']:.1f} / {res['median_n_turns']}"
        )

    traj_fh.close()

    np = results["no-persona"]
    bp = results["best-persona"]
    print(f"\n=== BEST-OF-{args.k} SUMMARY (test n={len(tasks)}, k={args.k}) ===")
    print(
        f"{'condition':<14} {'avg@k':>8} {'pass%all@k':>11} {'pass%any@k':>11} {'mean@k':>8}"
    )
    print(
        f"{'no-persona':<14} {np['avg_at_k']:>8.3f} {np['pass_pct_all_at_k']:>10.1%} {np['pass_pct_any_at_k']:>10.1%} {np['mean_at_k']:>8.3f}"
    )
    print(
        f"{'best-persona':<14} {bp['avg_at_k']:>8.3f} {bp['pass_pct_all_at_k']:>10.1%} {bp['pass_pct_any_at_k']:>10.1%} {bp['mean_at_k']:>8.3f}"
    )
    print(
        f"\n  Δ avg@k        (no-persona - best) = {np['avg_at_k'] - bp['avg_at_k']:+.3f}"
    )
    print(
        f"  Δ pass%all@k                       = {(np['pass_pct_all_at_k'] - bp['pass_pct_all_at_k']) * 100:+.1f} pp"
    )
    print(
        f"  Δ pass%any@k                       = {(np['pass_pct_any_at_k'] - bp['pass_pct_any_at_k']) * 100:+.1f} pp"
    )

    with open(args.out, "w") as fb:
        json.dump(results, fb, indent=2)
    print(f"\n  report       -> {args.out}")
    print(f"  trajectories -> {args.trajectories}")


if __name__ == "__main__":
    main()
