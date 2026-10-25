#!/usr/bin/env python3
"""Test-set evaluation with N trials: avg@N + pass^N.

avg@N   = per task mean reward over N trials, then mean over tasks.
pass^N  = per task 1 if ALL N trials reward>=success_threshold else 0, then mean
          over tasks (= fraction of tasks the agent solved correctly N/N times).

Runs two conditions: no-persona baseline vs best persona. Dumps conversations
so the offline judge can score validity without re-simulating.

Usage:
    .venv/bin/python scripts/vita_test_eval.py \
        --run-dir data/persona_opt/vita_pgepa_runs/run_delivery --trials 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vita.run import get_tasks, run_task  # noqa: E402
from vita.data_model.message import UserMessage  # noqa: E402
from vita.utils.llm_utils import generate  # noqa: E402

_SUCCESS_THR = 0.999  # same as adapter._success_threshold


def split_test_ids(cfg: dict) -> list[str]:
    all_ids = [t.id for t in get_tasks(cfg["domain"], None, None, cfg["language"])]
    rng = random.Random(cfg["seed"])
    rng.shuffle(all_ids)
    tr, va = cfg["train_size"], cfg["val_size"]
    ts = cfg["test_size"]
    return all_ids[tr + va:tr + va + ts]


def _sim_to_reward_and_msgs(sim) -> tuple[float | None, list[dict]]:
    if sim is None:
        return None, []
    ri = getattr(sim, "reward_info", None)
    reward = getattr(ri, "reward", None) if ri is not None else None
    msgs = []
    for m in getattr(sim, "messages", []):
        role = getattr(m, "role", "?")
        content = getattr(m, "content", None) or ""
        d = {"role": role, "content": content}
        if hasattr(m, "is_tool_call") and m.is_tool_call():
            d["tool_calls"] = [
                {"name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", {})}
                for tc in (getattr(m, "tool_calls", None) or [])
            ]
        msgs.append(d)
    return (float(reward) if reward is not None else None), msgs


def run_one(task, trial, seed, persona, cfg):
    try:
        sim = run_task(
            domain=cfg["domain"], task=task, agent="llm_agent", user="user_simulator",
            llm_agent=cfg["agent_llm"], llm_args_agent=cfg.get("agent_llm_args", {}),
            llm_user=cfg.get("user_llm", cfg["agent_llm"]),
            llm_args_user=cfg.get("user_llm_args", {}),
            max_steps=cfg["max_steps"], max_errors=100000, max_duration=0,
            seed=seed, enable_think=True,
            llm_evaluator=cfg.get("evaluator_llm", "gpt-4.1"),
            llm_args_evaluator={}, language=cfg["language"],
            user_persona_override=persona,
        )
        return _sim_to_reward_and_msgs(sim)
    except Exception as e:  # noqa: BLE001
        print(f"  task {task.id} trial {trial} failed: {e}")
        return None, []


def eval_condition(test_tasks, persona, label, cfg, trials, concurrency, seed_base, run_dir):
    print(f"\n[{label}] {len(test_tasks)} tasks x {trials} trials...")
    rewards_by_task: dict[str, list] = {}
    msgs_by_task: dict[str, list] = {}
    jobs = []
    for t in test_tasks:
        for tr in range(trials):
            jobs.append((t, tr, seed_base + tr * 7919))
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(run_one, t, tr, seed, persona, cfg): (t.id, tr)
                for (t, tr, seed) in jobs}
        for fut in futs:
            tid, tr = futs[fut]
            r, msgs = fut.result()
            rewards_by_task.setdefault(tid, [None] * trials)[tr] = r
            # keep the worst-reward (hardest) trial's conversation for the judge
            prev = msgs_by_task.get(tid)
            if prev is None or (r is not None and (prev[0] is None or r < prev[0])):
                msgs_by_task[tid] = (r, msgs)
    avg_per_task, pass_per_task = {}, {}
    for tid, rs in rewards_by_task.items():
        valid = [float(r) if r is not None else 0.0 for r in rs]
        avg_per_task[tid] = sum(valid) / len(valid)
        pass_per_task[tid] = 1.0 if all(v >= _SUCCESS_THR for v in valid) else 0.0
    avg_at_n = sum(avg_per_task.values()) / len(avg_per_task) if avg_per_task else 0.0
    pass_at_n = sum(pass_per_task.values()) / len(pass_per_task) if pass_per_task else 0.0
    n_fully = sum(pass_per_task.values())
    print(f"  avg@{trials}={avg_at_n:.4f}  pass^{trials}={pass_at_n:.4f}  "
          f"({n_fully:.0f}/{len(avg_per_task)} tasks solved {trials}/{trials})")
    # dump conversations for the offline judge
    cdir = run_dir / "conversations" / f"test{trials}_{label}"
    cdir.mkdir(parents=True, exist_ok=True)
    for tid, (r, msgs) in msgs_by_task.items():
        rec = {"task_id": tid, "domain": cfg["domain"], "reward": r,
               "messages": msgs}
        (cdir / f"{tid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return {
        "avg_at_n": avg_at_n, "pass_at_n": pass_at_n,
        "n_test": len(avg_per_task), "n_fully_solved": int(n_fully),
        "avg_per_task": avg_per_task, "pass_per_task": pass_per_task,
        "rewards_by_task": {k: v for k, v in rewards_by_task.items()},
    }


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=500)
    ap.add_argument("--persona-file", default=None,
                    help="default: {run-dir}/best_persona.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "pgepa_config.json").read_text())
    persona = (Path(args.persona_file).read_text()
               if args.persona_file else (run_dir / "best_persona.md").read_text())
    test_ids = split_test_ids(cfg)
    task_map = {t.id: t for t in get_tasks(cfg["domain"], None, None, cfg["language"])}
    test_tasks = [task_map[tid] for tid in test_ids if tid in task_map]
    print(f"test eval: domain={cfg['domain']} n={len(test_tasks)} trials={args.trials} "
          f"persona_len={len(persona)} agent={cfg['agent_llm']}")

    base = eval_condition(test_tasks, "", "no_persona", cfg, args.trials,
                          args.concurrency, args.seed_base, run_dir)
    best = eval_condition(test_tasks, persona, "best_persona", cfg, args.trials,
                          args.concurrency, args.seed_base, run_dir)

    report = {
        "domain": cfg["domain"], "trials": args.trials, "n_test": len(test_tasks),
        "no_persona": {k: base[k] for k in ("avg_at_n", "pass_at_n", "n_fully_solved")},
        "best_persona": {k: best[k] for k in ("avg_at_n", "pass_at_n", "n_fully_solved")},
        "delta_avg_no_minus_best": base["avg_at_n"] - best["avg_at_n"],
        "delta_pass_no_minus_best": base["pass_at_n"] - best["pass_at_n"],
        "avg_per_task": {"no_persona": base["avg_per_task"], "best_persona": best["avg_per_task"]},
        "pass_per_task": {"no_persona": base["pass_per_task"], "best_persona": best["pass_per_task"]},
        "rewards_by_task": {"no_persona": base["rewards_by_task"], "best_persona": best["rewards_by_task"]},
    }
    out = run_dir / f"test_eval_{args.trials}trials.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n=== TEST EVAL ({cfg['domain']}, {args.trials} trials) ===")
    print(f"  no_persona:   avg@{args.trials}={base['avg_at_n']:.4f}  "
          f"pass^{args.trials}={base['pass_at_n']:.4f}  "
          f"({base['n_fully_solved']}/{base['n_test']} solved {args.trials}/{args.trials})")
    print(f"  best_persona: avg@{args.trials}={best['avg_at_n']:.4f}  "
          f"pass^{args.trials}={best['pass_at_n']:.4f}  "
          f"({best['n_fully_solved']}/{best['n_test']} solved {args.trials}/{args.trials})")
    print(f"  Δ avg  = {report['delta_avg_no_minus_best']:+.4f}  "
          f"({'+ persona 更难' if report['delta_avg_no_minus_best'] > 0 else 'no improvement'})")
    print(f"  Δ pass = {report['delta_pass_no_minus_best']:+.4f}")
    print(f"  report -> {out}")
    print(f"  conversations -> {run_dir}/conversations/test{args.trials}_*")
    print(f"  Next: scripts/vita_offline_judge.py --run-dir {run_dir} "
          f"(reads test{args.trials}_* if present)")


if __name__ == "__main__":
    main()
