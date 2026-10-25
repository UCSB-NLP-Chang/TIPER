#!/usr/bin/env python3
"""Test-set certification for enriched mutation context GEPA run (seed01).

Evaluates seed persona + frontier top-K on test split (54 tasks × 3 trials).
Computes avg@3, pass@3, pass^3, fail^3 per persona.

    .venv/bin/python experiments/persona_search/scripts/run_test_cert_enriched_seed1.py
    .venv/bin/python experiments/persona_search/scripts/run_test_cert_enriched_seed1.py --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SIMULATIONS_DIR  # noqa: E402, DATA_ROOT
from eval_persona import _run_domain_with_retry  # noqa: E402


def load_test_tasks(split_path: str) -> list[str]:
    split = json.loads(Path(split_path).read_text())
    return split["test"]["telecom"]


def load_frontier_top_k(state_path: str, top_k: int) -> list[dict]:
    """Load Pareto frontier members, sorted by agg descending, return top-k."""
    state = json.loads(Path(state_path).read_text())
    cands = state.get("candidates", [])
    val_ids = list(cands[0].get("subscores", {}).keys()) if cands else []
    feasible = [c for c in cands if c.get("feasible", True) and c.get("subscores")]
    frontier = []
    for i, a in enumerate(feasible):
        sa = {k: a.get("subscores", {}).get(k, 0) for k in val_ids}
        dominated = False
        for j, b in enumerate(feasible):
            if i == j:
                continue
            sb = {k: b.get("subscores", {}).get(k, 0) for k in val_ids}
            if all(sb[k] >= sa[k] for k in val_ids) and any(
                sb[k] > sa[k] for k in val_ids
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    frontier.sort(key=lambda c: c.get("agg", 0), reverse=True)
    return frontier[:top_k]


def make_eval_args(persona_text: str, num_trials: int, args) -> SimpleNamespace:
    return SimpleNamespace(
        user_type=args.user_type,
        user_llm=args.user_llm,
        effort=args.effort,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        num_trials=num_trials,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        sim_timeout=args.sim_timeout,
        max_infra_retries=args.max_infra_retries,
        use_bedrock=False,
        persona_text=persona_text,
        retrieval_config="bm25",
    )


def score_results(results_path: Path) -> dict:
    """Compute avg@3, pass@3, pass^3, fail^3 from results.json."""
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])

    by_task: dict[str, list[float]] = defaultdict(list)
    for s in sims:
        tid = s.get("task_id", "")
        reward = s.get("reward") or s.get("reward_info", {}).get("reward")
        if reward is not None:
            by_task[tid].append(float(reward))

    n_tasks = len(by_task)
    avg_scores, pass_scores, pass3_scores, fail3_scores = [], [], [], []
    for tid, rewards in by_task.items():
        avg_scores.append(sum(rewards) / len(rewards))
        pass_scores.append(1.0 if any(r > 0 for r in rewards) else 0.0)
        pass3_scores.append(1.0 if all(r > 0 for r in rewards) else 0.0)
        fail3_scores.append(1.0 if all(r == 0 for r in rewards) else 0.0)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0

    return {
        "n_tasks": n_tasks,
        "n_sims": len(sims),
        "avg_at_3": _mean(avg_scores),
        "pass_at_3": _mean(pass_scores),
        "pass_cubed": _mean(pass3_scores),
        "fail_cubed": _mean(fail3_scores),
        "diff_avg": 1.0 - _mean(avg_scores),
        "diff_pass_cubed": 1.0 - _mean(pass3_scores),
    }


def run_one(label: str, persona_text: str, task_ids: list[str], num_trials: int, args):
    save_to = f"{args.out_base}/{label}"
    eval_args = make_eval_args(persona_text, num_trials, args)
    print(f"  [{label}] Running {num_trials} trials × {len(task_ids)} tasks...")
    _run_domain_with_retry(eval_args, "telecom", task_ids, save_to)

    cert_dir = SIMULATIONS_DIR / save_to
    rp = cert_dir / "results.json"
    if rp.exists():
        scores = score_results(rp)
        print(
            f"  [{label}] avg@3={scores['avg_at_3']:.3f}  pass@3={scores['pass_at_3']:.3f}  "
            f"pass^3={scores['pass_cubed']:.3f}  fail^3={scores['fail_cubed']:.3f}  "
            f"({scores['n_sims']} sims / {scores['n_tasks']} tasks)"
        )
        return scores
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state",
        default=str(
            DATA_ROOT
            / "data/persona_opt/pgepa_runs/enriched_seed1_gpt54user/enriched_brinks_01/pgepa_state.json"
        ),
    )
    ap.add_argument(
        "--split",
        default="experiments/persona_search/configs/telecom_island_split_fail2.json",
    )
    ap.add_argument(
        "--seed-file",
        default=str(
            DATA_ROOT
            / "data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md"
        ),
    )
    ap.add_argument("--out-base", default="gepa_opt/test_enriched_seed1_cert")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-concurrency", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    args = ap.parse_args()

    test_tasks = load_test_tasks(args.split)

    print(f"=== Test-set certification (enriched seed01) ===")
    print(f"  Test tasks: {len(test_tasks)}")
    print(f"  Trials: {args.num_trials}")

    all_scores = {}

    # 1. Seed persona
    seed_text = Path(args.seed_file).read_text()
    print(f"\n--- Seed: {args.seed_file} ---")
    scores = run_one("seed_brinks_01", seed_text, test_tasks, args.num_trials, args)
    if scores:
        all_scores["seed_brinks_01"] = scores
    print()

    # 2. Frontier top-k
    state_path = Path(args.state)
    if state_path.exists():
        frontier = load_frontier_top_k(args.state, args.top_k)
        labels = [f"idx{c['idx']}(agg={c.get('agg', 0):.3f})" for c in frontier]
        print(f"  Frontier top-{args.top_k}: {labels}")
        print()

        for c in frontier:
            idx = c["idx"]
            gen = c.get("gen", "?")
            agg = c.get("agg", 0)
            label = f"frontier_idx{idx}_gen{gen}"
            print(f"--- Frontier idx{idx} (gen={gen}, val_agg={agg:.3f}) ---")
            scores = run_one(
                label, c["persona_text"], test_tasks, args.num_trials, args
            )
            if scores:
                all_scores[label] = scores
            print()
    else:
        print(f"  WARNING: state file not found: {args.state}")
        print("  Skipping frontier evaluation.")

    # Summary table
    print("=" * 80)
    print(
        f"{'Persona':<30s} {'avg@3':>7s} {'pass@3':>7s} {'pass^3':>7s} {'fail^3':>7s} {'diff_avg':>9s}"
    )
    print("-" * 80)
    for name, sc in all_scores.items():
        print(
            f"{name:<30s} {sc['avg_at_3']:7.3f} {sc['pass_at_3']:7.3f} "
            f"{sc['pass_cubed']:7.3f} {sc['fail_cubed']:7.3f} {sc['diff_avg']:9.3f}"
        )

    # Save summary
    summary_path = SIMULATIONS_DIR / args.out_base / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
