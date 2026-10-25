#!/usr/bin/env python3
"""Test-set certification for fail² GEPA.

Evaluates top-K candidates + seed on the test split (54 tasks × 3 trials).
Seed already has 1 trial from a prior run → auto-resume adds 2 more.
Computes avg@3 and pass@3 per persona.

    .venv/bin/python experiments/persona_search/scripts/run_test_cert_f2.py
    .venv/bin/python experiments/persona_search/scripts/run_test_cert_f2.py --top-k 3 --max-concurrency 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def load_top_candidates(state_path: str, top_k: int) -> list[dict]:
    state = json.loads(Path(state_path).read_text())
    cands = state.get("candidates", [])
    cands.sort(key=lambda c: c.get("agg", 0), reverse=True)
    return cands[:top_k]


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
    """Compute avg@3 and pass@3 from results.json."""
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])

    by_task: dict[str, list[float]] = defaultdict(list)
    for s in sims:
        tid = s.get("task_id", "")
        reward = s.get("reward") or s.get("reward_info", {}).get("reward")
        if reward is not None:
            by_task[tid].append(float(reward))

    n_tasks = len(by_task)
    avg_scores = []
    pass_scores = []
    for tid, rewards in by_task.items():
        avg_scores.append(sum(rewards) / len(rewards))
        pass_scores.append(1.0 if any(r > 0 for r in rewards) else 0.0)

    return {
        "n_tasks": n_tasks,
        "n_sims": len(sims),
        "avg_at_3": sum(avg_scores) / len(avg_scores) if avg_scores else 0,
        "pass_at_3": sum(pass_scores) / len(pass_scores) if pass_scores else 0,
        "difficulty_avg": 1.0 - (sum(avg_scores) / len(avg_scores))
        if avg_scores
        else 0,
        "difficulty_pass": 1.0 - (sum(pass_scores) / len(pass_scores))
        if pass_scores
        else 0,
    }


def run_one(
    label: str,
    persona_text: str,
    task_ids: list[str],
    num_trials: int,
    args,
    existing_results: Path | None = None,
):
    save_to = f"{args.out_base}/{label}"
    cert_dir = SIMULATIONS_DIR / save_to

    if existing_results and existing_results.exists() and not cert_dir.exists():
        cert_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_results, cert_dir / "results.json")
        print(
            f"  [{label}] Copied existing results → auto-resume will add missing trials"
        )

    eval_args = make_eval_args(persona_text, num_trials, args)
    print(f"  [{label}] Running {num_trials} trials × {len(task_ids)} tasks...")
    _run_domain_with_retry(eval_args, "telecom", task_ids, save_to)

    rp = cert_dir / "results.json"
    if rp.exists():
        scores = score_results(rp)
        print(
            f"  [{label}] avg@3={scores['avg_at_3']:.3f}  pass@3={scores['pass_at_3']:.3f}  "
            f"difficulty_avg={scores['difficulty_avg']:.3f}  difficulty_pass={scores['difficulty_pass']:.3f}  "
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
            / "data/persona_opt/pgepa_runs/fail2_seed0_gpt54user/f2_brinks_00/pgepa_state.json"
        ),
    )
    ap.add_argument(
        "--split",
        default="experiments/persona_search/configs/telecom_island_split_fail2.json",
    )
    ap.add_argument(
        "--seed-file",
        default=str(
            DATA_ROOT / "data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
        ),
    )
    ap.add_argument("--out-base", default="gepa_opt/test_f2_cert")
    ap.add_argument("--top-k", type=int, default=2)
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
    top_cands = load_top_candidates(args.state, args.top_k)
    seed_text = Path(args.seed_file).read_text()

    print(f"=== Test-set certification ===")
    print(f"  Test tasks: {len(test_tasks)}")
    print(f"  Trials: {args.num_trials}")
    print(f"  Personas: seed + top {args.top_k} candidates")
    print()

    all_scores = {}

    # Seed: has 1 existing trial
    existing_seed = (
        SIMULATIONS_DIR
        / "gepa_opt"
        / "test_gpt54user"
        / "seed_brinks_00"
        / "results.json"
    )
    scores = run_one(
        "seed_brinks_00",
        seed_text,
        test_tasks,
        args.num_trials,
        args,
        existing_results=existing_seed,
    )
    if scores:
        all_scores["seed"] = scores

    # Top-K candidates
    for c in top_cands:
        idx = c["idx"]
        gen = c.get("gen", "?")
        agg = c.get("agg", 0)
        label = f"cand_idx{idx}_gen{gen}"
        print(f"\n--- Candidate idx{idx} (gen={gen}, val_agg={agg:.3f}) ---")
        scores = run_one(label, c["persona_text"], test_tasks, args.num_trials, args)
        if scores:
            all_scores[label] = scores

    # Summary
    print("\n" + "=" * 60)
    print(
        f"{'Persona':<30s} {'avg@3':>7s} {'pass@3':>7s} {'diff_avg':>9s} {'diff_pass':>10s}"
    )
    print("-" * 60)
    for name, sc in all_scores.items():
        print(
            f"{name:<30s} {sc['avg_at_3']:7.3f} {sc['pass_at_3']:7.3f} "
            f"{sc['difficulty_avg']:9.3f} {sc['difficulty_pass']:10.3f}"
        )

    # Save summary
    summary_path = SIMULATIONS_DIR / args.out_base / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
