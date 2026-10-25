#!/usr/bin/env python3
"""Evaluate Brinks seed and final optimized personas on the full telecom base set.

This is a raw-difficulty eval only:
  difficulty = 1 - mean(agent reward)

It intentionally does not run the validity judge; the optimization run already
used validity as a gate, and this script is for measuring full-set difficulty.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402, DATA_ROOT
    REPO_ROOT,
    SIMULATIONS_DIR,
    read_json,
    skill_name_for,
    write_json,
)
from eval_persona import _aggregate_results, _run_domain_with_retry  # noqa: E402
from lineage_loop import materialize  # noqa: E402

RUNS = {
    "brinks_03": {
        "seed": DATA_ROOT
        / "data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md",
        "run_dir": DATA_ROOT
        / "data/persona_opt/pgepa_runs/pgepa_brinks_03_verbose_polite",
    },
    "brinks_04": {
        "seed": DATA_ROOT
        / "data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md",
        "run_dir": DATA_ROOT
        / "data/persona_opt/pgepa_runs/pgepa_brinks_04_confused_drip_feeding",
    },
}


def load_telecom_base_tasks() -> list[str]:
    split_path = REPO_ROOT / "datasets/tau2/domains/telecom/split_tasks.json"
    split = read_json(split_path)
    return list(split["base"])


def load_optimized_persona(run_dir: Path) -> tuple[str, dict]:
    state = read_json(run_dir / "pgepa_state.json")
    best_idx = state["best_idx"]
    best = next(c for c in state["candidates"] if c["idx"] == best_idx)
    return best["persona_text"], best


def make_eval_args(args: argparse.Namespace, persona_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        skill_name=skill_name_for(persona_id),
        effort=args.effort,
        use_bedrock=args.use_bedrock,
        bedrock_profile=args.bedrock_profile,
        anthropic_base_url=args.anthropic_base_url,
        anthropic_api_key=args.anthropic_api_key,
        user_type=args.user_type,
        user_llm=args.user_llm,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        retrieval_config=args.retrieval_config,
        max_steps=args.max_steps,
        sim_timeout=args.sim_timeout,
        max_infra_retries=args.max_infra_retries,
    )


def evaluate_one(
    *,
    label: str,
    persona_id: str,
    persona_text: str,
    task_ids: list[str],
    args: argparse.Namespace,
) -> dict:
    materialize(persona_id, persona_text)
    eval_args = make_eval_args(args, persona_id)
    run_name = f"persona_opt/fullset_brinks/{persona_id}/telecom"
    if args.dry_run:
        return {
            "label": label,
            "persona_id": persona_id,
            "n_tasks": len(task_ids),
            "difficulty": None,
            "agent_success_rate": None,
            "infra_error_count": None,
            "timeout_stats": None,
            "results_path": str(SIMULATIONS_DIR / run_name / "results.json"),
        }

    results_path = _run_domain_with_retry(eval_args, "telecom", task_ids, run_name)
    per_task, infra, n_seen, timeout_stats = _aggregate_results(results_path, "telecom")
    agent_success_rate = sum(per_task.values()) / len(per_task) if per_task else None
    difficulty = 1.0 - agent_success_rate if agent_success_rate is not None else None
    return {
        "label": label,
        "persona_id": persona_id,
        "n_tasks": len(per_task),
        "n_seen": n_seen,
        "agent_success_rate": agent_success_rate,
        "difficulty": difficulty,
        "infra_error_count": infra,
        "timeout_stats": timeout_stats,
        "results_path": str(results_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Brinks seed + optimized personas on telecom full base split."
    )
    ap.add_argument(
        "--out",
        default=str(DATA_ROOT / "data/persona_opt/pgepa_runs/brinks_fullset_eval.json"),
    )
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="cap telecom base tasks for smoke"
    )
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-concurrency", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=False
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument("--anthropic-base-url", default=None)
    ap.add_argument("--anthropic-api-key", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    task_ids = load_telecom_base_tasks()
    if args.max_tasks:
        task_ids = task_ids[: args.max_tasks]

    jobs: list[tuple[str, str, str, dict | None]] = []
    for short, cfg in RUNS.items():
        seed_text = cfg["seed"].read_text()
        opt_text, meta = load_optimized_persona(cfg["run_dir"])
        jobs.append((f"{short}_seed", f"fullset_{short}_seed", seed_text, None))
        jobs.append(
            (f"{short}_optimized", f"fullset_{short}_optimized", opt_text, meta)
        )

    print(
        f"Evaluating {len(jobs)} personas on telecom base split: {len(task_ids)} tasks each"
    )
    print(f"Total rollouts: {len(jobs) * len(task_ids) * args.num_trials}")
    print(
        f"User route: {args.user_type} {args.user_llm} bedrock={args.use_bedrock} "
        f"base_url={args.anthropic_base_url if not args.use_bedrock else '-'}"
    )

    results = []
    for label, persona_id, text, meta in jobs:
        print(f"\n[{label}] persona_id={persona_id}")
        if meta:
            print(
                f"  exploration: idx={meta['idx']} diff={meta['agg']:.3f} "
                f"validity={meta['validity_agg']:.3f} direction={meta['mutation_direction']}"
            )
        res = evaluate_one(
            label=label,
            persona_id=persona_id,
            persona_text=text,
            task_ids=task_ids,
            args=args,
        )
        results.append({**res, "exploration_meta": meta})
        if res["difficulty"] is not None:
            print(
                f"  fullset difficulty={res['difficulty']:.3f} "
                f"agent_success={res['agent_success_rate']:.3f} "
                f"n={res['n_tasks']} infra={res['infra_error_count']}"
            )

    out = Path(args.out)
    write_json(
        out,
        {
            "task_split": "telecom.base",
            "n_tasks_requested": len(task_ids),
            "num_trials": args.num_trials,
            "models": {
                "agent": f"{args.agent_llm} {args.agent_llm_args}",
                "user": f"{args.user_type} {args.user_llm} effort={args.effort} "
                f"bedrock={args.use_bedrock}",
            },
            "results": results,
        },
    )
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
