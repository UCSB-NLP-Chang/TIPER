#!/usr/bin/env python3
"""Run adversarial user-persona search on sweet_rl ColBench `code` tasks via GEPA.

Pipeline: GEPA evolves a `persona_text` (injected into the user simulator) to
MINIMIZE the agent's reward (test-case pass rate). Score = 1 - reward, so
higher = harder = better for GEPA.

Settings (per request):
  - task split 30/30/50 (train / val / test) from the hardest 110 tasks
  - agent & user: gpt-5.4, reasoning_effort=high
  - mutation: gpt-5.5, reasoning_effort=xhigh
  - minibatch 10 tasks x 2 trials; valset 2 trials
  - start from no-persona (empty persona_text)
  - max_steps=100, 20 proposal iterations

Usage:
    export OPENAI_API_KEY=sk-...
    cd <sweet_rl repo>  # so sweet_rl/ and prompts/ resolve
    python <this script> --data data/backend_tasks/test_110_hard.jsonl \
        --run-dir data/persona_opt/gepa_runs/sweetrl_run_001

Smoke test:
    python <this script> --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the sweet_rl package importable from the parent venv (gepa lives there).
_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
_REPO_TOP = _SWEET_RL_REPO.parent
for _p in (str(_SWEET_RL_REPO), str(_HERE), str(_REPO_TOP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# minimax routing: patch openai.OpenAI BEFORE importing sweetrl_persona_adapter
# (which does `from openai import OpenAI` at module load). After this, every
# OpenAI(...) call returns a RoutingClient that routes by model name:
#   minimax-m3 → :4114 openai→fireworks shim (RR + exp backoff)
#   gpt-5.x    → standard OpenAI
import minimax_routing  # noqa: E402

minimax_routing.patch_openai()

from sweetrl_persona_adapter import (  # noqa: E402
    SweetRLPersonaAdapter,
    SweetRLTask,
    build_custom_proposer,
)


def load_tasks(path: str) -> list[dict]:
    with open(path, "r") as fb:
        return [json.loads(l) for l in fb]


def build_task_lists(data_path: str, train_n: int, val_n: int, test_n: int, seed: int):
    """Load tasks and split into train/val/test by deterministic shuffle."""
    import random

    raw = load_tasks(data_path)
    # Ensure we have enough tasks.
    need = train_n + val_n + test_n
    if len(raw) < need:
        raise RuntimeError(f"need {need} tasks, found {len(raw)} in {data_path}")
    random.Random(seed).shuffle(raw)
    raw = raw[:need]

    def make(split, items):
        out = []
        for i, t in enumerate(items):
            out.append(
                SweetRLTask(
                    task_id=f"{split}_{i:03d}",
                    problem_description=t["problem_description"],
                    ground_truth=t["ground_truth"],
                    hidden_information=t.get("hidden_information", t["ground_truth"]),
                    test_cases=t["test_cases"],
                    split=split,
                )
            )
        return out

    train = make("train", raw[:train_n])
    val = make("val", raw[train_n : train_n + val_n])
    test = make("test", raw[train_n + val_n : train_n + val_n + test_n])
    return train, val, test


def evaluate_reward(
    adapter: SweetRLPersonaAdapter,
    tasks: list[SweetRLTask],
    persona_text: str,
    label: str,
) -> dict:
    """Run the adapter on a task set with a given persona; report avg reward.

    The ONLINE JUDGE is disabled here (mirrors vita_pgepa.evaluate_reward) —
    test-set rewards must be the RAW agent reward (validity is audited
    separately); gating test scores to zero on severe violations would corrupt
    the no-persona vs best-persona comparison."""
    print(f"  [{label}] running {len(tasks)} tasks x {adapter.num_trials} trials...")
    _prev_judge = adapter.online_judge
    adapter.online_judge = False
    try:
        batch = adapter.evaluate(
            tasks, {"persona_text": persona_text}, capture_traces=False
        )
    finally:
        adapter.online_judge = _prev_judge
    rewards = [1.0 - s for s in batch.scores]  # score = 1 - reward
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    fully_correct = sum(1 for r in rewards if r >= 1.0)
    print(
        f"  [{label}] n={len(tasks)} avg_reward={avg_reward:.3f} "
        f"fully_correct={fully_correct}/{len(tasks)} "
        f"(difficulty={1 - avg_reward:.3f})"
    )
    return {
        "avg_reward": avg_reward,
        "fully_correct": fully_correct,
        "n": len(tasks),
        "rewards": rewards,
    }


def main():
    ap = argparse.ArgumentParser(
        description="GEPA adversarial persona search on sweet_rl code tasks."
    )
    ap.add_argument(
        "--data",
        default=str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl"),
    )
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--train-n", type=int, default=30)
    ap.add_argument("--val-n", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument("--split-seed", type=int, default=42)

    # Models
    ap.add_argument("--agent-model", default="gpt-5.4")
    ap.add_argument("--agent-effort", default="high")
    ap.add_argument("--human-model", default="gpt-5.4")
    ap.add_argument("--human-effort", default="high")
    ap.add_argument("--mutation-model", default="gpt-5.5")
    ap.add_argument("--mutation-effort", default="xhigh")

    # Optimization
    ap.add_argument("--minibatch-size", type=int, default=10)
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument(
        "--test-trials",
        type=int,
        default=None,
        help="trials for the final test-set eval; default = --num-trials. "
        "Set to e.g. 3 to run search at --num-trials 2 but test at 3.",
    )
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--iters", type=int, default=20, help="max candidate proposals")
    ap.add_argument("--max-metric-calls", type=int, default=1500)
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--mutation-edit-pct", default="10-15%")
    ap.add_argument("--seed", type=int, default=0)

    # Unified validity gate (tau/vita/lic semantics): failed conversations are
    # judged on 4 severe categories; severe_rate = severe/n_noninfra > 0.05 gates
    # the whole candidate; timeout_rate >= 0.15 gates likewise.
    ap.add_argument(
        "--judge-model",
        default="gpt-5.2",
        help="model for the shared severe judge (unified across adapters)",
    )
    ap.add_argument(
        "--judge-maj",
        type=int,
        default=3,
        help="validity-judge votes per judged trial, strict majority "
        "(unified across tau/vita/lic/sweetrl; 1 = single judgment)",
    )
    ap.add_argument(
        "--severe-gate-rate",
        type=float,
        default=0.05,
        help="severe_rate above this gates the candidate (0 disables; tau default 5%)",
    )
    ap.add_argument(
        "--timeout-attack-threshold",
        type=float,
        default=0.15,
        help="timeout_rate at/above this gates the candidate (0 disables)",
    )
    ap.add_argument(
        "--no-online-judge",
        action="store_true",
        help="disable the gpt-5.2 severe judge (no validity signal)",
    )

    # Paths
    ap.add_argument(
        "--agent-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "llm_agent_code_prompt.txt"),
    )
    ap.add_argument(
        "--human-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "human_simulator_code_prompt.txt"),
    )

    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end wiring test")
    args = ap.parse_args()

    if args.smoke:
        args.train_n, args.val_n, args.test_n = 3, 3, 4
        args.minibatch_size = 3
        args.num_trials = 1
        args.iters = 2
        args.max_metric_calls = 30
        args.max_steps = 12
        args.max_concurrency = 6

    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(
            _SWEET_RL_REPO / "data" / "persona_opt" / "gepa_runs" / f"sweetrl_{ts}"
        )
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    # 1. Build splits
    train, val, test = build_task_lists(
        args.data, args.train_n, args.val_n, args.test_n, args.split_seed
    )
    print(
        f"Splits: train={len(train)} val={len(val)} test={len(test)}  (data={args.data})"
    )
    print(
        f"Models: agent={args.agent_model}/{args.agent_effort}  human={args.human_model}/{args.human_effort}  "
        f"mutation={args.mutation_model}/{args.mutation_effort}"
    )
    print(
        f"Opt: minibatch={args.minibatch_size} trials={args.num_trials} max_steps={args.max_steps} "
        f"iters={args.iters} max_metric_calls={args.max_metric_calls}"
    )
    print(f"run_dir: {args.run_dir}")

    # 2. Adapter + proposer
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
        judge_model=args.judge_model,
        judge_maj=args.judge_maj,
        severe_gate_rate=(args.severe_gate_rate if args.severe_gate_rate > 0 else None),
        timeout_attack_threshold=(
            args.timeout_attack_threshold if args.timeout_attack_threshold > 0 else None
        ),
        online_judge=not args.no_online_judge,
    )
    print(
        f"Validity gate: judge={args.judge_model} (online={not args.no_online_judge}) "
        f"severe_rate>{args.severe_gate_rate} timeout_rate>={args.timeout_attack_threshold}"
    )
    proposer = build_custom_proposer(
        mutation_model=args.mutation_model,
        mutation_effort=args.mutation_effort,
        mutation_edit_pct=args.mutation_edit_pct,
        max_steps=args.max_steps,
    )

    # 3. No-persona seed candidate
    seed_candidate = {"persona_text": ""}

    # 4. Stopper: cap proposal iterations
    from gepa.utils import MaxCandidateProposalsStopper

    stoppers = [MaxCandidateProposalsStopper(args.iters)]

    # 5. Run GEPA
    from gepa.api import optimize as gepa_optimize

    print("\nStarting GEPA optimization (no-persona start)...\n")
    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=val,
        adapter=adapter,
        custom_candidate_proposer=proposer,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        reflection_minibatch_size=args.minibatch_size,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=stoppers,
        skip_perfect_score=False,
        perfect_score=1.0,
        cache_evaluation=True,
        run_dir=args.run_dir,
        seed=args.seed,
        raise_on_exception=False,
        display_progress_bar=True,
        module_selector="all",
    )

    # 6. Save candidates + best persona
    out_dir = Path(args.run_dir)
    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx]
    best_persona = result.candidates[best_idx]["persona_text"]

    candidates_out = []
    for i, (cand, score) in enumerate(
        zip(result.candidates, result.val_aggregate_scores)
    ):
        parents = result.parents[i]
        candidates_out.append(
            {
                "idx": i,
                "persona_text": cand["persona_text"],
                "val_difficulty": float(score),
                "val_reward": float(1.0 - score),
                "parent": parents[0] if parents and parents[0] is not None else None,
                "is_best": i == best_idx,
            }
        )
    with open(out_dir / "candidates.json", "w") as fb:
        json.dump(candidates_out, fb, indent=2)
    (out_dir / "best_persona.md").write_text(best_persona)

    print(f"\n{'=' * 60}\n  GEPA COMPLETE\n{'=' * 60}")
    print(f"  candidates explored: {result.num_candidates}")
    print(f"  total metric calls:  {result.total_metric_calls}")
    print(
        f"  best idx: {best_idx}  val difficulty={best_score:.3f} (agent reward≈{1 - best_score:.1%})"
    )
    print(f"  best persona -> {out_dir / 'best_persona.md'}")

    # 7. Test-set evaluation: no-persona baseline vs best persona
    if args.test_n > 0 and not args.smoke:
        test_trials = (
            args.test_trials if args.test_trials is not None else args.num_trials
        )
        adapter.num_trials = (
            test_trials  # search used num_trials; test uses test_trials
        )
        print(
            f"\n{'=' * 60}\n  TEST-SET EVALUATION (n={len(test)}, {test_trials} trials)\n{'=' * 60}"
        )
        print("  No-persona baseline:")
        base = evaluate_reward(adapter, test, "", "test/no-persona")
        print("  Best persona:")
        best = evaluate_reward(adapter, test, best_persona, "test/best-persona")
        delta = base["avg_reward"] - best["avg_reward"]
        print(
            f"\n  Δ reward (no-persona - best) = {delta:+.3f}  "
            f"({'persona HARDER ✓' if delta > 0 else 'no improvement ✗'})"
        )
        report = {
            "no_persona": base,
            "best_persona": best,
            "delta_reward": delta,
            "best_persona_path": str(out_dir / "best_persona.md"),
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
        }
        with open(out_dir / "test_report.json", "w") as fb:
            json.dump(report, fb, indent=2)
        print(f"  report -> {out_dir / 'test_report.json'}")
    elif args.smoke:
        print("\n  [smoke] skipping test-set eval")


if __name__ == "__main__":
    main()
