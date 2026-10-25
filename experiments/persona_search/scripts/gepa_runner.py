#!/usr/bin/env python3
"""Run persona optimization via GEPA.

Replaces lineage_loop.py's 3-lineage (1+1)-ES with GEPA's evolutionary
optimization. Uses the same eval/judge/mutation infrastructure.

    # Smoke test (dry-run, no API calls)
    python experiments/persona_search/scripts/gepa_runner.py \
        --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
        --max-metric-calls 10 --minibatch-size 3 --dry-run

    # Minimal live test (2 tasks, ~$1)
    python experiments/persona_search/scripts/gepa_runner.py \
        --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
        --max-metric-calls 4 --minibatch-size 2

    # Full run
    python experiments/persona_search/scripts/gepa_runner.py \
        --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
        --max-metric-calls 500 --minibatch-size 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    load_rep_tasks,
    write_json,
)
from tau_persona_adapter import (  # noqa: E402
    TauPersonaAdapter,
    TauTask,
    build_custom_proposer,
)


def build_task_list(domains: list[str], max_tasks: int = 0) -> list[TauTask]:
    rep = load_rep_tasks()
    tasks: list[TauTask] = []
    for dk in domains:
        ids = rep.get(dk, [])
        if max_tasks:
            ids = ids[:max_tasks]
        for tid in ids:
            tasks.append(TauTask(task_id=tid, domain=dk))
    return tasks


def main():
    ap = argparse.ArgumentParser(description="GEPA-based persona optimization.")
    ap.add_argument(
        "--seed-persona", required=True, help="path to initial persona .skill/.md file"
    )
    ap.add_argument(
        "--run-dir", default=None, help="GEPA state directory (default: auto-generated)"
    )
    ap.add_argument(
        "--max-metric-calls",
        type=int,
        default=500,
        help="budget in individual task evaluations",
    )
    ap.add_argument(
        "--minibatch-size",
        type=int,
        default=10,
        help="tasks per reflective mutation round",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--persona-id",
        default="gepa_eval",
        help="unique id for this run's .skill file and sim namespace "
        "(set distinct values when running multiple seeds in parallel)",
    )

    # Model / infra (same defaults as lineage_loop.py)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument("--mutation-model", default="bedrock/us.anthropic.claude-opus-4-7")
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])

    # Judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-maj", type=int, default=3)

    # Validity-gated optimization (opt-in). When --judge-valset is set, the full
    # valset eval (which decides best/Pareto) is also judged (failed-only), so a
    # cheating persona's severe sims score 0 difficulty there too — not just on
    # the minibatch. --severe-gate-rate disqualifies the whole persona (all
    # scores -> 0) when its severe_rate exceeds the rate (cert uses 0.05).
    ap.add_argument(
        "--judge-valset",
        action="store_true",
        default=False,
        help="also judge the full-valset eval (failed-only) so validity "
        "gates the score that decides best/Pareto",
    )
    ap.add_argument(
        "--severe-gate-rate",
        type=float,
        default=None,
        help="persona-level hard gate: zero all scores when "
        "severe_rate > this (e.g. 0.05). Default: legacy absolute "
        "severe_batch_threshold gate",
    )
    ap.add_argument(
        "--valset-judge-max-failures",
        type=int,
        default=15,
        help="cap judged failed sims per domain in valset eval (worst-first)",
    )

    # GEPA tuning
    ap.add_argument(
        "--frontier-type",
        default="instance",
        choices=["instance", "objective", "hybrid"],
    )
    ap.add_argument(
        "--selection-strategy",
        default="pareto",
        choices=["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
    )

    # Mutation
    ap.add_argument(
        "--mutation-edit-pct",
        default="10-15%",
        help="how much of the persona text to change per mutation "
        "(e.g. '10-15%%', '~30%%', '30-50%%')",
    )

    # Convergence / early-stop (composed with max-metric-calls via OR; whichever
    # fires first wins). Keyed off the full-valset best, not the noisy minibatch.
    ap.add_argument(
        "--patience",
        type=int,
        default=8,
        help="stop after N proposal rounds with no new full-valset best "
        "difficulty (0 = disabled; budget is still the hard cap)",
    )
    ap.add_argument(
        "--converge-threshold",
        type=float,
        default=0.0,
        help="stop early if full-valset best difficulty >= this "
        "(e.g. 0.95 = agent success <= 5%%; 0 = disabled)",
    )

    # Testing
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="cap tasks PER DOMAIN (for smoke/small tests)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="skip real sims, use fake results for wiring test",
    )

    args = ap.parse_args()

    # Auto-generate run_dir
    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(PERSONA_OPT_DIR / "gepa_runs" / f"run_{ts}")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    # 1. Build task list
    tasks = build_task_list(args.domains, args.max_tasks)
    print(
        f"Tasks: {len(tasks)} ({sum(1 for t in tasks if t.domain == 'telecom')}T "
        f"+ {sum(1 for t in tasks if t.domain == 'banking')}B)"
    )

    # 2. Load seed persona
    seed_text = Path(args.seed_persona).read_text()
    seed_candidate = {"persona_text": seed_text}
    print(f"Seed persona: {args.seed_persona} ({len(seed_text)} chars)")

    # 3. Create adapter
    adapter = TauPersonaAdapter(
        persona_id=args.persona_id,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        user_type=args.user_type,
        user_llm=args.user_llm,
        effort=args.effort,
        use_bedrock=args.use_bedrock,
        bedrock_profile=args.bedrock_profile,
        max_concurrency=args.max_concurrency,
        sim_timeout=args.sim_timeout,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
        max_infra_retries=args.max_infra_retries,
        retrieval_config=args.retrieval_config,
        domains=args.domains,
        judge_model=args.judge_model,
        judge_concurrency=args.judge_concurrency,
        judge_maj=args.judge_maj,
        always_judge_minibatch=True,
        minibatch_judge_cap=20,
        judge_valset=args.judge_valset,
        severe_gate_rate=args.severe_gate_rate,
        valset_judge_max_failures=args.valset_judge_max_failures,
    )

    # 4. Build custom proposer
    custom_proposer = build_custom_proposer(args.mutation_model, args.mutation_edit_pct)

    # 5. Run GEPA
    print(f"\nStarting GEPA optimization")
    print(f"  run_dir:          {args.run_dir}")
    print(f"  max_metric_calls: {args.max_metric_calls}")
    print(f"  minibatch_size:   {args.minibatch_size}")
    print(f"  frontier_type:    {args.frontier_type}")
    print(f"  selection:        {args.selection_strategy}")
    print(f"  mutation_model:   {args.mutation_model}")

    # Convergence: build optional early-stop callbacks. GEPA composes these with
    # the max_metric_calls budget stopper via CompositeStopper(mode="any"), so the
    # run halts on whichever condition trips first.
    stop_callbacks = []
    if args.patience and args.patience > 0:
        from gepa.utils import NoImprovementStopper

        stop_callbacks.append(NoImprovementStopper(args.patience))
        print(
            f"  patience:         {args.patience} rounds (no full-valset improvement)"
        )
    if args.converge_threshold and args.converge_threshold > 0:
        from gepa.utils import ScoreThresholdStopper

        stop_callbacks.append(ScoreThresholdStopper(args.converge_threshold))
        print(f"  converge_thresh:  {args.converge_threshold:.3f} difficulty")
    print()

    from gepa.api import optimize as gepa_optimize

    result = gepa_optimize(
        stop_callbacks=stop_callbacks or None,
        seed_candidate=seed_candidate,
        trainset=tasks,
        valset=tasks,
        adapter=adapter,
        custom_candidate_proposer=custom_proposer,
        candidate_selection_strategy=args.selection_strategy,
        frontier_type=args.frontier_type,
        reflection_minibatch_size=args.minibatch_size,
        max_metric_calls=args.max_metric_calls,
        skip_perfect_score=False,
        perfect_score=1.0,
        cache_evaluation=True,
        run_dir=args.run_dir,
        seed=args.seed,
        raise_on_exception=False,
        display_progress_bar=True,
        module_selector="all",
    )

    # 6. Post-hoc analysis
    print(f"\n{'=' * 60}")
    print("  GEPA OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")

    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx]
    best_persona = result.candidates[best_idx]["persona_text"]

    print(f"  Candidates explored: {result.num_candidates}")
    print(f"  Total metric calls:  {result.total_metric_calls}")
    print(f"  Exploration-best idx:  {best_idx}")
    print(
        f"  Exploration-best difficulty: {best_score:.3f} "
        f"(agent success rate ≈ {1 - best_score:.1%})"
    )
    print(
        f"\n  NOTE: This is the 'exploration_best' — GEPA's reprep-50 single-trial"
        f"\n  best. Run certify_gepa_candidates.py for proper certification."
    )

    # Per-candidate summary
    print(f"\n  {'idx':>3}  {'score':>7}  {'parent':>6}  {'persona_text[:60]'}")
    print(f"  {'---':>3}  {'-----':>7}  {'------':>6}  {'-' * 60}")
    for i, (cand, score) in enumerate(
        zip(result.candidates, result.val_aggregate_scores)
    ):
        parents = result.parents[i]
        parent_str = str(parents[0]) if parents and parents[0] is not None else "seed"
        text_preview = cand["persona_text"][:60].replace("\n", " ")
        marker = " <-- BEST" if i == best_idx else ""
        print(f"  {i:>3}  {score:>7.3f}  {parent_str:>6}  {text_preview}{marker}")

    # Save exploration-best persona
    out_dir = Path(args.run_dir)
    best_path = out_dir / "exploration_best_persona.md"
    best_path.write_text(best_persona)

    summary = {
        "exploration_best_idx": best_idx,
        "exploration_best_difficulty": best_score,
        "exploration_best_agent_success_rate": 1.0 - best_score,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "run_dir": args.run_dir,
        "seed_persona": args.seed_persona,
        "config": {
            "max_metric_calls": args.max_metric_calls,
            "minibatch_size": args.minibatch_size,
            "frontier_type": args.frontier_type,
            "selection_strategy": args.selection_strategy,
            "agent_llm": args.agent_llm,
            "user_type": args.user_type,
            "user_llm": args.user_llm,
            "mutation_model": args.mutation_model,
            "domains": args.domains,
        },
        "all_scores": result.val_aggregate_scores,
        "candidates_json": str(out_dir / "candidates.json"),
    }
    write_json(out_dir / "gepa_summary.json", summary)

    # Save all candidates for certification
    candidates_out = []
    for i, (cand, score) in enumerate(
        zip(result.candidates, result.val_aggregate_scores)
    ):
        parents = result.parents[i]
        candidates_out.append(
            {
                "idx": i,
                "persona_text": cand["persona_text"],
                "exploration_score": score,
                "parent": parents[0] if parents and parents[0] is not None else None,
                "is_exploration_best": i == best_idx,
            }
        )
    write_json(out_dir / "candidates.json", candidates_out)

    print(f"\n  Exploration-best -> {best_path}")
    print(f"  Candidates       -> {out_dir / 'candidates.json'}")
    print(f"  Summary          -> {out_dir / 'gepa_summary.json'}")

    # Materialize exploration-best as a usable .skill file (namespaced per run
    # so parallel seeds don't overwrite each other's exploration-best).
    from lineage_loop import materialize

    best_pid = f"{args.persona_id}_exploration_best"
    materialize(best_pid, best_persona)
    print(f"  Skill file       -> {PERSONAS_DIR / f'opt_{best_pid}.skill'}")
    print(
        f"\n  Next: python experiments/persona_search/scripts/certify_gepa_candidates.py "
        f"--run-dirs {args.run_dir}"
    )


if __name__ == "__main__":
    main()
