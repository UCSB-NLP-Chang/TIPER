#!/usr/bin/env python3
"""Adversarial user-persona search on VitaBench via GEPA.

Mirrors scripts/persona_opt/gepa_runner_sweetrl.py: gepa.optimize() drives a
parallel-generational, per-instance-Pareto search over a `persona_text`
(injected into the user simulator) to MINIMIZE the agent's rubric reward. GEPA's
score = 1 - reward, so higher = harder = better.

    seed  : {"persona_text": ""}  (no-persona start -> genesis mode at first proposal)
    search: gepa.optimize(trainset, valset, adapter, custom_candidate_proposer)
            pareto / instance frontier, reflection_minibatch_size = minibatch,
            MaxCandidateProposalsStopper(iters), cache_evaluation=True
    final : evaluate best persona on the TEST set vs no-persona baseline

Setting (user-specified):
    task split 30 train / 30 val / 40 test  (single domain = 100 tasks; the
        requested 30/30/50 = 110 exceeds one domain's 100, so test is trimmed to
        40 to keep train/val/test disjoint)
    agent / user  = gpt-5.4 high reasoning  (gpt-5.4 routes to /v1/responses)
    mutation      = gpt-5.5 xhigh reasoning
    minibatch 10 x 2 trials ; val 2 trials
    start from no-persona (empty seed persona -> genesis mode at first proposal)
    max_steps 100 ; iterations 20

Usage:
    nohup .venv/bin/python -u scripts/vita_pgepa.py --domain delivery \
        --run-dir data/persona_opt/vita_pgepa_runs/run_delivery \
        > /tmp/vita_pgepa_delivery.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# gepa is appended to sys.path by vita_persona_adapter's import block below; but
# make the injection explicit here too so direct `from gepa...` works regardless
# of import order.
_TAU_SITE = Path(__file__).resolve().parents[2] / ".venv" / "lib"
for _c in sorted(_TAU_SITE.glob("python3.*/site-packages")):
    if str(_c) not in sys.path:
        sys.path.append(str(_c))

from vita_persona_adapter import (  # noqa: E402
    VitaPersonaAdapter,
    VitaTask,
    build_custom_proposer,
)

from gepa.api import optimize as gepa_optimize  # noqa: E402
from gepa.utils import MaxCandidateProposalsStopper  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
RUN_DIR_DEFAULT = REPO_DIR / "data" / "persona_opt" / "vita_pgepa_runs"


# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------


def split_tasks(domain: str, train_size: int, val_size: int, test_size: int,
                language: str, seed: int):
    """Load all task ids for a domain, shuffle deterministically, split."""
    from vita.run import get_tasks
    all_tasks = get_tasks(domain, None, None, language)
    ids = [t.id for t in all_tasks]
    rng = random.Random(seed)
    rng.shuffle(ids)
    need = train_size + val_size + test_size
    if len(ids) < need:
        raise RuntimeError(f"domain {domain} has {len(ids)} tasks, need {need}")
    train = ids[:train_size]
    val = ids[train_size:train_size + val_size]
    test = ids[train_size + val_size:train_size + val_size + test_size]
    return train, val, test


def to_vitatasks(ids: list[str], domain: str) -> list[VitaTask]:
    return [VitaTask(task_id=tid, domain=domain) for tid in ids]


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_reward(adapter: VitaPersonaAdapter, tasks: list[VitaTask],
                    persona_text: str, label: str, run_dir: Path) -> dict:
    """Run the adapter on a task set with a given persona; report avg reward.

    Captures and dumps conversation traces so the offline judge can score
    validity without re-simulating. The ONLINE JUDGE is disabled here — test-set
    rewards must be the RAW agent reward (the offline judge handles validity
    separately); gating test scores to zero on severe violations would corrupt
    the no-persona vs best-persona comparison."""
    print(f"\n  [{label}] running {len(tasks)} tasks x {adapter._num_trials} trials (capture_traces)...")
    _prev_judge = adapter._online_judge_enabled
    adapter._online_judge_enabled = False
    try:
        eb = adapter.evaluate(tasks, {"persona_text": persona_text}, capture_traces=True)
    finally:
        adapter._online_judge_enabled = _prev_judge
    # score = 1 - reward  ->  reward = 1 - score
    rewards = [1.0 - s for s in eb.scores]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    fully_correct = sum(1 for r in rewards if r >= 1.0)
    print(f"  [{label}] n={len(tasks)} avg_reward={avg_reward:.3f} "
          f"fully_correct={fully_correct}/{len(tasks)} "
          f"(difficulty={1-avg_reward:.3f})")
    # dump traces for the offline judge
    if eb.trajectories:
        cdir = run_dir / "conversations" / f"test_{label}"
        cdir.mkdir(parents=True, exist_ok=True)
        for tr in eb.trajectories:
            rec = {"task_id": tr.task_id, "domain": tr.domain, "reward": tr.reward,
                   "termination_reason": tr.termination_reason, "outcome": tr.outcome.value,
                   "messages": tr.messages}
            (cdir / f"{tr.task_id}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2))
    return {"label": label, "avg_reward": avg_reward,
            "avg_difficulty": _mean(eb.scores), "fully_correct": fully_correct,
            "n": len(tasks), "rewards": rewards, "scores": eb.scores}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # parse llm_args from CLI strings to dicts
    for attr in ("agent_llm_args", "user_llm_args", "evaluator_llm_args"):
        v = getattr(args, attr)
        if isinstance(v, str):
            setattr(args, attr, json.loads(v) if v else {})

    train_ids, val_ids, test_ids = split_tasks(
        args.domain, args.train_size, args.val_size, args.test_size,
        args.language, args.seed)
    train_tasks = to_vitatasks(train_ids, args.domain)
    val_tasks = to_vitatasks(val_ids, args.domain)
    test_tasks = to_vitatasks(test_ids, args.domain)
    print(f"vita_pgepa: domain={args.domain} split train={len(train_ids)}/"
          f"val={len(val_ids)}/test={len(test_ids)}  run_dir={run_dir}")
    print(f"  agent={args.agent_llm}(high) user={args.user_llm}(high) "
          f"mut={args.mutation_model}(xhigh) eval={args.evaluator_llm}")
    print(f"  iters={args.iters} minibatch={args.minibatch} trials={args.num_trials} "
          f"max_steps={args.max_steps} max_metric_calls={args.max_metric_calls}")

    # save config so the offline judge can reconstruct the test split + models
    config = {
        "domain": args.domain, "seed": args.seed, "train_size": args.train_size,
        "val_size": args.val_size, "test_size": args.test_size,
        "num_trials": args.num_trials, "max_steps": args.max_steps,
        "agent_llm": args.agent_llm, "agent_llm_args": args.agent_llm_args,
        "user_llm": args.user_llm, "user_llm_args": args.user_llm_args,
        "mutation_model": args.mutation_model, "mutation_edit_pct": args.mutation_edit_pct,
        "evaluator_llm": args.evaluator_llm, "language": args.language,
        "persona_id_base": args.persona_id_base, "iters": args.iters,
        "minibatch": args.minibatch,
    }
    (run_dir / "pgepa_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    adapter = VitaPersonaAdapter(
        persona_id=args.persona_id_base,
        agent_llm=args.agent_llm, agent_llm_args=args.agent_llm_args,
        user_llm=args.user_llm, user_llm_args=args.user_llm_args,
        max_steps=args.max_steps, max_errors=args.max_errors,
        max_duration=args.max_duration, num_trials=args.num_trials,
        evaluator_llm=args.evaluator_llm, evaluator_llm_args=args.evaluator_llm_args,
        enable_think=args.enable_think, language=args.language, domain=args.domain,
        max_concurrency=args.max_concurrency, seed_base=args.seed_base,
        timeout_attack_threshold=args.timeout_attack_threshold,
        online_judge=args.online_judge, online_judge_model=args.online_judge_model,
        judge_maj=args.judge_maj,
        severe_gate_threshold=args.severe_gate_threshold,
        online_judge_concurrency=args.online_judge_concurrency,
    )
    proposer = build_custom_proposer(
        args.mutation_model, args.mutation_edit_pct,
        mutation_context="enriched", scoring_domain=args.domain,
        max_steps=args.max_steps,
    )

    seed_candidate = {"persona_text": ""}
    stoppers = [MaxCandidateProposalsStopper(args.iters)]

    print("\nStarting GEPA optimization (no-persona start)...\n")
    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=train_tasks,
        valset=val_tasks,
        adapter=adapter,
        custom_candidate_proposer=proposer,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        reflection_minibatch_size=args.minibatch,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=stoppers,
        skip_perfect_score=False,
        perfect_score=1.0,
        cache_evaluation=True,
        run_dir=str(run_dir),
        seed=args.seed,
        raise_on_exception=False,
        display_progress_bar=True,
        module_selector="all",
    )

    # ---- save candidates + best persona ----
    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx]
    best_persona = result.candidates[best_idx]["persona_text"]

    candidates_out = []
    for i, (cand, score) in enumerate(zip(result.candidates, result.val_aggregate_scores)):
        parents = result.parents[i]
        candidates_out.append({
            "idx": i, "persona_text": cand["persona_text"],
            "val_difficulty": float(score), "val_reward": float(1.0 - score),
            "parent": parents[0] if parents and parents[0] is not None else None,
            "is_best": i == best_idx,
        })
    with open(run_dir / "candidates.json", "w") as fb:
        json.dump(candidates_out, fb, ensure_ascii=False, indent=2)
    (run_dir / "best_persona.md").write_text(best_persona)

    print(f"\n{'=' * 60}\n  GEPA COMPLETE\n{'=' * 60}")
    print(f"  candidates explored: {result.num_candidates}")
    print(f"  total metric calls:  {result.total_metric_calls}")
    print(f"  best idx: {best_idx}  val difficulty={best_score:.3f} "
          f"(agent reward≈{1-best_score:.1%})")
    print(f"  best persona -> {run_dir / 'best_persona.md'}")

    # ---- TEST set: best persona vs no-persona baseline ----
    if args.test_size > 0:
        # test trials can differ from search trials (search uses --num-trials,
        # test uses --test-trials if given, else --num-trials). The online judge
        # is disabled inside evaluate_reward regardless, so test reward is raw.
        test_trials = args.test_trials if args.test_trials is not None else args.num_trials
        adapter._num_trials = test_trials
        print(f"\n{'=' * 60}\n  TEST-SET EVALUATION (n={len(test_tasks)}, "
              f"{test_trials} trials)\n{'=' * 60}")
        print("  No-persona baseline:")
        base = evaluate_reward(adapter, test_tasks, "", "no_persona", run_dir)
        print("  Best persona:")
        best = evaluate_reward(adapter, test_tasks, best_persona, "best_persona", run_dir)
        delta = base["avg_reward"] - best["avg_reward"]
        print(f"\n  Δ reward (no-persona - best) = {delta:+.3f}  "
              f"({'persona HARDER ✓' if delta > 0 else 'no improvement ✗'})")
        report = {
            "best_idx": best_idx, "best_val_difficulty": float(best_score),
            "test_trials": test_trials,
            "no_persona": base, "best_persona": best,
            "delta_reward_no_minus_best": delta,
            "best_persona_path": str(run_dir / "best_persona.md"),
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
        }
        with open(run_dir / "test_report.json", "w") as fb:
            json.dump(report, fb, ensure_ascii=False, indent=2)
        print(f"  report -> {run_dir / 'test_report.json'}")
        print(f"\n  Next: scripts/vita_offline_judge.py --run-dir {run_dir}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="VitaBench GEPA adversarial persona search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--domain", default="delivery",
                    help="delivery | instore | ota (single domain, 100 tasks)")
    ap.add_argument("--run-dir", default=str(RUN_DIR_DEFAULT / "run_delivery"))
    ap.add_argument("--persona-id-base", default="vita_pgepa")
    ap.add_argument("--seed", type=int, default=300)

    # task split (30/30/40; 30/30/50=110 > 100 single-domain)
    ap.add_argument("--train-size", type=int, default=30)
    ap.add_argument("--val-size", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=40)

    # optimization
    ap.add_argument("--iters", type=int, default=20, help="max candidate proposals")
    ap.add_argument("--minibatch", type=int, default=10)
    ap.add_argument("--max-metric-calls", type=int, default=2000)

    # models (gpt-5.4 agent + user: routed to /v1/responses via models.yaml
    # `endpoint: responses`; mutation gpt-5.5 xhigh is tool-free; evaluator
    # non-reasoning to avoid invalid_prompt on long support transcripts)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--user-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--mutation-model", default="gpt-5.5")
    ap.add_argument("--mutation-edit-pct", default="30-50%")
    ap.add_argument("--evaluator-llm", default="gpt-5.2",
                    help="VitaBench rubric scorer model (unified gpt-5.2 with the validity judge)")
    ap.add_argument("--evaluator-llm-args", default="{}")

    # sim
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--test-trials", type=int, default=None,
                    help="trials for the final test-set eval; default = --num-trials. "
                         "Set to e.g. 3 to run search at --num-trials but test at 3.")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-errors", type=int, default=100000)
    ap.add_argument("--max-duration", type=float, default=0)
    ap.add_argument("--enable-think", action="store_true", default=True)
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--seed-base", type=int, default=300)
    ap.add_argument("--timeout-attack-threshold", type=float, default=0.15)

    # online conversation-behavior judge (tau parallel_gepa severe gate).
    # Enabled by default during search: failed sims are judged on 4 severe
    # categories; severe_rate > threshold -> candidate gated (scores zeroed).
    # Disabled automatically during test-set eval (see evaluate_reward).
    ap.add_argument("--online-judge", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--online-judge-model", default="gpt-5.2",
                    help="model for the online severe judge (unified with tau/lic/sweetrl)")
    ap.add_argument("--severe-gate-threshold", type=float, default=0.05,
                    help="severe_rate = severe/n_noninfra above this gates the candidate "
                         "(unified with tau's rate gate, 5%)")
    ap.add_argument("--online-judge-concurrency", type=int, default=8)
    ap.add_argument("--judge-maj", type=int, default=3,
                    help="validity-judge votes per judged trial, strict majority "
                         "(unified across tau/vita/lic/sweetrl; 1 = single judgment)")

    # concurrency
    ap.add_argument("--max-concurrency", type=int, default=16)
    return ap.parse_args()


if __name__ == "__main__":
    main()
