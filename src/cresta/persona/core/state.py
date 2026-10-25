"""Run-state persistence: write/load pgepa_state.json and rebuild eval batches.

Extracted from parallel_gepa.py — write_state, write_candidates_json,
_rebuild_eval_batch, _collect_train_traces, and load_resume_state.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from types import SimpleNamespace

from gepa.core.adapter import EvaluationBatch

from cresta.persona.core._common import SIMULATIONS_DIR, read_json, write_json
from cresta.persona.core.candidate import Candidate
from cresta.persona.eval.scoring import Outcome

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_state(
    run_dir: Path,
    candidates: list[Candidate],
    instance_ids: list[str],
    gen_logs: list[dict],
    best_idx: int,
    args,
    invalid_cases: list[dict] | None = None,
) -> None:
    state = {
        "config": {
            "iterations": args.iterations,
            "minibatch": args.minibatch,
            "tie_tol": args.tie_tol,
            "seed": args.seed,
            "parent_selection": "pareto_instance_win (gepa.gepa_utils)",
            "min_dissimilarity": args.min_dissimilarity,
            "dissimilarity_metric": "layered: 1 - word-3gram Jaccard (>=min_dissimilarity) "
            "+ embedding cosine (<=max_semantic_similarity) vs all sampled personas",
            "embedding_model": (None if args.no_embedding else args.embedding_model),
            "max_semantic_similarity": args.max_semantic_similarity,
            "max_mutation_attempts": args.max_mutation_attempts,
            "no_task_specific_mutation": True,
            "lint_persona": args.lint_persona,
            "lint_model": (args.lint_model or args.mutation_model)
            if args.lint_persona
            else None,
            "lint_maj": args.lint_maj,
            "few_shot_examples": "mutation override + linter; invalid-case registry reused",
            "persona_id_base": args.persona_id_base,
            "mutation_edit_pct": args.mutation_edit_pct,
            "max_tasks": args.max_tasks,
            "search_score": "difficulty; validity/feasible are gates",
            "acceptance_rule": "minibatch>=parent, then full valset eval + feasible gate",
            "judge_failed_only": True,
            "agent_llm": args.agent_llm,
            "user_type": args.user_type,
            "user_llm": args.user_llm,
            "mutation_model": args.mutation_model,
            "mutation_effort": args.mutation_effort,
            "mutation_context": getattr(args, "mutation_context", "basic"),
            "reflect_model": getattr(args, "reflect_model", None),
            "reflect_effort": getattr(args, "reflect_effort", "medium"),
            "domains": args.domains,
            "dry_run": args.dry_run,
        },
        "instance_ids": instance_ids,
        "candidates": [
            {
                "idx": c.idx,
                "gen": c.gen,
                "parent": c.parent,
                "mutation_direction": c.direction,
                "dissimilarity": c.dissimilarity,
                "agg": c.agg,
                "difficulty_agg": c.difficulty_agg,
                "validity_agg": c.validity_agg,
                "feasible": c.feasible,
                "subscores": c.subscores,
                "difficulty_subscores": c.difficulty_subscores,
                "validity_subscores": c.validity_subscores,
                "persona_text": c.persona_text,
                "train_subscores": c.train_subscores,
                "train_agg": c.train_agg,
            }
            for c in candidates
        ],
        "generations": gen_logs,
        "invalid_cases": invalid_cases or [],
        "best_idx": best_idx,
        "best_agg": candidates[best_idx].agg if candidates else None,
    }
    write_json(run_dir / "pgepa_state.json", state)


def write_candidates_json(
    run_dir: Path, candidates: list[Candidate], best_idx: int
) -> None:
    """Exact certify_gepa_candidates.py schema (confirmed candidates only)."""
    rows = [
        {
            "idx": c.idx,
            "persona_text": c.persona_text,
            "exploration_score": c.agg if c.agg is not None else 0.0,
            "difficulty_score": c.difficulty_agg
            if c.difficulty_agg is not None
            else 0.0,
            "validity_score": c.validity_agg if c.validity_agg is not None else 0.0,
            "parent": c.parent,
            "mutation_direction": c.direction,
            "is_exploration_best": c.idx == best_idx,
        }
        for c in candidates
        if c.subscores is not None
    ]
    write_json(run_dir / "candidates.json", rows)


# ---------------------------------------------------------------------------
# Eval-batch reconstruction from disk
# ---------------------------------------------------------------------------


def _rebuild_eval_batch(
    persona_id_base: str, gen: int, domains: list[str] | None = None
) -> EvaluationBatch | None:
    """Try to rebuild an EvaluationBatch from on-disk results.json for a candidate.

    Outcome is derived from reward + termination_reason (same logic as
    classify_outcome but without judge info, which is acceptable for trace
    extraction — the proposer only needs success/failure classification).
    """
    sim_root = SIMULATIONS_DIR / "gepa_opt"
    suffix = "seed" if gen == 0 else "k00"
    prefix = f"{persona_id_base}_g{gen:02d}_{suffix}"
    eval_dir = sim_root / prefix
    if not eval_dir.is_dir():
        return None

    eval_subdirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("eval_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not eval_subdirs:
        return None

    trajectories = []
    for sub in eval_subdirs[:1]:
        for domain_dir in sub.iterdir():
            if not domain_dir.is_dir():
                continue
            if domains and domain_dir.name not in domains:
                continue
            rp = domain_dir / "results.json"
            if not rp.exists():
                continue
            try:
                data = _json.loads(rp.read_text())
            except Exception as _e:
                print(f"  [rebuild_eb] parse error {rp}: {_e}")
                continue
            for sim in data.get("simulations", []):
                term = sim.get("termination_reason", "")
                reward = (sim.get("reward_info") or {}).get("reward")
                if term in ("timeout", "max_steps"):
                    outcome = Outcome.TIMEOUT_OR_MAX_STEPS
                elif reward is not None and float(reward) >= 1.0:
                    outcome = Outcome.AGENT_SUCCESS
                elif reward is not None:
                    outcome = Outcome.GENUINE_AGENT_FAILURE
                else:
                    outcome = Outcome.UNKNOWN
                msgs = sim.get("messages", [])
                traj = SimpleNamespace(
                    task_id=sim.get("task_id", "?"),
                    domain=domain_dir.name,
                    outcome=outcome,
                    messages=msgs,
                    termination_reason=term,
                )
                trajectories.append(traj)

    if not trajectories:
        return None
    return EvaluationBatch(outputs=[], scores=[], trajectories=trajectories)


def _collect_train_traces(
    persona_id_base: str,
    parent_gen: int,
    task_ids: set[str],
    domains: list[str] | None = None,
) -> EvaluationBatch | None:
    """Collect parent's traces on specific train tasks from disk.

    Searches across all eval directories for this parent (seed_train for gen 0,
    k00 for children) and filters trajectories to only those whose task_id
    matches *task_ids* (bare ids, no domain:: prefix).
    """
    sim_root = SIMULATIONS_DIR / "gepa_opt"
    # candidate directories that may hold train-task rollouts for this parent
    candidate_dirs = []
    if parent_gen == 0:
        candidate_dirs.append(sim_root / f"{persona_id_base}_g00_seed_train")
        candidate_dirs.append(sim_root / f"{persona_id_base}_g00_seed")
    else:
        candidate_dirs.append(sim_root / f"{persona_id_base}_g{parent_gen:02d}_k00")

    trajectories = []
    seen_task_ids: set[str] = set()
    for eval_dir in candidate_dirs:
        if not eval_dir.is_dir():
            continue
        eval_subdirs = sorted(
            [
                d
                for d in eval_dir.iterdir()
                if d.is_dir() and d.name.startswith("eval_")
            ],
            key=lambda d: d.name,
            reverse=True,
        )
        if not eval_subdirs:
            continue
        for sub in eval_subdirs[:1]:
            for domain_dir in sub.iterdir():
                if not domain_dir.is_dir():
                    continue
                if domains and domain_dir.name not in domains:
                    continue
                rp = domain_dir / "results.json"
                if not rp.exists():
                    continue
                try:
                    data = _json.loads(rp.read_text())
                except Exception:
                    continue
                for sim in data.get("simulations", []):
                    tid = sim.get("task_id", "?")
                    if tid not in task_ids or tid in seen_task_ids:
                        continue
                    seen_task_ids.add(tid)
                    term = sim.get("termination_reason", "")
                    reward = (sim.get("reward_info") or {}).get("reward")
                    if term in ("timeout", "max_steps"):
                        outcome = Outcome.TIMEOUT_OR_MAX_STEPS
                    elif reward is not None and float(reward) >= 1.0:
                        outcome = Outcome.AGENT_SUCCESS
                    elif reward is not None:
                        outcome = Outcome.GENUINE_AGENT_FAILURE
                    else:
                        outcome = Outcome.UNKNOWN
                    trajectories.append(
                        SimpleNamespace(
                            task_id=tid,
                            domain=domain_dir.name,
                            outcome=outcome,
                            messages=sim.get("messages", []),
                            termination_reason=term,
                        )
                    )

    if not trajectories:
        return None
    return EvaluationBatch(outputs=[], scores=[], trajectories=trajectories)


def load_resume_state(
    run_dir: Path,
    instance_ids: list[str],
    persona_id_base: str | None = None,
    domains: list[str] | None = None,
) -> tuple[list[Candidate], list[str], list[dict], list[dict], int] | None:
    """Restore (candidates, seen_personas, invalid_cases, gen_logs, start_iter) from
    a prior pgepa_state.json. Returns None if there is nothing to resume from.

    When ``persona_id_base`` is provided, attempts to rebuild eval_batch from
    on-disk results.json so that resumed candidates carry traces for mutation.
    `seen_personas` is rebuilt from confirmed candidates only; rejected children
    (whose text was never persisted) no longer count toward the diversity gate.
    """
    state_p = run_dir / "pgepa_state.json"
    if not state_p.exists():
        return None
    state = read_json(state_p)
    raw = state.get("candidates") or []
    if not raw:
        return None

    saved_ids = state.get("instance_ids") or []
    if set(saved_ids) != set(instance_ids):
        raise SystemExit(
            f"--resume: valset mismatch (saved {len(saved_ids)} instances vs current "
            f"{len(instance_ids)}). Use the same --domains/--max-tasks, or a fresh --run-dir."
        )

    candidates: list[Candidate] = []
    for c in sorted(raw, key=lambda c: c["idx"]):
        cand = Candidate(
            idx=c["idx"],
            persona_text=c["persona_text"],
            parent=c.get("parent"),
            gen=c.get("gen", 0),
            direction=c.get("mutation_direction"),
            dissimilarity=c.get("dissimilarity"),
        )
        cand.subscores = c.get("subscores")
        cand.difficulty_subscores = c.get("difficulty_subscores")
        cand.train_subscores = c.get("train_subscores")
        cand.train_agg = c.get("train_agg")
        cand.validity_subscores = c.get("validity_subscores")
        cand.agg = c.get("agg")
        cand.difficulty_agg = c.get("difficulty_agg")
        cand.validity_agg = c.get("validity_agg")
        cand.feasible = c.get("feasible", True)
        if persona_id_base:
            cand.eval_batch = _rebuild_eval_batch(persona_id_base, cand.gen, domains)
        else:
            cand.eval_batch = None
        candidates.append(cand)

    seen_personas = [c.persona_text for c in candidates]
    invalid_cases = list(state.get("invalid_cases") or [])
    gen_logs = list(state.get("generations") or [])
    completed = max((g.get("gen", 0) for g in gen_logs), default=0)
    return candidates, seen_personas, invalid_cases, gen_logs, completed + 1
