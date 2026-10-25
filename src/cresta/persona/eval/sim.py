"""Simulation evaluation primitives for parallel GEPA.

Extracted from parallel_gepa.py — adapter construction, dry-run fakes,
reuse cache, and eval convenience wrappers.
"""

from __future__ import annotations

import glob as _glob
import hashlib
import json

from gepa.core.adapter import EvaluationBatch

from cresta.persona.core._common import read_json
from cresta.persona.eval.adapter import TauPersonaAdapter
from cresta.persona.eval.scoring import (
    Outcome,
    TauEvalArtifacts,
    TauOutput,
    TauTask,
    TauTrajectory,
)


def build_tasks_from_split(
    split_path: str, which: str, domains: list[str] | None = None
) -> list[TauTask]:
    """Build a TauTask list from a train/val/test split file.

    Split format: {"train": {"telecom": [ids...]}, "val": {...}, "test": {...}}.
    `which` is one of train/val/test. `domains` filters which domains to include
    (None = all domains present in that split section)."""
    data = json.load(open(split_path))
    section = data[which]
    doms = domains if domains else list(section)
    tasks: list[TauTask] = []
    for dk in doms:
        for tid in section.get(dk, []):
            tasks.append(TauTask(task_id=tid, domain=dk))
    return tasks


def make_child_adapter(
    args,
    persona_id: str,
    always_judge_minibatch: bool = True,
    judge_valset: bool | None = None,
    num_trials_override: int | None = None,
    reuse_cache: dict | None = None,
    scoring_mode_override: str | None = None,
    mutation_context: str = "basic",
) -> TauPersonaAdapter:
    """Mirror gepa_runner's adapter construction, overriding only persona_id.
    Distinct persona_id => disjoint opt_{id}.skill + sim dir."""
    return TauPersonaAdapter(
        persona_id=persona_id,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        user_type=args.user_type,
        user_llm=args.user_llm,
        effort=args.effort,
        use_bedrock=args.use_bedrock,
        bedrock_profile=args.bedrock_profile,
        anthropic_base_url=getattr(args, "anthropic_base_url", None),
        anthropic_api_key=getattr(args, "anthropic_api_key", None),
        max_concurrency=args.max_concurrency,
        sim_timeout=args.sim_timeout,
        max_steps=args.max_steps,
        num_trials=num_trials_override
        if num_trials_override is not None
        else args.num_trials,
        max_infra_retries=args.max_infra_retries,
        retrieval_config=args.retrieval_config,
        domains=args.domains,
        judge_model=args.judge_model,
        judge_concurrency=args.judge_concurrency,
        judge_maj=args.judge_maj,
        always_judge_minibatch=always_judge_minibatch,
        minibatch_judge_cap=20,
        judge_valset=args.judge_valset if judge_valset is None else judge_valset,
        judge_failed_only=True,
        severe_gate_rate=args.severe_gate_rate,
        valset_judge_max_failures=args.valset_judge_max_failures,
        reuse_cache=reuse_cache,
        scoring_mode=scoring_mode_override
        if scoring_mode_override is not None
        else getattr(args, "scoring_mode", "mean"),
        mutation_context=mutation_context,
    )


def _stable_unit(s: str) -> float:
    """Deterministic float in [0,1) from a string (process-independent)."""
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def fake_evaluate(batch: list[TauTask], text: str, capture: bool) -> EvaluationBatch:
    """Synthetic EvaluationBatch: per-(instance, text) difficulty in {0,0.5,1.0};
    ~1-in-10 instances infeasible to exercise the validity-gate path."""
    outputs, scores, objective_scores = [], [], []
    trajectories: list[TauTrajectory] | None = [] if capture else None
    text_key = hashlib.md5(text.encode()).hexdigest()[:8]
    gate = _stable_unit("gate::" + text_key) < 0.10
    for t in batch:
        inst = f"{t.domain}::{t.task_id}"
        d = [0.0, 0.5, 1.0][int(_stable_unit(inst + "::" + text_key) * 3) % 3]
        if gate:
            d = 0.0
        reward = 1.0 - d
        outcome = Outcome.GENUINE_AGENT_FAILURE if d > 0 else Outcome.AGENT_SUCCESS
        outputs.append(TauOutput(t.task_id, t.domain, reward, "user_stop", outcome))
        scores.append(d)
        objective_scores.append(
            {
                "difficulty": d,
                "validity": 1.0,
                "validity_known": 1.0,
                "timeout_safety": 1.0,
                "semantic": 1.0,
                "infra_error": 0.0,
                "feasible": 0.0 if gate else 1.0,
            }
        )
        if trajectories is not None:
            trajectories.append(
                TauTrajectory(
                    t.task_id,
                    t.domain,
                    reward,
                    "user_stop",
                    [{"role": "user", "content": "(dry-run)"}],
                    outcome,
                )
            )
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=objective_scores,
    )


def build_reuse_cache(globs: list[str], allowed_ids) -> dict:
    """Build a gen-0 warm-start cache {task_id: {"sims": [...], "task": {...}}}
    from PRIOR results.json files (cross-run reuse of the SEED's sims).

    Only tasks in ``allowed_ids`` (the current split) are kept; the first file
    that supplies a usable (non-infra, reward-bearing) sim for a task wins. The
    seed eval then skips those tasks and only simulates the misses. Reuse is
    sound only because the seed persona is identical across runs — never apply
    this to mutated children.
    """
    allowed = set(allowed_ids)
    cache: dict[str, dict] = {}
    files: list[str] = []
    for pat in globs:
        files.extend(sorted(_glob.glob(pat)))
    for fp in files:
        try:
            doc = read_json(fp)
        except Exception:
            continue
        task_by_id = {t.get("id"): t for t in doc.get("tasks", [])}
        sims_by_id: dict[str, list] = {}
        for sim in doc.get("simulations", []):
            sims_by_id.setdefault(sim.get("task_id"), []).append(sim)
        for tid, sims in sims_by_id.items():
            if tid not in allowed or tid in cache:
                continue
            usable = [
                s
                for s in sims
                if (s.get("reward_info") or {}).get("reward") is not None
            ]
            if usable:
                cache[tid] = {"sims": usable, "task": task_by_id.get(tid)}
    return cache


def eval_one(
    args,
    persona_id: str,
    batch: list[TauTask],
    text: str,
    capture: bool,
    num_trials_override: int | None = None,
    reuse_cache: dict | None = None,
    scoring_mode_override: str | None = None,
) -> EvaluationBatch:
    if args.dry_run:
        return fake_evaluate(batch, text, capture)
    adapter = make_child_adapter(
        args,
        persona_id,
        num_trials_override=num_trials_override,
        reuse_cache=reuse_cache,
        scoring_mode_override=scoring_mode_override,
    )
    return adapter.evaluate(batch, {"persona_text": text}, capture_traces=capture)


def eval_one_difficulty_only(
    args,
    persona_id: str,
    batch: list[TauTask],
    text: str,
    num_trials_override: int | None = None,
    reuse_cache: dict | None = None,
    scoring_mode_override: str | None = None,
) -> tuple[EvaluationBatch, TauEvalArtifacts | None]:
    """Full-set difficulty screen without validity judge or trace capture."""
    if args.dry_run:
        return fake_evaluate(batch, text, False), None
    adapter = make_child_adapter(
        args,
        persona_id,
        always_judge_minibatch=False,
        judge_valset=False,
        num_trials_override=num_trials_override,
        reuse_cache=reuse_cache,
        scoring_mode_override=scoring_mode_override,
    )
    artifacts = adapter.run_simulations(batch, {"persona_text": text})
    eb = adapter.aggregate_artifacts(
        batch, artifacts, capture_traces=False, should_judge=False, failed_only=False
    )
    return eb, artifacts


def eval_existing_with_validity(
    args,
    persona_id: str,
    batch: list[TauTask],
    text: str,
    artifacts: TauEvalArtifacts | None,
    num_trials_override: int | None = None,
    scoring_mode_override: str | None = None,
) -> EvaluationBatch:
    """Run failed-only validity judge + capture traces on an existing sim pass."""
    if args.dry_run or artifacts is None:
        return fake_evaluate(batch, text, True)
    adapter = make_child_adapter(
        args,
        persona_id,
        always_judge_minibatch=False,
        judge_valset=False,
        num_trials_override=num_trials_override,
        scoring_mode_override=scoring_mode_override,
    )
    return adapter.aggregate_artifacts(
        batch, artifacts, capture_traces=True, should_judge=True, failed_only=True
    )
