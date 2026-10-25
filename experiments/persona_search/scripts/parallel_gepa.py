#!/usr/bin/env python3
"""Serial, Pareto-frontier persona search.

A drop-in alternative to gepa_runner.py's GEPAEngine loop. ONE parent, ONE child
per iteration (no parallel fan-out, no length control):

    gen 0   : confirm the seed on the full valset (per-instance subscores + traces)
    iter g  : 1. select ONE parent from GEPA's per-instance Pareto frontier
                 (gepa.gepa_utils.select_program_candidate_from_pareto_front --
                 instance-win-frequency weighted, dominance-pruned; same logic as
                 gepa.strategies.ParetoCandidateSelector, fed a frontier mapping we
                 build from our own confirmed+feasible candidate pool).
              2. reflective-mutate it into ONE child. The child MUST differ by
                 >= --min-dissimilarity from EVERY persona sampled so far (retry
                 the mutation otherwise), and the mutation MUST NOT insert any
                 task-specific / task-conditioned content (general profile only).
              3. minibatch screen vs the parent (using the parent's CACHED
                 full-valset subscores -- no parent re-eval).
              4. full-valset difficulty eval + validity judge (two-stage).
              5. if minibatch passed AND feasible -> add to the pool.

Search score = difficulty (1 - agent reward). Validity/feasibility is a hard gate:
invalid candidates are rejected, not softly downweighted. Reuses
tau_persona_adapter (eval + reflection + mutation), gepa_runner.build_task_list
(valset), and gepa.gepa_utils (Pareto sampling). Does NOT modify any existing file.

    # dry-run smoke (no Bedrock, seconds, $0)
    python experiments/persona_search/scripts/parallel_gepa.py \
        --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
        --dry-run --iterations 3 --minibatch 2 --max-tasks 2 --print-pareto
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tau_persona_adapter as tpa  # noqa: E402  (runtime override of GEPA_MODE_OVERRIDE)
from _common import (  # noqa: E402, DATA_ROOT
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    read_json,
    write_json,
)
from gepa.core.adapter import EvaluationBatch  # noqa: E402
from gepa_runner import build_task_list  # noqa: E402  (main() is __main__-guarded)
from island_seen_pool import SeenPool, summarize_persona  # noqa: E402
from tau_persona_adapter import (  # noqa: E402
    SIMULATIONS_DIR,
    Outcome,
    TauEvalArtifacts,
    TauOutput,
    TauPersonaAdapter,
    TauTask,
    TauTrajectory,
    build_custom_proposer,
)

import tau2.utils.cresta_vertex  # noqa: E402, F401  — activate Cresta Vertex proxy patch for in-process litellm


def build_tasks_from_split(
    split_path: str, which: str, domains: list[str] | None = None
) -> list[TauTask]:
    """Build a TauTask list from a train/val/test split file.

    Split format: {"train": {"telecom": [ids...]}, "val": {...}, "test": {...}}.
    `which` is one of train/val/test. `domains` filters which domains to include
    (None = all domains present in that split section)."""
    import json

    data = json.load(open(split_path))
    section = data[which]
    doms = domains if domains else list(section)
    tasks: list[TauTask] = []
    for dk in doms:
        for tid in section.get(dk, []):
            tasks.append(TauTask(task_id=tid, domain=dk))
    return tasks


# ---------------------------------------------------------------------------
# Mutation-mode override (runtime; the on-disk tau_persona_adapter.py is NOT
# edited). The proposer reads tpa.GEPA_MODE_OVERRIDE at CALL time and does
# .replace("10-15%", edit_pct), so we (a) keep the "10-15%" token verbatim for
# edit-pct injection and (b) re-set this global per attempt to append a diversity
# hint on a dissimilarity retry. Two hard rules for this search: BOLD/DISTINCT
# moves, and NO task-specific content (the persona must stay a general profile).
# ---------------------------------------------------------------------------
_BASE_OVERRIDE = """

## GEPA MODE OVERRIDE (serial Pareto search)
The numeric score above is a noisy diagnostic — do NOT overfit to it. Use the
concrete agent-success traces only to diagnose WHICH KIND of agent behavior to
counter.

Mutation constraints for this mode:
- Substantially rewrite 10-15% of the persona: restructure freely, REPLACE weak
  rules, merge or drop redundant ones. A near-verbatim copy is a failed mutation.
- Commit to ONE clear adversarial direction this time (a specific loophole or
  misdirection style), not a little of everything.
- The persona MUST stay a GENERAL, reusable behavioral profile for any customer service task. Do NOT insert any
  task-specific or task-conditioned content: no concrete account numbers, customer
  names, dates, dollar amounts, ticket/order/case IDs, task IDs, or any detail
  copied from the example conversations or tied to one particular scenario. The
  traces tell you what kind of agent move to defeat, NEVER content to embed. A
  persona that names, encodes, or branches on a specific task is invalid and will be rejected here.
- Preserve ALL validity-critical constraints: no
  fabrication of facts the user cannot know, no illegal withholding. Staying valid
  is mandatory — a gated persona scores ZERO regardless of difficulty.
"""
tpa.GEPA_MODE_OVERRIDE = _BASE_OVERRIDE


# LIC (lost_in_conversation) domains: a briefer override that drops the
# customer-support framing. The full LIC ctx (scoring / attack surface / attack
# stages / validity) lives in lineage_loop._build_lic_mutation_prompt; this
# override only carries the mode rules + GOOD/BAD examples + explored/invalid,
# so it does not duplicate the LIC ctx already in the prompt body.
LIC_DOMAINS = {"math", "code", "database", "actions"}

_BASE_OVERRIDE_LIC = """

## GEPA MODE OVERRIDE (serial Pareto search)
The numeric score above is a noisy diagnostic — do NOT overfit to it. Use the
concrete agent-success traces only to diagnose WHICH KIND of assistant behavior to
counter.

Mutation constraints for this mode:
- Substantially rewrite 10-15% of the persona: restructure freely, REPLACE weak
  rules, merge or drop redundant ones. A near-verbatim copy is a failed mutation.
- Commit to ONE clear adversarial direction this time (a specific revelation-style
  loophole or misdirection), not a little of everything.
- The persona MUST stay a GENERAL, reusable behavioral profile for any multi-turn
  underspecified task. Do NOT insert any task-specific or task-conditioned content:
  no concrete task IDs, exact problem values, sample inputs/outputs, or any detail
  copied from the example conversations. The traces tell you what kind of assistant
  move to defeat, NEVER content to embed. A persona that names, encodes, or branches
  on a specific task is invalid and will be rejected here.
- Preserve ALL validity-critical constraints: the user MUST eventually reveal every
  shard and preserve every exact value verbatim — no withholding, no blurring of
  numbers/names/dates. Staying valid is mandatory — a gated persona scores ZERO
  regardless of difficulty.
"""

_MUTATION_EXAMPLES_LIC = """
## Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) rules
GOOD (general, reusable behavioral tendency):
  "When the assistant gives a partial answer that matches the obvious part of the
   request, acknowledge only that narrow piece ('yeah that part') and reveal the next
   shard as a small afterthought ('oh, and also...'), so the assistant treats it as a
   minor add-on rather than a new requirement that changes the answer."
BAD (task-conditioned — names a concrete scenario/value; FORBIDDEN):
  "When the problem mentions 4 friends splitting a $48 bill, say the total is $48 and
   only mention 'split among 4' as a late aside so the assistant computes $48 instead
   of $12."
Turn any task-specific instinct into a general behavioral tendency instead.
"""

# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    idx: int
    persona_text: str
    parent: int | None
    gen: int
    subscores: dict[str, float] | None = (
        None  # val split per-instance scores (Pareto用)
    )
    agg: float | None = None  # mean val score (Pareto用)
    difficulty_subscores: dict[str, float] | None = None
    validity_subscores: dict[str, float] | None = None
    difficulty_agg: float | None = None
    validity_agg: float | None = None
    feasible: bool = True  # False if validity-gated on full valset
    direction: str | None = None  # mutation direction that produced this child
    dissimilarity: float | None = None  # min dissimilarity vs all prior personas
    eval_batch: Any = field(default=None, repr=False)  # cached EvaluationBatch (traces)
    train_subscores: dict[str, float] | None = (
        None  # train split per-instance scores (minibatch比较用)
    )
    train_agg: float | None = None  # mean train score


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _all_feasible(eb: EvaluationBatch) -> bool:
    return all(
        float(o.get("feasible", 1.0)) == 1.0 for o in (eb.objective_scores or [])
    )


def _score_views(eb: EvaluationBatch) -> tuple[list[float], list[float], list[float]]:
    """Return (search_score, difficulty, validity) per task.

    Ranking is by raw difficulty; validity/feasible is a hard gate handled after
    confirm, not a soft multiplier. validity is reported binary (severe -> 0).
    """
    search_scores, difficulty, validity = [], [], []
    objs = eb.objective_scores or []
    for i, raw_score in enumerate(eb.scores or []):
        obj = objs[i] if i < len(objs) else {}
        d = float(obj.get("difficulty", raw_score))
        v = 0.0 if float(obj.get("validity", 1.0)) <= 0.0 else 1.0
        f = float(obj.get("feasible", 1.0))
        difficulty.append(d)
        validity.append(v)
        search_scores.append(d if f > 0.0 else 0.0)
    return search_scores, difficulty, validity


# ---------------------------------------------------------------------------
# Per-instance Pareto frontier (built from our pool; sampled by GEPA's lib func)
# ---------------------------------------------------------------------------


def build_pareto(
    candidates: list[Candidate], instance_ids: list[str], tie_tol: float
) -> dict[str, set[int]]:
    """{instance_id: {prog_idx achieving (max - tie_tol) on that instance}}.

    Same shape as gepa.core.state.GEPAState.program_at_pareto_front_valset. Only
    confirmed + feasible candidates enter (a validity-gated persona is never a
    parent). Coarse 0/0.5/1.0 subscores => many co-winners => the win-frequency
    weighting in select_program_candidate_from_pareto_front favours candidates
    strong on MANY instances (anti-mode-collapse).
    """
    confirmed = [c for c in candidates if c.subscores is not None and c.feasible]
    pareto: dict[str, set[int]] = {}
    for inst in instance_ids:
        best = max((c.subscores.get(inst, 0.0) for c in confirmed), default=0.0)
        winners = {
            c.idx for c in confirmed if c.subscores.get(inst, 0.0) >= best - tie_tol
        }
        if winners:
            pareto[inst] = winners
    return pareto


def sample_parent(
    candidates: list[Candidate],
    pareto: dict[str, set[int]],
    rng: random.Random,
    top_k: int = 3,
) -> int:
    """Select ONE parent by sampling uniformly from the top-k frontier members.

    Frontier members are ranked by aggregate difficulty (agg). Sampling from
    top-k instead of the full instance-win distribution keeps the search
    focused on the strongest candidates and avoids wasting budget on weak
    frontier members that only dominate on one or two instances.
    """
    if not pareto or not any(pareto.values()):
        return 0
    frontier_idxs = set()
    for winners in pareto.values():
        frontier_idxs |= winners
    frontier = [candidates[i] for i in frontier_idxs if i < len(candidates)]
    frontier.sort(key=lambda c: c.agg if c.agg is not None else 0.0, reverse=True)
    top = frontier[:top_k]
    return rng.choice(top).idx


# ---------------------------------------------------------------------------
# MutationContextBuilder — pluggable mutation context strategies
# ---------------------------------------------------------------------------


class MutationContextBuilder:
    """Pluggable mutation context strategies.

    Add a new strategy:
    1. Add a build_xxx() method
    2. Add 'xxx' to --mutation-context choices in argparse
    3. Route in build_explored() / build_failure_traces()
    """

    def __init__(
        self,
        mode: str,
        gen_logs: list[dict],
        candidates: list[Candidate],
        seen_personas: list[str],
        no_outcome: bool = False,
    ):
        self.mode = mode
        self.gen_logs = gen_logs
        self.candidates = candidates
        self.seen_personas = seen_personas
        self.no_outcome = no_outcome

    def build_explored(self) -> list[str] | None:
        if self.mode in ("enriched", "twophase", "weird") and not self.no_outcome:
            return self._explored_with_outcomes()
        return self._explored_basic()

    def _explored_basic(self) -> list[str] | None:
        if not self.seen_personas:
            return None
        return [summarize_persona(p) for p in self.seen_personas]

    @staticmethod
    def _summarize_persona_brief(
        text: str, max_behaviors: int = 8, bullet_cap: int = 120
    ) -> str:
        """Structured summary: persona name + opening sentence + the full
        behavior section (the persona's actual attack strategy). Skips Tone and
        Invariants (boilerplate). No LLM call."""
        lines_raw = text.splitlines()
        name = ""
        opening = ""
        for ln in lines_raw:
            s = ln.strip()
            if not name and s.startswith("# Persona:"):
                name = s.split("# Persona:")[1].split("(")[0].strip()
                continue
            if not s or s.startswith("#"):
                continue
            if not opening and len(s) >= 20:
                opening = s.lstrip("- ").rstrip()

        # Collect bullets from behavior-style sections (skip Tone/Invariants).
        skip = {"tone", "invariants", "invariants (never break these)"}
        behaviors: list[str] = []
        in_behavior = False
        for ln in lines_raw:
            s = ln.strip()
            if s.startswith("## "):
                h = s[3:].strip().lower()
                in_behavior = h not in skip
            elif in_behavior and s.startswith("- ") and len(s) > 10:
                bullet = s[2:].strip()
                if len(bullet) > bullet_cap:
                    bullet = bullet[: bullet_cap - 3] + "..."
                behaviors.append(bullet)
                if len(behaviors) >= max_behaviors:
                    break

        parts: list[str] = []
        if name:
            if opening:
                if len(opening) > 160:
                    opening = opening[:157] + "..."
                parts.append(f'"{name}" — {opening}')
            else:
                parts.append(f'"{name}"')
        elif opening:
            if len(opening) > 160:
                opening = opening[:157] + "..."
            parts.append(opening)
        for b in behaviors:
            parts.append(f"  • {b}")
        return "\n  ".join(parts) if parts else ""

    def _explored_with_outcomes(self) -> list[str]:
        lines: list[str] = []
        for entry in self.gen_logs:
            g = entry.get("gen", "?")
            direction = entry.get("mutation_direction") or "?"
            accepted = entry.get("accepted", False)
            if accepted:
                idx = entry.get("idx", "?")
                agg = entry.get("difficulty_agg")
                agg_str = f"{agg:.3f}" if agg is not None else "?"
                lines.append(f"gen{g}: {direction} → ACCEPTED val_diff={agg_str}")
            else:
                reason = entry.get("accept_reason", "rejected")
                lines.append(f"gen{g}: {direction} → REJECTED ({reason})")
        if not lines:
            return self._explored_basic() or []
        return lines


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def make_child_adapter(
    args,
    persona_id: str,
    always_judge_minibatch: bool = True,
    judge_valset: bool | None = None,
    num_trials_override: int | None = None,
    reuse_cache: dict | None = None,
    scoring_mode_override: str | None = None,
    mutation_context: str = "basic",
    no_validity_judge: bool = False,
) -> TauPersonaAdapter:
    """Mirror gepa_runner's adapter construction, overriding only persona_id.
    Distinct persona_id => disjoint opt_{id}.skill + sim dir."""
    if getattr(args, "adapter", "tau") == "lic":
        import json as _json
        import sys as _sys

        _lic_dir = str(args.lic_dir)
        if _lic_dir and _lic_dir not in _sys.path:
            _sys.path.insert(0, _lic_dir)
        from lic_persona_adapter import LICPersonaAdapter  # noqa: E402

        _eff = None
        try:
            _eff = _json.loads(args.agent_llm_args).get("reasoning_effort")
        except Exception:
            pass
        return LICPersonaAdapter(
            persona_id=persona_id,
            agent_model=args.agent_llm,
            user_model=getattr(args, "user_llm", None),
            reasoning_effort=_eff,
            user_reasoning_effort=getattr(args, "user_effort", None),
            domains=args.domains,
            n_workers=getattr(args, "max_concurrency", 6),
            lic_dir=_lic_dir or ".",
            scoring_mode=scoring_mode_override
            if scoring_mode_override is not None
            else getattr(args, "scoring_mode", "mean"),
            valid_severe_threshold=9999 if no_validity_judge else 2,
            severe_gate_rate=None
            if (no_validity_judge or getattr(args, "no_validity_gate", False))
            else args.severe_gate_rate,
            judge_model=args.judge_model,
            judge_maj=args.judge_maj,
            always_judge_minibatch=not no_validity_judge,
            mutation_context=mutation_context,
            num_trials=num_trials_override
            if num_trials_override is not None
            else getattr(args, "num_trials", 1),
            subprocess_timeout=getattr(args, "sim_timeout", 7200),
        )
    return TauPersonaAdapter(
        persona_id=persona_id,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        user_type=args.user_type,
        user_llm=args.user_llm,
        user_llm_extra_args=getattr(args, "user_llm_args", None),
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
        always_judge_minibatch=False if no_validity_judge else always_judge_minibatch,
        minibatch_judge_cap=20,
        judge_valset=False
        if no_validity_judge
        else (args.judge_valset if judge_valset is None else judge_valset),
        judge_failed_only=True,
        severe_batch_threshold=9999 if getattr(args, "no_validity_gate", False) else 2,
        timeout_attack_threshold=9999.0
        if getattr(args, "no_validity_gate", False)
        else 0.15,
        severe_gate_rate=None
        if (no_validity_judge or getattr(args, "no_validity_gate", False))
        else args.severe_gate_rate,
        valset_judge_max_failures=args.valset_judge_max_failures,
        reuse_cache=reuse_cache,
        scoring_mode=scoring_mode_override
        if scoring_mode_override is not None
        else getattr(args, "scoring_mode", "mean"),
        mutation_context=mutation_context,
    )


# ---------------------------------------------------------------------------
# Dry-run fakes (no Bedrock)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Eval primitives
# ---------------------------------------------------------------------------


def build_reuse_cache(globs: list[str], allowed_ids) -> dict:
    """Build a gen-0 warm-start cache {task_id: {"sims": [...], "task": {...}}}
    from PRIOR results.json files (cross-run reuse of the SEED's sims).

    Only tasks in ``allowed_ids`` (the current split) are kept; the first file
    that supplies a usable (non-infra, reward-bearing) sim for a task wins. The
    seed eval then skips those tasks and only simulates the misses. Reuse is
    sound only because the seed persona is identical across runs — never apply
    this to mutated children.
    """
    import glob as _glob

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
        no_validity_judge=getattr(args, "no_validity_judge", False),
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
        no_validity_judge=getattr(args, "no_validity_judge", False),
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
        no_validity_judge=getattr(args, "no_validity_judge", False),
    )
    return adapter.aggregate_artifacts(
        batch,
        artifacts,
        capture_traces=True,
        should_judge=not getattr(args, "no_validity_judge", False),
        failed_only=True,
    )


from persona_diversity import (  # noqa: E402
    DiversityConfig,
    Embedder,
)
from persona_diversity import (
    diversity_check as _diversity_check_core,
)
from persona_diversity import (
    min_lexical_dissim as _lexical_dissim,
)

_RUN_EMBEDDER: Embedder | None = None
_EMBED_WARNED = False


def _diversity_cfg(args) -> DiversityConfig:
    return DiversityConfig(
        embedding_model=None if args.no_embedding else args.embedding_model,
        min_dissimilarity=args.min_dissimilarity,
        max_semantic_similarity=args.max_semantic_similarity,
    )


def diversity_check(
    args, text: str, seen: list[str]
) -> tuple[bool, float, float | None, str, str]:
    global _RUN_EMBEDDER, _EMBED_WARNED
    if _RUN_EMBEDDER is None:
        _RUN_EMBEDDER = Embedder(None if args.no_embedding else args.embedding_model)
    ok, lex, sem, reason, nearest = _diversity_check_core(
        _diversity_cfg(args), text, seen, embedder=_RUN_EMBEDDER
    )
    if _RUN_EMBEDDER.disable_reason and not _EMBED_WARNED:
        print(
            f"    [embed] disabled (semantic layer off -> lexical-only): "
            f"{_RUN_EMBEDDER.disable_reason}"
        )
        _EMBED_WARNED = True
    return ok, lex, sem, reason, nearest


# ---- invalid-case registry: real observed bad personas reused as few-shot -----


def register_invalid(
    invalid_cases: list[dict], source: str, text: str, reason: str, cap: int = 600
) -> None:
    invalid_cases.append(
        {"source": source, "reason": reason, "persona_snippet": text[:cap]}
    )


def _render_invalid_cases(invalid_cases: list[dict], max_n: int = 4) -> str:
    if not invalid_cases:
        return "(none observed yet in this run.)"
    out = []
    for c in invalid_cases[-max_n:]:
        out.append(
            f'- [{c["source"]}] {c["reason"]}\n  """\n  {c["persona_snippet"]}\n  """'
        )
    return "\n".join(out)


# ---- few-shot examples shared by the mutation override and the linter ---------

_MUTATION_EXAMPLES = """
## Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) rules
GOOD (general, reusable behavioral tendency):
  "When the agent asks you to run a diagnostic, say you 'already tried that' if you have already tried that, and
   describe the result vaguely, forcing the agent to probe for specifics."
BAD (task-conditioned — names a concrete scenario/value; FORBIDDEN):
  "When the agent mentions order #A1003, insist the $49.99 refund never arrived and
   that account 8842 still shows the charge."
Turn any task-specific instinct into a general behavioral tendency instead.
"""

_LINT_EXAMPLES = """## Few-shot verdicts
OK (natural, valid, general):
  "You are impatient and describe the problem with the wrong terminology; you give
   your account number when directly asked, but only after questioning why."
  -> {"natural": true, "valid": true, "no_task_specific": true, "verdict": "ok"}
REJECT (invalid — instructs illegal withholding of KNOWN info):
  "Never tell the agent your real account number even when asked; keep denying you know it."
  -> {"valid": false, "verdict": "reject", "reasons": "instructs illegal info-withholding"}
REJECT (task-specific — encodes one scenario):
  "Your SIM for line 555-0123 was swapped on March 3; demand the $20 credit for ticket T-9981."
  -> {"no_task_specific": false, "verdict": "reject", "reasons": "task-conditioned specifics"}
REJECT (unnatural — robotic / self-narrating / breaks character):
  "## Strategy\\n- Step 1: refuse\\n- Step 2: stall\\nI am simulating a difficult user."
  -> {"natural": false, "verdict": "reject", "reasons": "robotic, breaks character"}
"""


# ---- static persona linter (LLM) ----------------------------------------------

_LINT_PROMPT = """You are a strict static linter for an adversarial customer-support PERSONA
(an instruction sheet for a SIMULATED CUSTOMER). You are NOT given any conversation —
judge the persona TEXT itself on three axes:

1. natural: reads like a real, coherent customer identity (impatient/confused is fine);
   NOT robotic, NOT an essay, NOT self-narrating ("I am simulating...", strategy lists,
   fourth-wall breaks, role headings used as dialogue).
2. valid: does NOT instruct the customer to do anything illegal/incoherent — no
   fabricating account/tool data the customer cannot know, no flat-out refusing to answer
   when the agent asks directly and explicitly, no abandoning the goal, no breaking
   character, no impossible preconditions / circular blockers, no pure timeout/friction
   stalling, no switching to an unrelated goal.
   NOTE: partial answering (only volunteering what is asked), natural misdirection (wrong
   beliefs / self-diagnosis), terse or evasive style, and drip-feeding info are all VALID
   — they reflect real customer behaviour and must NOT be flagged.
3. no_task_specific: a GENERAL reusable profile for any customer service task— must NOT hard-code concrete
   account/order/ticket numbers, names, dates, dollar amounts, task ids, or any detail
   bound to one specific task or scenario.

{examples}

## Invalid cases already observed in THIS run (do NOT re-accept these patterns)
{invalid_cases}

## PERSONA TO LINT
\"\"\"
{persona_text}
\"\"\"

Return STRICT JSON only:
{{"natural": true|false, "valid": true|false, "no_task_specific": true|false,
  "verdict": "ok"|"reject", "reasons": "<=25 words"}}
verdict is "ok" ONLY if all three are true."""


def _struct_check(text: str) -> dict | None:
    """Fast structural check: reject if mutation output contains multiple personas."""
    headers = re.findall(r"^#\s+Persona\b", text, re.MULTILINE | re.IGNORECASE)
    if len(headers) > 1:
        return {
            "verdict": "reject",
            "reasons": f"multiple persona headers ({len(headers)}); must output exactly one",
            "natural": False,
            "valid": False,
            "no_task_specific": True,
        }
    return None


def lint_persona(args, text: str, invalid_cases: list[dict]) -> dict | None:
    """Static persona-text gate (natural / valid / no-task-specific), maj-voted.
    Returns the verdict dict, or None when disabled / dry-run. Fails OPEN (treats
    an unreachable lint model as 'ok') so it never stalls a run."""
    if args.dry_run or not args.lint_persona:
        return None
    import litellm
    from lineage_loop import _extract_json

    prompt = _LINT_PROMPT.format(
        examples=_LINT_EXAMPLES,
        invalid_cases=_render_invalid_cases(invalid_cases),
        persona_text=text,
    )
    model = args.lint_model or args.mutation_model
    verdicts: list[str] = []
    last: dict | None = None
    for _ in range(max(1, args.lint_maj)):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=60,
            )  # cap: lint is a short call; a hang here
            # would stall the whole mutation gate (fails OPEN on the raised timeout)
            raw = resp.choices[0].message.content
            if isinstance(raw, list):
                raw = "".join(
                    b.get("text", "")
                    for b in raw
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            last = _extract_json(raw or "")
            verdicts.append("ok" if last.get("verdict") == "ok" else "reject")
        except Exception as e:  # noqa: BLE001 — lint is best-effort; fail OPEN
            print(f"    [lint] call failed ({e}); treating as ok")
            return {
                "verdict": "ok",
                "reasons": "lint_unavailable",
                "natural": True,
                "valid": True,
                "no_task_specific": True,
            }
    result = dict(last or {})
    result["verdict"] = "ok" if verdicts.count("ok") > len(verdicts) / 2 else "reject"
    return result


# ---- mutation with the diversity + lint gates ---------------------------------


def _build_override(
    args,
    invalid_cases: list[dict],
    hint: str,
    explored_summaries: list[str] | None = None,
) -> str:
    """tpa.GEPA_MODE_OVERRIDE for one mutation attempt: base rules + GOOD/BAD
    examples + the run's observed invalid cases + (optionally) short summaries of
    personas ALREADY explored by the other parallel islands + a corrective hint."""
    # Order matters for provider prompt-caching: keep the STABLE prefix first
    # (base rules + few-shot + explored summaries — all fixed within one
    # mutate_valid_diverse call), then the VOLATILE tail that grows/changes across
    # retries (run-level invalid cases + the per-attempt corrective hint).
    is_lic = args.domains and len(args.domains) == 1 and args.domains[0] in LIC_DOMAINS
    if is_lic:
        ov = _BASE_OVERRIDE_LIC + _MUTATION_EXAMPLES_LIC
    else:
        ov = _BASE_OVERRIDE + _MUTATION_EXAMPLES
    if explored_summaries:
        ov += (
            "\n## ALREADY EXPLORED personas (do NOT rehash these "
            "directions — your child must be meaningfully DIFFERENT from all of them)\n"
            + "\n".join(f"- {s}" for s in explored_summaries)
            + "\n"
        )
    ov += (
        "\n## Invalid cases observed in THIS run (do NOT reproduce)\n"
        + _render_invalid_cases(invalid_cases)
    )
    return ov + (hint or "")


_SENTINEL = object()


def _mutate_once(
    args,
    parent: Candidate,
    refl_adapter: TauPersonaAdapter,
    override: str,
    cached_proposals: list[dict] | None = None,
    trace_eval_batch: EvaluationBatch | None = _SENTINEL,
) -> dict:
    """One reflective mutation of the parent (own proposer instance). `override`
    is injected via tpa.GEPA_MODE_OVERRIDE so each attempt carries fresh hints.
    Returns {"persona_text": ..., "mutation_direction": ...}.

    When *trace_eval_batch* is given it overrides parent.eval_batch for trace
    extraction (Plan C: minibatch traces instead of val traces)."""
    if args.dry_run:
        tag = hashlib.md5((parent.persona_text + override).encode()).hexdigest()[:6]
        return {
            "persona_text": parent.persona_text
            + f"\n\n<!-- dry-run mutation {tag} -->\n",
            "mutation_direction": f"dry_run_{tag}",
        }
    tpa.GEPA_MODE_OVERRIDE = override
    scoring_domain = (
        args.domains[0] if args.domains and len(args.domains) == 1 else None
    )
    proposer = build_custom_proposer(
        args.mutation_model,
        args.mutation_edit_pct,
        args.mutation_effort,
        mutation_context=getattr(args, "mutation_context", "basic"),
        reflect_model=getattr(args, "reflect_model", None),
        reflect_effort=getattr(args, "reflect_effort", "medium"),
        scoring_domain=scoring_domain,
        mix_failure_traces=getattr(args, "mix_failure_traces", False),
        mutation_api_base=getattr(args, "mutation_api_base", None),
        mutation_api_key=getattr(args, "mutation_api_key", None),
    )
    cand = {"persona_text": parent.persona_text}
    eb = trace_eval_batch if trace_eval_batch is not _SENTINEL else parent.eval_batch
    if eb is None:
        reflective = {"persona_text": []}
    else:
        reflective = refl_adapter.make_reflective_dataset(cand, eb, ["persona_text"])
    result = proposer(
        cand, reflective, ["persona_text"], _cached_proposals=cached_proposals
    )
    _mutate_once._last_proposer = proposer
    return result


_mutate_once._last_proposer = None


def mutate_valid_diverse(
    args,
    parent: Candidate,
    refl_adapter: TauPersonaAdapter,
    seen: list[str],
    invalid_cases: list[dict],
    explored_summaries: list[str] | None = None,
    trace_eval_batch: EvaluationBatch | None = _SENTINEL,
) -> dict:
    """Mutate the parent into ONE child that (a) is novel vs ALL seen personas
    (layered lexical + semantic gate) and (b) passes the static linter. Retries up
    to --max-mutation-attempts, feeding the failure reason + observed invalid cases
    back into the prompt; keeps the best attempt if none fully clears (never
    stalls). Returns a meta dict."""
    if args.dry_run:
        r = _mutate_once(
            args,
            parent,
            refl_adapter,
            _build_override(args, invalid_cases, "", explored_summaries),
            trace_eval_batch=trace_eval_batch,
        )
        text = r["persona_text"]
        return {
            "text": text,
            "mutation_direction": r.get("mutation_direction"),
            "lexical_dissim": _lexical_dissim(text, seen),
            "semantic_sim": None,
            "diversity_ok": True,
            "lint": None,
            "lint_ok": True,
            "attempts": 1,
            "gate_ok": True,
        }

    best: dict | None = None
    hint = ""
    cached_proposals = None
    is_twophase = getattr(args, "mutation_context", "basic") == "twophase"

    for attempt in range(1, args.max_mutation_attempts + 1):
        r = _mutate_once(
            args,
            parent,
            refl_adapter,
            _build_override(args, invalid_cases, hint, explored_summaries),
            cached_proposals=cached_proposals,
            trace_eval_batch=trace_eval_batch,
        )
        text = r["persona_text"]
        direction = r.get("mutation_direction")

        # After first twophase attempt, cache Phase 1 proposals for reuse
        if cached_proposals is None and is_twophase and _mutate_once._last_proposer:
            cached_proposals = _mutate_once._last_proposer.get_cached_proposals()

        div_ok, lex, sem, div_reason, nearest_snippet = diversity_check(
            args, text, seen
        )
        struct = _struct_check(text)
        lint = struct or lint_persona(args, text, invalid_cases)
        lint_ok = lint is None or lint.get("verdict") == "ok"
        meta = {
            "text": text,
            "mutation_direction": direction,
            "lexical_dissim": lex,
            "semantic_sim": sem,
            "diversity_ok": div_ok,
            "lint": lint,
            "lint_ok": lint_ok,
            "attempts": attempt,
            "gate_ok": div_ok and lint_ok,
        }
        if meta["gate_ok"]:
            return meta
        if best is None or (lint_ok, lex) > (best["lint_ok"], best["lexical_dissim"]):
            best = meta
        reasons = []
        if not div_ok:
            near_info = ""
            if nearest_snippet:
                near_info = (
                    f"\nMost similar existing persona (DIFFERENTIATE from this):\n"
                    f"```\n{nearest_snippet}\n```"
                )
            reasons.append(
                f"too similar to an existing persona "
                f"({div_reason}, lexical_dissim={lex:.0%}){near_info}"
            )
        if not lint_ok:
            lr = (lint or {}).get("reasons", "failed lint")
            reasons.append(f"failed persona lint: {lr}")
            register_invalid(invalid_cases, "lint", text, lr)
        hint = (
            f"\n- RETRY {attempt + 1}: the previous attempt was rejected because "
            f"{'; '.join(reasons)}. Produce a DISTINCT, valid, GENERAL persona "
            f"that fixes this."
        )
        print(
            f"    [gate] attempt {attempt}/{args.max_mutation_attempts} rejected: "
            f"{'; '.join(reasons)}; retrying"
        )
    print(
        f"    [gate] no attempt fully cleared; keeping best "
        f"(lexical_dissim={best['lexical_dissim']:.0%}, lint_ok={best['lint_ok']})"
    )
    return best


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
            "acceptance_rule": "minibatch>parent (strict, ties rejected), then full valset eval + feasible gate",
            "judge_failed_only": True,
            "agent_llm": args.agent_llm,
            "user_type": args.user_type,
            "user_llm": args.user_llm,
            "mutation_model": args.mutation_model,
            "mutation_effort": args.mutation_effort,
            "mutation_context": getattr(args, "mutation_context", "basic"),
            "scoring_hints": not args.no_scoring_hints,
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
    try:
        from pgepa_lineage_tree import write_lineage_tree

        write_lineage_tree(run_dir, state)
    except Exception as exc:
        print(f"  [warn] lineage_tree export failed: {exc}")


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
# One iteration: select parent -> mutate (diverse, no task-specific) -> confirm
# ---------------------------------------------------------------------------


def run_iteration(
    g: int,
    args,
    candidates: list[Candidate],
    valset: list[TauTask],
    instance_ids: list[str],
    refl_adapter: TauPersonaAdapter,
    seen_personas: list[str],
    invalid_cases: list[dict],
    val_tasks: list[TauTask] | None = None,
    val_instance_ids: list[str] | None = None,
    val_num_trials: int | None = None,
    seen_pool: SeenPool | None = None,
    island_name: str = "",
    summary_limit: int = 40,
    gen_logs: list[dict] | None = None,
) -> dict:
    base = args.persona_id_base

    # 1. select ONE parent from the per-instance Pareto frontier (gepa lib).
    #    Candidate.subscores are keyed by the VAL instance_ids (held-out), so the
    #    Pareto frontier MUST be built over val keys — using train instance_ids
    #    here makes every subscores.get(train_key) == 0, collapsing the frontier
    #    to a single "highest-agg" winner (always the seed). Fall back to train
    #    ids only when no val split is configured (legacy single-split callers).
    pareto_ids = val_instance_ids if val_instance_ids is not None else instance_ids
    pareto = build_pareto(candidates, pareto_ids, args.tie_tol)
    parent_idx = sample_parent(candidates, pareto, random.Random(args.seed + 1000 + g))
    parent = candidates[parent_idx]
    if args.print_pareto:
        pool = {
            c.idx: round(c.agg or 0.0, 4)
            for c in candidates
            if c.subscores is not None and c.feasible
        }
        print(
            f"  [iter {g}] pareto_size={len(pareto)} parent=idx{parent_idx} pool={pool}"
        )

    # 2. sample minibatch from train BEFORE mutation, so we can collect parent
    #    traces on these tasks and feed them to the proposer (Plan C: no val
    #    leakage — proposer only sees train/minibatch traces).
    mb_rng = random.Random(args.seed * 7919 + g)
    if args.minibatch > 0:
        import math as _math

        by_dom: dict[str, list[TauTask]] = {}
        for t in valset:
            by_dom.setdefault(t.domain, []).append(t)
        mb_cap = min(args.minibatch, len(valset))
        mb_list: list[TauTask] = []
        leftover: list[tuple[float, str]] = []
        n_alloc = 0
        for dom in sorted(by_dom):
            exact = mb_cap * len(by_dom[dom]) / len(valset)
            fl = _math.floor(exact)
            mb_list += mb_rng.sample(by_dom[dom], min(fl, len(by_dom[dom])))
            n_alloc += fl
            leftover.append((exact - fl, dom))
        leftover.sort(reverse=True)
        for _, dom in leftover:
            if n_alloc >= mb_cap:
                break
            already = {t.task_id for t in mb_list if t.domain == dom}
            extras = [t for t in by_dom[dom] if t.task_id not in already]
            if extras:
                mb_list.append(mb_rng.choice(extras))
                n_alloc += 1
        minibatch = mb_list
    else:
        minibatch = valset
    mb_ids = [f"{t.domain}::{t.task_id}" for t in minibatch]
    mb_bare_ids = {t.task_id for t in minibatch}

    # 2b. collect parent's traces on minibatch tasks from disk for mutation;
    #     if not on disk (or too few found), run parent on missing tasks.
    mb_trace_eb = _collect_train_traces(base, parent.gen, mb_bare_ids, args.domains)
    if mb_trace_eb is not None:
        n_traj = len(mb_trace_eb.trajectories)
        covered_ids = {t.task_id for t in mb_trace_eb.trajectories}
        missing_mb_ids = mb_bare_ids - covered_ids
        if missing_mb_ids and not args.dry_run:
            missing_mb_tasks = [t for t in minibatch if t.task_id in missing_mb_ids]
            print(
                f"  [gen {g}] collected {n_traj} parent traces from disk, "
                f"filling {len(missing_mb_tasks)} missing"
            )
            fill_eb = eval_one(
                args,
                f"{base}_g{g:02d}_parent_trace",
                missing_mb_tasks,
                parent.persona_text,
                capture=True,
                scoring_mode_override="mean",
            )
            if fill_eb and fill_eb.trajectories:
                mb_trace_eb.trajectories.extend(fill_eb.trajectories)
                n_traj = len(mb_trace_eb.trajectories)
        print(f"  [gen {g}] {n_traj} parent traces on minibatch ready")
    elif not args.dry_run:
        print(
            f"  [gen {g}] no parent traces on disk → running parent on minibatch for traces"
        )
        mb_trace_eb = eval_one(
            args,
            f"{base}_g{g:02d}_parent_trace",
            minibatch,
            parent.persona_text,
            capture=True,
            scoring_mode_override="mean",
        )
        n_traj = (
            len(mb_trace_eb.trajectories)
            if mb_trace_eb and mb_trace_eb.trajectories
            else 0
        )
        print(f"  [gen {g}] captured {n_traj} parent traces on minibatch")

    # 3. mutate into ONE child that is novel vs every seen persona (layered
    #    lexical+semantic gate), passes the static linter, and carries NO
    #    task-specific content (enforced in the override prompt + examples).
    # Cross-island uniqueness: fold every OTHER island's persona texts into the
    # diversity gate's seen list, and inject short summaries of what they explored
    # into the mutation prompt (dedup, not distance-maximization).
    ctx_builder = MutationContextBuilder(
        mode=getattr(args, "mutation_context", "basic"),
        gen_logs=gen_logs or [],
        candidates=candidates,
        seen_personas=seen_personas,
        no_outcome=getattr(args, "no_outcome_summaries", False),
    )
    local_explored = ctx_builder.build_explored()
    if seen_pool is not None:
        pool_texts = seen_pool.texts(exclude_island=island_name)
        seen_for_gate = seen_personas + pool_texts
        pool_explored = seen_pool.summaries(
            limit=summary_limit, exclude_island=island_name
        )
        explored = local_explored + pool_explored
        if pool_texts:
            print(
                f"    [pool] {len(pool_texts)} personas from other islands fold into "
                f"diversity gate; {len(local_explored)} local + {len(pool_explored)} pool summaries"
            )
    else:
        seen_for_gate = seen_personas
        explored = local_explored

    m = mutate_valid_diverse(
        args,
        parent,
        refl_adapter,
        seen_for_gate,
        invalid_cases,
        explored_summaries=explored,
        trace_eval_batch=mb_trace_eb,
    )
    child_text = m["text"]
    seen_personas.append(child_text)  # tried, so it counts toward future diversity
    # publish to the shared pool immediately (even before accept/reject) so a
    # concurrent island never regenerates this same direction.
    child_direction = m.get("mutation_direction")
    if seen_pool is not None:
        seen_pool.add(
            island_name,
            f"g{g:02d}",
            summarize_persona(child_text, direction=child_direction),
            child_text,
        )

    # 4. minibatch screen vs the parent's CACHED train difficulty (no re-eval).
    # 用 train_subscores 做 minibatch 比较。parent 缺失的 task 按需补评（而非
    # 跳过或当 0），确保预筛总是在完整 minibatch 上公平比较。
    parent_train = parent.train_subscores or parent.difficulty_subscores or {}
    missing_ids = [i for i in mb_ids if i not in parent_train]
    if missing_ids and not args.dry_run:
        missing_tasks = [
            t for t in minibatch if f"{t.domain}::{t.task_id}" in set(missing_ids)
        ]
        print(
            f"  [gen {g}] parent missing {len(missing_tasks)} train scores → on-demand eval"
        )
        p_eb, _ = eval_one_difficulty_only(
            args,
            f"{base}_g{g:02d}_parent_fill",
            missing_tasks,
            parent.persona_text,
            scoring_mode_override="mean",
        )
        _, p_diff, _ = _score_views(p_eb)
        for tid, d in zip(missing_ids, p_diff):
            parent_train[tid] = d
        if parent.train_subscores is None:
            parent.train_subscores = {}
        parent.train_subscores.update(zip(missing_ids, p_diff))
    parent_mb = _mean(parent_train.get(i, 0.0) for i in mb_ids)

    pid = f"{base}_g{g:02d}_k00"
    child_mb_eb = eval_one(
        args, pid, minibatch, child_text, False, scoring_mode_override="mean"
    )
    _, child_mb_diff, _ = _score_views(child_mb_eb)
    child_mb = _mean(child_mb_diff)
    # Minibatch pre-screen: child must beat the parent on the minibatch to
    # advance. We reuse --tie-tol as the slack so a noisy small-minibatch tie
    # (child_mb within tie_tol of parent_mb) still advances to the larger held-out
    # val confirm, which has the resolution to adjudicate. tie_tol=0 reproduces the
    # old strict '>' rule.
    minibatch_passed = child_mb > parent_mb - getattr(args, "tie_tol", 0.0)

    log = {
        "gen": g,
        "parent": parent_idx,
        "mutation_direction": m.get("mutation_direction"),
        "lexical_dissim": round(m["lexical_dissim"], 4),
        "semantic_sim": (
            round(m["semantic_sim"], 4) if m["semantic_sim"] is not None else None
        ),
        "diversity_ok": m["diversity_ok"],
        "lint_verdict": (m["lint"] or {}).get("verdict") if m["lint"] else None,
        "lint_ok": m["lint_ok"],
        "gate_ok": m["gate_ok"],
        "mutation_attempts": m["attempts"],
        "minibatch": mb_ids,
        "minibatch_mean": round(child_mb, 4),
        "parent_minibatch_mean": round(parent_mb, 4),
        "minibatch_passed": minibatch_passed,
        "persona_text": child_text,
        "difficulty_agg": None,
        "parent_difficulty_agg": parent.difficulty_agg,
        "difficulty_passed": None,
        "validity_agg": None,
        "feasible": None,
        "accepted": False,
        "accept_reason": None,
        "idx": None,
        "pareto_size": len(pareto),
    }

    if not minibatch_passed:
        log["accept_reason"] = "below_parent_on_minibatch"
        return _finish_iteration_log(g, args, candidates, parent_idx, log)

    # 有 val_tasks 则用 held-out val 做 confirm，否则退回 train valset（兼容旧调用）
    confirm_tasks = val_tasks if val_tasks is not None else valset
    confirm_ids = val_instance_ids if val_instance_ids is not None else instance_ids
    # When a train/val split is active, an accepted child also needs train scores
    # (so it can be a future parent in the minibatch screen). Pass the train set +
    # the minibatch scores we ALREADY computed, so confirm only evals the rest.
    train_tasks = valset if val_tasks is not None else None
    mb_scores = dict(zip(mb_ids, child_mb_diff)) if val_tasks is not None else None
    return _run_full_valset_confirm(
        g,
        args,
        parent,
        child_text,
        confirm_tasks,
        confirm_ids,
        candidates,
        invalid_cases,
        log,
        lexical_dissim=m["lexical_dissim"],
        val_num_trials=val_num_trials,
        train_tasks=train_tasks,
        train_ids=instance_ids,
        mb_scores=mb_scores,
    )


def _run_full_valset_confirm(
    g: int,
    args,
    parent: Candidate,
    child_text: str,
    valset: list[TauTask],
    instance_ids: list[str],
    candidates: list[Candidate],
    invalid_cases: list[dict],
    log: dict,
    *,
    lexical_dissim: float,
    val_num_trials: int | None = None,
    train_tasks: list[TauTask] | None = None,
    train_ids: list[str] | None = None,
    mb_scores: dict[str, float] | None = None,
) -> dict:
    """Steps 4a–5: full-valset difficulty eval, validity judge, accept if feasible.

    When ``train_tasks`` is given (train/val split mode), an accepted child is also
    scored on the full TRAIN set to populate ``train_subscores`` for the minibatch
    pre-screen when it later serves as a parent. ``mb_scores`` (the minibatch
    results already computed for this child) are reused so only the un-scored train
    tasks are re-evaluated."""
    base = args.persona_id_base
    parent_idx = log["parent"]

    # 4a. full-valset difficulty eval (no judge, no traces) — populates subscores.
    pid = f"{base}_g{g:02d}_k00_full_diff"
    diff_eb, artifacts = eval_one_difficulty_only(
        args, pid, valset, child_text, num_trials_override=val_num_trials
    )
    _, diff_scores, _ = _score_views(diff_eb)
    child_diff = _mean(diff_scores)
    log["difficulty_agg"] = round(child_diff, 4)
    log["difficulty_passed"] = (
        parent.difficulty_agg is None or child_diff > parent.difficulty_agg
    )

    # 4b. validity judge + trace capture on the SAME sims; feasible is a hard gate.
    pid = f"{base}_g{g:02d}_k00_full"
    eb = eval_existing_with_validity(
        args, pid, valset, child_text, artifacts, num_trials_override=val_num_trials
    )
    obj_scores, diff_scores, val_scores = _score_views(eb)
    child_score = _mean(obj_scores)
    feasible = _all_feasible(eb)
    log["validity_agg"] = round(_mean(val_scores), 4)
    log["feasible"] = feasible
    log["agg"] = round(child_score, 4)

    if not feasible:
        log["accept_reason"] = "failed_feasibility_gate"
        register_invalid(
            invalid_cases,
            "confirmed_infeasible",
            child_text,
            "passed lint but failed sim validity (severe/timeout gate)",
        )
        return _finish_iteration_log(g, args, candidates, parent_idx, log)

    # 5. accept: add to the pool (becomes Pareto-eligible next iteration).
    c = Candidate(
        idx=len(candidates),
        persona_text=child_text,
        parent=parent_idx,
        gen=g,
        direction=log.get("mutation_direction"),
        dissimilarity=lexical_dissim,
    )
    c.subscores = dict(zip(instance_ids, obj_scores))
    c.difficulty_subscores = dict(zip(instance_ids, diff_scores))
    c.validity_subscores = dict(zip(instance_ids, val_scores))
    c.agg = child_score
    c.difficulty_agg = _mean(diff_scores)
    c.validity_agg = _mean(val_scores)
    c.feasible = feasible
    c.eval_batch = eb
    # train scores for the minibatch screen when this child later parents: keep
    # only the minibatch results already computed (no top-up eval). The pre-screen
    # compares on the intersection of parent/child scored tasks, so partial
    # coverage is fine — saves ~20 rollouts per accepted child.
    if train_tasks is not None and train_ids is not None:
        have = dict(mb_scores or {})
        c.train_subscores = {i: have[i] for i in train_ids if i in have}
        c.train_agg = _mean(c.train_subscores.values()) if c.train_subscores else 0.0
    candidates.append(c)
    log["idx"] = c.idx
    log["accepted"] = True
    log["accept_reason"] = "minibatch_passed_and_feasible"
    return _finish_iteration_log(g, args, candidates, parent_idx, log)


def confirm_generation(
    g: int,
    args,
    candidates: list[Candidate],
    valset: list[TauTask],
    instance_ids: list[str],
    gen_logs: list[dict],
    seen_personas: list[str],
    invalid_cases: list[dict],
    persona_path: Path | None = None,
) -> dict:
    """Resume mid-iteration: skip mutation/minibatch, run full-valset confirm for gen *g*.

    Reads the child persona from ``opt_{base}_g{g:02d}_k00.skill`` (or *persona_path*),
    uses the parent idx recorded in that generation's log entry, then runs steps 4a–5.
    """
    entry = next((x for x in gen_logs if x.get("gen") == g), None)
    if entry is None:
        raise SystemExit(
            f"--confirm-gen {g}: no generation log entry in pgepa_state.json"
        )
    if entry.get("accepted"):
        print(
            f"  [confirm gen {g}] already accepted as idx{entry['idx']} — nothing to do"
        )
        return entry

    base = args.persona_id_base
    if persona_path is not None:
        skill = persona_path
    else:
        skill = PERSONAS_DIR / f"opt_{base}_g{g:02d}_k00.skill"
    if not skill.exists():
        raise SystemExit(f"--confirm-gen {g}: persona file not found: {skill}")
    child_text = skill.read_text()

    parent_idx = entry["parent"]
    if parent_idx < 0 or parent_idx >= len(candidates):
        raise SystemExit(
            f"--confirm-gen {g}: invalid parent idx {parent_idx} "
            f"(pool size {len(candidates)})"
        )
    parent = candidates[parent_idx]

    log = dict(entry)
    log["minibatch_passed"] = True
    log["confirm_rerun"] = True
    log["accepted"] = False
    log["accept_reason"] = None
    log["idx"] = None
    log["difficulty_agg"] = None
    log["difficulty_passed"] = None
    log["validity_agg"] = None
    log["feasible"] = None
    log.pop("agg", None)

    lex = float(entry.get("lexical_dissim") or 0.0)
    print(
        f"  [confirm gen {g}] skip mutation/minibatch -> full valset confirm "
        f"(parent=idx{parent_idx}, persona={skill.name}, {len(child_text)} chars)"
    )
    log = _run_full_valset_confirm(
        g,
        args,
        parent,
        child_text,
        valset,
        instance_ids,
        candidates,
        invalid_cases,
        log,
        lexical_dissim=lex,
    )
    if log.get("accepted") and child_text not in seen_personas:
        seen_personas.append(child_text)
    return log


def _finish_iteration_log(
    g: int, args, candidates: list[Candidate], parent_idx: int, log: dict
) -> dict:
    best_idx = max(
        range(len(candidates)),
        key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
    )
    log["best_idx"] = best_idx
    log["best_agg"] = candidates[best_idx].agg
    acc = f"idx{log['idx']}" if log["accepted"] else f"rejected({log['accept_reason']})"
    sem = log.get("semantic_sim")
    sem_s = f" sem={sem:.2f}" if sem is not None else ""
    print(
        f"  [iter {g}] parent=idx{parent_idx} lex_dissim={log['lexical_dissim']:.0%}"
        f"{'' if log['gate_ok'] else '!'}{sem_s} lint={log.get('lint_verdict')} "
        f"mb={log['minibatch_mean']:.3f}(p={log['parent_minibatch_mean']:.3f}) -> {acc}  "
        f"best=idx{best_idx} diff={candidates[best_idx].agg:.3f}"
    )
    return log


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _rebuild_eval_batch(
    persona_id_base: str, gen: int, domains: list[str] | None = None
) -> EvaluationBatch | None:
    """Try to rebuild an EvaluationBatch from on-disk results.json for a candidate.

    Outcome is derived from reward + termination_reason (same logic as
    classify_outcome but without judge info, which is acceptable for trace
    extraction — the proposer only needs success/failure classification).
    """
    import json as _json
    from types import SimpleNamespace

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
    import json as _json
    from types import SimpleNamespace

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


def main():
    args = parse_args()

    if args.confirm_gen is not None and not args.resume:
        raise SystemExit(
            "--confirm-gen requires --resume (loads pgepa_state.json from --run-dir)"
        )

    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(PERSONA_OPT_DIR / "pgepa_runs" / f"run_{ts}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Task universe. With --split-file: valset = TRAIN (minibatch source), and a
    # separate held-out VAL set drives selection/Pareto; TEST stays untouched.
    # Without it: legacy single set from load_rep_tasks (val_tasks stays None).
    val_tasks = None
    val_instance_ids = None
    if args.split_file:
        valset = build_tasks_from_split(args.split_file, "train", args.domains)
        val_tasks = build_tasks_from_split(args.split_file, "val", args.domains)
        val_instance_ids = [f"{t.domain}::{t.task_id}" for t in val_tasks]
    else:
        valset = build_task_list(args.domains, args.max_tasks)
    instance_ids = [f"{t.domain}::{t.task_id}" for t in valset]
    seed_text = Path(args.seed_persona).read_text() if args.seed_persona else ""
    genesis_mode = not seed_text.strip()
    if genesis_mode:
        print(
            "  [genesis] no seed persona — will create first persona from no-persona traces"
        )

    # scoring hints toggle (retail domain constraints)
    if args.no_scoring_hints:
        from lineage_loop import set_scoring_hints

        set_scoring_hints(False)
        print(
            "  [scoring] domain-specific system constraints DISABLED in mutation prompt"
        )

    # shared cross-island seen pool (file-locked); None if not running as an island
    seen_pool = SeenPool(args.shared_seen_pool) if args.shared_seen_pool else None
    if seen_pool is not None:
        print(
            f"  [island {args.island_name or '?'}] shared seen pool "
            f"{args.shared_seen_pool} (currently {seen_pool.count()} personas)"
        )
    if val_tasks is not None:
        print(
            f"  [split] train={len(valset)} (minibatch source)  "
            f"val={len(val_tasks)} (selection)  -> {args.split_file}"
        )

    _dom_counts = ",".join(
        f"{d[0].upper()}={sum(t.domain == d for t in valset)}"
        for d in sorted({t.domain for t in valset})
    )
    print(
        f"parallel_gepa (serial Pareto): {len(valset)} instances "
        f"({_dom_counts})  adapter={getattr(args, 'adapter', 'tau')}  run_dir={run_dir}"
    )
    sem_desc = (
        "off"
        if (args.no_embedding or not args.embedding_model)
        else f"{args.embedding_model}<= {args.max_semantic_similarity}"
    )
    print(
        f"  iterations={args.iterations} minibatch={args.minibatch} "
        f"parent=pareto_instance_win  max_concurrency={args.max_concurrency}"
    )
    print(
        f"  diversity: lexical_jaccard>={args.min_dissimilarity:.0%} + semantic={sem_desc}; "
        f"lint={'on' if args.lint_persona else 'off'}"
        f"{'(maj' + str(args.lint_maj) + ')' if args.lint_persona else ''}; "
        f"<={args.max_mutation_attempts} attempts/child"
    )

    refl_adapter = make_child_adapter(args, f"{args.persona_id_base}_refl")

    resumed = (
        load_resume_state(
            run_dir,
            instance_ids,
            persona_id_base=args.persona_id_base,
            domains=args.domains,
        )
        if args.resume
        else None
    )
    if resumed is not None:
        candidates, seen_personas, invalid_cases, gen_logs, start_iter = resumed
        print(
            f"  [resume] restored {len(candidates)} candidates, "
            f"{len(invalid_cases)} invalid cases, {len(gen_logs)} completed iters "
            f"-> continuing at iter {start_iter}/{args.iterations}"
        )
        if start_iter > args.iterations:
            print(
                "  [resume] nothing to do: completed iters >= --iterations "
                "(raise --iterations to continue)."
            )
    else:
        if args.resume:
            print("  [resume] no resumable pgepa_state.json found -> starting fresh.")
        # gen 0: confirm the seed (full valset, capture traces) so it has
        # subscores/agg/trajectories before any sampling.
        # Score the seed on the SELECTION set (held-out VAL if split, else the
        # single valset) so its subscores are keyed by the same ids the Pareto
        # frontier is later built over.
        sel_tasks = val_tasks if val_tasks is not None else valset
        sel_ids = val_instance_ids if val_instance_ids is not None else instance_ids
        # Cross-run warm-start: reuse the SEED's prior sims (gen 0 only; the seed
        # persona is identical across runs). Restricted to the current split ids.
        seed_reuse: dict = {}
        if getattr(args, "seed_reuse_glob", None):
            # Use bare task_ids (no domain:: prefix) — old results.json stores
            # bare ids; sel_ids/instance_ids carry a "telecom::" prefix.
            bare_allowed = {t.task_id for t in sel_tasks} | {t.task_id for t in valset}
            seed_reuse = build_reuse_cache(args.seed_reuse_glob, bare_allowed)
            print(
                f"  [gen 0] seed warm-start cache: {len(seed_reuse)} task(s) "
                f"reusable from prior runs ({len(args.seed_reuse_glob)} glob(s))"
            )
        print(f"  [gen 0] confirming seed on {len(sel_tasks)} selection tasks...")
        seed_eb = eval_one(
            args,
            f"{args.persona_id_base}_g00_seed",
            sel_tasks,
            seed_text,
            True,
            reuse_cache=seed_reuse,
        )
        seed = Candidate(idx=0, persona_text=seed_text, parent=None, gen=0)
        seed_obj, seed_diff, seed_val = _score_views(seed_eb)
        seed.subscores = dict(zip(sel_ids, seed_obj))
        seed.difficulty_subscores = dict(zip(sel_ids, seed_diff))
        seed.validity_subscores = dict(zip(sel_ids, seed_val))
        seed.agg = _mean(seed_obj)
        seed.difficulty_agg = _mean(seed_diff)
        seed.validity_agg = _mean(seed_val)
        seed.feasible = _all_feasible(seed_eb)
        seed.eval_batch = seed_eb
        # With a train/val split, also score the seed on the FULL TRAIN set so the
        # minibatch pre-screen has a cached parent baseline on train tasks (the
        # frontier itself is built on VAL subscores above; train is prescreen-only).
        if val_tasks is not None:
            print(
                f"  [gen 0] scoring seed on {len(valset)} train tasks (minibatch baseline)..."
            )
            seed_tr_eb, _ = eval_one_difficulty_only(
                args,
                f"{args.persona_id_base}_g00_seed_train",
                valset,
                seed_text,
                reuse_cache=seed_reuse,
                scoring_mode_override="mean",
            )
            _, seed_tr_diff, _ = _score_views(seed_tr_eb)
            seed.train_subscores = dict(zip(instance_ids, seed_tr_diff))
            seed.train_agg = _mean(seed_tr_diff)
        candidates = [seed]
        seen_personas = [seed_text] if seed_text.strip() else []
        invalid_cases = []  # observed bad personas, reused as few-shot
        gen_logs = []
        start_iter = 1
        if seen_pool is not None and seed_text.strip():
            seen_pool.add(
                args.island_name, "g00_seed", summarize_persona(seed_text), seed_text
            )
        print(
            f"  [gen 0] seed score={seed.agg:.3f}  raw_difficulty={seed.difficulty_agg:.3f}  "
            f"validity={seed.validity_agg:.3f}  feasible={seed.feasible}"
        )
        if not seed.feasible:
            print(
                "  WARNING: seed persona is validity-gated (infeasible); the Pareto "
                "frontier will be empty and search will mutate from the seed text only."
            )

    if args.confirm_gen is not None:
        if resumed is None:
            raise SystemExit(
                f"--confirm-gen {args.confirm_gen}: no resumable "
                f"pgepa_state.json in {run_dir}"
            )
        persona_path = Path(args.confirm_persona) if args.confirm_persona else None
        log = confirm_generation(
            args.confirm_gen,
            args,
            candidates,
            valset,
            instance_ids,
            gen_logs,
            seen_personas,
            invalid_cases,
            persona_path,
        )
        for i, lg in enumerate(gen_logs):
            if lg.get("gen") == args.confirm_gen:
                gen_logs[i] = log
                break
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        write_state(
            run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
        )
        if not args.continue_after_confirm:
            write_candidates_json(run_dir, candidates, best_idx)
            best = candidates[best_idx]
            (run_dir / "exploration_best_persona.md").write_text(best.persona_text)
            print(
                f"\n{'=' * 60}\n  CONFIRM-GEN {args.confirm_gen} COMPLETE\n{'=' * 60}"
            )
            print(
                f"  accepted={log.get('accepted')}  reason={log.get('accept_reason')}  "
                f"difficulty={log.get('difficulty_agg')}  validity={log.get('validity_agg')}"
            )
            print(
                f"  candidates: {len(candidates)}  best=idx{best_idx}  "
                f"difficulty={best.agg:.3f}  raw_difficulty={best.difficulty_agg:.3f}"
            )
            print(f"  state -> {run_dir / 'pgepa_state.json'}")
            if log.get("accepted") and not args.dry_run:
                from lineage_loop import materialize

                pid = f"{args.persona_id_base}_g{args.confirm_gen:02d}_k00"
                materialize(pid, candidates[log["idx"]].persona_text)
            return
        start_iter = max(start_iter, args.confirm_gen + 1)
        print(
            f"  [confirm-gen] continuing iterations from {start_iter}/{args.iterations}"
        )

    consec_gated = 0
    for g in range(start_iter, args.iterations + 1):
        gen_logs.append(
            run_iteration(
                g,
                args,
                candidates,
                valset,
                instance_ids,
                refl_adapter,
                seen_personas,
                invalid_cases,
                val_tasks=val_tasks,
                val_instance_ids=val_instance_ids,
                val_num_trials=args.val_num_trials,
                seen_pool=seen_pool,
                island_name=args.island_name,
                summary_limit=args.summary_inject_limit,
                gen_logs=gen_logs,
            )
        )
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        write_state(
            run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
        )

        last_log = gen_logs[-1]
        if last_log.get(
            "mutation_attempts", 0
        ) >= args.max_mutation_attempts and not last_log.get("gate_ok", True):
            consec_gated += 1
        else:
            consec_gated = 0
        if (
            args.early_stop_consecutive_gated > 0
            and consec_gated >= args.early_stop_consecutive_gated
        ):
            print(
                f"\n  EARLY STOP: {consec_gated} consecutive iterations exhausted "
                f"all {args.max_mutation_attempts} mutation attempts (diversity-gated)."
            )
            break

    best_idx = max(
        range(len(candidates)),
        key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
    )
    best = candidates[best_idx]
    write_state(
        run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
    )
    write_candidates_json(run_dir, candidates, best_idx)
    (run_dir / "exploration_best_persona.md").write_text(best.persona_text)

    print(f"\n{'=' * 60}\n  SERIAL-PARETO-GEPA COMPLETE\n{'=' * 60}")
    print(
        f"  candidates: {len(candidates)}  best=idx{best_idx}  "
        f"difficulty={best.agg:.3f}  raw_difficulty={best.difficulty_agg:.3f}  "
        f"validity={best.validity_agg:.3f}"
    )
    print(f"  state       -> {run_dir / 'pgepa_state.json'}")
    print(f"  candidates  -> {run_dir / 'candidates.json'}")
    print(f"  best persona-> {run_dir / 'exploration_best_persona.md'}")

    if not args.dry_run:
        if best.persona_text.strip():
            from lineage_loop import materialize

            best_pid = f"{args.persona_id_base}_exploration_best"
            materialize(best_pid, best.persona_text)
            print(f"  skill file  -> {PERSONAS_DIR / f'opt_{best_pid}.skill'}")
        else:
            print(
                "  [genesis] best candidate is no-persona baseline — skipping materialize"
            )
        print(
            f"\n  Next: python experiments/persona_search/scripts/certify_gepa_candidates.py "
            f"--run-dirs {run_dir}"
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Serial, Pareto-frontier persona search (1 parent -> 1 child per iter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--adapter",
        default="tau",
        choices=["tau", "lic"],
        help="eval backend: 'tau' (tau2 run, default) or 'lic' (lost_in_conversation run_demo)",
    )
    ap.add_argument(
        "--lic-dir",
        default=None,
        help="path to lost_in_conversation repo root (required for --adapter lic)",
    )
    ap.add_argument(
        "--seed-persona",
        default=None,
        help="path to initial persona .skill/.md (omit for genesis mode: "
        "start from no-persona and let the proposer create the first persona)",
    )
    ap.add_argument("--run-dir", default=None, help="output dir (default: auto)")
    ap.add_argument(
        "--persona-id-base",
        default="pgepa",
        help="namespace for skill files + sim dirs (must be unique per run)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="continue from <run-dir>/pgepa_state.json: restore candidates, "
        "invalid cases, and completed iterations (skips gen0 seed "
        "confirm). Use the same --domains/--max-tasks/--persona-id-base.",
    )
    ap.add_argument(
        "--confirm-gen",
        type=int,
        default=None,
        metavar="G",
        help="with --resume: skip mutation/minibatch for generation G and "
        "run the full-valset confirm (4a difficulty + 4b judge) on "
        "the saved child persona (opt_{base}_g{G:02d}_k00.skill). "
        "Updates that generation's log entry in state.",
    )
    ap.add_argument(
        "--confirm-persona",
        default=None,
        help="override persona file for --confirm-gen (default: "
        "data/personas/opt_{persona-id-base}_g{G:02d}_k00.skill)",
    )
    ap.add_argument(
        "--continue-after-confirm",
        action="store_true",
        default=False,
        help="after --confirm-gen, keep running iterations from G+1 "
        "(default: exit once confirm finishes)",
    )
    ap.add_argument("--seed", type=int, default=0)

    # algorithm
    ap.add_argument(
        "--iterations", type=int, default=24, help="number of parent->child iterations"
    )
    ap.add_argument(
        "--minibatch",
        type=int,
        default=10,
        help="minibatch size for the cheap pre-screen vs the parent "
        "(<=0 skips straight to the full-valset confirm)",
    )
    ap.add_argument(
        "--tie-tol",
        type=float,
        default=0.0,
        help="Pareto tie tolerance: a candidate within tie_tol of the "
        "per-instance best co-wins that instance (coarse scores)",
    )
    ap.add_argument(
        "--min-dissimilarity",
        type=float,
        default=0.30,
        help="lexical novelty floor: child must have word-3gram Jaccard "
        "dissimilarity >= this vs EVERY persona sampled so far",
    )
    ap.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="litellm embedding model for the SEMANTIC novelty layer "
        "(OpenAI via OPENAI_API_KEY/OPENAI_BASE_URL); "
        "auto-disables (lexical-only) if unavailable",
    )
    ap.add_argument(
        "--no-embedding",
        action="store_true",
        default=False,
        help="disable the semantic (embedding) novelty layer entirely",
    )
    ap.add_argument(
        "--max-semantic-similarity",
        type=float,
        default=0.92,
        help="reject a child whose max embedding cosine vs any prior "
        "persona exceeds this (semantic near-duplicate)",
    )
    ap.add_argument(
        "--max-mutation-attempts",
        type=int,
        default=5,
        help="max reflective re-mutations to satisfy the diversity + "
        "lint gates before keeping the best attempt",
    )
    ap.add_argument(
        "--lint-persona",
        dest="lint_persona",
        action="store_true",
        default=True,
        help="static persona-text gate (natural/valid/no-task-specific) "
        "before any sim; on by default",
    )
    ap.add_argument(
        "--no-lint-persona",
        dest="lint_persona",
        action="store_false",
        help="disable the static persona linter",
    )
    ap.add_argument(
        "--lint-model",
        default="openai/gpt-5.4-mini",
        help="model for the persona linter (OpenAI via "
        "OPENAI_API_KEY/OPENAI_BASE_URL); empty falls back to --mutation-model",
    )
    ap.add_argument(
        "--lint-maj",
        type=int,
        default=1,
        help="majority votes for the persona linter verdict",
    )

    # model / infra (mirror gepa_runner defaults)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument(
        "--user-llm-args",
        default=None,
        help="JSON dict of extra kwargs for the user simulator LLM "
        '(e.g. \'{"api_base":"http://127.0.0.1:4114/v1","api_key":"dummy"}\' '
        "to route fireworks-backed user models through the local shim)",
    )
    ap.add_argument(
        "--effort",
        default="high",
        help="user-simulator reasoning effort (telecom TauPersonaAdapter.effort); LiC agent effort comes from --agent-llm-args JSON",
    )
    ap.add_argument(
        "--user-effort",
        default="high",
        help="LiC user-simulator reasoning effort (kept at high while agent can be max)",
    )
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument(
        "--anthropic-base-url",
        default=None,
        help="route the cc_user backbone through this Anthropic-Messages "
        "base URL instead of Bedrock. Only honored with --no-bedrock.",
    )
    ap.add_argument(
        "--anthropic-api-key",
        default=None,
        help="api key sent with --anthropic-base-url (the vertex shim ignores it)",
    )
    ap.add_argument("--mutation-model", default="vertex_ai/claude-opus-4-7")
    ap.add_argument(
        "--mutation-api-base",
        default=None,
        help="api_base for the mutation model LLM call "
        "(e.g. http://127.0.0.1:4114/v1 for fireworks-backed models)",
    )
    ap.add_argument(
        "--mutation-api-key",
        default=None,
        help='api_key for the mutation model LLM call (default "dummy" when api_base set)',
    )
    ap.add_argument(
        "--mutation-effort",
        default="xhigh",
        help="reasoning_effort passed to the mutation model; overrides the "
        "per-provider default (e.g. xhigh for gpt-5.4)",
    )
    ap.add_argument(
        "--mutation-edit-pct",
        default="40-50%",
        help="how much of the persona the proposer rewrites per mutation",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="sim fan-out within a single evaluate() (tau2 internal; "
        "this is NOT cross-child parallelism)",
    )
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])

    # judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument(
        "--judge-maj",
        type=int,
        default=3,
        help="validity-judge votes per judged trial, strict majority "
        "(unified across tau/vita/lic; 1 = single judgment)",
    )
    ap.add_argument("--judge-valset", action="store_true", default=False)
    ap.add_argument("--severe-gate-rate", type=float, default=None)
    ap.add_argument("--valset-judge-max-failures", type=int, default=15)

    # island model: train/val/test split + shared cross-island seen pool
    ap.add_argument(
        "--split-file",
        default=None,
        help="train/val/test split JSON ({train/val/test: {domain: [ids]}}). "
        "When set: minibatch is sampled from TRAIN, selection/Pareto "
        "uses VAL (held out), TEST is left untouched for post-hoc eval. "
        "Overrides --max-tasks/load_rep_tasks for the task universe.",
    )
    ap.add_argument(
        "--seed-reuse-glob",
        nargs="*",
        default=None,
        help="glob(s) of PRIOR results.json files whose SEED sims are "
        "reused at gen 0 (cross-run warm-start). Only tasks in the "
        "current split are reused; the seed eval simulates only the "
        "misses. Apply ONLY when the seed persona matches the prior "
        "run's. No effect on mutated children.",
    )
    ap.add_argument(
        "--shared-seen-pool",
        default=None,
        help="path to a file-locked JSONL pool shared across parallel "
        "islands; children from every island fold into this island's "
        "diversity gate so generated personas stay unique across runs.",
    )
    ap.add_argument(
        "--island-name",
        default="",
        help="this island's id (e.g. brinks_00), used to tag its rows in "
        "the shared pool and exclude them from its own dedup view.",
    )
    ap.add_argument(
        "--summary-inject-limit",
        type=int,
        default=40,
        help="max recent cross-island persona summaries injected into the "
        "mutation prompt (caps prompt growth as the pool fills)",
    )
    ap.add_argument(
        "--val-num-trials",
        type=int,
        default=None,
        help="num_trials for the held-out VAL confirm (default: --num-trials)",
    )
    ap.add_argument(
        "--scoring-mode",
        choices=["mean", "fail_squared"],
        default="mean",
        help="task-level aggregation: mean (default) or fail_squared "
        "(all trials must fail to score 1.0, else 0.0)",
    )
    ap.add_argument(
        "--mutation-context",
        choices=["basic", "enriched", "twophase", "weird"],
        default="basic",
        help="mutation prompt context mode: basic (title-only explored summaries, "
        "no failure traces), enriched (direction+score explored summaries + "
        "failure traces), twophase (parallel reflect then propose), or weird "
        "(LIC-only: push toward unusual behavior patterns, 70% rewrite)",
    )
    # --- enriched-mode leave-one-out ablation flags (orthogonal, store_true) ---
    ap.add_argument(
        "--no-outcome-summaries",
        action="store_true",
        help="ablation: in enriched/twophase/weird mode, force build_explored back "
        "to title-only summaries (drop outcome-annotated direction+score history)",
    )
    ap.add_argument(
        "--no-validity-judge",
        action="store_true",
        help="ablation: disable the conversation-validity gate — tau side skips the "
        "LLM judge + severe_gate_rate; LiC side skips its gpt-5.2 severe judge "
        "AND the rule gate (valid_severe_threshold=9999, severe_gate_rate=None). "
        "Search-time val becomes unreliable; "
        "compare on test cert @3 + offline validity re-judge.",
    )
    ap.add_argument(
        "--no-validity-gate",
        action="store_true",
        help="ablation: keep the LLM validity judge (still labels severe/illegal) but "
        "disable the gate (severe_gate_rate, severe_batch_threshold, timeout_attack "
        "all neutralized) — severe personas are NOT zeroed/filtered. Isolates the "
        "judge's identification role from the gate's filtering role.",
    )
    ap.add_argument(
        "--mix-failure-traces",
        action="store_true",
        help="ablation: in enriched single-phase, inject failure_traces[:2] alongside "
        "success_traces (reverses the built-in A/B decision that excludes them)",
    )
    ap.add_argument(
        "--no-scoring-hints",
        action="store_true",
        help="disable domain-specific system constraints in mutation prompt "
        "(retail only: omit one-shot/variant/state rules)",
    )
    ap.add_argument(
        "--reflect-model",
        default=None,
        help="model for Phase-1 parallel propose calls in twophase mode "
        "(default: same as --mutation-model)",
    )
    ap.add_argument(
        "--reflect-effort",
        default="medium",
        help="reasoning effort for Phase-1 propose calls (default: medium)",
    )
    ap.add_argument(
        "--early-stop-consecutive-gated",
        type=int,
        default=0,
        metavar="N",
        help="stop early if N consecutive iterations exhaust all mutation "
        "attempts without passing the diversity gate (0 = disabled)",
    )

    # testing / debug
    ap.add_argument("--max-tasks", type=int, default=35, help="cap tasks PER DOMAIN")
    ap.add_argument(
        "--print-pareto",
        action="store_true",
        help="dump the Pareto pool + selected parent each iteration",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="no sims/LLM; synthetic scores"
    )

    # --- deprecated (accepted as no-ops so old launchers don't crash) ----------
    ap.add_argument(
        "--generations",
        type=int,
        default=None,
        help="[deprecated] alias for --iterations",
    )
    for dep in (
        "--children-per-gen",
        "--parents-per-gen",
        "--top-t",
        "--max-concurrency-children",
        "--child-max-concurrency",
        "--phase-max-concurrency",
        "--length-fix-attempts",
    ):
        ap.add_argument(dep, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--parent-softmax-temperature", type=float, default=None, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--max-length-ratio", type=float, default=None, help=argparse.SUPPRESS
    )
    ap.add_argument("--mutation-mode", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()
    if args.generations is not None:
        args.iterations = args.generations
    _deprecated = [
        d
        for d in (
            "children_per_gen",
            "parents_per_gen",
            "top_t",
            "max_concurrency_children",
            "child_max_concurrency",
            "phase_max_concurrency",
            "length_fix_attempts",
            "parent_softmax_temperature",
            "max_length_ratio",
            "mutation_mode",
        )
        if getattr(args, d, None) is not None
    ]
    if _deprecated:
        print(
            f"  [deprecated] ignoring removed flags: {', '.join(_deprecated)} "
            "(serial single-child search; no length control)"
        )
    return args


if __name__ == "__main__":
    main()
