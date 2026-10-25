"""Candidate dataclass and Pareto-frontier helpers.

Extracted from parallel_gepa.py — the Candidate pool, scoring helpers, and
per-instance Pareto frontier construction / parent sampling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

# ---------------------------------------------------------------------------
# Candidate pool
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
        None  # train split per-instance scores (minibatch比較用)
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
