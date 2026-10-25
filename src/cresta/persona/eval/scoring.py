"""Scoring primitives for tau2 persona evaluation.

Extracted from tau_persona_adapter.py — outcome classification, data types,
feedback builders, and per-task trial aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cresta.persona.core._common import is_infra_broken
from cresta.persona.mutation.prompt import _format_msg

# ---------------------------------------------------------------------------
# Outcome taxonomy
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    AGENT_SUCCESS = "agent_success"
    GENUINE_AGENT_FAILURE = "genuine_agent_failure"
    ILLEGAL_USER_BEHAVIOR = "illegal_user_behavior"
    TIMEOUT_OR_MAX_STEPS = "timeout_or_max_steps"
    INFRA_ERROR = "infra_error"
    UNKNOWN = "unknown"


def classify_outcome(sim: dict | None, judge_info: dict | None) -> Outcome:
    if sim is None or is_infra_broken(sim):
        return Outcome.INFRA_ERROR

    term = sim.get("termination_reason", "")
    if term in ("timeout", "max_steps"):
        return Outcome.TIMEOUT_OR_MAX_STEPS

    if judge_info and judge_info.get("verdict") == "severe_failure":
        violations = judge_info.get("violations", [])
        only_abandonment = violations and all(
            v.get("type") == "task_abandonment" for v in violations
        )
        n_msgs = len(sim.get("messages", [])) if sim else 0
        if only_abandonment and n_msgs >= 50:
            return Outcome.GENUINE_AGENT_FAILURE
        return Outcome.ILLEGAL_USER_BEHAVIOR

    reward = (sim.get("reward_info") or {}).get("reward")
    if reward is None:
        return Outcome.UNKNOWN

    if float(reward) >= 1.0:
        return Outcome.AGENT_SUCCESS

    return Outcome.GENUINE_AGENT_FAILURE


# ---------------------------------------------------------------------------
# Data types (opaque to GEPA)
# ---------------------------------------------------------------------------


@dataclass
class TauTask:
    task_id: str
    domain: str  # "telecom" or "banking"


@dataclass
class TauTrajectory:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
    messages: list[dict]
    outcome: Outcome = Outcome.UNKNOWN
    judge_verdict: str | None = None
    judge_violations: list[dict] = field(default_factory=list)


@dataclass
class TauOutput:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
    outcome: Outcome = Outcome.UNKNOWN


@dataclass
class TauEvalArtifacts:
    """Reusable outputs from one tau2 simulation pass."""

    run_label: str
    results_paths: dict[str, str]
    sims_by_task: dict[str, list[dict]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_snippet(head: list[dict], tail: list[dict], total: int) -> str:
    """Full conversation format with tool call info."""
    all_msgs = head + tail
    return "\n".join(_format_msg(m) for m in all_msgs)


_OUTCOME_FEEDBACK = {
    Outcome.AGENT_SUCCESS: (
        "Agent SUCCEEDED. Persona failed to create enough valid difficulty. "
        "Focus on where the user accepted the agent's framing, gave information "
        "too easily, or stopped pushing back too early."
    ),
    Outcome.GENUINE_AGENT_FAILURE: (
        "Agent FAILED genuinely. Persona was effective here. "
        "Preserve this behavior if it stayed valid and natural."
    ),
    Outcome.TIMEOUT_OR_MAX_STEPS: (
        "TIMEOUT/MAX_STEPS. This does NOT count as agent failure. "
        "Reduce stalling and keep the user cooperative enough to progress."
    ),
    Outcome.UNKNOWN: "UNKNOWN outcome — treat conservatively.",
}


def _build_feedback(traj: TauTrajectory) -> str:
    if traj.outcome == Outcome.ILLEGAL_USER_BEHAVIOR:
        violations_str = "; ".join(
            v.get("type", "unknown") for v in traj.judge_violations
        )
        return (
            f"SEVERE FAILURE: persona was ILLEGAL. Violations: {violations_str}. "
            f"Do not repeat this behavior."
        )
    return _OUTCOME_FEEDBACK.get(traj.outcome, "UNKNOWN outcome.")


def _score_for(outcome: Outcome, reward: float | None) -> float:
    """GEPA difficulty for ONE trial: genuine failure and timeout both score.
    Batch-level timeout_rate gate (>=30% -> zero all) prevents friction abuse."""
    if (
        outcome in (Outcome.GENUINE_AGENT_FAILURE, Outcome.TIMEOUT_OR_MAX_STEPS)
        and reward is not None
    ):
        return 1.0 - float(reward)
    return 0.0


def _aggregate_task_trials(
    task_id: str,
    trial_sims: list[dict],
    judge_by_key: dict[tuple[str, int], dict],
    scoring_mode: str = "mean",
) -> dict[str, Any]:
    """Collapse >=1 trial sims for one task into a single scored record.

    - difficulty = mean per-trial difficulty over non-infra trials (denoising)
    - severe if ANY non-infra trial is illegal (conservative on safety)
    - timeout only if ALL non-infra trials timed out (soft quality signal)
    - validity: invalid if any severe; valid if all non-infra trials judged and
      none severe; otherwise unknown (unjudged != valid)
    - representative trial (for trace/reporting) by outcome priority:
      illegal > agent_success > genuine_failure > timeout > unknown
    Works transparently for num_trials==1 (single-element list).
    """

    def _empty_infra() -> dict:
        return {
            "outcome": Outcome.INFRA_ERROR,
            "opt_score": 0.0,
            "validity": None,
            "validity_known": 0.0,
            "reward": None,
            "term": "",
            "msgs": [],
            "is_severe": False,
            "is_timeout": False,
            "is_infra": True,
            "jinfo": None,
        }

    per_trial: list[dict] = []
    for sim in trial_sims:
        trial = sim.get("trial", 0)
        jinfo = judge_by_key.get((task_id, trial))
        outcome = classify_outcome(sim, jinfo)
        broken = sim is None or is_infra_broken(sim)
        reward = None if broken else (sim.get("reward_info") or {}).get("reward")
        per_trial.append(
            {
                "outcome": outcome,
                "score": _score_for(outcome, reward),
                "reward": reward,
                "term": "" if broken else sim.get("termination_reason", ""),
                "msgs": [] if broken else sim.get("messages", []),
                "judged": jinfo is not None,
                "jinfo": jinfo,
            }
        )

    if not per_trial:
        return _empty_infra()
    non_infra = [t for t in per_trial if t["outcome"] != Outcome.INFRA_ERROR]
    if not non_infra:
        return _empty_infra()

    if scoring_mode == "fail_squared":
        opt_score = 1.0 if all(t["score"] > 0 for t in non_infra) else 0.0
    else:
        opt_score = sum(t["score"] for t in non_infra) / len(non_infra)
    any_severe = any(t["outcome"] == Outcome.ILLEGAL_USER_BEHAVIOR for t in non_infra)
    all_timeout = all(t["outcome"] == Outcome.TIMEOUT_OR_MAX_STEPS for t in non_infra)
    all_judged = all(t["judged"] for t in non_infra)

    rep, agg_outcome = non_infra[0], non_infra[0]["outcome"]
    for oc in (
        Outcome.ILLEGAL_USER_BEHAVIOR,
        Outcome.AGENT_SUCCESS,
        Outcome.GENUINE_AGENT_FAILURE,
        Outcome.TIMEOUT_OR_MAX_STEPS,
        Outcome.UNKNOWN,
    ):
        match = next((t for t in non_infra if t["outcome"] == oc), None)
        if match:
            rep, agg_outcome = match, oc
            break

    if any_severe:
        validity, validity_known = 0.0, 1.0
    elif all_judged:
        validity, validity_known = 1.0, 1.0
    else:
        validity, validity_known = None, 0.0

    return {
        "outcome": agg_outcome,
        "opt_score": opt_score,
        "validity": validity,
        "validity_known": validity_known,
        "reward": rep["reward"],
        "term": rep["term"],
        "msgs": rep["msgs"],
        "is_severe": any_severe,
        "is_timeout": all_timeout,
        "is_infra": False,
        "jinfo": rep["jinfo"],
    }
