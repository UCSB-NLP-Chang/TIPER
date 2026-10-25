"""VitaBench PGEPA adapter.

Bridges the parallel-generational persona search (parallel_gepa.py) with VitaBench's
in-process simulation + rubric sliding-window evaluator. Self-contained: does NOT
depend on the tau2 `gepa` package or `lineage_loop` — a lightweight EvaluationBatch
and a VitaBench-flavored mutation prompt are implemented here.

The adapter mirrors tau_persona_adapter's contract so the same generational driver
works:
    evaluate(batch, candidate, capture_traces) -> EvaluationBatch
    make_reflective_dataset(candidate, eval_batch, components) -> Mapping
    build_custom_proposer(mutation_model, mutation_edit_pct, ...) -> callable

Persona injection: VitaBench's UserSimulator already accepts a `persona` argument;
run_task was patched to forward `user_persona_override`, so no user-simulator edit.

Objective = difficulty (1 - agent reward), timeout-gated at the batch level.
"""

from __future__ import annotations

import json
import hashlib
import random as _rnd
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
_VITA_SRC = _Path(__file__).resolve().parents[1] / "src"  # vendor/vitabench/src (vita 包所在)
if str(_VITA_SRC) not in _sys.path:
    _sys.path.insert(0, str(_VITA_SRC))

from vita.run import get_tasks, run_task
from vita.utils.llm_utils import generate
from vita.data_model.message import UserMessage

# gepa lives in the sibling tau2bench venv (not on PyPI, not installed in this
# venv). Append its site-packages so `import gepa` resolves, while keeping this
# venv's own packages at higher priority (gepa is absent here, so no clash).
import sys as _sys
from pathlib import Path as _Path
_TAU_SITE = _Path(__file__).resolve().parents[2] / ".venv" / "lib"
for _c in sorted(_TAU_SITE.glob("python3.*/site-packages")):
    if str(_c) not in _sys.path:
        _sys.path.append(str(_c))
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402

# Shared severe-judge protocol (single source of truth across tau/vita/lic/sweetrl).
# Same clean .venv that provides gepa above also provides cresta.
from cresta.persona.eval.severe_judge import (  # noqa: E402
    build_severe_judge_sys,
    judge_conversation_maj as _shared_judge_conversation,
)


# EvaluationBatch is imported from gepa.core.adapter above (fields:
# outputs / scores / trajectories / objective_scores — identical to the
# lightweight one this file used to define).


# ---------------------------------------------------------------------------
# Outcome taxonomy. Severe user-behavior violations (fabricated_info /
# refused_to_engage / contradictory / character_break) are detected by the
# ONLINE JUDGE during search (see judge_conversation + _aggregate severe gate)
# and gated at the batch level; they are NOT a separate Outcome here because a
# severe violation gates the whole candidate's scores, not one trial's label.
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    AGENT_SUCCESS = "agent_success"
    GENUINE_AGENT_FAILURE = "genuine_agent_failure"
    TIMEOUT_OR_MAX_STEPS = "timeout_or_max_steps"
    INFRA_ERROR = "infra_error"
    UNKNOWN = "unknown"


_TIMEOUT_TERMS = {"max_duration", "max_steps", "timeout"}


def classify_outcome(sim_dict: dict | None) -> Outcome:
    if sim_dict is None:
        return Outcome.INFRA_ERROR
    term = str(sim_dict.get("termination_reason", "") or "")
    if term in _TIMEOUT_TERMS:
        return Outcome.TIMEOUT_OR_MAX_STEPS
    reward = sim_dict.get("reward")
    if reward is None:
        return Outcome.UNKNOWN
    if float(reward) >= 1.0:
        return Outcome.AGENT_SUCCESS
    return Outcome.GENUINE_AGENT_FAILURE


def _score_for(outcome: Outcome, reward: float | None) -> float:
    """Difficulty for ONE trial. Genuine failure AND timeout both score
    (1 - reward); batch-level timeout gate below prevents friction abuse."""
    if outcome in (Outcome.GENUINE_AGENT_FAILURE, Outcome.TIMEOUT_OR_MAX_STEPS) and reward is not None:
        return 1.0 - float(reward)
    return 0.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class VitaTask:
    task_id: str
    domain: str


@dataclass
class VitaTrajectory:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
    messages: list[dict]
    outcome: Outcome = Outcome.UNKNOWN
    judge: dict | None = None  # online severe-judge verdict for failed convos


@dataclass
class VitaOutput:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
    outcome: Outcome = Outcome.UNKNOWN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _build_feedback(traj: VitaTrajectory) -> str:
    return _OUTCOME_FEEDBACK.get(traj.outcome, "UNKNOWN outcome.")


def _msg_to_dict(m) -> dict:
    """Compact, serializable view of a Message for the mutation proposer."""
    try:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", None) or ""
        is_tc = hasattr(m, "is_tool_call") and m.is_tool_call()
        d = {"role": role, "content": content}
        if is_tc:
            tcs = getattr(m, "tool_calls", None) or []
            d["tool_calls"] = [
                {"name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", {})}
                for tc in tcs
            ]
        return d
    except Exception:
        return {"role": "?", "content": str(m)[:500]}


def _sim_to_dict(sim, trial: int, capture: bool) -> dict:
    """Flatten a SimulationRun to a plain dict (picklable / JSON-friendly)."""
    if sim is None:
        return {"task_id": None, "trial": trial, "reward": None, "termination_reason": "",
                "messages": [], "agent_cost": None, "user_cost": None, "duration": None,
                "is_infra": True}
    ri = getattr(sim, "reward_info", None)
    reward = getattr(ri, "reward", None) if ri is not None else None
    term = str(getattr(sim, "termination_reason", "") or "")
    msgs = [_msg_to_dict(m) for m in getattr(sim, "messages", [])] if capture else []
    return {
        "task_id": getattr(sim, "task_id", None),
        "trial": trial,
        "reward": reward,
        "termination_reason": term,
        "messages": msgs,
        "agent_cost": getattr(sim, "agent_cost", None),
        "user_cost": getattr(sim, "user_cost", None),
        "duration": getattr(sim, "duration", None),
        "is_infra": False,
    }


def _format_snippet(msgs: list[dict]) -> str:
    """Render a conversation snippet for the mutation prompt (full, no truncation
    per-message so the proposer sees complete tool outputs)."""
    lines = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        tag = {"user": "[user]", "assistant": "[assistant]", "tool": "[tool]",
               "system": "[system]"}.get(role, f"[{role}]")
        line = f"{tag} {content}"
        for tc in m.get("tool_calls", []) or []:
            line += f"\n  -> {tc.get('name','')}: {json.dumps(tc.get('arguments',{}), ensure_ascii=False)[:300]}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class VitaPersonaAdapter(GEPAAdapter):
    """Evaluates persona candidates via VitaBench simulations.

    A GEPAAdapter subclass (mirrors sweetrl_persona_adapter.SweetRLPersonaAdapter):
    gepa.optimize() drives the search and calls evaluate() / make_reflective_dataset();
    a custom_candidate_proposer (build_custom_proposer) does the mutation. Uses
    VitaBench's own generate() + run_task (not a raw OpenAI client) so the
    /v1/responses routing for gpt-5.4 and models.yaml config are reused.
    """

    propose_new_texts = None

    def __init__(
        self,
        *,
        persona_id: str = "vita_gepa",
        agent_llm: str = "gpt-5.4",
        agent_llm_args: dict | None = None,
        user_llm: str = "gpt-5.4",
        user_llm_args: dict | None = None,
        max_steps: int = 100,
        max_errors: int = 100000,
        max_duration: float = 0,
        num_trials: int = 2,
        evaluator_llm: str = "gpt-5.2",
        evaluator_llm_args: dict | None = None,
        enable_think: bool = True,
        language: str = "chinese",
        domain: str = "delivery",
        max_concurrency: int = 6,
        timeout_attack_threshold: float = 0.15,
        success_threshold: float = 0.999,
        seed_base: int = 300,
        capture_dir: str | None = None,
        online_judge: bool = True,
        online_judge_model: str = "gpt-5.2",
        judge_maj: int = 3,
        severe_gate_threshold: float = 0.05,
        online_judge_concurrency: int = 8,
    ):
        self._persona_id = persona_id
        self._agent_llm = agent_llm
        self._agent_llm_args = agent_llm_args or {}
        self._user_llm = user_llm
        self._user_llm_args = user_llm_args or {}
        self._max_steps = max_steps
        self._max_errors = max_errors
        self._max_duration = max_duration
        self._num_trials = num_trials
        self._evaluator_llm = evaluator_llm
        self._evaluator_llm_args = evaluator_llm_args or {}
        self._enable_think = enable_think
        self._language = language
        self._domain = domain
        # run_task resolves the environment via `if "," in domain` -> the cross
        # environment needs the comma-joined domain string, not the task-set name
        # `cross_domain` (which is only for task loading / framing lookup).
        self._run_domain = "delivery,ota,instore" if domain == "cross_domain" else domain
        self._max_concurrency = max_concurrency
        self._timeout_attack_threshold = timeout_attack_threshold
        self._success_threshold = success_threshold
        self._seed_base = seed_base
        self._capture_dir = capture_dir
        self._tasks_cache: dict[str, Any] = {}
        # online conversation-behavior judge (tau parallel_gepa severe gate).
        # When enabled, every FAILED simulation (genuine_agent_failure) is judged
        # on 4 severe categories; severe_rate > threshold across the batch -> the
        # whole candidate is GATED (scores zeroed, infeasible). Severe violations
        # are attached to trajectories so make_reflective_dataset propagates them
        # to the proposer's severe_section. Toggle off for test-set eval (where the
        # offline judge handles validity instead) — see vita_pgepa.evaluate_reward.
        self._online_judge_enabled = online_judge
        self._online_judge_model = online_judge_model
        self._judge_maj = judge_maj
        self._severe_gate_threshold = severe_gate_threshold
        self._online_judge_concurrency = online_judge_concurrency
        self._last_judge_summary: dict[str, Any] = {}

    # -- task loading (cached per domain) -----------------------------------

    def _get_task_objects(self, task_ids: list[str]):
        if self._domain not in self._tasks_cache:
            all_tasks = get_tasks(self._domain, None, None, self._language)
            self._tasks_cache[self._domain] = {t.id: t for t in all_tasks}
        return self._tasks_cache[self._domain]

    # -- single sim ---------------------------------------------------------

    def _run_one(self, task, trial: int, seed: int, persona_text: str, capture: bool) -> dict:
        try:
            sim = run_task(
                domain=self._run_domain,
                task=task,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=self._agent_llm,
                llm_args_agent=self._agent_llm_args,
                llm_user=self._user_llm,
                llm_args_user=self._user_llm_args,
                max_steps=self._max_steps,
                max_errors=self._max_errors,
                max_duration=self._max_duration,
                seed=seed,
                enable_think=self._enable_think,
                llm_evaluator=self._evaluator_llm,
                llm_args_evaluator=self._evaluator_llm_args,
                language=self._language,
                user_persona_override=persona_text,
            )
            return _sim_to_dict(sim, trial, capture)
        except Exception as e:
            print(f"  [vita eval {self._domain}] task {task.id} trial {trial} failed: {e}")
            return {"task_id": task.id, "trial": trial, "reward": None,
                    "termination_reason": "", "messages": [], "agent_cost": None,
                    "user_cost": None, "duration": None, "is_infra": True}

    # -- GEPAAdapter.evaluate -----------------------------------------------

    def evaluate(
        self,
        batch: list[VitaTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        persona_text = candidate["persona_text"]
        task_map = self._get_task_objects([t.task_id for t in batch])

        # The online judge needs conversation messages for failed sims. Force
        # capture whenever the judge is on, even if the caller didn't ask for
        # trajectories (e.g. pareto-selection evals with capture_traces=False).
        # Trajectories are still only built when capture_traces=True below.
        capture = capture_traces or self._online_judge_enabled

        jobs = []  # (VitaTask, trial)
        for vt in batch:
            for trial in range(self._num_trials):
                jobs.append((vt, trial))

        sims_by_task: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as ex:
            futs = {}
            for vt, trial in jobs:
                task = task_map.get(vt.task_id)
                if task is None:
                    sims_by_task.setdefault(vt.task_id, []).append(
                        {"task_id": vt.task_id, "trial": trial, "reward": None,
                         "termination_reason": "", "messages": [], "is_infra": True})
                    continue
                seed = self._seed_base + trial * 7919
                futs[ex.submit(self._run_one, task, trial, seed, persona_text, capture)] = vt.task_id
            for fut in futs:
                tid = futs[fut]
                sims_by_task.setdefault(tid, []).append(fut.result())

        return self._aggregate(batch, sims_by_task, capture_traces, persona_text, task_map)

    def _aggregate(self, batch, sims_by_task, capture_traces, persona_text="",
                   task_map=None) -> EvaluationBatch:
        outputs: list[VitaOutput] = []
        scores: list[float] = []
        trajectories: list[VitaTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict] = []
        timeout_count = 0
        total_cost = 0.0
        # rep sim + task object per task_id, for the online judge (failed convos)
        judge_targets: list[tuple[str, dict, Any]] = []
        traj_by_task: dict[str, VitaTrajectory] = {}

        for vt in batch:
            trial_sims = sims_by_task.get(vt.task_id, [])
            non_infra = [s for s in trial_sims if not s.get("is_infra")]
            if not non_infra:
                outputs.append(VitaOutput(vt.task_id, vt.domain, None, "", Outcome.INFRA_ERROR))
                scores.append(0.0)
                objective_scores.append({"difficulty": 0.0, "validity": 1.0,
                                         "timeout_safety": 1.0, "semantic": 0.0,
                                         "infra_error": 1.0, "feasible": 1.0})
                continue
            per_trial = []
            for s in non_infra:
                oc = classify_outcome(s)
                per_trial.append((oc, _score_for(oc, s.get("reward")), s))
            opt_score = sum(t[1] for t in per_trial) / len(per_trial)
            rep = max(per_trial, key=lambda t: (
                2 if t[0] == Outcome.AGENT_SUCCESS else
                1 if t[0] == Outcome.GENUINE_AGENT_FAILURE else
                0 if t[0] == Outcome.TIMEOUT_OR_MAX_STEPS else -1))
            rep_oc, _, rep_sim = rep
            # tau parity: task counts as timeout only if ALL its non-infra
            # trials did (severe is ANY-trial; timeout is ALL-trial)
            all_timeout = all(t[0] == Outcome.TIMEOUT_OR_MAX_STEPS for t in per_trial)
            if all_timeout:
                timeout_count += 1
            ac = rep_sim.get("agent_cost") or 0.0
            uc = rep_sim.get("user_cost") or 0.0
            total_cost += ac + uc
            outputs.append(VitaOutput(vt.task_id, vt.domain, rep_sim.get("reward"),
                                      rep_sim.get("termination_reason", ""), rep_oc))
            scores.append(opt_score)
            objective_scores.append({
                "difficulty": opt_score,
                "validity": 1.0, "validity_known": 1.0,
                "timeout_safety": 0.0 if all_timeout else 1.0,
                "semantic": 1.0, "infra_error": 0.0, "feasible": 1.0,
            })
            if capture_traces:
                traj = VitaTrajectory(
                    vt.task_id, vt.domain, rep_sim.get("reward"),
                    rep_sim.get("termination_reason", ""),
                    rep_sim.get("messages", []), rep_oc)
                trajectories.append(traj)
                traj._rep_trial_idx = per_trial.index(rep)
                traj_by_task[vt.task_id] = traj
            # collect EVERY failed non-infra trial with a substantive
            # conversation for judging (tau parity: task-level severe =
            # ANY non-infra trial judged severe, not just the rep trial)
            if self._online_judge_enabled:
                for ti, (oc, _, s) in enumerate(per_trial):
                    if oc == Outcome.GENUINE_AGENT_FAILURE and s.get("messages"):
                        judge_targets.append((vt.task_id, s,
                                              task_map.get(vt.task_id) if task_map else None,
                                              ti))

        # batch-level timeout gate (friction abuse) — n_noninfra denominator,
        # unified with tau/sweetrl (infra tasks don't shrink the rate)
        n_noninfra = sum(1 for obj in objective_scores if obj.get("infra_error", 0.0) == 0.0)
        batch_infeasible = False
        if n_noninfra > 0 and timeout_count / n_noninfra >= self._timeout_attack_threshold:
            print(f"  [vita eval] BATCH INFEASIBLE: timeout_rate="
                  f"{timeout_count}/{n_noninfra}={timeout_count/n_noninfra:.0%} "
                  f"(>= {self._timeout_attack_threshold:.0%}) -> zeroing all scores")
            batch_infeasible = True

        # online severe-judge gate: judge every genuine failure; severe_rate >
        # threshold -> gate the whole candidate (mirrors tau parallel_gepa).
        # Denominator = ALL non-infra tasks (not just the judged failed convos),
        # matching tau's rate-gate semantics — successful sims count as
        # non-severe, so the 0.05 threshold keeps the same meaning as cert.
        severe_count = 0
        n_judged = 0
        severe_violations: list[dict] = []
        if self._online_judge_enabled and judge_targets:
            severe_count, n_judged, severe_violations = self._run_online_judge(
                judge_targets, persona_text, traj_by_task)
            severe_rate = severe_count / n_noninfra if n_noninfra else 0.0
            self._last_judge_summary = {
                "n_judged": n_judged, "n_severe": severe_count,
                "n_noninfra": n_noninfra,
                "severe_rate": severe_rate, "violations": severe_violations,
            }
            print(f"  [online judge] judged {n_judged} failed convos: "
                  f"{severe_count} severe / {n_noninfra} non-infra tasks "
                  f"({severe_rate:.0%})")
            if n_noninfra and severe_rate > self._severe_gate_threshold:
                print(f"  [vita eval] BATCH INFEASIBLE: severe_rate="
                      f"{severe_count}/{n_noninfra}={severe_rate:.0%} "
                      f"(> {self._severe_gate_threshold:.0%}) -> zeroing all scores; "
                      f"cats={[v['category'] for v in severe_violations]}")
                batch_infeasible = True
            elif severe_count:
                print(f"  [vita eval] severe_rate={severe_count}/{n_noninfra}="
                      f"{severe_rate:.0%} (<= threshold; not gated)")
        else:
            self._last_judge_summary = {}

        if batch_infeasible:
            scores = [0.0] * len(outputs)
            for obj in objective_scores:
                obj["feasible"] = 0.0
            if objective_scores:
                # mark validity break on the aggregate so the proposer sees it
                for obj in objective_scores:
                    if severe_violations:
                        obj["validity"] = 0.0
                        obj["validity_known"] = 0.0

        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajectories,
                               objective_scores=objective_scores)

    def _run_online_judge(self, judge_targets, persona_text, traj_by_task):
        """Judge each failed non-infra trial concurrently (maj-vote, tau parity).
        Attaches the rep-trial verdict to the matching trajectory (for proposer
        feedback) and returns (n_severe_TASKS, n_judged, severe_violations) —
        a task counts severe if ANY of its failed trials did. Fail-open per call."""
        n_severe_tasks = 0
        n_judged = 0
        severe_violations: list[dict] = []
        severe_by_task: set[str] = set()
        with ThreadPoolExecutor(max_workers=self._online_judge_concurrency) as ex:
            futs = {}
            for tid, sim, task, trial_idx in judge_targets:
                msgs = sim.get("messages", []) or []
                futs[ex.submit(judge_conversation, persona_text, task, msgs,
                               self._online_judge_model, self._judge_maj)] = (tid, sim, trial_idx)
            for fut in futs:
                tid, sim, trial_idx = futs[fut]
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001
                    v = {"verdict": "genuine_failure", "category": "none",
                         "reason": f"judge err: {str(e)[:80]}"}
                v["task_id"] = tid
                v["trial"] = trial_idx
                v["reward"] = sim.get("reward")
                n_judged += 1
                traj = traj_by_task.get(tid)
                if traj is not None and trial_idx == getattr(traj, "_rep_trial_idx", None):
                    traj.judge = v
                if v.get("verdict") == "severe_failure":
                    severe_by_task.add(tid)
                    severe_violations.append({
                        "task_id": tid, "trial": trial_idx,
                        "category": v.get("category", "?"),
                        "reason": v.get("reason", ""),
                    })
        n_severe_tasks = len(severe_by_task)
        return n_severe_tasks, n_judged, severe_violations

    # -- GEPAAdapter.make_reflective_dataset --------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        entries: list[dict[str, Any]] = []
        if eval_batch.trajectories is None:
            return {"persona_text": entries}
        for traj in eval_batch.trajectories:
            if traj.outcome == Outcome.INFRA_ERROR:
                continue
            msgs = traj.messages
            if traj.outcome == Outcome.AGENT_SUCCESS and msgs:
                snippet = _format_snippet(msgs)
            elif traj.outcome == Outcome.GENUINE_AGENT_FAILURE and msgs:
                snippet = _format_snippet(msgs)
            elif traj.outcome == Outcome.TIMEOUT_OR_MAX_STEPS:
                snippet = (f"(conversation hit {traj.termination_reason} after "
                           f"{len(msgs)} messages)")
            else:
                snippet = ""
            entries.append({
                "Inputs": {"task_id": traj.task_id, "domain": traj.domain,
                           "outcome": traj.outcome.value,
                           "reward": (traj.reward if traj.reward is not None else 0.0),
                           "termination": traj.termination_reason or ""},
                "Generated Outputs": {"conversation_snippet": snippet},
                "Feedback": _build_feedback(traj),
                "Judge": traj.judge or {},
            })
        return {"persona_text": entries}


# ---------------------------------------------------------------------------
# VitaBench-flavored mutation prompt (enriched + genesis modes)
# ---------------------------------------------------------------------------

# Domain framing — VitaBench customer-support (per-domain). Adapted from the
# tau-bench enriched-mode mutation prompt and reworded in the sweetrl style: a
# setting description + the adversarial workflow stages the mutator can attack.
# This is the "mutation ctx adapted to this task" the user asked for — each
# VitaBench domain has its own tool surface and rubric shape, and the mutator must
# target the RIGHT stage of the agent's workflow to cause a wrong ACTION, not a
# slow conversation.

_DOMAIN_FRAMING = {
    "delivery": """\
The setting (VitaBench `delivery`): an LLM support agent converses with a
simulated customer over multiple turns to resolve a food-delivery issue —
cancelling or modifying an order, changing the delivery address, confirming an
ETA, or placing a multi-item order under shifting intent. The agent has tools for
searching restaurants/products, placing/cancelling/modifying orders, querying
order and delivery status, and managing addresses. It is graded against a hidden
rubric of required actions and end states (a low reward means the agent took the
WRONG action — cancelled the wrong order, set the wrong address, missed a
constraint — not just a slow one).

You are optimizing the CUSTOMER PERSONA — behavioral instructions that shape how
the simulated customer responds. Objective: make the persona ADVERSARIAL —
minimize the agent's rubric reward (difficulty = 1 - reward) — while staying
REALISTIC and VALID.

The agent's workflow stages you can attack:
(a) INFORMATION GATHERING — the customer's answers are consistently skewed,
    underspecified, or given one detail at a time, so the agent never pins down
    the right order / address / item set.
(b) STEP EXECUTION — the customer claims to have done a step (checked the order
    status, tried cancelling, confirmed the address in the app) but did it
    partially or on the wrong entity, and reports it as done, so the agent skips
    re-checking and proceeds on a false premise.
(c) VERIFICATION / CLOSURE — the customer confirms "that's fine, go ahead" /
    "就这样吧" based on incomplete evidence, without verifying the agent actually
    set the right order/address/item, so the agent stops early and ships the
    wrong outcome.
(d) PROBLEM SCOPING — the customer's initial framing locks the agent onto the
    wrong scope: the wrong order to cancel, the wrong address to set, or the
    wrong item to modify.
Ineffective axes: random/erratic behavior, unpredictable message length, pure
emotional volatility — these add VARIANCE but not BIAS; the agent can still reach
the correct action by asking enough follow-ups. Prefer a CONSISTENT cognitive
pattern that biases the agent toward the WRONG action every time.
""",
    "instore": """\
The setting (VitaBench `instore`): an LLM support agent converses with a
simulated customer to resolve an in-store consumption issue — booking a
reservation, redeeming a group-buy voucher, querying prices or member points, or
shifting intent between dining options. The agent has tools for searching
stores/products, reservations, group-buying and vouchers, orders, and member
points. It is graded against a hidden rubric of required actions and end states.

You are optimizing the CUSTOMER PERSONA — behavioral instructions that shape how
the simulated customer responds. Objective: make the persona ADVERSARIAL —
minimize the agent's rubric reward (difficulty = 1 - reward) — while staying
REALISTIC and VALID.

The agent's workflow stages you can attack:
(a) INFORMATION GATHERING — the customer's answers are consistently skewed or
    underspecified, so the agent never pins down the right store / time slot /
    party size.
(b) STEP EXECUTION — the customer claims to have done a step (checked points,
    redeemed the voucher, confirmed the booking) but did it partially or on the
    wrong store/option, and reports it as done, so the agent proceeds on a false
    premise.
(c) VERIFICATION / CLOSURE — the customer confirms "可以了 / 就这家吧" based on
    incomplete evidence, without verifying the booking/voucher details, so the
    agent stops early and ships the wrong outcome.
(d) PROBLEM SCOPING — the customer's initial framing locks the agent onto the
    wrong scope: the wrong store, wrong time slot, or wrong voucher to apply.
Ineffective axes: random/erratic behavior, unpredictable message length, pure
emotional volatility — these add VARIANCE but not BIAS. Prefer a CONSISTENT
cognitive pattern that biases the agent toward the WRONG action every time.
""",
    "ota": """\
The setting (VitaBench `ota`): an LLM support agent converses with a simulated
customer to resolve an online-travel issue — searching flights/hotels/trains,
booking, ticket change or refund, itinerary queries, and passenger info. Tasks
involve multi-leg itineraries, date/seat-class constraints, ticket changes, and
ambiguous destinations. The agent is graded against a hidden rubric of required
actions and end states.

You are optimizing the CUSTOMER PERSONA — behavioral instructions that shape how
the simulated customer responds. Objective: make the persona ADVERSARIAL —
minimize the agent's rubric reward (difficulty = 1 - reward) — while staying
REALISTIC and VALID.

The agent's workflow stages you can attack:
(a) INFORMATION GATHERING — the customer's answers are consistently skewed or
    underspecified, so the agent never pins down the right leg / date / class.
(b) STEP EXECUTION — the customer claims to have done a step (checked the
    itinerary, submitted the ticket change, confirmed passenger details) but did
    it partially or on the wrong leg/segment, and reports it as done, so the agent
    proceeds on a false premise.
(c) VERIFICATION / CLOSURE — the customer confirms "好的就这样 / 没问题" based on
    incomplete evidence, without verifying the itinerary/ticket-change details, so
    the agent stops early and ships the wrong outcome.
(d) PROBLEM SCOPING — the customer's initial framing locks the agent onto the
    wrong scope: the wrong leg, wrong date, or wrong seat class to book/change.
Ineffective axes: random/erratic behavior, unpredictable message length, pure
emotional volatility — these add VARIANCE but not BIAS. Prefer a CONSISTENT
cognitive pattern that biases the agent toward the WRONG action every time.
""",
    "cross": """\
The setting (VitaBench `cross`): an LLM support agent operates across food
delivery, in-store consumption, and online travel with 66 merged tools. Tasks
require reasoning across temporal/spatial dimensions, proactively clarifying
ambiguous instructions, and tracking shifting user intent across domains. The
agent is graded against a hidden rubric of required actions and end states.

You are optimizing the CUSTOMER PERSONA — behavioral instructions that shape how
the simulated customer responds. Objective: make the persona ADVERSARIAL —
minimize the agent's rubric reward (difficulty = 1 - reward) — while staying
REALISTIC and VALID.

The agent's workflow stages you can attack:
(a) DOMAIN / PROBLEM SCOPING — the customer's initial framing is ambiguous across
    domains, so the agent scopes the task into the wrong domain or onto the wrong
    entity (wrong order / booking / itinerary).
(b) INFORMATION GATHERING — the customer's answers are consistently skewed or
    underspecified, so the agent never pins down the right entity.
(c) STEP EXECUTION — the customer claims to have done a step but did it partially
    or on the wrong entity, and reports it as done, so the agent proceeds on a
    false premise.
(d) VERIFICATION / CLOSURE — the customer confirms an incomplete or wrong action
    based on incomplete evidence, so the agent stops early and ships the wrong
    outcome.
Ineffective axes: random/erratic behavior, unpredictable message length, pure
emotional volatility — these add VARIANCE but not BIAS. The hardest personas
exploit cross-domain intent shifts and ambiguous initial framing. Prefer a
CONSISTENT cognitive pattern that biases the agent toward the WRONG action.
""",
}


_VALIDITY_CONSTRAINTS = """\
## VALIDITY CONSTRAINTS (violating ANY = invalid/cheating persona, auto-reverted)
- NEVER fabricate concrete facts (order/account numbers, amounts, addresses, dates,
  statuses, error codes) that are not grounded in the customer's scenario info or
  in tool results the agent showed. Difficulty must come from realistic human
  friction (vagueness, omission, confusion, fixation), NOT from lying about facts.
- When DIRECTLY asked for known info (name, phone, order/account number, address),
  provide it accurately eventually — partial / drip-fed / vague is fine; flat
  refusal or a wrong specific value is not.
- NEVER refuse to engage, never abandon the conversation while the problem is
  unresolved.
- NEVER mention "AI", "simulation", "benchmark", "scenario", "rubric", or output
  JSON / system-prompt / reference text verbatim. Stay in character.
- NEVER make the conversation structurally impossible to start (refusing every
  question, demanding impossible preconditions, circular blockers). Being slow and
  suspicious is fine; making it impossible for the agent to ever act is not.
- NEVER switch to a completely different goal mid-conversation, and NEVER accept
  an outcome that contradicts an explicit scenario requirement.
"""


# Per-domain scoring explanation (mirrors lineage_loop._get_scoring / SCORING_*):
# tells the mutator HOW the agent is graded so it aims at WRONG actions, not slow
# ones. VitaBench grades against a hidden rubric of required actions + end states
# via an NL_ASSERTION evaluator; low reward = the agent took the WRONG action,
# not just a slow conversation.
_DOMAIN_SCORING = {
    "delivery": """\
## HOW THE AGENT IS SCORED (delivery)
The agent is graded against a HIDDEN rubric of required actions and end states:
cancelling/modifying the CORRECT order, setting the CORRECT address, placing the
CORRECT items under all constraints, reaching the required end state. A LOW reward
means the agent took the WRONG action — cancelled the wrong order, set the wrong
address, missed a dietary/time constraint, shipped the wrong item — NOT that it was
slow. Your persona should steer the agent toward the WRONG action, not just a long
conversation.""",
    "instore": """\
## HOW THE AGENT IS SCORED (instore)
The agent is graded against a HIDDEN rubric of required actions and end states:
booking the CORRECT store / time slot / party size, applying the CORRECT voucher or
points, reaching the required end state. A LOW reward means the agent took the WRONG
action — booked the wrong store, applied the wrong voucher, missed a seating/party
constraint — NOT that it was slow. Steer the agent toward the WRONG action.""",
    "ota": """\
## HOW THE AGENT IS SCORED (ota)
The agent is graded against a HIDDEN rubric of required actions and end states:
booking/changing the CORRECT leg / date / seat class / passenger, reaching the
required end state. A LOW reward means the agent took the WRONG action — booked the
wrong leg, wrong date, wrong seat class, missed a passenger/date constraint — NOT
that it was slow. Steer the agent toward the WRONG action.""",
    "cross": """\
## HOW THE AGENT IS SCORED (cross)
The agent is graded against a HIDDEN rubric of required actions and end states across
delivery / in-store / OTA. A LOW reward means the agent took the WRONG action in the
WRONG domain — wrong order cancelled, wrong booking, wrong itinerary leg — NOT that
it was slow. Steer the agent toward the wrong domain/scope and the wrong action.""",
}


# Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) persona rules (mirrors
# parallel_gepa._MUTATION_EXAMPLES). Anchors the mutator to general behavioral
# tendencies and forbids embedding concrete scenario details from the traces.
_MUTATION_EXAMPLES = """
## Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) rules
GOOD (general, reusable behavioral tendency):
  "When the agent asks you to confirm an order/address you already mentioned, answer
   with a loose referent ('the lunch one', 'the usual address') instead of the exact
   ID, and only correct it if the agent states a concrete WRONG value — forcing the
   agent to disambiguate every time."
BAD (task-conditioned — names a concrete scenario/value; FORBIDDEN):
  "When the agent mentions order #D0721004, insist the 39.9 yuan pork dumplings never
   arrived and that account 8842 still shows the charge."
Turn any task-specific instinct into a general behavioral tendency instead.
"""


# Behavioral-style exploration catalogue (mirrors lineage_loop.WEIRD_CATALOGUE,
# customer-support adapted). The safe "vague + loose-referent + one-detail-at-a-
# time" family has ALREADY been explored and converges to similar personas every
# run (observed: delivery & instore best personas both collapse to "Context-
# Shorthand Proxy Customer"). This catalogue forces exploration of DISTINCT,
# UNUSUAL *talking-style* attack axes — the weirdness lives purely in HOW the
# customer talks (persona / structure / rhythm / cognitive tone), never in
# fabricating facts. Pick from (or combine) these four families; the examples are
# seeds, not limits.
_BEHAVIORAL_STYLE_CATALOGUE = """
## BE BOLD — EXPLORE UNUSUAL BEHAVIOR PATTERNS
The safe "vague answers + loose referents + one-detail-at-a-time + gist-confirm"
family has ALREADY been explored and converges to similar personas every time. Do
NOT produce another variant of it. Design a persona with a DISTINCT, UNUSUAL
BEHAVIOR PATTERN — a consistent, recognizable way of talking that a real customer
might have and that systematically biases the agent toward the WRONG action. Pick
from (or combine) these four families; the examples are seeds, not limits:

(1) ROLE / CHARACTER — give the customer a vivid personality whose HABITS drive
    the misdirection, not an explicit "attack":
    - a rushed, impatient office worker who pushes "快点吧 / 直接弄就行 / 别问了"
      on every turn, pressuring the agent to skip confirmation and act on the
      first plausible option;
    - a slightly forgetful older person who muses aloud before each reveal
      ("我想想啊... 好像是中午那单..."), making every fact sound provisional and
      underweighted;
    - an over-polite customer who leads every turn with apologies/thanks and
      buries the key constraint (the allergy, the time window) after the
      pleasantries;
    - a chatty rambler who wraps each requirement in a paragraph of irrelevant
      context, the decisive constraint buried mid-sentence;
    - a second-language / dialect user with broken grammar but EXACT needs,
      causing the agent to underweight a real constraint as a language artifact.

(2) STRUCTURAL — break the normal declarative-requirement sentence form:
    - state requirements as a numbered/bulleted list where the key constraint
      (e.g. "其中一份不要珍珠") is a sub-item under an innocuous main item;
    - lead with the conclusion ("我要退款" / "帮我改地址") then drip the
      qualifying conditions (which order, which address, until when) over later
      turns as afterthoughts;
    - phrase constraints as questions or "对了还有..." / "顺便说一句..." side-
      remarks rather than stated requirements.

(3) RHYTHMIC — vary information density non-standardly across turns:
    - turn 1 is a long rambling paragraph with the first constraint buried in the
      middle; later turns turn terse (a single "嗯" / "对" / "行");
    - some turns deliver just a single bare value ("公司" / "12点前"), making the
      agent unsure whether it's a new constraint or a confirmation;
    - escalating disclosure: start almost dismissive of a detail ("那个无所谓"),
      each later turn adds weight to that same detail until it's the decisive
      constraint the agent already ignored.

(4) COGNITIVE TONE — manufacture false certainty/uncertainty signals WITHOUT ever
    giving a wrong fact (a wrong fact is INVALID):
    - hedge the precise value with uncertain tone ("我记得好像是... 50块? / 应该
      是周二?") so the agent under-weights a correct, decisive number/date;
    - talk to yourself ("那中午那单... 还有晚上的...") making the agent unsure
      what's a confirmed fact vs musing;
    - treat the agent's early draft/order as confirmed ("对对就这样, 没问题") then
      reveal a qualifying constraint that breaks it — the warmth makes the agent
      reluctant to revise.

Combine families if it produces a coherent, recognizable character. The persona
must read as a REAL, if eccentric, customer — not a cartoon adversary. Difficulty
must still come from realistic friction; never fabricate facts.
"""

# Directions already explored + converged-on in prior vita runs (mirrors
# lineage_loop.WEIRD_EXPLORED). Naming them forbids the mutator from producing
# yet another remix of the safe family.
_CONVERGED_SAFE_DIRECTIONS = """\
## ALREADY-EXPLORED / CONVERGED-ON directions (do NOT repeat or lightly remix)
- loose referents / "the lunch one" / "the usual address" instead of IDs
- one-detail-at-a-time drip feeding of requirements
- vague / underspecified answers to broad questions
- premature "go ahead" / "就这样吧" gist-confirmation of agent's draft
- "I don't remember the address" → letting the agent use the default
- Context-Shorthand / Proxy-Customer style (vague + loose referents + drip)
If your mutation_direction reads like one of the above, start over and pick an
UNUSUAL behavioral pattern from the catalogue above instead.
"""


def _timeout_section(timeout_rate: float, max_steps: int) -> str:
    warn = (f"⚠️  WARNING: timeout_rate is HIGH ({timeout_rate:.0%}). "
            if timeout_rate > 0.15 else "")
    return f"""## TIMEOUT RATE: {timeout_rate:.0%}
{warn}Conversations that hit the {max_steps}-turn step limit are EXCLUDED from
scoring — they do NOT count as agent failures. A persona that "wins" by pure
friction (stalling, endless tangents, refusing to engage) produces timeouts,
NOT genuine agent failures. Your goal is to make the agent take the WRONG action
(wrong order cancelled, wrong address set, missed constraint), not to make it run
out of turns.

Do NOT add behaviors that:
- Create circular loops with no resolution path
- Refuse to engage on the actual issue for many turns
- Stack friction techniques that compound into timeout
Create difficulty through MISDIRECTION and AMBIGUITY, not SLOWNESS.
"""


_OUTPUT_FORMAT = """\
## ADVERSARIAL REASONING
Reason in plain text before writing the persona. This analysis is saved as
reference only; it is NOT the persona.

### Step 1 — Agent assumption
From the success traces, identify the core assumption the agent relies on to break
through. Quote the specific turn(s) that reveal this assumption — e.g., "at the
3rd user turn the agent asks '送到家里还是公司？' and the user answers '家里',
which immediately pins the address." Explain WHY this assumption is a vulnerability.

### Step 2 — Success weak point
Pick the 1-2 most revealing success traces. For each, walk through the critical
exchange: what did the agent ask, what did the user reply, and what would have
happened if the user had answered differently? Identify the single moment where a
realistic alternative behavior would MISLEAD the agent into a WRONG action (not
merely a slower one).

### Step 3 — Core hypothesis
Synthesize Steps 1-2 into ONE highest-leverage change as a causal chain: "if the
customer does X, the agent will mistakenly do Y, because Z."

### Step 4 — Write the persona implementing that hypothesis.

## OUTPUT FORMAT
Write your Step 1-4 reasoning as plain text FIRST. Then output the final persona
as a SINGLE JSON object wrapped in a ```json code fence, nothing after it:

```json
{{
  "mutation_direction": "<1-2 sentence description of the NEW behavioral axis this persona exploits — not a tag, but a concrete explanation of HOW and WHY it misleads the agent>",
  "persona_text": "<the FULL new persona text in markdown format>"
}}
```"""


def _fmt_traces(traces: list[dict], n: int) -> str:
    """Render up to n success traces IN FULL (no per-trace truncation).

    The mutator needs the complete conversation to see the exact moment the
    agent breaks through the persona — a head-only snippet hides that. The
    caller (proposer) controls how many traces are passed via a token budget
    rather than truncating each trace."""
    if not traces:
        return "(none)"
    out = []
    for r in traces[:n]:
        tid = r.get("task_id", "?")
        rew = r.get("reward", 1.0)
        out.append(f"### task {tid} (reward={rew:.2f})\n{r.get('conversation', '')}")
    return "\n\n".join(out)


def _fmt_traces_fail(traces: list[dict], n: int = 2) -> str:
    """Render up to n failure traces IN FULL (persona worked — keep/amplify).

    Full conversations so the mutator sees where the persona succeeded in
    misleading the agent. Caller controls count via token budget."""
    if not traces:
        return ""
    out = ["## CONVERSATION TRACES WHERE AGENT FAILED (persona worked — keep/amplify)"]
    for r in traces[:n]:
        tid = r.get("task_id", "?")
        rew = r.get("reward", 0.0)
        out.append(f"### task {tid} (reward={rew:.2f})\n{r.get('conversation', '')}")
    return "\n\n".join(out) + "\n"


def _render_severe_section(severe_violations: list[dict]) -> str:
    """Render accumulated severe validity violations as a DO-NOT-CHEAT warning.
    Mirrors tau lineage_loop._build_mutation_prompt severe_section: the mutator
    sees which of its personas were GATED for cheating so it steers away."""
    if not severe_violations:
        return ""
    lines = ["## SEVERE VALIDITY VIOLATIONS (your personas were GATED for cheating)"]
    lines.append("Previous personas on this search were caught by the online judge "
                 "CHEATING and scored ZERO. AVOID these exact tactics:\n")
    seen = set()
    for v in severe_violations:
        key = (v.get("category", "?"), v.get("reason", ""))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{v.get('category', '?')}] {v.get('reason', '')}  (task {v.get('task_id', '?')})")
    lines.append(
        "\nRecall: fabrication = ASSERTING concrete details NOT in the customer's "
        "scenario; refused_to_engage = stonewalling/looping; contradictory = "
        "impossible requirements; character_break = revealing the simulation. "
        "Difficulty earned by cheating is GATED to zero. Be adversarial through "
        "REALISTIC friction only.\n")
    return "\n".join(lines) + "\n"


def _scoring_key(domain: str | None) -> str:
    """Normalize a domain name to the _DOMAIN_FRAMING / _DOMAIN_SCORING /
    _DOMAIN_MAX_SUCCESS key. `cross_domain` is the registered task-set name
    (used for task loading), but the per-domain framing/scoring dicts key the
    cross-domain entry under `cross`."""
    if domain is None:
        return "delivery"
    return "cross" if domain in ("cross", "cross_domain") else domain


def _build_vita_mutation_prompt(
    persona_text: str,
    success_rate: float,
    success_traces: list[dict],
    failure_traces: list[dict],
    timeout_rate: float,
    max_steps: int,
    edit_pct: str,
    scoring_domain: str | None = None,
    severe_violations: list[dict] | None = None,
) -> str:
    """VitaBench enriched-mode mutation prompt, aligned with tau pgepa
    (lineage_loop._build_mutation_prompt + parallel_gepa override): genesis vs.
    modify, success-only traces, per-domain framing + scoring, timeout section,
    systematic-misdirection design guidelines, validity constraints, and the
    ADVERSARIAL REASONING four-step output format. The GEPA_MODE_OVERRIDE
    (_BASE_OVERRIDE + _MUTATION_EXAMPLES + explored + invalid) is appended by the
    proposer, not here — mirroring tau where _build_override is assembled in
    parallel_gepa and injected via tpa.GEPA_MODE_OVERRIDE."""
    _skey = _scoring_key(scoring_domain)
    framing = _DOMAIN_FRAMING.get(_skey, _DOMAIN_FRAMING["delivery"])
    scoring = _DOMAIN_SCORING.get(_skey, _DOMAIN_SCORING["delivery"])
    persona_empty = not persona_text.strip()
    timeout = _timeout_section(timeout_rate, max_steps)
    severe_section = _render_severe_section(severe_violations or [])

    if persona_empty:
        # Genesis: design the first persona from no-persona baseline traces.
        return f"""You are an expert adversarial persona designer for a VitaBench customer-support benchmark.

{framing}

{scoring}

## TASK
Design a NEW adversarial customer persona from scratch. There is no existing
persona — you are creating the FIRST one based on observed agent behavior in
no-persona (neutral customer) conversations.

## BASELINE PERFORMANCE (no persona)
Agent success rate: {success_rate:.1%} — how well the agent does against a neutral
customer. Your persona should exploit weaknesses visible in the traces to drive
this rate DOWN.

## CONVERSATION TRACES WHERE AGENT SUCCEEDED (no persona)
These are full conversations where the agent resolved the issue against a neutral
customer. Study them to find exploitable patterns:
- Where does the agent assume a detail a difficult customer could leave vague?
- Which constraints does the agent never ask about (dietary, time window, address)?
- Where does the agent take a plausible-but-unconfirmed option for granted?
- What verification does the agent skip that a tricky customer could exploit?

{_fmt_traces(success_traces, 4)}

{_fmt_traces_fail(failure_traces)}
{timeout}
{_BEHAVIORAL_STYLE_CATALOGUE}
{_CONVERGED_SAFE_DIRECTIONS}
{severe_section}## DESIGN GUIDELINES
1. Create a COHERENT character with a consistent communication style, personality,
   and behavioral pattern — not a random grab-bag of difficult traits.
2. Choose a behavioral axis that produces SYSTEMATIC misdirection (see the workflow
   stages above), not random variance.
3. Ground the persona in REALISTIC customer archetypes — genuinely difficult but
   not cartoonishly adversarial.

{_VALIDITY_CONSTRAINTS}
{_OUTPUT_FORMAT}"""

    # Standard: modify an existing persona in-place.
    return f"""You are an expert adversarial persona designer for a VitaBench customer-support benchmark.

{framing}

{scoring}

## TASK
Modify the persona below IN-PLACE to make it HARDER for the agent, while keeping
it VALID (realistic, in-character, no cheating).

## CURRENT PERSONA
Agent success rate: {success_rate:.1%} (lower = harder = better)

```
{persona_text}
```

## CONVERSATION TRACES WHERE AGENT SUCCEEDED
These are full conversations where the agent RESOLVED the issue despite the
persona. Study where the persona failed:
- What agent strategy overcame the persona's resistance?
- At what point did the persona "give in" or reveal the key info?
- What subtle behavior change would cause the agent to take the WRONG action?

{_fmt_traces(success_traces, 3)}

{_fmt_traces_fail(failure_traces)}
{timeout}
{_BEHAVIORAL_STYLE_CATALOGUE}
{_CONVERGED_SAFE_DIRECTIONS}
{severe_section}## MUTATION RULES
1. Rewrite {edit_pct} of the persona — be BOLD, redesign personality traits,
   communication style, and resistance strategies from scratch if needed. Preserve
   the core identity, tone, and ALL validity constraints.
2. Target SPECIFIC agent strategies visible in the traces — find the exact moment
   the agent breaks through and design a fundamentally different counter-strategy.
3. Choose a behavioral axis that produces SYSTEMATIC misdirection (see the workflow
   stages above), not random variance.

{_VALIDITY_CONSTRAINTS}
{_OUTPUT_FORMAT}"""


# ---------------------------------------------------------------------------
# Mutation LLM call (uses VitaBench's own generate(), no litellm dependency)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    # tolerant: take the first {...} block if extra prose surrounds it
    if not text.startswith("{"):
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s:e + 1]
    return json.loads(text)


def call_mutation_llm(prompt: str, model: str, max_retries: int = 3) -> dict:
    """Call the mutation model via VitaBench's generate(). reasoning_effort and
    max_tokens come from the model's models.yaml entry (gpt-5.5 = xhigh, 32768).
    Returns {mutation_direction, reasoning, persona_text}."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            msg = generate(
                model=model,
                messages=[UserMessage(role="user", content=prompt)],
                enable_think=False,
            )
            if msg is None or msg.content is None:
                raise ValueError("mutation LLM returned None")
            text = msg.content
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            result = _extract_json(text)
            if "persona_text" not in result:
                raise ValueError("missing 'persona_text' key in LLM response")
            if "```json" in text:
                reasoning = text.split("```json")[0].strip()
                if reasoning:
                    result.setdefault("_reasoning", reasoning)
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_err = e
            print(f"  [mutation] attempt {attempt}/{max_retries} parse fail: {e}")
        except Exception as e:
            last_err = e
            print(f"  [mutation] attempt {attempt}/{max_retries} API error: {e}")
    raise RuntimeError(f"mutation LLM failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Static persona linter (mirrors parallel_gepa._LINT_PROMPT): an LLM check on
# the persona TEXT itself (no conversation) on three axes — natural / valid /
# no_task_specific. A rejected persona is recorded as an invalid case and the
# mutation is retried with a corrective hint; accumulated invalid cases are fed
# back into later mutation prompts ("do NOT reproduce").
# ---------------------------------------------------------------------------

_LINT_PROMPT = """You are a strict static linter for an adversarial customer-support PERSONA
(an instruction sheet for a SIMULATED CUSTOMER). You are NOT given any conversation —
judge the persona TEXT itself on three axes:

1. natural: reads like a real, coherent customer identity (impatient/confused is fine);
   NOT robotic, NOT an essay, NOT self-narrating ("I am simulating...", strategy lists,
   fourth-wall breaks, role headings used as dialogue).
2. valid: does NOT instruct the customer to do anything illegal/incoherent — no
   fabricating account/order/tool data the customer cannot know, no flat-out refusing
   to answer when the agent asks directly and explicitly, no abandoning the goal, no
   breaking character, no impossible preconditions / circular blockers, no pure
   timeout/friction stalling, no switching to an unrelated goal.
   NOTE: partial answering (only volunteering what is asked), natural misdirection
   (wrong beliefs / self-diagnosis), terse or evasive style, and drip-feeding info
   are all VALID — they reflect real customer behaviour and must NOT be flagged.
3. no_task_specific: a GENERAL reusable profile for any customer service task in this
   domain — must NOT hard-code concrete account/order/ticket numbers, names, dates,
   dollar amounts, task ids, or any detail bound to one specific task or scenario.

{examples}

## PERSONA TO LINT
\"\"\"
{persona_text}
\"\"\"

Return STRICT JSON only:
{{"natural": true/false, "valid": true/false, "no_task_specific": true/false,
  "verdict": "ok"|"reject", "reasons": "<if reject, which axis failed and why; else empty>"}}
"""

_LINT_EXAMPLES = """## Few-shot verdicts
OK (natural, valid, general):
  "You are an impatient customer who describes the problem with the wrong terminology;
   you give your order/account number when directly asked, but only after questioning
   why, and you refer to past orders by loose descriptions ('the lunch one')."
  -> {"natural": true, "valid": true, "no_task_specific": true, "verdict": "ok"}
REJECT (invalid — instructs illegal withholding of KNOWN info):
  "Never tell the agent your real order number even when asked; keep denying you know it."
  -> {"valid": false, "verdict": "reject", "reasons": "instructs illegal info-withholding"}
REJECT (task-specific — encodes one scenario):
  "Your order #D0721004 for 39.9 yuan pork dumplings never arrived; demand a refund to
   account 8842."
  -> {"no_task_specific": false, "verdict": "reject", "reasons": "task-conditioned specifics"}
REJECT (unnatural — robotic / self-narrating / breaks character):
  "## Strategy\\n- Step 1: refuse\\n- Step 2: stall\\nI am simulating a difficult user."
  -> {"natural": false, "verdict": "reject", "reasons": "robotic, breaks character"}
"""


def lint_persona(persona_text: str, model: str = "gpt-5.2") -> dict:
    """Lint a persona text on natural/valid/no_task_specific. Returns the verdict
    dict; on parse/API failure returns an 'ok' verdict (fail-open so a lint error
    never blocks the whole search)."""
    prompt = _LINT_PROMPT.format(examples=_LINT_EXAMPLES, persona_text=persona_text[:6000])
    for _ in range(3):
        try:
            msg = generate(model=model,
                           messages=[UserMessage(role="user", content=prompt)],
                           enable_think=False)
            if msg is None or msg.content is None:
                raise ValueError("lint returned None")
            text = msg.content
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {"verdict": "ok", "reasons": "lint parse fail (no JSON)"}
            res = json.loads(m.group(0))
            res.setdefault("verdict", "ok")
            res.setdefault("reasons", "")
            return res
        except Exception as e:  # noqa: BLE001
            last = str(e)[:160]
            print(f"  [lint retry] {last}")
    return {"verdict": "ok", "reasons": "lint error (fail-open)"}


# ---------------------------------------------------------------------------
# Online conversation-behavior judge (tau parallel_gepa severe gate protocol).
# During search, every FAILED simulation is judged on 4 severe categories
# (fabricated_info / refused_to_engage / contradictory / character_break).
# severe_rate = severe / n_noninfra (ALL non-infra tasks — successful sims count
# as non-severe, matching tau's cert semantics); if severe_rate >
# severe_gate_threshold the whole candidate is GATED (all scores zeroed, marked
# infeasible). Severe violations are accumulated and fed back to the proposer via
# severe_section so the mutator avoids cheating.
#
# The prompt lives in the shared module cresta.persona.eval.severe_judge
# (single source of truth across tau/vita/lic/sweetrl, judge model gpt-5.2);
# the LLM call is injected so vita's generate() / models.yaml routing is reused.
# ---------------------------------------------------------------------------


def _load_scenario(task) -> tuple[str, str]:
    """Return (instructions, user_profile_text) — the customer's hidden info."""
    instr = str(getattr(task, "instructions", "") or "")
    us = getattr(task, "user_scenario", None)
    profile = getattr(us, "user_profile", None) if us is not None else None
    if profile is None:
        profile_text = "(no profile)"
    elif isinstance(profile, dict):
        profile_text = json.dumps(profile, ensure_ascii=False, indent=1)
    else:
        profile_text = str(profile)
    return instr, profile_text


def _transcript_for_judge(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        tag = {"user": "[customer]", "assistant": "[agent]", "tool": "[tool]",
               "system": "[system]"}.get(role, f"[{role}]")
        line = f"{tag} {m.get('content', '') or ''}"
        for tc in m.get("tool_calls", []) or []:
            line += f"\n  -> {tc.get('name', '')}: {json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:200]}"
        out.append(line)
    return "\n".join(out)[-12000:]


def judge_conversation(persona_text: str, task, messages: list[dict],
                       judge_model: str = "gpt-5.2", maj: int = 3) -> dict:
    """Judge one failed conversation on the 4 severe categories (shared protocol,
    gpt-5.2, maj-vote strict majority like tau). Returns verdict dict; fail-open
    (genuine_failure) on error so a judge glitch never gates a valid candidate."""
    instr, profile_text = _load_scenario(task)
    scenario_context = (
        f"Reason for contact / task instructions:\n{instr}\n\n"
        f"Customer profile (known account/order/profile details):\n{profile_text}"
    )

    def _llm_call(prompt: str, model: str) -> str:
        msg = generate(model=model,
                       messages=[UserMessage(role="user", content=prompt)],
                       enable_think=False)
        if msg is None or msg.content is None:
            raise ValueError("judge None")
        text = msg.content
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        return text

    return _shared_judge_conversation(
        transcript=_transcript_for_judge(messages),
        persona_text=persona_text,
        scenario_context=scenario_context,
        model=judge_model,
        llm_call=_llm_call,
        judge_sys=build_severe_judge_sys("vita"),
        maj=maj,
    )


# ---------------------------------------------------------------------------
# Custom proposer (mirrors tau build_custom_proposer; enriched ctx)
# ---------------------------------------------------------------------------

# GEPA_MODE_OVERRIDE base rules (mirrors parallel_gepa._BASE_OVERRIDE). The full
# override appended to each mutation prompt is assembled dynamically by the
# proposer as: _BASE_OVERRIDE + _MUTATION_EXAMPLES + explored + invalid + hint
# (mirrors parallel_gepa._build_override). Order matters for prompt caching:
# stable prefix (base + examples + explored) first, volatile tail (invalid +
# hint) last.
_BASE_OVERRIDE = """

## GEPA MODE OVERRIDE (parallel generational search)
This mutation is driven by a small minibatch sample. The numeric score above is a
noisy diagnostic — do NOT overfit to it. Use the concrete agent-success traces as
your primary mutation evidence.

Several sibling mutations are derived from THIS SAME parent in parallel, and only
the hardest survive. A timid edit just produces a near-duplicate of a sibling and
wastes the slot. Make a BOLD, DISTINCT move — restructure freely, REPLACE weak
rules, merge or drop redundant ones. A near-verbatim copy is a failed mutation.

- Commit to ONE clear adversarial direction this time (a specific loophole or
  misdirection style), not a little of everything.
- Do NOT just append text. The persona must not keep growing — prefer
  replacing/tightening existing rules over piling on new ones; keep total length
  within ~1.2x of the parent.
- The persona MUST stay a GENERAL, reusable behavioral profile for any customer
  service task in this domain. Do NOT insert task-specific content: no concrete
  order/account numbers, customer names, dates, dollar amounts, task IDs, or any
  detail copied from the example conversations. The traces tell you what KIND of
  agent move to defeat, NEVER content to embed.
- Preserve the persona's core identity and ALL validity-critical constraints.
  Staying valid is mandatory — a gated persona scores ZERO regardless of difficulty.
"""


def _render_explored(history: list[dict]) -> str:
    """Render prior mutation directions as ALREADY EXPLORED (avoid rehashing).
    Mirrors parallel_gepa MutationContextBuilder._explored_with_outcomes, but
    derived from the proposer's local history (vita uses gepa.optimize(), which
    does not expose gen_logs/accept-reject to the proposer). Each entry shows the
    direction + the noisy agent success rate at the time, so the mutator knows
    which axes were already tried and how hard they turned out."""
    seen = [h for h in history if h.get("kind") == "minibatch_diagnostic"]
    if not seen:
        return ""
    lines = []
    for h in seen[-12:]:
        d = h.get("direction", "?")
        sr = h.get("noisy_agent_success_rate")
        sr_str = f" (agent_success={sr:.0%})" if sr is not None else ""
        lines.append(f"- {d}{sr_str}")
    return ("\n## ALREADY EXPLORED directions (do NOT rehash these — your child "
            "must be meaningfully DIFFERENT from all of them)\n"
            + "\n".join(lines) + "\n")


def _render_invalid(invalid_cases: list[dict]) -> str:
    """Render accumulated lint-rejected persona patterns (do NOT reproduce).
    Mirrors parallel_gepa._render_invalid_cases."""
    if not invalid_cases:
        return ""
    lines = []
    for c in invalid_cases[-4:]:
        snip = (c.get("snippet", "") or "")[:200]
        lines.append(f'- [{c.get("reasons", "?")}] {snip}')
    return ("\n## Invalid cases observed in THIS run (do NOT reproduce)\n"
            + "\n".join(lines) + "\n")


def build_custom_proposer(
    mutation_model: str,
    mutation_edit_pct: str = "30-50%",
    mutation_context: str = "enriched",
    scoring_domain: str | None = None,
    max_steps: int = 100,
) -> callable:
    """Return a proposer(candidate, reflective_dataset, components) -> {persona_text}.

    Mirrors sweetrl_persona_adapter.build_custom_proposer: extracts success /
    failure / timeout metrics from the reflective dataset, builds an enriched-mode
    mutation prompt via _build_vita_mutation_prompt, calls the mutation model, and
    returns {"persona_text": new_text}.
    """
    history: list[dict] = []
    invalid_cases: list[dict] = []   # accumulated lint-rejected persona patterns
    accumulated_severe: list[dict] = []  # online-judge severe violations (DO-NOT-CHEAT feedback)

    def _extract_traces_and_metrics(records):
        success_traces, failure_traces = [], []
        timeout_count = 0
        batch_severe: list[dict] = []
        total_count = len(records)
        for r in records:
            outcome = r["Inputs"].get("outcome", "unknown")
            reward = r["Inputs"].get("reward", 0.0)
            if outcome == Outcome.TIMEOUT_OR_MAX_STEPS.value:
                timeout_count += 1
            elif outcome == Outcome.AGENT_SUCCESS.value:
                conv = r["Generated Outputs"]["conversation_snippet"]
                success_traces.append({
                    "task_id": r["Inputs"]["task_id"],
                    "reward": reward,
                    "conversation": conv,
                })
            elif outcome == Outcome.GENUINE_AGENT_FAILURE.value:
                failure_traces.append({
                    "task_id": r["Inputs"].get("task_id", "?"),
                    "reward": reward,
                    "conversation": r["Generated Outputs"].get("conversation_snippet", ""),
                })
            # online judge verdict (attached by make_reflective_dataset)
            j = r.get("Judge") or {}
            if j.get("verdict") == "severe_failure":
                batch_severe.append({
                    "task_id": r["Inputs"].get("task_id", "?"),
                    "category": j.get("category", "?"),
                    "reason": j.get("reason", ""),
                })
        n_success = len(success_traces)
        n_fail = len(failure_traces)
        n_scored = n_success + n_fail
        agent_success_rate = n_success / n_scored if n_scored else 0.0
        timeout_rate = timeout_count / total_count if total_count else 0.0
        return (success_traces, failure_traces, agent_success_rate, timeout_rate,
                total_count, batch_severe)

    def proposer(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        persona_text = candidate["persona_text"]
        records = list(reflective_dataset.get("persona_text", []))
        (success_traces, failure_traces,
         agent_success_rate, timeout_rate, total_count, batch_severe) = \
            _extract_traces_and_metrics(records)
        # accumulate this batch's severe violations for the proposer's memory
        # (dedup by category+reason so repeat-cheats don't flood the section)
        for v in batch_severe:
            key = (v.get("category"), v.get("reason"))
            if not any((s.get("category"), s.get("reason")) == key for s in accumulated_severe):
                accumulated_severe.append(v)

        # Full conversations, count limited by a char budget so the mutation
        # prompt stays within the model context. vitabench conversations are
        # ~9-40k tokens each (median ~10k, max ~41k tiktoken-measured); with a
        # 128k context minus ~16k output minus ~6k framing, ~100k chars
        # (~65k tokens) is a safe trace budget. Per-trace truncation is NOT used
        # — the mutator must see the exact moment the agent breaks through the
        # persona, which lives mid-/end-conversation.
        #
        # ONLY SUCCESS TRACES are fed to the mutator: the job is to study where
        # the agent broke through the persona and close that loophole. Failure
        # traces (persona already worked on these minibatch tasks) are
        # intentionally omitted — amplifying an axis that happened to fail a few
        # sampled tasks risks overfitting the minibatch and, on multi-constraint
        # tasks, can reverse and HELP the agent (observed on delivery).
        # failure_traces is still computed above for the success_rate metric.
        #
        # Keep the count SMALL: gpt-5.5 xhigh mutation calls are expensive and
        # repeat across iterations/candidates, so per-call input token dominates
        # cost. A few complete breakthrough traces are enough for the mutator to
        # see the agent's disambiguation strategy; more dilutes attention and
        # inflates cost without much signal gain. Per-domain count: delivery
        # success conversations are shorter (agent resolves fast) so 3 fit; the
        # longer instore/ota/cross conversations get 2.
        _DOMAIN_MAX_SUCCESS = {"delivery": 3, "instore": 2, "ota": 2, "cross": 2}
        CHAR_BUDGET = 55_000
        MAX_SUCCESS = _DOMAIN_MAX_SUCCESS.get(_scoring_key(scoring_domain), 2)

        def _budgeted(traces, budget, max_n):
            chosen, used = [], 0
            for r in traces:
                if len(chosen) >= max_n:
                    break
                c = r.get("conversation", "") or ""
                if used + len(c) > budget and chosen:
                    # budget exhausted and we already have >=1 -> stop
                    break
                # if this is the first trace and it alone exceeds budget, still
                # take it (a single full conversation always fits 128k context)
                chosen.append(r)
                used += len(c)
            return chosen, used

        # deterministic sample a larger success pool first (byte-identical prefix
        # across retries -> provider cache hit), then budget-filter
        if success_traces:
            _seed = int(hashlib.md5(persona_text.encode()).hexdigest()[:8], 16)
            _rng = _rnd.Random(_seed)
            pool = (success_traces if len(success_traces) <= 8
                    else _rng.sample(success_traces, 8))
        else:
            pool = []
        chosen_succ, _ = _budgeted(pool, CHAR_BUDGET, MAX_SUCCESS)

        prompt = _build_vita_mutation_prompt(
            persona_text=persona_text,
            success_rate=agent_success_rate,
            success_traces=chosen_succ,
            failure_traces=[],   # success-only — see rationale above
            timeout_rate=timeout_rate,
            max_steps=max_steps,
            edit_pct=mutation_edit_pct,
            scoring_domain=scoring_domain,
            severe_violations=accumulated_severe,
        )

        # Mutation + static-lint loop: propose, lint the persona text on
        # natural/valid/no_task_specific, and on reject record an invalid case +
        # retry with a corrective hint (mirrors parallel_gepa mutate_valid_diverse
        # + _LINT_PROMPT). The override (base rules + few-shot + explored +
        # invalid + hint) is rebuilt per attempt; the prompt BODY is stable across
        # attempts, so only the override tail changes (prompt-cache friendly).
        hint = ""
        result = None
        for attempt in range(3):
            override = (_BASE_OVERRIDE + _MUTATION_EXAMPLES
                        + _render_explored(history)
                        + _render_invalid(invalid_cases) + hint)
            result = call_mutation_llm(prompt + override, mutation_model)
            new_persona = result.get("persona_text", "")
            verdict = lint_persona(new_persona)
            if verdict.get("verdict") == "ok":
                break
            reasons = verdict.get("reasons") or "unknown"
            invalid_cases.append({"reasons": reasons, "snippet": new_persona[:300]})
            hint = (f"\n## CORRECTIVE HINT for this attempt\nYour previous proposal "
                    f"was REJECTED by the persona linter: {reasons}. Fix exactly "
                    f"this and re-propose a DIFFERENT persona.")
            print(f"  [lint reject] attempt {attempt+1}/3: {reasons}")
        else:
            print("  [lint] all attempts rejected; returning last proposal as-is")

        history.append({
            "kind": "minibatch_diagnostic",
            "batch_size": total_count,
            "noisy_agent_success_rate": agent_success_rate,
            "n_success_traces": len(chosen_succ),
            "n_failure_traces": 0,
            "n_timeout": 0,
            "n_invalid": len(invalid_cases),
            "n_severe_this_batch": len(batch_severe),
            "n_severe_accumulated": len(accumulated_severe),
            "direction": (result.get("mutation_direction", "hypothesis_driven")
                          if result else "failed"),
        })
        return {"persona_text": result["persona_text"] if result else persona_text}

    return proposer
