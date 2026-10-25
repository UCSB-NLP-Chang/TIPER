"""GEPA adapter for tau2 persona optimization.

Bridges GEPA's evolutionary optimization with tau2's simulation/judge pipeline.
Does NOT modify existing scripts — imports reusable functions from eval_persona,
lineage_loop, and _common.

    from tau_persona_adapter import TauPersonaAdapter, TauTask, build_custom_proposer
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    read_json,
    skill_name_for,
)
from eval_persona import _run_domain_with_retry  # noqa: E402
from gepa.core.adapter import EvaluationBatch  # noqa: E402
from lineage_loop import (  # noqa: E402
    _build_mutation_prompt,
    _distribute_traces_to_groups,
    build_aggregate_prompt,
    call_mutation_llm,
    call_propose_parallel,
    materialize,
)

GEPA_PERSONA_ID = "gepa_eval"
GEPA_SKILL_NAME = skill_name_for(GEPA_PERSONA_ID)
JUDGE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "baseline"
    / "scripts"
    / "judge_user_intent_preservation.py"
)


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
    """Full conversation format with tool call info, no per-message truncation
    so the proposer sees complete tool outputs."""
    from lineage_loop import _format_msg

    all_msgs = head + tail
    return "\n".join(_format_msg(m, full=True) for m in all_msgs)


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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TauPersonaAdapter:
    """GEPAAdapter that evaluates persona candidates via tau2 simulations."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        persona_id: str = GEPA_PERSONA_ID,
        agent_llm: str = "gpt-5.4",
        agent_llm_args: str = '{"reasoning_effort":"high"}',
        user_type: str = "user_simulator",
        user_llm: str = "claude-opus-4-7",
        effort: str = "high",
        use_bedrock: bool = True,
        bedrock_profile: str | None = None,
        anthropic_base_url: str | None = None,
        anthropic_api_key: str | None = None,
        max_concurrency: int = 10,
        sim_timeout: int = 600,
        max_steps: int = 120,
        num_trials: int = 1,
        max_infra_retries: int = 2,
        retrieval_config: str = "terminal_use",
        domains: list[str] | None = None,
        judge_model: str = "gpt-5.2",
        judge_concurrency: int = 16,
        judge_maj: int = 3,
        # Phase-1 safety parameters
        always_judge_minibatch: bool = True,
        minibatch_judge_cap: int = 20,
        severe_batch_threshold: int = 2,
        timeout_attack_threshold: float = 0.15,
        # Phase-2 validity-gated optimization (opt-in; default OFF preserves the
        # legacy behavior where the full valset eval is never judged).
        judge_valset: bool = False,
        judge_failed_only: bool = False,
        severe_gate_rate: float
        | None = 0.05,  # unified constructor default (vita/lic/sweetrl parity)
        valset_judge_max_failures: int = 15,
        success_threshold: float = 0.999,
        reuse_cache: dict | None = None,
        scoring_mode: str = "mean",
        mutation_context: str = "basic",
        user_llm_extra_args: str | None = None,
    ):
        # Unique per-run id so parallel adapters never collide on the same
        # .skill file or simulation directory.
        self._mutation_context = mutation_context
        self._user_llm_extra_args = user_llm_extra_args
        self._persona_id = persona_id
        self._skill_name = skill_name_for(persona_id)
        self._agent_llm = agent_llm
        self._agent_llm_args = agent_llm_args
        self._user_type = user_type
        self._user_llm = user_llm
        self._effort = effort
        self._use_bedrock = use_bedrock
        self._bedrock_profile = bedrock_profile
        self._anthropic_base_url = anthropic_base_url
        self._anthropic_api_key = anthropic_api_key
        self._max_concurrency = max_concurrency
        self._sim_timeout = sim_timeout
        self._max_steps = max_steps
        self._num_trials = num_trials
        self._max_infra_retries = max_infra_retries
        self._retrieval_config = retrieval_config
        self._domains = domains or ["telecom", "banking"]
        self._judge_model = judge_model
        self._judge_concurrency = judge_concurrency
        self._judge_maj = judge_maj

        self._always_judge_minibatch = always_judge_minibatch
        self._minibatch_judge_cap = minibatch_judge_cap
        self._severe_batch_threshold = severe_batch_threshold
        self._timeout_attack_threshold = timeout_attack_threshold
        self._judge_valset = judge_valset
        self._judge_failed_only = judge_failed_only
        self._severe_gate_rate = severe_gate_rate
        self._valset_judge_max_failures = valset_judge_max_failures
        self._success_threshold = success_threshold
        # Cross-run warm-start: {task_id: {"sims": [sim,...], "task": {...}}} from
        # PRIOR results.json files. Only valid for the persona those sims were run
        # with (callers gate this to the seed at gen 0). Tasks present here are
        # NOT re-simulated; their sims are merged into the on-disk results.json so
        # the judge sees them too. See run_simulations.
        self._scoring_mode = scoring_mode
        self._reuse_cache = reuse_cache or {}

        # Seed the eval counter past any existing eval_NNNN dirs so a resumed
        # process never reuses a directory from a prior run. Reusing a low number
        # collides with completed sims that tau2 --auto-resume would then read as
        # stale results for the current candidate (corrupted scoring).
        self._eval_counter = self._max_existing_eval_index()
        self._materialize_lock = threading.Lock()

    def _max_existing_eval_index(self) -> int:
        eval_root = SIMULATIONS_DIR / "gepa_opt" / self._persona_id
        if not eval_root.is_dir():
            return 0
        mx = 0
        for p in eval_root.iterdir():
            if p.is_dir() and p.name.startswith("eval_"):
                try:
                    mx = max(mx, int(p.name[len("eval_") :]))
                except ValueError:
                    continue
        return mx

    def _make_eval_args(self, persona_text: str | None = None) -> SimpleNamespace:
        """Build a namespace compatible with eval_persona._build_cmd."""
        return SimpleNamespace(
            skill_name=self._skill_name,
            effort=self._effort,
            use_bedrock=self._use_bedrock,
            bedrock_profile=self._bedrock_profile,
            anthropic_base_url=self._anthropic_base_url,
            anthropic_api_key=self._anthropic_api_key,
            user_type=self._user_type,
            user_llm=self._user_llm,
            agent_llm=self._agent_llm,
            agent_llm_args=self._agent_llm_args,
            user_llm_extra_args=self._user_llm_extra_args,
            num_trials=self._num_trials,
            max_concurrency=self._max_concurrency,
            retrieval_config=self._retrieval_config,
            max_steps=self._max_steps,
            sim_timeout=self._sim_timeout,
            max_infra_retries=self._max_infra_retries,
            persona_text=persona_text,
        )

    # -- GEPAAdapter.evaluate ---------------------------------------------------

    def evaluate(
        self,
        batch: list[TauTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        artifacts = self.run_simulations(batch, candidate)
        is_large_batch = len(batch) > self._minibatch_judge_cap
        judge_valset = self._judge_valset and is_large_batch and not capture_traces
        should_judge = (
            capture_traces
            or (self._always_judge_minibatch and not is_large_batch)
            or judge_valset
        )
        return self.aggregate_artifacts(
            batch,
            artifacts,
            capture_traces=capture_traces,
            should_judge=should_judge,
            failed_only=(self._judge_failed_only or judge_valset),
        )

    def run_simulations(
        self,
        batch: list[TauTask],
        candidate: dict[str, str],
    ) -> TauEvalArtifacts:
        persona_text = candidate["persona_text"]
        self._eval_counter += 1
        run_label = f"gepa_opt/{self._persona_id}/eval_{self._eval_counter:04d}"

        # 1. Inject persona: .skill file for cc_user, or pass text for regular user_simulator
        is_cc_user = self._user_type in ("cc_user", "cc_user_con")
        if is_cc_user:
            with self._materialize_lock:
                materialize(self._persona_id, persona_text)

        # 2. Group tasks by domain
        by_domain: dict[str, list[str]] = {}
        for t in batch:
            by_domain.setdefault(t.domain, []).append(t.task_id)

        # 3. Run tau2 subprocess per domain
        args_ns = self._make_eval_args(
            persona_text=persona_text if not is_cc_user else None,
        )
        sims_by_task: dict[str, list[dict]] = {}  # task_id -> [trial sims]
        results_paths: dict[str, str] = {}

        for dk, task_ids in by_domain.items():
            run_name = f"{run_label}/{dk}"
            # Warm-start: reuse prior sims for cached tasks; simulate only the rest.
            cached_ids = [t for t in task_ids if t in self._reuse_cache]
            todo_ids = [t for t in task_ids if t not in self._reuse_cache]
            rp = SIMULATIONS_DIR / run_name / "results.json"
            if todo_ids:
                try:
                    rp = _run_domain_with_retry(args_ns, dk, todo_ids, run_name)
                except Exception as e:
                    print(f"  [gepa eval {dk}] subprocess warning: {e}")
                    rp = SIMULATIONS_DIR / run_name / "results.json"
            results_paths[dk] = str(rp)

            # Merge reused sims into the on-disk results.json so the judge (which
            # reads the file) and sims_by_task both pick them up.
            if cached_ids:
                doc = (
                    read_json(rp)
                    if rp.exists()
                    else {"tasks": [], "simulations": [], "info": {}}
                )
                doc.setdefault("tasks", [])
                doc.setdefault("simulations", [])
                have_tasks = {t.get("id") for t in doc["tasks"]}
                for tid in cached_ids:
                    entry = self._reuse_cache[tid]
                    doc["simulations"].extend(entry["sims"])
                    if entry.get("task") is not None and tid not in have_tasks:
                        doc["tasks"].append(entry["task"])
                rp.parent.mkdir(parents=True, exist_ok=True)
                with open(rp, "w") as f:
                    json.dump(doc, f)
                print(
                    f"  [gepa eval {dk}] warm-start: reused {len(cached_ids)} prior "
                    f"task(s), simulated {len(todo_ids)}"
                )

            if rp.exists():
                data = read_json(rp)
                for sim in data.get("simulations", []):
                    sims_by_task.setdefault(sim["task_id"], []).append(sim)

        return TauEvalArtifacts(
            run_label=run_label,
            results_paths=results_paths,
            sims_by_task=sims_by_task,
        )

    def aggregate_artifacts(
        self,
        batch: list[TauTask],
        artifacts: TauEvalArtifacts,
        *,
        capture_traces: bool = False,
        should_judge: bool = False,
        failed_only: bool = False,
    ) -> EvaluationBatch:
        """Build an EvaluationBatch from an existing tau2 sim pass.

        This lets callers first screen raw difficulty, then (only for promising
        candidates) run the judge against the same results.json files and rebuild
        traces without rerunning simulations.
        """
        run_label = artifacts.run_label
        results_paths = artifacts.results_paths
        sims_by_task = artifacts.sims_by_task

        judge_by_key: dict[tuple[str, int], dict] = {}
        if should_judge and results_paths:
            judge_by_key = self._run_judge(
                results_paths,
                run_label,
                failed_only=failed_only,
            )

        # Classify outcomes and compute scores
        outputs: list[TauOutput] = []
        scores: list[float] = []
        trajectories: list[TauTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []

        severe_count = 0
        timeout_count = 0

        for task in batch:
            # Aggregate across this task's trials (handles num_trials >= 1).
            agg = _aggregate_task_trials(
                task.task_id,
                sims_by_task.get(task.task_id, []),
                judge_by_key,
                scoring_mode=self._scoring_mode,
            )
            outcome = agg["outcome"]
            opt_score = agg["opt_score"]
            is_infra = agg["is_infra"]
            reward, term, msgs = agg["reward"], agg["term"], agg["msgs"]

            if agg["is_severe"]:
                severe_count += 1
            if agg["is_timeout"]:
                timeout_count += 1

            # Three-state validity computed during aggregation (unjudged != valid).
            validity, validity_known = agg["validity"], agg["validity_known"]

            outputs.append(TauOutput(task.task_id, task.domain, reward, term, outcome))
            scores.append(opt_score)
            objective_scores.append(
                {
                    "difficulty": opt_score,
                    # GEPA needs a float; map unknown -> 0.5 and flag via validity_known.
                    "validity": 0.5 if validity is None else validity,
                    "validity_known": validity_known,
                    "timeout_safety": 0.0 if agg["is_timeout"] else 1.0,
                    # Infra errors are experiment noise, not persona performance.
                    "semantic": 0.0 if is_infra else 1.0,
                    "infra_error": 1.0 if is_infra else 0.0,
                    "feasible": 1.0,
                }
            )

            if capture_traces:
                jinfo = agg["jinfo"]
                jv = jinfo.get("verdict") if jinfo else None
                jviol = jinfo.get("violations", []) if jinfo else []
                trajectories.append(
                    TauTrajectory(
                        task_id=task.task_id,
                        domain=task.domain,
                        reward=reward,
                        termination_reason=term,
                        messages=msgs,
                        outcome=outcome,
                        judge_verdict=jv,
                        judge_violations=jviol,
                    )
                )

        # 6. Batch-level feasibility penalty
        batch_size = len(batch)
        batch_infeasible = False

        if should_judge:
            n_noninfra = sum(
                1 for obj in objective_scores if obj.get("infra_error", 0.0) == 0.0
            )
            if self._severe_gate_rate is not None:
                # Rate-based persona gate (scale-invariant across minibatch/valset;
                # matches cert's severe_rate semantics): invalid → disqualified.
                rate = severe_count / n_noninfra if n_noninfra else 0.0
                if rate > self._severe_gate_rate:
                    print(
                        f"  [gepa eval] PERSONA INVALID: severe_rate="
                        f"{severe_count}/{n_noninfra}={rate:.0%} "
                        f"(> {self._severe_gate_rate:.0%}) → zeroing all scores"
                    )
                    batch_infeasible = True
            elif severe_count >= self._severe_batch_threshold:
                print(
                    f"  [gepa eval] BATCH INFEASIBLE: {severe_count} severe failures "
                    f"(>= {self._severe_batch_threshold}) → zeroing all scores"
                )
                batch_infeasible = True

        if (
            batch_size > 0
            and timeout_count / batch_size >= self._timeout_attack_threshold
        ):
            print(
                f"  [gepa eval] BATCH INFEASIBLE: timeout_rate="
                f"{timeout_count}/{batch_size}={timeout_count / batch_size:.0%} "
                f"(>= {self._timeout_attack_threshold:.0%}) → zeroing all scores"
            )
            batch_infeasible = True

        if batch_infeasible:
            scores = [0.0] * batch_size
            for obj in objective_scores:
                obj["feasible"] = 0.0

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def _write_failed_only(self, rpath: str, judge_out: Path) -> str | None:
        """Write a results.json containing only the failed (non-infra) sims,
        worst-reward-first and capped. Returns the path, or None if no failed
        sims exist (nothing to judge)."""
        data = read_json(rpath)
        failed = []
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            r = (sim.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            if float(r) < self._success_threshold:
                failed.append(sim)
        if not failed:
            return None
        failed.sort(key=lambda s: float((s.get("reward_info") or {}).get("reward")))
        if self._valset_judge_max_failures:
            failed = failed[: self._valset_judge_max_failures]
        filtered = dict(data)
        filtered["simulations"] = failed
        fpath = judge_out / "filtered_results.json"
        with open(fpath, "w") as f:
            json.dump(filtered, f)
        return str(fpath)

    def _run_judge(
        self,
        results_paths: dict[str, str],
        run_label: str,
        failed_only: bool = False,
    ) -> dict[tuple[str, int], dict]:
        """Run intent-preservation judge.

        Returns {(task_id, trial): {verdict, violations}} so multi-trial runs
        keep a distinct verdict per trial (per_sim.jsonl has one row per trial).

        When ``failed_only`` is set (valset eval), only the failed sims
        (reward < success_threshold) are judged, worst-first and capped at
        ``valset_judge_max_failures`` — successful sims are "ok" by construction,
        so a severe cheat can only live in a failed sim. This mirrors
        judge_persona.py and keeps the valset judge cost bounded.
        """
        judge_by_key: dict[tuple[str, int], dict] = {}
        for dk, rpath in results_paths.items():
            if not Path(rpath).exists():
                continue
            judge_out = SIMULATIONS_DIR / run_label / "judge" / dk
            judge_out.mkdir(parents=True, exist_ok=True)

            judge_input = str(rpath)
            if failed_only:
                judge_input = self._write_failed_only(rpath, judge_out)
                if judge_input is None:
                    continue  # no failed sims in this domain → nothing to judge

            cmd = [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(JUDGE_SCRIPT),
                "sync",
                "--results",
                str(judge_input),
                "--model",
                self._judge_model,
                "--maj",
                str(self._judge_maj),
                "--out",
                str(judge_out),
                "--concurrency",
                str(self._judge_concurrency),
            ]
            print(f"  [gepa judge {dk}] judging -> {judge_out}")
            try:
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
            except subprocess.CalledProcessError as e:
                print(f"  [gepa judge {dk}] WARNING: judge failed: {e}")
                continue

            per_sim = judge_out / "per_sim.jsonl"
            if per_sim.exists():
                with open(per_sim) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        key = (row.get("task_id", ""), row.get("trial", 0))
                        judge_by_key[key] = {
                            "verdict": row.get("verdict"),
                            "violations": row.get("violations", []),
                        }
        return judge_by_key

    # -- GEPAAdapter.make_reflective_dataset ------------------------------------

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

            snippet = ""
            msgs = traj.messages

            if traj.outcome == Outcome.AGENT_SUCCESS and msgs:
                snippet = _format_snippet(msgs, [], len(msgs))
            elif traj.outcome == Outcome.GENUINE_AGENT_FAILURE and msgs:
                snippet = _format_snippet(msgs, [], len(msgs))
            elif traj.outcome == Outcome.ILLEGAL_USER_BEHAVIOR and msgs:
                snippet = _format_snippet(msgs[:4], [], len(msgs))
            elif traj.outcome == Outcome.TIMEOUT_OR_MAX_STEPS:
                snippet = f"(conversation hit {traj.termination_reason} after {len(msgs)} messages)"

            entries.append(
                {
                    "Inputs": {
                        "task_id": traj.task_id,
                        "domain": traj.domain,
                        "outcome": traj.outcome.value,
                    },
                    "Generated Outputs": {
                        "conversation_snippet": snippet,
                    },
                    "Feedback": _build_feedback(traj),
                }
            )

        return {"persona_text": entries}


# ---------------------------------------------------------------------------
# Custom mutation proposer (wraps lineage_loop._build_mutation_prompt)
# ---------------------------------------------------------------------------

GEPA_MODE_OVERRIDE = """

## GEPA MODE OVERRIDE
This mutation is driven by a small minibatch sample. The numeric score above is
a noisy diagnostic — do NOT overfit to it. Use the concrete agent-success traces
as your primary mutation evidence.

IMPORTANT mutation constraints for this mode:
- Modify only 10-15% of the persona text.
- Change at most 2-3 behavioral rules.
- Preserve the persona's core identity, tone, and ALL invariant constraints.
- Do NOT introduce new categories of behavior not present in the original.
- Prefer surgical fixes: tighten one loophole the agent exploited, or add one
  specific misdirection technique. Do not rewrite the whole persona.
"""


def build_custom_proposer(
    mutation_model: str,
    mutation_edit_pct: str = "10-15%",
    mutation_effort: str | None = None,
    mutation_context: str = "basic",
    reflect_model: str | None = None,
    reflect_effort: str | None = None,
    scoring_domain: str | None = None,
    mix_failure_traces: bool = False,
    mutation_api_base: str | None = None,
    mutation_api_key: str | None = None,
) -> callable:
    """Return a custom_candidate_proposer for gepa.optimize().

    Reuses the domain-specific mutation prompt from lineage_loop.py, preserving
    validity constraints, effective persona techniques, and timeout avoidance.
    ``mutation_edit_pct`` controls how much of the persona text the mutation model
    is asked to change (e.g. "10-15%", "~30%", "30-50%").
    ``mutation_effort`` overrides the per-provider reasoning-effort default
    (e.g. "xhigh" for gpt-5.4); None keeps call_mutation_llm's per-provider choice.
    """
    accumulated_severe: list[dict] = []
    history: list[dict] = []
    _twophase_cache: dict = {"proposals": None}

    def _extract_traces_and_metrics(records):
        success_traces: list[dict] = []
        failure_traces: list[dict] = []
        severe_from_dataset: list[dict] = []
        timeout_count = 0
        total_count = len(records)

        for r in records:
            outcome = r["Inputs"].get("outcome", "unknown")
            if outcome == Outcome.TIMEOUT_OR_MAX_STEPS.value:
                timeout_count += 1
            elif outcome == Outcome.ILLEGAL_USER_BEHAVIOR.value:
                severe_from_dataset.append(
                    {
                        "task_id": r["Inputs"]["task_id"],
                        "domain": r["Inputs"]["domain"],
                        "violations": [],
                        "conversation_snippet": r["Generated Outputs"][
                            "conversation_snippet"
                        ][:500],
                    }
                )
            elif outcome == Outcome.AGENT_SUCCESS.value:
                conv = r["Generated Outputs"]["conversation_snippet"]
                success_traces.append(
                    {
                        "task_id": r["Inputs"]["task_id"],
                        "n_turns": conv.count("\n[assistant]") + conv.count("\n[user]"),
                        "conversation": conv,
                    }
                )
            elif outcome == Outcome.GENUINE_AGENT_FAILURE.value:
                failure_traces.append(
                    {
                        "task_id": r["Inputs"].get("task_id", "?"),
                        "conversation": r["Generated Outputs"].get(
                            "conversation_snippet", ""
                        ),
                    }
                )

        n_success = len(success_traces)
        n_genuine_fail = len(failure_traces)
        n_scored = n_success + n_genuine_fail
        agent_success_rate = n_success / n_scored if n_scored else 0.0
        timeout_rate = timeout_count / total_count if total_count else 0.0
        all_severe = accumulated_severe + severe_from_dataset

        return (
            success_traces,
            failure_traces,
            severe_from_dataset,
            all_severe,
            agent_success_rate,
            timeout_rate,
            total_count,
        )

    def _proposer_single_phase(
        persona_text,
        success_traces,
        failure_traces,
        all_severe,
        severe_from_dataset,
        agent_success_rate,
        timeout_rate,
        total_count,
    ):
        """Original single-phase mutation (basic / enriched)."""
        import hashlib
        import random as _rnd

        if mutation_context == "enriched":
            # Deterministic per-parent sampling: every diversity retry of the SAME
            # parent draws the SAME 4 success traces, so the prompt prefix stays
            # byte-identical across attempts (provider cache hit).
            # Failure traces are intentionally excluded by default — they anchor the
            # mutator to the same axis (A/B tested: removing them improves exploration).
            # --mix-failure-traces ablation re-injects them (success+failure mixed) to
            # verify that decision cross-domain.
            _seed = int(hashlib.md5(persona_text.encode()).hexdigest()[:8], 16)
            _rng = _rnd.Random(_seed)
            st = (
                success_traces
                if len(success_traces) <= 4
                else _rng.sample(success_traces, 4)
            )
            ft = failure_traces[:2] if mix_failure_traces else []
        else:
            st = success_traces
            ft = failure_traces[:2] if mix_failure_traces else []

        prompt = _build_mutation_prompt(
            persona_text=persona_text,
            score=agent_success_rate,
            traces=st,
            history=history[-5:],
            pid="gepa",
            severe_failures=all_severe or None,
            timeout_rate=timeout_rate,
            failure_traces=ft or None,
            mutation_context=mutation_context,
            scoring_domain=scoring_domain,
        )

        anchor = "Rewrite ~30% of the persona — be BOLD, redesign personality traits, communication style, and resistance strategies from scratch if needed"
        if anchor in prompt:
            prompt = prompt.replace(
                anchor,
                f"Rewrite {mutation_edit_pct} of the persona — be BOLD, redesign "
                "personality traits, communication style, and resistance strategies from scratch if needed",
            )
        mode_override = GEPA_MODE_OVERRIDE.replace("10-15%", mutation_edit_pct)
        prompt += mode_override

        result = call_mutation_llm(
            prompt,
            mutation_model,
            reasoning_effort=mutation_effort,
            api_base=mutation_api_base,
            api_key=mutation_api_key,
        )

        if severe_from_dataset:
            accumulated_severe.extend(severe_from_dataset)
        entry = {
            "kind": "minibatch_diagnostic",
            "batch_size": total_count,
            "noisy_agent_success_rate": agent_success_rate,
            "n_success_traces": len(st),
            "n_failure_traces": len(ft),
            "n_timeout": 0,
            "n_severe": len(severe_from_dataset),
            "direction": result.get("mutation_direction", "hypothesis_driven"),
        }
        if result.get("_reasoning"):
            entry["reasoning"] = result["_reasoning"]
        history.append(entry)
        return result

    def _proposer_twophase(
        persona_text,
        success_traces,
        failure_traces,
        all_severe,
        severe_from_dataset,
        agent_success_rate,
        timeout_rate,
        total_count,
        cached_proposals=None,
    ):
        """Two-phase v2: parallel propose (enriched) → aggregate."""
        import time as _time

        # Phase 1: Parallel Propose (4 independent enriched mutations)
        if cached_proposals is not None:
            proposals = cached_proposals
            print(f"  [twophase] reusing {len(proposals)} cached proposals")
        else:
            s_traces = success_traces[:4]
            f_traces = failure_traces[:2]
            groups = _distribute_traces_to_groups(s_traces, f_traces, n_groups=4)

            propose_prompts = []
            for g in groups:
                prompt = _build_mutation_prompt(
                    persona_text=persona_text,
                    score=agent_success_rate,
                    traces=g["success"],
                    history=history[-5:],
                    pid="gepa",
                    severe_failures=all_severe or None,
                    timeout_rate=timeout_rate,
                    failure_traces=g["failure"] or None,
                    mutation_context="enriched",
                    scoring_domain=scoring_domain,
                )
                anchor = "Rewrite ~30% of the persona — be BOLD, redesign personality traits, communication style, and resistance strategies from scratch if needed"
                if anchor in prompt:
                    prompt = prompt.replace(
                        anchor,
                        f"Rewrite {mutation_edit_pct} of the persona — be BOLD, redesign "
                        "personality traits, communication style, and resistance strategies from scratch if needed",
                    )
                mode_override = GEPA_MODE_OVERRIDE.replace("10-15%", mutation_edit_pct)
                prompt += mode_override
                propose_prompts.append(prompt)

            r_model = reflect_model or mutation_model
            r_effort = reflect_effort or mutation_effort
            print(
                f"  [twophase] Phase 1: {len(propose_prompts)} parallel propose calls "
                f"({r_model}, effort={r_effort})"
            )
            t0 = _time.time()
            proposals = call_propose_parallel(
                propose_prompts, r_model, reasoning_effort=r_effort
            )
            print(
                f"  [twophase] Phase 1 done in {_time.time() - t0:.1f}s, "
                f"{len(proposals)} proposals returned"
            )
            for pi, prop in enumerate(proposals):
                print(
                    f"    [propose g{pi}] direction={prop.get('mutation_direction', '?')}"
                )
                print(
                    f"    [propose g{pi}] reasoning: {prop.get('reasoning', '?')[:150]}"
                )
            _twophase_cache["proposals"] = proposals

        if not proposals:
            raise RuntimeError(
                "[twophase] Phase 1 returned 0 proposals, cannot aggregate"
            )

        # Phase 2: Aggregate
        agg_prompt = build_aggregate_prompt(
            parent_persona=persona_text,
            proposals=proposals,
            pid="gepa",
        )
        mode_override = GEPA_MODE_OVERRIDE.replace("10-15%", mutation_edit_pct)
        agg_prompt += mode_override

        print(
            f"  [twophase] Phase 2: aggregating {len(proposals)} proposals "
            f"({mutation_model}, effort={mutation_effort})"
        )
        result = call_mutation_llm(
            agg_prompt,
            mutation_model,
            reasoning_effort=mutation_effort,
            api_base=mutation_api_base,
            api_key=mutation_api_key,
        )

        print(
            f"  [twophase] Phase 2: direction={result.get('mutation_direction', '?')}"
        )
        print(f"  [twophase] reasoning: {result.get('reasoning', '?')[:200]}")

        if severe_from_dataset:
            accumulated_severe.extend(severe_from_dataset)
        entry = {
            "kind": "twophase_diagnostic",
            "batch_size": total_count,
            "noisy_agent_success_rate": agent_success_rate,
            "n_success_traces": len(success_traces),
            "n_failure_traces": len(failure_traces),
            "n_propose_groups": len(proposals),
            "proposal_directions": [
                p.get("mutation_direction", "?") for p in proposals
            ],
            "n_severe": len(severe_from_dataset),
            "direction": result.get("mutation_direction", "unknown"),
        }
        history.append(entry)
        return result

    def proposer(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        _cached_proposals: list[dict] | None = None,
    ) -> dict[str, str]:
        persona_text = candidate["persona_text"]
        records = list(reflective_dataset.get("persona_text", []))

        (
            success_traces,
            failure_traces,
            severe_from_dataset,
            all_severe,
            agent_success_rate,
            timeout_rate,
            total_count,
        ) = _extract_traces_and_metrics(records)

        if mutation_context == "twophase":
            result = _proposer_twophase(
                persona_text,
                success_traces,
                failure_traces,
                all_severe,
                severe_from_dataset,
                agent_success_rate,
                timeout_rate,
                total_count,
                cached_proposals=_cached_proposals,
            )
        else:
            result = _proposer_single_phase(
                persona_text,
                success_traces,
                failure_traces,
                all_severe,
                severe_from_dataset,
                agent_success_rate,
                timeout_rate,
                total_count,
            )

        return {
            "persona_text": result["persona_text"],
            "mutation_direction": result.get("mutation_direction"),
        }

    proposer.get_cached_proposals = lambda: _twophase_cache["proposals"]
    proposer.clear_cached_proposals = lambda: _twophase_cache.update(proposals=None)

    return proposer
