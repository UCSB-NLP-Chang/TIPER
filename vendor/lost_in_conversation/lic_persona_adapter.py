"""GEPA adapter for lost_in_conversation (LIC) persona optimization.

Mirrors scripts/persona_opt/tau_persona_adapter.py but drives LIC's
run_demo.py / simulator_sharded.py instead of `tau2 run`. The GEPA main loop
(parallel_gepa.py), mutation proposer, Pareto selection and state persistence
are all reused unchanged — only the simulation/eval backend is swapped here.

Objective = difficulty (1 - agent is_correct).

Validity gate (unified with tau/vita/sweetrl, tau semantics):
  - severe signal = shared LLM severe judge ONLY (gpt-5.2, 4 categories via
    cresta.persona.eval.severe_judge, maj-vote strict majority): EVERY failed
    non-infra trial is judged; a task counts severe if ANY of its trials did;
  - severe_gate_rate mode: severe_count / n_noninfra > rate -> gate the whole
    candidate (scores zeroed, infeasible) — matches tau's rate gate;
  - legacy fallback (severe_gate_rate=None): absolute valid_severe_threshold;
  - rule signal (user did NOT transmit all shards) is STRUCTURAL, not severe:
    it populates timeout_safety and gates at timeout_attack_threshold (0.15),
    matching tau/vita/sweetrl's timeout gate (v1 counted it as severe).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from gepa.core.adapter import EvaluationBatch  # noqa: E402
except Exception:  # pragma: no cover - gepa is provided by the runner venv
    EvaluationBatch = None  # type: ignore

# Shared severe-judge protocol (single source of truth across tau/vita/lic/sweetrl).
# The adapter runs in the parallel_gepa process (tau2 venv), where cresta is installed.
from cresta.persona.eval.severe_judge import (  # noqa: E402
    build_severe_judge_sys,
)
from cresta.persona.eval.severe_judge import (
    judge_conversation_maj as _shared_judge_conversation,
)

# ---------------------------------------------------------------------------
# Outcome taxonomy (mirrors tau_persona_adapter.Outcome semantics)
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    AGENT_SUCCESS = "agent_success"            # agent solved it; persona failed to create difficulty
    GENUINE_AGENT_FAILURE = "genuine_agent_failure"  # agent failed; persona was effective (or task just hard)
    ILLEGAL_USER_BEHAVIOR = "illegal_user_behavior"  # user did NOT transmit all shards -> breaks task
    INFRA_ERROR = "infra_error"                # run crashed / timeout
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data types (opaque to GEPA)
# ---------------------------------------------------------------------------


@dataclass
class LICTask:
    task_id: str
    domain: str  # math / code / database / actions


@dataclass
class LICTrajectory:
    task_id: str
    domain: str
    is_correct: bool | None
    messages: list[dict]            # user/assistant turns from the trace
    outcome: Outcome = Outcome.UNKNOWN
    n_shards: int = 0
    n_revealed: int = 0
    judge: dict | None = None       # shared severe-judge verdict for failed convos


@dataclass
class LICOutput:
    task_id: str
    domain: str
    is_correct: bool | None
    outcome: Outcome = Outcome.UNKNOWN


@dataclass
class LICEvalArtifacts:
    run_label: str
    results_paths: dict[str, str]          # domain -> demo_runs.json path
    runs_by_task: dict[str, list[dict]]    # task_id -> [run record per trial]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_for(outcome: Outcome) -> float:
    """GEPA difficulty for ONE trial: 1.0 when the agent genuinely failed, else 0.0."""
    if outcome == Outcome.GENUINE_AGENT_FAILURE:
        return 1.0
    return 0.0


def _aggregate_outcome(outcomes: list[Outcome]) -> Outcome:
    """Collapse per-trial outcomes into one task-level outcome."""
    if not outcomes:
        return Outcome.INFRA_ERROR
    non_infra = [o for o in outcomes if o != Outcome.INFRA_ERROR]
    if not non_infra:
        return Outcome.INFRA_ERROR
    if all(o == Outcome.ILLEGAL_USER_BEHAVIOR for o in non_infra):
        return Outcome.ILLEGAL_USER_BEHAVIOR
    non_illegal = [o for o in non_infra if o != Outcome.ILLEGAL_USER_BEHAVIOR]
    if non_illegal and all(o == Outcome.AGENT_SUCCESS for o in non_illegal):
        return Outcome.AGENT_SUCCESS
    return Outcome.GENUINE_AGENT_FAILURE


def classify_outcome(run: dict | None) -> Outcome:
    if run is None:
        return Outcome.INFRA_ERROR
    if "error" in run and run.get("is_correct") is None:
        return Outcome.INFRA_ERROR
    is_correct = run.get("is_correct")
    nshards = run.get("shards") or 0
    trace = run.get("trace") or []
    n_revealed = sum(
        1 for m in trace
        if m.get("role") == "log" and (m.get("content") or {}).get("type") == "shard_revealed"
    )
    if is_correct is True:
        return Outcome.AGENT_SUCCESS
    # agent did not solve it. Did the user transmit all shards?
    if nshards and n_revealed < nshards:
        return Outcome.ILLEGAL_USER_BEHAVIOR
    if is_correct in (False, None):
        return Outcome.GENUINE_AGENT_FAILURE
    return Outcome.UNKNOWN


_OUTCOME_FEEDBACK = {
    Outcome.AGENT_SUCCESS: (
        "Agent SUCCEEDED (solved the task). Persona failed to create enough difficulty. "
        "Look at where the user gave information too cleanly, let the agent finalize too "
        "early, or failed to obscure/misdirect. Find a harder misdirection axis."
    ),
    Outcome.GENUINE_AGENT_FAILURE: (
        "Agent FAILED genuinely. Persona was effective here. "
        "Preserve this behavior if it stayed within the invariants."
    ),
    Outcome.ILLEGAL_USER_BEHAVIOR: (
        "ILLEGAL: the user did NOT transmit all shards (broke the task). "
        "Do NOT drop/withhold shard content. The persona must still reveal every shard "
        "eventually — only the STYLE/TIMING of revelation is the attack surface."
    ),
    Outcome.INFRA_ERROR: "INFRA ERROR — experiment noise, ignore for mutation.",
    Outcome.UNKNOWN: "UNKNOWN outcome — treat conservatively.",
}


def _build_feedback(outcome: Outcome) -> str:
    return _OUTCOME_FEEDBACK.get(outcome, _OUTCOME_FEEDBACK[Outcome.UNKNOWN])


def _format_conversation(trace: list[dict], char_limit: int = 4000) -> str:
    """Render user/assistant turns (skip log/bookkeeping entries) for the proposer."""
    lines = []
    used = 0
    for m in trace:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            text = str(content)
        elif role == "assistant":
            text = str(content)
        else:
            continue
        if used + len(text) > char_limit:
            text = text[: char_limit - used] + " …[truncated]"
        lines.append(f"\n[{role}] {text}")
        used += len(text)
        if used >= char_limit:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persona wrapper — the structural guarantee that the task is never broken
# ---------------------------------------------------------------------------

_INVARIANTS = """# Invariants (do NOT change — these keep the task well-formed)

- [Reveal Entire Shard] If you reveal a shard, you must include ALL of its information.
- [Keep Exact Values] You MUST preserve every exact value verbatim: names, numbers, strings, dates, enum values. Never blur or approximate them.
- [One Shard at a Time] Reveal at most one shard per response.
- [No Repeated Shards] Never reveal a shard whose shard_id was already revealed.
- [No Fabrication] Never state information beyond what the shards contain.
- [Do Not Ask Questions] Use declarative sentences only, never questions.
- [Always Reveal] You MUST keep revealing unrevealed shards across turns until all shards are revealed; do not stall or refuse.

Here's the conversation so far:
[[CONVERSATION_SO_FAR]]

Here are the shards that have already been revealed:
[[SHARDS_REVEALED]]

Here are all the shards that have not been revealed yet:
[[SHARDS_NOT_REVEALED]]

You must generate a response to the conversation so far. Apply your persona style (above) to HOW you reveal the next shard, but obey every Invariant.
- [Format] Respond as a JSON object: {"response": "<your message>", "shard_id": <int or -1>}
"""


def wrap_persona(persona_text: str) -> str:
    """Behavior description (optimizable) + locked invariants (task-preserving)."""
    return f"{persona_text.strip()}\n\n{_INVARIANTS}"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LICPersonaAdapter:
    """GEPAAdapter that evaluates persona candidates via LIC simulations."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        persona_id: str = "lic_eval",
        agent_model: str = "gpt-5.4",
        user_model: str | None = None,
        reasoning_effort: str | None = "high",
        user_reasoning_effort: str | None = None,
        domains: list[str] | None = None,
        n_workers: int = 6,
        dataset_file: str = "data/sharded_instructions_600.json",
        lic_dir: str | Path = ".",
        eval_root: str | Path | None = None,
        valid_severe_threshold: int = 2,   # legacy fallback: >= this many judge-severe outcomes -> batch infeasible
        severe_gate_rate: float | None = 0.05,  # rate mode (tau semantics): severe/n_noninfra > rate -> infeasible
        timeout_attack_threshold: float | None = 0.15,  # rule-ILLEGAL (shards not delivered) gate — tau timeout parity
        judge_model: str = "gpt-5.2",      # shared 4-category severe judge (LLM signal, on failed convos)
        judge_maj: int = 3,                # strict-majority votes per judged trial (tau parity)
        judge_concurrency: int = 8,
        scoring_mode: str = "mean",
        mutation_context: str = "basic",
        sim_timeout_per_conv: int = 60,    # run_demo assistant timeout is set in simulator
        subprocess_timeout: int = 7200,    # per run_demo subprocess wall budget (gen0 = valset×trials convs; minimax via :4114 is rate-limited ~19 req/min shared across domains, so 30min was too tight)
        num_trials: int = 1,               # repeats per task_id (multi-trial sampling)
        always_judge_minibatch: bool = True,  # judge failed convos on every eval (off => rule-only validity)
    ):
        self._persona_id = persona_id
        self._agent_model = agent_model
        # user/system model: default to agent_model (legacy single-model behavior)
        # unless explicitly separated. When separated, run_demo routes via
        # minimax_routing (agent→:4114, user/system gpt-5.x→standard OpenAI).
        self._user_model = user_model or agent_model
        self._system_model = user_model or agent_model
        self._reasoning_effort = reasoning_effort
        # Separate user-simulator effort (defaults to the agent effort when unset,
        # preserving legacy behavior). Set to 'high' while the agent runs at 'max'
        # so the gpt-5.x user simulator stays at the aligned harness effort.
        self._user_reasoning_effort = user_reasoning_effort
        self._domains = domains or ["math", "code", "database", "actions"]
        self._n_workers = n_workers
        self._num_trials = max(1, num_trials)
        self._dataset_file = dataset_file
        self._lic_dir = Path(lic_dir).resolve()
        self._eval_root = (
            Path(eval_root).resolve() if eval_root
            else self._lic_dir / "data" / "persona_opt" / "lic_eval"
        )
        self._valid_severe_threshold = valid_severe_threshold
        self._severe_gate_rate = severe_gate_rate
        self._timeout_attack_threshold = timeout_attack_threshold
        self._judge_model = judge_model
        self._judge_maj = judge_maj
        self._judge_concurrency = judge_concurrency
        self._judge_sys = build_severe_judge_sys("lic")
        self._samples_cache: dict[str, dict] | None = None
        self._scoring_mode = scoring_mode
        self._mutation_context = mutation_context
        self._subprocess_timeout = subprocess_timeout
        self._eval_counter = self._max_existing_eval_index()
        self._materialize_lock = threading.Lock()
        self._always_judge_minibatch = always_judge_minibatch

    # -- paths / persona injection -------------------------------------------

    def _max_existing_eval_index(self) -> int:
        root = self._eval_root / self._persona_id
        if not root.is_dir():
            return 0
        mx = 0
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("eval_"):
                try:
                    mx = max(mx, int(p.name[len("eval_"):]))
                except ValueError:
                    continue
        return mx

    def _skill_prompt_path(self) -> Path:
        return self._lic_dir / "prompts" / f"user_agent_opt_{self._persona_id}.txt"

    def _materialize(self, persona_text: str) -> Path:
        path = self._skill_prompt_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(wrap_persona(persona_text))
        return path

    def _python(self) -> str:
        venv_py = self._lic_dir / ".venv" / "bin" / "python"
        return str(venv_py) if venv_py.exists() else sys.executable

    # -- severe-judge support (shared protocol, gpt-5.2) ----------------------

    def _load_sample(self, task_id: str) -> dict | None:
        """The dataset sample for a task_id (scenario context for the judge:
        the original instruction + shard texts the user holds)."""
        if self._samples_cache is None:
            path = Path(self._dataset_file)
            if not path.is_absolute():
                path = self._lic_dir / path
            try:
                data = json.load(open(path))
            except Exception as e:  # noqa: BLE001
                print(f"  [lic judge] dataset load failed: {e}")
                self._samples_cache = {}
                return None
            self._samples_cache = {s["task_id"]: s for s in data}
        return self._samples_cache.get(task_id)

    @staticmethod
    def _scenario_context(sample: dict | None) -> str:
        if not sample:
            return "(scenario unavailable)"
        shards_txt = "\n".join(
            f"- shard {s.get('shard_id')}: {s.get('shard', '')}"
            for s in sample.get("shards", [])
        )
        return (
            f"Original (fully-specified) instruction the shards decompose:\n"
            f"{sample.get('prompt', '')}\n\n"
            f"Information shards the simulated user holds (must ALL be revealed):\n{shards_txt}"
        )

    @staticmethod
    def _transcript_for_judge(trace: list[dict]) -> str:
        out = []
        for m in trace:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue  # log/bookkeeping entries were never sent to the model
            out.append(f"[{'user' if role == 'user' else 'assistant'}] {m.get('content', '') or ''}")
        return "\n".join(out)[-12000:]

    def _run_judge(self, judge_targets: list[tuple[str, dict, int]]) -> dict[tuple[str, int], dict]:
        """Judge each failed run's conversation concurrently (persona text stashed
        by run_simulations). Targets are (task_id, run, trial_idx); returns
        {(task_id, trial_idx): verdict}. Fail-open per call
        (see severe_judge.judge_conversation_maj)."""
        persona_text = self._persona_text
        verdicts: dict[tuple[str, int], dict] = {}

        def _one(tid: str, run: dict, trial_idx: int) -> tuple[tuple[str, int], dict]:
            sample = self._load_sample(tid)
            v = _shared_judge_conversation(
                transcript=self._transcript_for_judge(run.get("trace") or []),
                persona_text=persona_text,
                scenario_context=self._scenario_context(sample),
                model=self._judge_model,
                judge_sys=self._judge_sys,
                maj=self._judge_maj,
            )
            v["task_id"] = tid
            v["trial"] = trial_idx
            v["reward"] = run.get("score")
            return (tid, trial_idx), v

        with ThreadPoolExecutor(max_workers=self._judge_concurrency) as ex:
            for key, v in ex.map(lambda t: _one(*t), judge_targets):
                verdicts[key] = v
        return verdicts

    # -- GEPAAdapter.evaluate ------------------------------------------------

    def evaluate(
        self,
        batch: list[LICTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        artifacts = self.run_simulations(batch, candidate)
        # Judge failed conversations (shared gpt-5.2 severe judge) unless disabled
        # (always_judge_minibatch=False => rule-only validity, legacy v1 behavior).
        should_judge = capture_traces or self._always_judge_minibatch
        return self.aggregate_artifacts(
            batch, artifacts, capture_traces=capture_traces, should_judge=should_judge,
        )

    def run_simulations(
        self,
        batch: list[LICTask],
        candidate: dict[str, str],
    ) -> LICEvalArtifacts:
        persona_text = candidate["persona_text"]
        with self._materialize_lock:
            prompt_path = self._materialize(persona_text)
        self._persona_text = persona_text  # stashed for the severe judge

        self._eval_counter += 1
        run_label = f"{self._persona_id}/eval_{self._eval_counter:04d}"
        out_root = self._eval_root / run_label

        by_domain: dict[str, list[str]] = {}
        for t in batch:
            by_domain.setdefault(t.domain, []).append(t.task_id)

        results_paths: dict[str, str] = {}
        runs_by_task: dict[str, dict] = {}

        for dk, task_ids in by_domain.items():
            run_dir = out_root / dk
            run_dir.mkdir(parents=True, exist_ok=True)
            out_json = run_dir / "demo_runs.json"
            cmd = [
                self._python(),
                str(self._lic_dir / "run_demo.py"),
                "--assistant-model", self._agent_model,
                "--user-model", self._user_model,
                "--system-model", self._system_model,
                "--model", self._user_model,  # fallback for any path still reading --model
                "--tasks", dk,
                "--task-ids", ",".join(task_ids),
                "--user_persona", str(prompt_path),
                "--workers", str(self._n_workers),
                "--output-json", str(out_json),
            ]
            if self._num_trials > 1:
                cmd += ["--n-trials", str(self._num_trials)]
            if self._reasoning_effort:
                cmd += ["--reasoning_effort", self._reasoning_effort]
            if self._user_reasoning_effort:
                cmd += ["--user_reasoning_effort", self._user_reasoning_effort]
            env = dict(os.environ)
            env.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
            try:
                subprocess.run(
                    cmd, check=False, cwd=str(self._lic_dir), env=env,
                    timeout=self._subprocess_timeout,
                    capture_output=True, text=True,
                )
            except subprocess.TimeoutExpired:
                print(f"  [lic eval {dk}] run_demo subprocess TIMEOUT ({self._subprocess_timeout}s)")
            except Exception as e:
                print(f"  [lic eval {dk}] subprocess error: {e}")

            results_paths[dk] = str(out_json)
            if out_json.exists():
                try:
                    doc = json.load(open(out_json))
                except Exception:
                    doc = {"runs": []}
                for r in doc.get("runs", []):
                    runs_by_task.setdefault(r["task_id"], []).append(r)

        return LICEvalArtifacts(
            run_label=run_label,
            results_paths=results_paths,
            runs_by_task=runs_by_task,
        )

    def aggregate_artifacts(
        self,
        batch: list[LICTask],
        artifacts: LICEvalArtifacts,
        *,
        capture_traces: bool = False,
        should_judge: bool = True,
        failed_only: bool = True,    # tau-compat: the judge only ever sees failed runs
    ) -> EvaluationBatch:
        runs_by_task = artifacts.runs_by_task
        outputs: list[LICOutput] = []
        scores: list[float] = []
        trajectories: list[LICTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []
        severe_count = 0
        rule_illegal_count = 0

        # Representative (first non-error) run per task, for trace/is_correct.
        rep_by_task: dict[str, dict] = {}
        for task in batch:
            trials = runs_by_task.get(task.task_id) or []
            rep = next((r for r in trials if "error" not in r), trials[0] if trials else None)
            if rep is not None:
                rep_by_task[task.task_id] = rep

        # Trial-level judging (tau parity): EVERY failed non-infra trial is
        # judged (maj-vote); a task counts severe if ANY of its trials did.
        judge_by_key: dict[tuple[str, int], dict] = {}
        if should_judge:
            judge_targets = []
            for task in batch:
                for ti, r in enumerate(runs_by_task.get(task.task_id) or []):
                    if "error" not in r and r.get("is_correct") is not True and r.get("trace"):
                        judge_targets.append((task.task_id, r, ti))
            if judge_targets:
                judge_by_key = self._run_judge(judge_targets)
                n_severe_trials = sum(
                    1 for v in judge_by_key.values() if v.get("verdict") == "severe_failure")
                print(f"  [lic judge] judged {len(judge_targets)} failed trial(s) "
                      f"({n_severe_trials} severe verdicts) on {self._judge_model} "
                      f"maj={self._judge_maj}")

        for task in batch:
            trials = runs_by_task.get(task.task_id) or []
            trial_outcomes = [classify_outcome(r) for r in trials] or [Outcome.INFRA_ERROR]
            outcome = _aggregate_outcome(trial_outcomes)
            # severe = JUDGE-ONLY signal (any failed trial); the rule signal
            # (not ALL shards transmitted) is structural — it lives in the
            # timeout_safety dimension at the 0.15 gate (tau timeout parity).
            non_infra_trials = [r for r in trials if "error" not in r]
            rep_idx = next((i for i, r in enumerate(trials) if "error" not in r), 0)
            judge_severe_task = any(
                v.get("verdict") == "severe_failure"
                for (tid, _ti), v in judge_by_key.items() if tid == task.task_id)
            rule_illegal = (
                bool(non_infra_trials) and all(
                    classify_outcome(r) == Outcome.ILLEGAL_USER_BEHAVIOR
                    for r in non_infra_trials))
            if judge_severe_task:
                severe_count += 1
            if rule_illegal:
                rule_illegal_count += 1
            # per-trial difficulty, aggregated by scoring_mode
            per_score = [_score_for(o) for o in trial_outcomes]
            if self._scoring_mode == "fail_squared":
                opt_score = 1.0 if per_score and all(s == 1.0 for s in per_score) else 0.0
            else:  # mean
                opt_score = sum(per_score) / len(per_score) if per_score else 0.0
            # representative run for trace/is_correct (first non-infra trial, else first)
            rep = rep_by_task.get(task.task_id)
            is_correct = None if rep is None else rep.get("is_correct")
            trace = (rep or {}).get("trace") or []
            nsh = (rep or {}).get("shards") or 0
            n_rev = sum(
                1 for m in trace
                if m.get("role") == "log" and (m.get("content") or {}).get("type") == "shard_revealed"
            )

            # per-task validity: judge-severe -> 0 (known), infra -> unknown,
            # else valid. Rule-ILLEGAL no longer zeroes validity — it gates via
            # timeout_safety below (v1 counted it as severe; that double-counted
            # structural nondelivery as a "cheat").
            if judge_severe_task:
                validity, validity_known = 0.0, 1.0
            elif outcome == Outcome.INFRA_ERROR:
                validity, validity_known = 0.5, 0.0
            else:
                validity, validity_known = 1.0, 1.0

            outputs.append(LICOutput(task.task_id, task.domain, is_correct, outcome))
            scores.append(opt_score)
            objective_scores.append({
                "difficulty": opt_score,
                "validity": validity,
                "validity_known": validity_known,
                "timeout_safety": 0.0 if rule_illegal else 1.0,
                "feasible": 1.0,
                "infra_error": 1.0 if outcome == Outcome.INFRA_ERROR else 0.0,
            })
            if capture_traces:
                jinfo = judge_by_key.get((task.task_id, rep_idx))
                trajectories.append(LICTrajectory(
                    task_id=task.task_id, domain=task.domain,
                    is_correct=is_correct, messages=trace,
                    outcome=outcome, n_shards=nsh, n_revealed=n_rev,
                    judge=jinfo,
                ))

        # batch-level feasibility gates: (1) severe judge rate (tau semantics;
        # legacy absolute-threshold fallback when severe_gate_rate is None) and
        # (2) structural timeout gate on the rule signal (>= threshold rate).
        batch_infeasible = False
        n_noninfra = sum(1 for o in objective_scores if o.get("infra_error", 0.0) == 0.0)
        if n_noninfra and self._severe_gate_rate is not None:
            rate = severe_count / n_noninfra
            if rate > self._severe_gate_rate:
                print(f"  [lic eval] PERSONA INVALID: severe_rate="
                      f"{severe_count}/{n_noninfra}={rate:.0%} "
                      f"(> {self._severe_gate_rate:.0%}) -> zeroing all scores")
                batch_infeasible = True
        elif n_noninfra and severe_count >= self._valid_severe_threshold:
            print(f"  [lic eval] BATCH INFEASIBLE: {severe_count} severe "
                  f"(>= {self._valid_severe_threshold}) -> zeroing all scores")
            batch_infeasible = True
        if (not batch_infeasible and self._timeout_attack_threshold is not None
                and n_noninfra
                and rule_illegal_count / n_noninfra >= self._timeout_attack_threshold):
            print(f"  [lic eval] BATCH INFEASIBLE: rule_illegal_rate="
                  f"{rule_illegal_count}/{n_noninfra}="
                  f"{rule_illegal_count/n_noninfra:.0%} "
                  f"(>= {self._timeout_attack_threshold:.0%}) -> zeroing all scores")
            batch_infeasible = True
        if batch_infeasible:
            scores = [0.0] * len(batch)
            for obj in objective_scores:
                obj["feasible"] = 0.0

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    # -- GEPAAdapter.make_reflective_dataset ---------------------------------

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
            snippet = _format_conversation(traj.messages)
            entries.append({
                "Inputs": {
                    "task_id": traj.task_id,
                    "domain": traj.domain,
                    "outcome": traj.outcome.value,
                    "n_shards": traj.n_shards,
                    "n_revealed": traj.n_revealed,
                },
                "Generated Outputs": {"conversation_snippet": snippet},
                "Feedback": _build_feedback(traj.outcome),
                "Judge": traj.judge or {},
            })
        return {"persona_text": entries}
