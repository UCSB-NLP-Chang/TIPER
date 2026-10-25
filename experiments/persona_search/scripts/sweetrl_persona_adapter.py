#!/usr/bin/env python3
"""GEPA adapter for adversarial user-persona search on sweet_rl ColBench `code` tasks.

Goal: find a *user-simulator persona* that minimizes the agent's reward (i.e.
makes the multi-turn requirement-elicitation harder), while staying realistic
(not cheating / not lying about hidden info). GEPA's score = `1 - reward`, so
higher = harder = better, which matches "降低 reward".

Validity gate (unified with tau/vita/lic): every failed conversation is judged
on 4 severe categories (fabricated_info / refused_to_engage / contradictory /
character_break) by the shared gpt-5.2 severe judge
(cresta.persona.eval.severe_judge); severe_rate = severe/n_noninfra > 0.05 gates
the whole candidate, as does timeout_rate >= 0.15.

Self-contained: uses the OpenAI Chat Completions API directly with
reasoning-model-compatible params (`max_completion_tokens`, `reasoning_effort`,
no `temperature`), so it works with gpt-5.4 / gpt-5.5. It does NOT touch tau2 or
vLLM. Scoring reuses `sweet_rl.utils.code_utils.check_correctness` (run in the
main thread because it uses SIGALRM).
"""

from __future__ import annotations

import concurrent
import json
import os
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402
from openai import OpenAI

# sweet_rl lives in the sibling repo one directory down from the tau2 repo root.
# The runner injects the right sys.path; import defensively here.
from sweet_rl.utils.code_utils import check_correctness

# Shared severe-judge protocol (single source of truth across tau/vita/lic/sweetrl;
# judge model gpt-5.2, 4 severe categories). The LLM call is injected so the
# adapter's own OpenAI client (reasoning-model-compatible params) is reused.
from cresta.persona.eval.severe_judge import (  # noqa: E402
    build_severe_judge_sys,
)
from cresta.persona.eval.severe_judge import (
    judge_conversation_maj as _shared_judge_conversation,
)

HUMAN_RESPONSE_CHARACTER_LIMIT = 400
ANSWER_MARKER = "I WANT TO ANSWER:"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """Trial outcome taxonomy, aligned with vita_persona_adapter.Outcome."""

    AGENT_SUCCESS = "agent_success"
    GENUINE_AGENT_FAILURE = "genuine_agent_failure"
    TIMEOUT_OR_MAX_STEPS = "timeout_or_max_steps"
    INFRA_ERROR = "infra_error"
    UNKNOWN = "unknown"


def classify_outcome(traj: "SweetRLTrajectory") -> Outcome:
    if traj.termination == "error":
        return Outcome.INFRA_ERROR
    if traj.termination == "max_steps":
        return Outcome.TIMEOUT_OR_MAX_STEPS
    if traj.reward >= 1.0:
        return Outcome.AGENT_SUCCESS
    return Outcome.GENUINE_AGENT_FAILURE


@dataclass
class SweetRLTask:
    task_id: str
    problem_description: str
    ground_truth: str  # reference function (also the human's hidden info)
    hidden_information: str
    test_cases: dict
    split: str = "train"  # train | val | test


@dataclass
class SweetRLTrajectory:
    task_id: str
    reward: float  # 0..1 test-case pass rate
    messages: list[dict]  # [{"role","content"}, ...]
    answer: str
    n_turns: int
    termination: str  # "answered" | "max_steps" | "error"
    judge: dict | None = None  # shared severe-judge verdict for failed convos


@dataclass
class SweetROutput:
    task_id: str
    reward: float
    termination: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt(path: str) -> str:
    with open(path, "r") as fb:
        return fb.read()


def _format_dialogue(history: list[dict]) -> str:
    """Render dialogue for the human-simulator prompt's {dialogue_history}."""
    out = ""
    for d in history:
        out += str(d["role"]) + ":" + str(d["content"]) + "\n\n\n\n"
    return out + "agent:"


def _transcript_for_judge(history: list[dict]) -> str:
    """Render the conversation for the severe judge (agent / human turns)."""
    out = []
    for d in history:
        role = str(d.get("role", "?"))
        tag = (
            "agent"
            if role == "assistant"
            else ("developer" if role == "user" else role)
        )
        out.append(f"[{tag}] {d.get('content', '') or ''}")
    return "\n".join(out)[-12000:]


def _bookend(messages: list[dict], head: int = 4, tail: int = 8) -> str:
    if len(messages) <= head + tail:
        sel = messages
    else:
        sel = (
            messages[:head]
            + [
                {
                    "role": "...",
                    "content": f"[{len(messages) - head - tail} turns omitted]",
                }
            ]
            + messages[-tail:]
        )
    return "\n".join(f"[{m['role']}] {m['content'][:400]}" for m in sel)


def _extract_code(answer: str) -> str:
    """Match check_correctness' fence stripping so scoring is consistent."""
    f = answer
    if "```python" in f:
        f = f.split("```python")[1].split("```")[0]
    elif "```" in f:
        f = f.split("```")[1].split("```")[0]
    return f


# ---------------------------------------------------------------------------
# Conversation runner (reasoning-model compatible)
# ---------------------------------------------------------------------------


class _ConversationRunner:
    """Runs one agent<->user conversation for a single sweet_rl task.

    Both sides call the OpenAI Chat Completions API with reasoning-model params.
    The persona is injected into the human simulator's prompt.
    """

    def __init__(
        self,
        client: OpenAI,
        agent_model: str,
        agent_effort: str,
        human_model: str,
        human_effort: str,
        agent_prompt: str,
        human_prompt_base: str,
        persona_text: str,
        max_steps: int,
        max_completion_tokens: int = 4096,
    ):
        self.client = client
        self.agent_model = agent_model
        self.agent_effort = agent_effort
        self.human_model = human_model
        self.human_effort = human_effort
        self.agent_prompt = agent_prompt
        self.human_prompt_base = human_prompt_base
        self.persona_text = persona_text
        self.max_steps = max_steps
        self.max_completion_tokens = max_completion_tokens

    def _chat(self, model: str, effort: str, messages: list[dict]) -> str:
        for _ in range(3):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": self.max_completion_tokens,
                    "reasoning_effort": effort,
                }
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if content:
                    return content
                # Reasoning models can return empty content when the budget is
                # eaten by thinking; retry once with a bigger budget.
                kwargs["max_completion_tokens"] = self.max_completion_tokens * 2
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return content or ""
            except Exception as e:  # noqa: BLE001
                print(f"    [chat retry] {model}: {str(e)[:160]}")
        return ""

    def _human_prompt(self) -> str:
        base = self.human_prompt_base
        if self.persona_text and self.persona_text.strip():
            base = (
                base
                + "\n\n## YOUR PERSONA (adopt this when responding to the agent)\n"
                + "Stay realistic: do NOT fabricate information that is not in the "
                "hidden information above, and do NOT refuse to engage. Express "
                "yourself through this persona's personality and communication style:\n\n"
                + self.persona_text.strip()
            )
        return base

    def run(self, task: SweetRLTask) -> SweetRLTrajectory:
        history: list[dict] = [{"role": "user", "content": task.problem_description}]
        answer = "No answer"
        termination = "max_steps"
        human_prompt = self._human_prompt()

        for _ in range(self.max_steps):
            # Agent turn
            agent_msgs = [{"role": "system", "content": self.agent_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in history
            ]
            agent_resp = self._chat(self.agent_model, self.agent_effort, agent_msgs)
            if not agent_resp:
                agent_resp = "I WANT TO ANSWER:"

            history.append({"role": "assistant", "content": agent_resp})

            if ANSWER_MARKER in agent_resp:
                answer = agent_resp.split(ANSWER_MARKER, 1)[1]
                termination = "answered"
                break

            # Human turn
            user_msg = human_prompt.format(
                problem_description=task.problem_description,
                hidden_information=task.hidden_information,
                dialogue_history=_format_dialogue(history),
            )
            human_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_msg},
            ]
            human_resp = self._chat(self.human_model, self.human_effort, human_msgs)
            if not human_resp:
                human_resp = "I don't know."
            history.append(
                {"role": "user", "content": human_resp[:HUMAN_RESPONSE_CHARACTER_LIMIT]}
            )

        # Score (run in whatever thread; check_correctness uses a subprocess for
        # the actual exec, and SIGALRM only in the main thread of this process.
        # The runner scores in the main thread to be safe — see adapter.)
        return SweetRLTrajectory(
            task_id=task.task_id,
            reward=-1.0,  # filled in by the adapter (main-thread scoring)
            messages=history,
            answer=answer,
            n_turns=len(history),
            termination=termination,
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SweetRLPersonaAdapter(GEPAAdapter):
    """GEPAAdapter that evaluates persona candidates via sweet_rl conversations."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        agent_model: str = "gpt-5.4",
        agent_effort: str = "high",
        human_model: str = "gpt-5.4",
        human_effort: str = "high",
        agent_prompt_path: str = "sweet_rl/prompts/llm_agent_code_prompt.txt",
        human_prompt_path: str = "sweet_rl/prompts/human_simulator_code_prompt.txt",
        max_steps: int = 100,
        num_trials: int = 2,
        max_concurrency: int = 16,
        max_completion_tokens: int = 4096,
        agent_round_budget: int = 100,
        # unified validity gate (tau/vita/lic semantics): failed conversations are
        # judged on 4 severe categories by gpt-5.2; severe_rate = severe/n_noninfra
        # > 0.05 gates the whole candidate; timeout_rate >= 0.15 gates likewise.
        judge_model: str = "gpt-5.2",
        judge_maj: int = 3,
        severe_gate_rate: float | None = 0.05,
        timeout_attack_threshold: float | None = 0.15,
        online_judge: bool = True,
        online_judge_concurrency: int = 8,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
        )
        self.agent_model = agent_model
        self.agent_effort = agent_effort
        self.human_model = human_model
        self.human_effort = human_effort
        self.max_steps = max_steps
        self.num_trials = num_trials
        self.max_concurrency = max_concurrency
        self.max_completion_tokens = max_completion_tokens
        self.judge_model = judge_model
        self.judge_maj = judge_maj
        self.severe_gate_rate = severe_gate_rate
        self.timeout_attack_threshold = timeout_attack_threshold
        self.online_judge = online_judge
        self.online_judge_concurrency = online_judge_concurrency
        self._judge_sys = build_severe_judge_sys("sweetrl")
        self._last_judge_summary: dict[str, Any] = {}

        agent_prompt = _load_prompt(agent_prompt_path)
        # Align the agent's stated round budget with max_steps (minimal edit).
        agent_prompt = agent_prompt.replace(
            "WITHIN 10 back-and-forth rounds",
            f"WITHIN {agent_round_budget} back-and-forth rounds",
        )
        self.agent_prompt = agent_prompt
        self.human_prompt_base = _load_prompt(human_prompt_path)
        self._scoring_lock = threading.Lock()

    # -- GEPAAdapter.evaluate -------------------------------------------------

    def evaluate(
        self,
        batch: list[SweetRLTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        persona_text = candidate.get("persona_text", "")
        # Run num_trials conversations per task, concurrently across tasks.
        jobs = []
        for t in batch:
            for trial in range(self.num_trials):
                runner = _ConversationRunner(
                    self.client,
                    self.agent_model,
                    self.agent_effort,
                    self.human_model,
                    self.human_effort,
                    self.agent_prompt,
                    self.human_prompt_base,
                    persona_text,
                    self.max_steps,
                    self.max_completion_tokens,
                )
                jobs.append((t, trial, runner))

        trajectories_by_task: dict[str, list[SweetRLTrajectory]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrency
        ) as ex:
            futs = {ex.submit(r.run, t): (t, tr) for (t, tr, r) in jobs}
            for fut in concurrent.futures.as_completed(futs):
                t, tr = futs[fut]
                try:
                    traj = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [sweetrl eval] task {t.task_id} trial {tr} failed: {e}")
                    traj = SweetRLTrajectory(
                        task_id=t.task_id,
                        reward=0.0,
                        messages=[],
                        answer="No answer",
                        n_turns=0,
                        termination="error",
                    )
                trajectories_by_task.setdefault(t.task_id, []).append(traj)

        # Score in the MAIN thread (check_correctness uses SIGALRM). GEPA calls
        # evaluate single-threaded, so we are in the main thread here.
        outputs: list[SweetROutput] = []
        scores: list[float] = []
        trajectories: list[SweetRLTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []
        # rep (worst-reward non-infra) trial per failed task, for the judge
        judge_targets: list[tuple[SweetRLTask, SweetRLTrajectory]] = []

        for t in batch:
            trajs = trajectories_by_task.get(t.task_id, [])
            non_infra = [tr for tr in trajs if tr.termination != "error"]
            rewards = []
            for traj in trajs:
                if traj.termination == "error":
                    r = 0.0
                else:
                    try:
                        with self._scoring_lock:
                            r = check_correctness(
                                t.ground_truth, _extract_code(traj.answer), t.test_cases
                            )
                    except Exception as e:  # noqa: BLE001
                        print(f"  [sweetrl score] {t.task_id}: {e}")
                        r = 0.0
                traj.reward = r
                rewards.append(r)
            # difficulty over NON-INFRA trials only (infra errors are experiment
            # noise, not persona performance — matches tau aggregation)
            if non_infra:
                avg_reward = sum(tr.reward for tr in non_infra) / len(non_infra)
                opt_score = 1.0 - avg_reward
                worst = min(non_infra, key=lambda x: x.reward)
                outcome = classify_outcome(worst)
                # tau parity: task counts as timeout only if ALL its non-infra
                # trials did (severe is ANY-trial; timeout is ALL-trial)
                all_timeout = all(
                    classify_outcome(tr) == Outcome.TIMEOUT_OR_MAX_STEPS
                    for tr in non_infra
                )
            else:
                avg_reward = 0.0
                opt_score = 0.0
                outcome = Outcome.INFRA_ERROR
                all_timeout = False
            # GEPA: higher = better. difficulty = 1 - reward.
            score = opt_score
            outputs.append(
                SweetROutput(
                    t.task_id, avg_reward, trajs[0].termination if trajs else "error"
                )
            )
            scores.append(score)
            objective_scores.append(
                {
                    "difficulty": opt_score,
                    # three-state validity (tau semantics): severe -> 0; judged clean
                    # -> 1; unjudged -> 0.5 unknown (flagged via validity_known)
                    "validity": 0.5,
                    "validity_known": 0.0,
                    "timeout_safety": 0.0 if all_timeout else 1.0,
                    "semantic": 0.0 if outcome == Outcome.INFRA_ERROR else 1.0,
                    "infra_error": 1.0 if outcome == Outcome.INFRA_ERROR else 0.0,
                    "feasible": 1.0,
                }
            )
            if capture_traces and trajs:
                # Keep the worst-reward (hardest) non-infra trial as the trace.
                trajectories.append((worst if non_infra else trajs[0]))  # type: ignore[arg-type]
            # collect EVERY failed non-infra trial with a substantive
            # conversation to judge (tau parity: task severe = ANY trial severe)
            if self.online_judge:
                for tr in non_infra:
                    if (
                        classify_outcome(tr) == Outcome.GENUINE_AGENT_FAILURE
                        and tr.messages
                    ):
                        judge_targets.append((t, tr))

        # online severe-judge gate (unified with tau/vita/lic): judge every failed
        # task's rep conversation on 4 severe categories; severe_rate =
        # severe/n_noninfra > threshold gates the whole candidate.
        severe_violations: list[dict] = []
        if self.online_judge and judge_targets:
            obj_by_task = {t.task_id: obj for t, obj in zip(batch, objective_scores)}
            severe_count, n_judged = self._run_online_judge(
                judge_targets, persona_text, obj_by_task, severe_violations
            )
        else:
            severe_count, n_judged = 0, 0
        n_noninfra = sum(
            1 for obj in objective_scores if obj.get("infra_error", 0.0) == 0.0
        )
        timeout_count = sum(
            1 for obj in objective_scores if obj.get("timeout_safety", 1.0) == 0.0
        )

        batch_infeasible = False
        if n_noninfra and self.severe_gate_rate is not None:
            severe_rate = severe_count / n_noninfra
            self._last_judge_summary = {
                "n_judged": n_judged,
                "n_severe": severe_count,
                "n_noninfra": n_noninfra,
                "severe_rate": severe_rate,
                "violations": severe_violations,
            }
            if severe_rate > self.severe_gate_rate:
                print(
                    f"  [sweetrl eval] PERSONA INVALID: severe_rate="
                    f"{severe_count}/{n_noninfra}={severe_rate:.0%} "
                    f"(> {self.severe_gate_rate:.0%}) -> zeroing all scores; "
                    f"cats={[v['category'] for v in severe_violations]}"
                )
                batch_infeasible = True
            elif severe_count:
                print(
                    f"  [sweetrl eval] severe_rate={severe_count}/{n_noninfra}="
                    f"{severe_rate:.0%} (<= threshold; not gated)"
                )
        else:
            self._last_judge_summary = {}

        if (
            n_noninfra
            and self.timeout_attack_threshold is not None
            and timeout_count / n_noninfra >= self.timeout_attack_threshold
        ):
            print(
                f"  [sweetrl eval] BATCH INFEASIBLE: timeout_rate="
                f"{timeout_count}/{n_noninfra}={timeout_count / n_noninfra:.0%} "
                f"(>= {self.timeout_attack_threshold:.0%}) -> zeroing all scores"
            )
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

    def _run_online_judge(
        self,
        judge_targets: list[tuple[SweetRLTask, SweetRLTrajectory]],
        persona_text: str,
        obj_by_task: dict[str, dict],
        severe_violations: list[dict],
    ) -> tuple[int, int]:
        """Judge each failed non-infra trial concurrently (maj-vote, tau parity).
        Attaches each verdict to its trajectory and updates the task's
        objective_scores validity fields (a task is invalid if ANY trial was
        severe). Returns (n_severe_TASKS, n_judged). Fail-open per call."""
        n_judged = 0
        severe_by_task: set[str] = set()

        def _llm_call(prompt: str, model: str) -> str:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            return resp.choices[0].message.content or ""

        def _one(task: SweetRLTask, traj: SweetRLTrajectory) -> dict:
            scenario_context = (
                f"Task description given to the agent:\n{task.problem_description}\n\n"
                f"Hidden information the simulated developer legitimately knows "
                f"(only source of truth for fabrication checks):\n{task.hidden_information}"
            )
            v = _shared_judge_conversation(
                transcript=_transcript_for_judge(traj.messages),
                persona_text=persona_text,
                scenario_context=scenario_context,
                model=self.judge_model,
                llm_call=_llm_call,
                judge_sys=self._judge_sys,
                maj=self.judge_maj,
            )
            v["task_id"] = task.task_id
            v["reward"] = traj.reward
            return v

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.online_judge_concurrency
        ) as ex:
            futs = {ex.submit(_one, t, traj): (t, traj) for t, traj in judge_targets}
            for fut in futs:
                task, traj = futs[fut]
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001
                    v = {
                        "verdict": "genuine_failure",
                        "category": "none",
                        "reason": f"judge err: {str(e)[:80]}",
                    }
                n_judged += 1
                traj.judge = v
                if v.get("verdict") == "severe_failure":
                    severe_by_task.add(task.task_id)
                    obj = obj_by_task.get(task.task_id)
                    severe_violations.append(
                        {
                            "task_id": task.task_id,
                            "category": v.get("category", "?"),
                            "reason": v.get("reason", ""),
                        }
                    )
        for tid in severe_by_task:
            obj = obj_by_task.get(tid)
            if obj is not None:
                obj["validity"] = 0.0
                obj["validity_known"] = 1.0
        # tasks whose trials were all judged clean (at least one judged trial)
        # become validity=1/known=1
        judged_tasks = {t.task_id for t, _traj in judge_targets}
        clean_tasks = judged_tasks - severe_by_task
        for tid in clean_tasks:
            obj = obj_by_task.get(tid)
            if obj is not None and obj.get("validity_known", 0.0) == 0.0:
                obj["validity"] = 1.0
                obj["validity_known"] = 1.0
        return len(severe_by_task), n_judged

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
            if traj.termination == "error":
                continue
            snippet = _bookend(traj.messages)
            if traj.reward >= 1.0:
                feedback = (
                    f"Agent SUCCEEDED (reward=1.0) on task {traj.task_id} after "
                    f"{traj.n_turns} turns — the persona FAILED to make elicitation "
                    f"hard. The agent extracted what it needed and wrote a correct "
                    f"function. Identify what the persona let slip too easily and "
                    f"tighten that loophole."
                )
            else:
                feedback = (
                    f"Agent FAILED (reward={traj.reward:.2f}) on task {traj.task_id} "
                    f"after {traj.n_turns} turns — the persona worked here. Keep this "
                    f"behavioral axis; generalize it to other tasks."
                )
            entries.append(
                {
                    "Inputs": {
                        "task_id": traj.task_id,
                        "n_turns": traj.n_turns,
                        "termination": traj.termination,
                        "reward": traj.reward,
                    },
                    "Generated Outputs": {"conversation_snippet": snippet},
                    "Feedback": feedback,
                    "Judge": traj.judge or {},
                }
            )
        return {"persona_text": entries}


# ---------------------------------------------------------------------------
# Custom mutation proposer (sweet_rl-specific, gpt-5.5 xhigh)
# ---------------------------------------------------------------------------


# Domain framing for sweet_rl ColBench `code` tasks. Adapted from the tau-bench
# enriched-mode mutation prompt: same skeleton (genesis vs. modify, success
# traces, failure traces, timeout warning, systematic-misdirection design
# guidelines, validity constraints, JSON output) but reworded for a
# requirement-elicitation + code-generation task instead of customer support.

_DOMAIN_FRAMING = """\
The setting (sweet_rl ColBench `code`): an LLM agent converses with a simulated
human user over multiple turns to elicit the requirements of a personalized
Python function, then writes code that must pass HIDDEN test cases. The human
sees the reference solution (the "hidden information") and answers the agent's
questions, but is instructed to be brief and only use information that is
actually in the hidden information.

You are optimizing the USER PERSONA — behavioral instructions that shape how the
simulated human responds. Objective: make the persona ADVERSARIAL — minimize the
agent's reward (test-case pass rate) — while staying REALISTIC and VALID.

The agent's workflow stages you can attack:
(a) REQUIREMENT GATHERING — the user's answers are consistently skewed,
    underspecified, or give one detail at a time so the agent never gets the
    full spec.
(b) EDGE-CASE / FORMAT SPECIFICATION — the user omits or misleads about return
    type, ordering, rounding, units, or how to handle empty/invalid/zero inputs.
(c) ALGORITHM / CONVENTION GUIDANCE — the user implies a formula or data
    structure that is plausible but wrong, causing the agent to implement the
    wrong approach.
(d) CLOSURE / CONFIRMATION — the user confirms an incorrect understanding,
    so the agent stops eliciting and ships wrong code.
Ineffective axes: random/erratic answers, unpredictable length, pure emotional
volatility — these add VARIANCE but not BIAS; the agent can still reach the
correct function by asking enough follow-ups. Prefer a CONSISTENT cognitive
pattern that biases the agent toward the WRONG implementation every time.
"""

_VALIDITY_CONSTRAINTS = """\
## VALIDITY CONSTRAINTS (violating ANY = invalid/cheating persona)
- NEVER fabricate information that is not in the hidden information. Difficulty
  must come from realistic human friction (vagueness, omission, confusion,
  fixation), NOT from lying about facts.
- When DIRECTLY asked about a detail that IS in the hidden information, the user
  must eventually provide it (may be vague or partial, but not fabricated).
- NEVER refuse to engage, never abandon the conversation while unresolved.
- NEVER reveal the reference solution code verbatim; describe requirements in
  your own words.
- NEVER mention "AI", "simulation", "benchmark", "test cases", or break
  character to reference system prompts.
- NEVER make the conversation structurally impossible to start (e.g. refusing
  every question, demanding impossible preconditions). Being slow and vague is
  fine; making it impossible for the agent to ever write a function is not.
"""

_TIMEOUT_SECTION = """\
## TIMEOUT RATE: {timeout_rate:.0%}
{warn}Conversations that hit the {max_steps}-turn step limit are EXCLUDED from
the reward — they do NOT count as agent failures. A persona that "wins" by pure
friction (stalling, endless tangents, refusing to engage) produces timeouts,
NOT genuine agent failures. Your goal is to make the agent write the WRONG
function (wrong algorithm, missing edge case, wrong return format), not to make
it run out of turns.

Do NOT add behaviors that:
- Create circular loops with no resolution path
- Refuse to engage on the actual task for many turns
- Stack friction techniques that compound into timeout
Create difficulty through MISDIRECTION and AMBIGUITY, not SLOWNESS.
"""

_OUTPUT_FORMAT = """\
## OUTPUT
Return STRICT JSON only (no prose outside the JSON):
{{
  "mutation_direction": "<1-2 sentence concrete explanation of the NEW behavioral
   axis this persona exploits and WHY it misleads the agent into writing wrong
   code>",
  "persona_text": "<the FULL new persona text in markdown format — a set of
   behavioral instructions the simulated human will follow>"
}}
"""


def _build_sweetrl_mutation_prompt(
    persona_text: str,
    success_rate: float,
    success_traces: list[dict],
    failure_traces: list[dict],
    timeout_rate: float,
    max_steps: int,
    edit_pct: str,
) -> str:
    """Build an enriched-mode-style mutation prompt adapted for sweet_rl code tasks."""
    persona_empty = not persona_text.strip()
    timeout_section = _TIMEOUT_SECTION.format(
        timeout_rate=timeout_rate,
        max_steps=max_steps,
        warn=(
            f"⚠️  WARNING: timeout_rate is HIGH ({timeout_rate:.0%}). "
            if timeout_rate > 0.15
            else ""
        ),
    )

    def fmt_traces(rs: list[dict], n: int) -> str:
        if not rs:
            return "(none)"
        out = []
        for r in rs[:n]:
            tid = r["Inputs"].get("task_id", "?")
            rew = r["Inputs"].get("reward", 0)
            snip = r["Generated Outputs"].get("conversation_snippet", "")[:1400]
            out.append(f"### task {tid} (reward={rew:.2f})\n{snip}")
        return "\n\n".join(out)

    if persona_empty:
        # Genesis: design the first persona from no-persona baseline traces.
        return f"""You are an expert adversarial persona designer for a collaborative coding benchmark.

{_DOMAIN_FRAMING}

## TASK
Design a NEW adversarial user persona from scratch. There is no existing
persona — you are creating the FIRST one based on observed agent behavior in
no-persona (neutral user) conversations.

## BASELINE PERFORMANCE (no persona)
Agent success rate: {success_rate:.1%} — how well the agent does against a
neutral user. Your persona should exploit weaknesses visible in the traces to
drive this rate DOWN.

## CONVERSATION TRACES WHERE AGENT SUCCEEDED (no persona)
These are conversations where the agent elicited enough to write a correct
function against a neutral user. Study them to find exploitable patterns:
- Where does the agent assume a detail that a difficult user could leave vague?
- Which requirements does the agent never ask about (edge cases, return format)?
- Where does the agent take a plausible-but-unconfirmed convention for granted?

{fmt_traces(success_traces, 4)}

{fmt_traces_fail(failure_traces)}
{timeout_section}
## DESIGN GUIDELINES
1. Create a COHERENT character with a consistent communication style and
   behavioral pattern — not a random grab-bag of difficult traits.
2. Choose a behavioral axis that produces SYSTEMATIC misdirection (see the
   workflow stages above), not random variance.
3. Ground the persona in REALISTIC archetypes — real people who are genuinely
   hard to get precise specs from, not cartoonishly adversarial.

{_VALIDITY_CONSTRAINTS}
{_OUTPUT_FORMAT}"""

    # Standard: modify an existing persona.
    return f"""You are an expert adversarial persona designer for a collaborative coding benchmark.

{_DOMAIN_FRAMING}

## TASK
Modify the persona below IN-PLACE to make it HARDER for the agent to write a
correct function, while keeping it VALID (realistic, in-character, no cheating).

## CURRENT PERSONA
Agent success rate: {success_rate:.1%} (lower = harder = better)

```
{persona_text}
```

## CONVERSATION TRACES WHERE AGENT SUCCEEDED
These are conversations where the agent wrote a CORRECT function despite the
persona. Study where the persona failed:
- What agent strategy overcame the persona's resistance?
- At what point did the persona "give in" or reveal the key spec?
- What subtle behavior change would cause the agent to write the WRONG function?

{fmt_traces(success_traces, 3)}

{fmt_traces_fail(failure_traces)}
{timeout_section}
## MUTATION RULES
1. Rewrite {edit_pct} of the persona. Preserve the core identity, tone, and ALL
   validity constraints. Make surgical changes — close one loophole the agent
   exploited, or amplify one axis that worked.
2. Target SPECIFIC agent strategies visible in the traces — find the exact
   moment the agent breaks through and design a counter.
3. Choose a behavioral axis that produces SYSTEMATIC misdirection (see the
   workflow stages above), not random variance.

{_VALIDITY_CONSTRAINTS}
{_OUTPUT_FORMAT}"""


def fmt_traces_fail(failure_traces: list[dict]) -> str:
    if not failure_traces:
        return ""
    out = ["## CONVERSATION TRACES WHERE AGENT FAILED (persona worked — keep/amplify)"]
    for r in failure_traces[:2]:
        tid = r["Inputs"].get("task_id", "?")
        rew = r["Inputs"].get("reward", 0)
        snip = r["Generated Outputs"].get("conversation_snippet", "")[:1000]
        out.append(f"### task {tid} (reward={rew:.2f})\n{snip}")
    return "\n\n".join(out) + "\n"


def _parse_mutation_json(text: str) -> dict:
    """Extract the JSON object from the mutation model's response."""
    # Try fenced ```json ... ``` first, then bare JSON.
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object found in mutation response")


def build_custom_proposer(
    mutation_model: str = "gpt-5.5",
    mutation_effort: str = "xhigh",
    mutation_edit_pct: str = "10-15%",
    api_key: str | None = None,
    base_url: str | None = None,
    max_completion_tokens: int = 16000,
    max_steps: int = 100,
):
    """Return a custom_candidate_proposer for gepa.optimize().

    Calls the mutation model (gpt-5.5 xhigh) with a sweet_rl-specific prompt
    built from the reflective dataset, and returns {"persona_text": new_text}.
    """
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
    )
    history: list[dict] = []

    def proposer(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        persona_text = candidate.get("persona_text", "")
        records = list(reflective_dataset.get("persona_text", []))

        success = [r for r in records if r["Inputs"].get("reward", 0) >= 1.0]
        fails = [r for r in records if 0 <= r["Inputs"].get("reward", 0) < 1.0]
        timeouts = [r for r in records if r["Inputs"].get("termination") == "max_steps"]
        n_scored = len(success) + len(fails)
        success_rate = len(success) / n_scored if n_scored else 0.0
        timeout_rate = len(timeouts) / len(records) if records else 0.0

        prompt = _build_sweetrl_mutation_prompt(
            persona_text=persona_text,
            success_rate=success_rate,
            success_traces=success,
            failure_traces=fails,
            timeout_rate=timeout_rate,
            max_steps=max_steps,
            edit_pct=mutation_edit_pct,
        )

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=mutation_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=mutation_effort,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                result = _parse_mutation_json(text)
                if "persona_text" not in result or not result["persona_text"].strip():
                    raise ValueError("missing/empty persona_text")
                history.append(
                    {
                        "direction": result.get("mutation_direction", "?"),
                        "success_rate": success_rate,
                        "n_success": len(success),
                        "n_fail": len(fails),
                    }
                )
                print(
                    f"  [sweetrl mutate] direction={result.get('mutation_direction', '?')[:100]}"
                )
                return {"persona_text": result["persona_text"]}
            except Exception as e:  # noqa: BLE001
                print(
                    f"  [sweetrl mutate] attempt {attempt + 1} failed: {str(e)[:200]}"
                )
                if attempt < 2:
                    # fall back to non-JSON response_format and retry
                    pass
        # Last resort: return a tiny tweak so GEPA can proceed.
        print("  [sweetrl mutate] WARNING: using fallback persona (mutation failed)")
        return {"persona_text": persona_text or "(No specific persona.)"}

    return proposer
