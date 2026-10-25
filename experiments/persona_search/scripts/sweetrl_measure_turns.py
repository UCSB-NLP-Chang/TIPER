#!/usr/bin/env python3
"""Measure conversation length (n_turns) for no-persona vs best-persona on test tasks."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
for _p in (str(_SWEET_RL_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweetrl_persona_adapter import SweetRLPersonaAdapter, SweetRLTask


def build_tasks(data_path, split_seed, n, offset=60):
    import random

    raw = [json.loads(l) for l in open(data_path)]
    random.Random(split_seed).shuffle(raw)
    raw = raw[:110]
    out = []
    for i, t in enumerate(raw[offset : offset + n]):
        out.append(
            SweetRLTask(
                task_id=f"test_{i:03d}",
                problem_description=t["problem_description"],
                ground_truth=t["ground_truth"],
                hidden_information=t.get("hidden_information", t["ground_truth"]),
                test_cases=t["test_cases"],
                split="test",
            )
        )
    return out


def main():
    data = str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl")
    best = Path(
        _SWEET_RL_REPO
        / "data"
        / "persona_opt"
        / "gepa_runs"
        / "sweetrl_full_001"
        / "best_persona.md"
    ).read_text()
    tasks = build_tasks(data, 42, 20)
    adapter = SweetRLPersonaAdapter(
        agent_model="gpt-5.4",
        agent_effort="high",
        human_model="gpt-5.4",
        human_effort="high",
        agent_prompt_path=str(_SWEET_RL_REPO / "prompts" / "llm_agent_code_prompt.txt"),
        human_prompt_path=str(
            _SWEET_RL_REPO / "prompts" / "human_simulator_code_prompt.txt"
        ),
        max_steps=100,
        num_trials=2,
        max_concurrency=20,
        agent_round_budget=100,
        online_judge=False,
        severe_gate_rate=None,
        timeout_attack_threshold=None,
    )
    for label, persona in (("no-persona", ""), ("best-persona", best)):
        batch = adapter.evaluate(tasks, {"persona_text": persona}, capture_traces=True)
        # trajectories = worst-reward trial per task; but we want all trials' turn counts.
        # Re-collect from the batch: we only kept worst trial. Instead, re-run capturing all?
        # Simpler: the worst-trial trace has n_turns; use those as representative.
        turns = [tr.n_turns for tr in (batch.trajectories or [])]
        terms = [tr.termination for tr in (batch.trajectories or [])]
        answered = sum(1 for t in terms if t == "answered")
        maxstep = sum(1 for t in terms if t == "max_steps")
        print(
            f"[{label}] n={len(turns)} turns: avg={statistics.mean(turns):.1f} med={statistics.median(turns)} min={min(turns)} max={max(turns)} | answered={answered} max_steps={maxstep}"
        )
        print(f"  per-task turns: {sorted(turns)}")


if __name__ == "__main__":
    main()
