#!/usr/bin/env python3
"""Test certification for ev2_3islands_v4: sequential phases.

Phase 1: brinks_00 top1 + brinks_04 top1 (2 × 54 × 3 = 324 sims)
Phase 2: seed_04 1 trial (54 × 1 = 54 sims)
Phase 3: brinks_00 top2 + brinks_04 top2 (2 × 54 × 3 = 324 sims)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
RUN_ROOT = REPO / "data/persona_opt/pgepa_runs/ev2_3islands_v4"
SPLIT = REPO / "experiments/persona_search/configs/telecom_island_split.json"
CERT_DIR = RUN_ROOT / "test_cert"
CERT_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 25
AGENT_LLM = "gpt-5.4"
AGENT_LLM_ARGS = json.dumps({"reasoning_effort": "high"})
USER_TYPE = "user_simulator"
USER_LLM = "gpt-5.4"
EFFORT = "xhigh"
SIM_TIMEOUT = 600
MAX_STEPS = 120

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def load_test_tasks() -> list[str]:
    d = json.loads(SPLIT.read_text())
    return d["test"]["telecom"]


def load_top_n(island: str, n: int) -> list[dict]:
    state = json.loads((RUN_ROOT / f"ev2v4_{island}" / "pgepa_state.json").read_text())
    feasible = sorted(
        [c for c in state["candidates"] if c.get("feasible", True)],
        key=lambda c: c.get("agg", 0),
        reverse=True,
    )
    return feasible[:n]


def run_persona(
    label: str, persona_text: str, save_to: str, task_ids: list[str], num_trials: int
) -> None:
    ts = time.strftime("%H:%M:%S")
    print(
        f"\n[{ts}] === {label}: {len(task_ids)} tasks × {num_trials} trials "
        f"= {len(task_ids) * num_trials} sims ==="
    )

    user_args = json.dumps({"reasoning_effort": EFFORT})
    user_persona = json.dumps({"custom_guidelines": ROLEPLAY_PREAMBLE + persona_text})

    cmd = [
        str(REPO / ".venv/bin/tau2"),
        "run",
        "--domain",
        "telecom",
        "--user",
        USER_TYPE,
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        user_args,
        "--agent-llm",
        AGENT_LLM,
        "--agent-llm-args",
        AGENT_LLM_ARGS,
        "--num-trials",
        str(num_trials),
        "--max-concurrency",
        str(CONCURRENCY),
        "--save-to",
        save_to,
        "--auto-resume",
        "--user-persona",
        user_persona,
        "--max-steps",
        str(MAX_STEPS),
        "--timeout",
        str(SIM_TIMEOUT),
        "--task-ids",
        *task_ids,
    ]

    subprocess.run(cmd, check=True, cwd=str(REPO))
    print(f"[{time.strftime('%H:%M:%S')}] Done: {label}")


def main():
    task_ids = load_test_tasks()
    print(f"Test tasks: {len(task_ids)}")

    b00_top = load_top_n("brinks_00", 2)
    b04_top = load_top_n("brinks_04", 2)
    seed_04 = (RUN_ROOT / "seeds/brinks_04.md").read_text()

    print(f"brinks_00 top1: idx={b00_top[0]['idx']} val_agg={b00_top[0]['agg']:.3f}")
    print(f"brinks_00 top2: idx={b00_top[1]['idx']} val_agg={b00_top[1]['agg']:.3f}")
    print(f"brinks_04 top1: idx={b04_top[0]['idx']} val_agg={b04_top[0]['agg']:.3f}")
    print(f"brinks_04 top2: idx={b04_top[1]['idx']} val_agg={b04_top[1]['agg']:.3f}")

    # Phase 1
    print("\n" + "=" * 50)
    print("Phase 1: top-1 candidates (324 sims)")
    print("=" * 50)
    run_persona(
        "b00_top1", b00_top[0]["persona_text"], "ev2v4_cert/b00_top1", task_ids, 3
    )
    run_persona(
        "b04_top1", b04_top[0]["persona_text"], "ev2v4_cert/b04_top1", task_ids, 3
    )

    # Phase 2
    print("\n" + "=" * 50)
    print("Phase 2: seed_04 (54 sims)")
    print("=" * 50)
    run_persona("seed_04", seed_04, "ev2v4_cert/seed_04", task_ids, 1)

    # Phase 3
    print("\n" + "=" * 50)
    print("Phase 3: top-2 candidates (324 sims)")
    print("=" * 50)
    run_persona(
        "b00_top2", b00_top[1]["persona_text"], "ev2v4_cert/b00_top2", task_ids, 3
    )
    run_persona(
        "b04_top2", b04_top[1]["persona_text"], "ev2v4_cert/b04_top2", task_ids, 3
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] All phases complete.")
    print(f"Results: data/simulations/ev2v4_cert/")


if __name__ == "__main__":
    main()
