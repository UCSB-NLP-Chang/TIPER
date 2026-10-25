#!/usr/bin/env python3
"""Run airline v5 personas WITH human-style turn-taking preamble.

Compares original persona vs humanstyle-augmented persona on airline val set.
Uses same config as cert_alaska v5 (gpt-5.4 agent+user, reasoning_effort=high).

Usage:
  python experiments/persona_search/scripts/run_humanstyle_experiment.py --dry-run
  python experiments/persona_search/scripts/run_humanstyle_experiment.py --smoke 2     # 2 tasks only
  python experiments/persona_search/scripts/run_humanstyle_experiment.py               # full val set
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = str(REPO / ".venv/bin/tau2")
SIM = REPO / "data/simulations"
EXPERIMENT_DIR = REPO / "data/persona_opt/humanstyle_experiment"

SPLIT_FILE = REPO / "experiments/persona_search/configs/airline_alaska_split_v5.json"
SPLIT = json.loads(SPLIT_FILE.read_text())
VAL_IDS = SPLIT["val"]["airline"]

PGEPA_DIR = REPO / "data/persona_opt/pgepa_runs/airline_v5"

NUM_TRIALS = 3
AGENT_LLM = "gpt-5.4"
USER_LLM = "gpt-5.4"
EFFORT = "high"
CONCURRENCY = 30


def get_personas(version: str = "v1") -> dict[str, str]:
    """Return {label: persona_text} for all 4 humanstyle conditions."""
    personas = {}
    suffix = "humanstyle" if version == "v1" else "hsv2"
    tag = "hs" if version == "v1" else "hsv2"
    for island in ["alaska_01", "alaska_03"]:
        for cond in ["seed", "best"]:
            path = EXPERIMENT_DIR / f"{island}_{cond}_{suffix}.md"
            personas[f"{island}_{cond}_{tag}"] = path.read_text()
    return personas


def run_cell(
    label: str, persona_text: str, task_ids: list[str], num_trials: int, dry_run: bool
) -> None:
    save_to = f"humanstyle_exp/{label}"
    save_path = SIM / save_to / "results.json"

    persona_cfg = json.dumps(
        {
            "custom_guidelines": persona_text,
            "verbosity": "standard",
        }
    )

    cmd = [
        TAU2,
        "run",
        "--domain",
        "airline",
        "--agent-llm",
        AGENT_LLM,
        "--agent-llm-args",
        json.dumps({"reasoning_effort": EFFORT}),
        "--user",
        "user_simulator",
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        json.dumps({"reasoning_effort": EFFORT}),
        "--user-persona",
        persona_cfg,
        "--num-trials",
        str(num_trials),
        "--task-ids",
        *task_ids,
        "--max-steps",
        "120",
        "--max-concurrency",
        str(CONCURRENCY),
        "--save-to",
        save_to,
        "--auto-resume",
    ]

    print(f"\n{'=' * 60}")
    print(
        f"[{label}] {len(task_ids)} tasks × {num_trials} trials = {len(task_ids) * num_trials} sims"
    )
    print(f"  save: {save_path}")
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(cmd)}")
        return

    print(f"  Running...")
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  [WARN] {label} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Only run this many tasks (for smoke testing)",
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Only run specific cells, e.g. alaska_01_seed_hs",
    )
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="v1=original humanstyle preamble, v2=style-only preamble",
    )
    args = parser.parse_args()

    task_ids = VAL_IDS[: args.smoke] if args.smoke else VAL_IDS
    personas = get_personas(args.version)

    if args.cells:
        personas = {k: v for k, v in personas.items() if k in args.cells}

    print(f"Airline humanstyle experiment")
    print(f"  Val tasks: {len(task_ids)}")
    print(f"  Trials: {args.num_trials}")
    print(f"  Cells: {list(personas.keys())}")

    for label, persona_text in personas.items():
        run_cell(label, persona_text, task_ids, args.num_trials, args.dry_run)

    print(f"\n{'=' * 60}")
    print("All cells done.")


if __name__ == "__main__":
    main()
