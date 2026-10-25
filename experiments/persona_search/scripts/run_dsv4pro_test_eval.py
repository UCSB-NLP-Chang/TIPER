#!/usr/bin/env python3
"""Test-set evaluation for DeepSeek v4-pro GEPA best personas.
Runs best + seed persona from each island + no_persona baseline
on the test split (54 tasks) x 3 trials.
"""

import json
import os
import subprocess
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)

RUN_ROOT = DATA_ROOT / "data/persona_opt/pgepa_runs/islands_telecom_dsv4pro_v2"
SPLIT_FILE = REPO_ROOT / "experiments/persona_search/configs/telecom_island_split.json"
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "3"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))

AGENT_LLM = "anthropic/accounts/fireworks/models/deepseek-v4-pro"
USER_TYPE = "user_simulator"
USER_LLM = "gpt-5.4"
EFFORT = "high"
FIREWORKS_SHIM_PORT = os.environ.get("FIREWORKS_SHIM_PORT", "4114")

os.environ["ANTHROPIC_API_BASE"] = f"http://127.0.0.1:{FIREWORKS_SHIM_PORT}"
os.environ["ANTHROPIC_API_KEY"] = "dummy"
if "OPENAI_API_KEY" not in os.environ:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1]
                break

split = json.loads(SPLIT_FILE.read_text())
test_tasks = split["test"]["telecom"]
print(f"Test set: {len(test_tasks)} tasks x {NUM_TRIALS} trials")

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def run_eval(label: str, persona_json: str | None, save_to: str):
    print(f"\n{'=' * 50}\n  {label} -> {save_to}\n{'=' * 50}")
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "tau2"),
        "run",
        "--domain",
        "telecom",
        "--user",
        USER_TYPE,
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        json.dumps({"reasoning_effort": EFFORT}),
        "--agent-llm",
        AGENT_LLM,
        "--agent-llm-args",
        "{}",
        "--num-trials",
        str(NUM_TRIALS),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--save-to",
        save_to,
        "--auto-resume",
        "--task-ids",
        *test_tasks,
    ]
    if persona_json:
        cmd.extend(["--user-persona", persona_json])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# 1. No-persona baseline
run_eval("no_persona baseline", None, "gepa_opt/dsv4pro_v2_test/no_persona")

# 2+3. Each island's best and seed personas
for island in ["brinks_00", "p0003"]:
    state_path = RUN_ROOT / f"isl_ds_{island}" / "pgepa_state.json"
    tree_path = RUN_ROOT / f"isl_ds_{island}" / "lineage_tree.json"
    if not tree_path.exists():
        print(f"SKIP {island}: no lineage_tree.json")
        continue

    state = json.loads(state_path.read_text())
    tree = json.loads(tree_path.read_text())
    best_idx = state.get("best_idx", 0)
    best_agg = state.get("best_agg", 0)

    # Best persona
    best_text = tree["personas"][f"idx{best_idx}"]
    best_persona = json.dumps({"custom_guidelines": ROLEPLAY_PREAMBLE + best_text})
    run_eval(
        f"{island} best (idx{best_idx}, diff={best_agg:.3f})",
        best_persona,
        f"gepa_opt/dsv4pro_v2_test/{island}_best",
    )

    # Seed persona
    seed_text = tree["personas"]["idx0"]
    seed_persona = json.dumps({"custom_guidelines": ROLEPLAY_PREAMBLE + seed_text})
    run_eval(
        f"{island} seed (idx0)",
        seed_persona,
        f"gepa_opt/dsv4pro_v2_test/{island}_seed",
    )

print(f"\n=== All test evals complete ===")
print(f"Results in data/simulations/gepa_opt/dsv4pro_v2_test/")
