#!/usr/bin/env python3
"""Re-run banking tasks affected by v1.0.1 fixes (T053, T074, T077-T086).

For each condition in fullset/banking/, strips affected sims from results.json
(backed up as results.json.pre_v101), then runs tau2 with --auto-resume to fill
the gaps.

Usage:
  # Dry run — show what would change, run nothing
  .venv/bin/python experiments/persona_search/scripts/rerun_banking_v101.py --dry-run

  # Re-run gpt-5.4 canonical fullset only (default)
  .venv/bin/python experiments/persona_search/scripts/rerun_banking_v101.py

  # Re-run a specific fullset
  .venv/bin/python experiments/persona_search/scripts/rerun_banking_v101.py --fullset fullset_glm

  # Re-run all fullsets
  .venv/bin/python experiments/persona_search/scripts/rerun_banking_v101.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
SIM = REPO / "data/simulations"
TAU2 = str(REPO / ".venv/bin/tau2")

AFFECTED_TASKS = {
    f"task_{i:03d}" for i in [53, 74, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
}

# Per-fullset agent config (model + args). Extracted from fullset_build.py / run scripts.
AGENT_CONFIGS = {
    "fullset": {"llm": "gpt-5.5", "args": {"reasoning_effort": "high"}},
    "fullset_sonnet46": {
        "llm": "claude-sonnet-4-6",
        "args": {"reasoning_effort": "high"},
    },
    "fullset_glm": {
        "llm": "anthropic/accounts/fireworks/models/glm-5p2",
        "args": {"reasoning_effort": "high"},
    },
    "fullset_max_glm": {
        "llm": "anthropic/accounts/fireworks/models/glm-5p2",
        "args": {"reasoning_effort": "max"},
    },
    "fullset_ds": {
        "llm": "anthropic/claude-deepseek-v3",
        "args": {"reasoning_effort": "high"},
    },
    "fullset_ds4pro_aliyun": {
        "llm": "anthropic/claude-deepseek-v4-pro",
        "args": {"reasoning_effort": "high"},
    },
    "fullset_qwen38max": {
        "llm": "anthropic/claude-qwen-3.8-max",
        "args": {"reasoning_effort": "high"},
    },
}

# User config is consistent across all fullsets for banking
USER_LLM = "gpt-5.4"
USER_ARGS = {"reasoning_effort": "high"}
RETRIEVAL = "bm25"


def task_is_affected(task_id: str) -> bool:
    for t in AFFECTED_TASKS:
        if t in task_id:
            return True
    return False


def strip_affected(results_path: Path, dry: bool) -> tuple[int, int]:
    """Remove affected sims from results.json. Returns (total_before, n_removed)."""
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])
    before = len(sims)
    kept = [s for s in sims if not task_is_affected(s.get("task_id", ""))]
    removed = before - len(kept)

    if removed > 0 and not dry:
        backup = results_path.with_suffix(".json.pre_v101")
        if not backup.exists():
            shutil.copy2(results_path, backup)
        data["simulations"] = kept
        results_path.write_text(json.dumps(data, ensure_ascii=False, indent=None))

    return before, removed


def extract_persona(results_path: Path) -> str | None:
    data = json.loads(results_path.read_text())
    pc = ((data.get("info") or {}).get("user_info") or {}).get("persona_config") or {}
    return pc.get("custom_guidelines") if isinstance(pc, dict) else None


def merge_new_sims(orig_path: Path, tmp_path: Path, dry: bool) -> int:
    """Merge sims from tmp results into original, replacing affected tasks."""
    tmp_rj = tmp_path / "results.json"
    if not tmp_rj.exists():
        print(f"  [WARN] no tmp results to merge from {tmp_path}")
        return 0
    new_data = json.loads(tmp_rj.read_text())
    new_sims = new_data.get("simulations", [])
    if not new_sims:
        print(f"  [WARN] tmp results empty")
        return 0

    orig_data = json.loads(orig_path.read_text())
    orig_sims = orig_data.get("simulations", [])
    # Keep non-affected + append new
    kept = [s for s in orig_sims if not task_is_affected(s.get("task_id", ""))]
    merged = kept + new_sims

    if not dry:
        orig_data["simulations"] = merged
        orig_path.write_text(json.dumps(orig_data, ensure_ascii=False))
        shutil.rmtree(tmp_path)

    print(f"  Merged: {len(kept)} kept + {len(new_sims)} new = {len(merged)} total")
    return len(new_sims)


def run_cell(fullset_name: str, cond_dir: Path, dry: bool) -> None:
    results_path = cond_dir / "results.json"
    if not results_path.exists():
        print(f"  [skip] {cond_dir.name}: no results.json")
        return

    # Check if affected tasks are missing (already stripped or never present)
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])
    present_tasks = {s.get("task_id", "").split("[")[0].split("|")[0] for s in sims}
    missing = sorted(t for t in AFFECTED_TASKS if t not in present_tasks)
    already_there = sorted(t for t in AFFECTED_TASKS if t in present_tasks)

    if not missing and not already_there:
        print(f"  [skip] {cond_dir.name}: no affected tasks in this set")
        return
    if not missing:
        print(
            f"  [skip] {cond_dir.name}: all 12 affected tasks already present with new code"
        )
        return

    print(
        f"  {cond_dir.name}: {len(missing)} affected tasks missing, {len(sims)} existing sims"
    )

    # Backup original
    backup = results_path.with_suffix(".json.pre_v101")
    if not backup.exists() and not dry:
        shutil.copy2(results_path, backup)

    agent_cfg = AGENT_CONFIGS.get(fullset_name)
    if agent_cfg is None:
        print(f"  [ERROR] no agent config for {fullset_name}, skipping run")
        return

    persona = extract_persona(results_path)
    # Run to a temp subdir, then merge back
    tmp_name = f"_v101_rerun_{cond_dir.name}"
    tmp_rel = f"{fullset_name}/banking/{tmp_name}"
    tmp_abs = SIM / tmp_rel
    task_ids = sorted(AFFECTED_TASKS)

    cmd = [
        TAU2,
        "run",
        "--domain",
        "banking_knowledge",
        "--user",
        "user_simulator",
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        json.dumps(USER_ARGS),
        "--agent-llm",
        agent_cfg["llm"],
        "--agent-llm-args",
        json.dumps(agent_cfg["args"]),
        "--num-trials",
        "3",
        "--seed",
        "300",
        "--max-concurrency",
        "20",
        "--max-steps",
        "120",
        "--timeout",
        "600",
        "--retrieval-config",
        RETRIEVAL,
        "--save-to",
        tmp_rel,
        "--task-ids",
        *task_ids,
    ]
    if persona is not None:
        cmd += [
            "--user-persona",
            json.dumps({"custom_guidelines": persona}, ensure_ascii=False),
        ]

    if dry:
        show = " ".join(c if len(c) < 80 else c[:60] + f"...<{len(c)}c>" for c in cmd)
        print(f"  [would run] {show}")
    else:
        print(f"  Running tau2 for {len(task_ids)} tasks × 3 trials → {tmp_rel}")
        subprocess.run(cmd, cwd=str(REPO), check=False)
        merge_new_sims(results_path, tmp_abs, dry=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fullset",
        default="fullset",
        help="which fullset to re-run (default: fullset)",
    )
    ap.add_argument(
        "--all", action="store_true", help="re-run ALL fullsets that have banking data"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="show plan without making changes"
    )
    args = ap.parse_args()

    if args.all:
        targets = sorted(AGENT_CONFIGS.keys())
    else:
        targets = [args.fullset]

    for fs_name in targets:
        banking_dir = SIM / fs_name / "banking"
        if not banking_dir.exists():
            print(f"\n{fs_name}: no banking/ directory, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"{fs_name} (agent={AGENT_CONFIGS.get(fs_name, {}).get('llm', '?')})")
        print(f"{'=' * 60}")

        for cond_dir in sorted(banking_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            run_cell(fs_name, cond_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
