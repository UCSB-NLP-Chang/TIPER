#!/usr/bin/env python3
"""Retail fullset (步骤②): v4 gc_00/01 + v2 gc_00, agent high + user xhigh, 4 trials.
跑完整个 retail set (split v4: train30+val30+test54 = 114 task × 4 trials, seed 300).
v4_gc_01 best=gen0 (seed=opt 退化), seed/opt 合并跑一次 -> 5 cells.
persona = ROLEPLAY_PREAMBLE + pgepa_state persona_text (byte-identical to search).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = str(REPO / ".venv/bin/tau2")

PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def persona(state_rel, kind):
    st = json.load(open(REPO / state_rel))
    cands = st["candidates"]
    if kind == "seed":
        return PREAMBLE + cands[0]["persona_text"]
    bidx = st["best_idx"]
    return PREAMBLE + cands[bidx]["persona_text"]


split = json.load(
    open(REPO / "experiments/persona_search/configs/retail_guitar_center_split_v4.json")
)
ids = split["test"]["retail"]  # 只跑 test 54 task 选 optim, train/val 不需要

CELLS = [
    (
        "v2_gc_00_seed",
        str(DATA_ROOT / "data/persona_opt/pgepa_runs/retail_v2/gc_00/pgepa_state.json"),
        "seed",
    ),
    (
        "v2_gc_00_opt",
        str(DATA_ROOT / "data/persona_opt/pgepa_runs/retail_v2/gc_00/pgepa_state.json"),
        "best",
    ),
]


def run_cell(name, state_rel, kind):
    cg = persona(state_rel, kind)
    cmd = [
        TAU2,
        "run",
        "--domain",
        "retail",
        "--agent-llm",
        "gpt-5.4",
        "--agent-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--user-llm",
        "gpt-5.4",
        "--user-llm-args",
        json.dumps({"reasoning_effort": "xhigh"}),
        "--num-trials",
        "4",
        "--seed",
        "300",
        "--max-concurrency",
        "10",
        "--max-steps",
        "120",
        "--user-persona",
        json.dumps({"custom_guidelines": cg}, ensure_ascii=False),
        "--task-ids",
        *ids,
        "--save-to",
        f"retail_fullset_v4/{name}",
        "--auto-resume",
    ]
    print(
        f"=== {name}: {len(ids)} task × 4 trials -> retail_fullset_v4/{name} (guidelines {len(cg)} chars) ===",
        flush=True,
    )
    return subprocess.Popen(cmd, cwd=str(REPO))


def main():
    procs = []
    for name, state_rel, kind in CELLS:
        procs.append((name, run_cell(name, state_rel, kind)))
    failed = 0
    for name, p in procs:
        rc = p.wait()
        if rc:
            print(f"WARNING: {name} exited rc={rc}", flush=True)
            failed += 1
    print(f"\n===== ALL DONE (failed: {failed}/{len(procs)}) =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
