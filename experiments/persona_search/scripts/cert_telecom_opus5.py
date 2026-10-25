#!/usr/bin/env python3
"""Telecom 认证：no-persona vs best persona, USER = claude-opus-5 (high).

口径同 cert_telecom_s5.py:
  no-persona = tau2 run 不传 --user-persona (user_simulator 默认无 persona)
  best       = tau2 run --user-persona {"custom_guidelines": roleplay_preamble + text}
agent = gpt-5.4 (high) 标准OpenAI; user = claude-opus-5 (high) 经 :4112 vertex shim.
3 trials, conc=24, test split = 54 telecom tasks (telecom_island_split test).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = REPO / ".venv" / "bin" / "tau2"
RUN_DIR = (
    REPO
    / "data"
    / "persona_opt"
    / "pgepa_runs"
    / "model_ablation"
    / "telecom"
    / "user_opus5"
)
CAND = RUN_DIR / "candidates.json"
SPLIT = REPO / "experiments/persona_search/configs/telecom_island_split.json"
OUT_ROOT = REPO / "data" / "simulations" / "cert_telecom_opus5"
REPORT = RUN_DIR / "cert" / "cert_report.json"

AGENT = "openai/gpt-5.4"
AGENT_ARGS = '{"reasoning_effort":"high"}'
USER = "user_simulator"
USER_LLM = "claude-opus-5"
EFFORT = "high"
NUM_TRIALS = 3
CONC = 24
MAX_STEPS = 120
TIMEOUT = 7200

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def best_persona_text() -> tuple[str, str]:
    cands = json.loads(CAND.read_text())
    best, best_sc = None, -1.0
    for c in cands:
        t = (c.get("persona_text") or "").strip()
        if len(t) < 50:
            continue
        sc = float(c.get("exploration_score", 0.0))
        if sc > best_sc:
            best, best_sc = c, sc
    if best is None:
        sys.exit("ERROR: no non-empty candidate in candidates.json")
    return best.get("persona_id") or "best", best["persona_text"]


def test_task_ids() -> list[str]:
    sp = json.loads(SPLIT.read_text())
    return sp["test"]["telecom"]


def build_cmd(
    task_ids: list[str], run_name: str, persona_text: str | None
) -> list[str]:
    user_args = {"reasoning_effort": EFFORT}
    cmd = [
        str(TAU2),
        "run",
        "--domain",
        "telecom",
        "--user",
        USER,
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        json.dumps(user_args),
        "--agent-llm",
        AGENT,
        "--agent-llm-args",
        AGENT_ARGS,
        "--num-trials",
        str(NUM_TRIALS),
        "--max-concurrency",
        str(CONC),
        "--save-to",
        run_name,
        "--auto-resume",
        "--max-steps",
        str(MAX_STEPS),
        "--timeout",
        str(TIMEOUT),
    ]
    if persona_text:
        cmd += [
            "--user-persona",
            json.dumps({"custom_guidelines": ROLEPLAY_PREAMBLE + persona_text}),
        ]
    cmd += ["--task-ids", *task_ids]
    return cmd


def run(cmd: list[str], label: str) -> Path:
    print(
        f"\n{'=' * 70}\n[{label}] running {len(cmd) - 2} tasks x{NUM_TRIALS} conc={CONC}\n{'=' * 70}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(REPO))
    save_to = cmd[cmd.index("--save-to") + 1]
    rp = REPO / "data" / "simulations" / save_to / "results.json"
    if not rp.exists():
        sys.exit(f"ERROR: no results.json at {rp}")
    return rp


def avg_reward(rp: Path) -> dict:
    data = json.loads(rp.read_text())
    sims = data.get("simulations", [])
    rewards, infra, per_task = [], 0, {}
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            infra += 1
            continue
        rewards.append(float(r))
        per_task.setdefault(s["task_id"], []).append(float(r))
    n = len(rewards)
    avg = sum(rewards) / n if n else 0.0
    task_means = [sum(v) / len(v) for v in per_task.values()]
    task_avg = sum(task_means) / len(task_means) if task_means else 0.0
    return {
        "n_sims": len(sims),
        "n_rewarded": n,
        "n_infra_none": infra,
        "avg_reward_per_sim": round(avg, 4),
        "avg_reward_per_task": round(task_avg, 4),
        "n_tasks": len(per_task),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tids = test_task_ids()
    pid, ptext = best_persona_text()
    print(
        f"best persona: id={pid}  chars={len(ptext)}  test_tasks={len(tids)}",
        flush=True,
    )

    np_cmd = build_cmd(tids, "cert_telecom_opus5/no_persona/telecom", None)
    np_rp = run(np_cmd, "NO-PERSONA baseline")
    np_m = avg_reward(np_rp)
    print(f"[NO-PERSONA] {np_m}", flush=True)

    bp_cmd = build_cmd(tids, "cert_telecom_opus5/best/telecom", ptext)
    bp_rp = run(bp_cmd, "BEST persona")
    bp_m = avg_reward(bp_rp)
    print(f"[BEST] {bp_m}", flush=True)

    delta = round(np_m["avg_reward_per_task"] - bp_m["avg_reward_per_task"], 4)
    report = {
        "domain": "telecom",
        "agent": AGENT,
        "agent_args": AGENT_ARGS,
        "user": USER,
        "user_llm": USER_LLM,
        "effort": EFFORT,
        "num_trials": NUM_TRIALS,
        "concurrency": CONC,
        "n_test_tasks": len(tids),
        "best_persona_id": pid,
        "best_persona_chars": len(ptext),
        "no_persona": np_m,
        "best_persona": bp_m,
        "delta_reward": delta,
        "delta_per_sim": round(
            np_m["avg_reward_per_sim"] - bp_m["avg_reward_per_sim"], 4
        ),
        "interpretation": "delta>0 => persona lowers agent success (harder)",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n{'=' * 70}\nCERT REPORT -> {REPORT}")
    print(json.dumps(report, indent=2))
    print(
        f"\nΔ (no_persona - best) = {delta}  ({'persona HARDER' if delta > 0 else 'no effect / easier'})"
    )


if __name__ == "__main__":
    main()
