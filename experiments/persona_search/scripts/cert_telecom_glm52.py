#!/usr/bin/env python3
"""Telecom 认证：no-persona vs best persona (GLM-5.2 agent, 经 fireworks :4114 shim)。

口径与 cert_telecom_gpt56luna.py / cert_telecom_s5.py 一致，仅 agent model 换
GLM-5.2 + 路由改 fireworks shim。agent = openai/accounts/fireworks/models/glm-5p2
(high) 经 litellm openai provider -> :4114 openai_to_fireworks shim -> fireworks
/v1/messages; user = gpt-5.4 (high) 经标准 OpenAI。3 trials, conc=8, test split=54。
额外报转人工率 (messages 含 transfer_to_human_agents 的 sim 比例)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = REPO / ".venv" / "bin" / "tau2"
RUN_TAG = "glm52_4domain"
SPLIT = REPO / "experiments/persona_search/configs/telecom_island_split.json"
CAND = (
    REPO
    / "data"
    / "persona_opt"
    / "pgepa_runs"
    / RUN_TAG
    / "telecom"
    / "isl_glm52_telecom_g0"
    / "candidates.json"
)
OUT_ROOT = REPO / "data" / "simulations" / "cert_telecom_glm52"
REPORT = (
    REPO
    / "data"
    / "persona_opt"
    / "pgepa_runs"
    / RUN_TAG
    / "telecom"
    / "cert"
    / "cert_report.json"
)

AGENT = "openai/accounts/fireworks/models/glm-5p2"
AGENT_ARGS = '{"api_base":"http://127.0.0.1:4114/v1","api_key":"dummy","reasoning_effort":"high"}'
USER = "user_simulator"
USER_LLM = "gpt-5.4"
EFFORT = "high"
NUM_TRIALS = 3
CONC = 8
MAX_STEPS = 120
TIMEOUT = 600

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
        sc = float(c.get("score", c.get("exploration_score", 0.0)))
        if sc > best_sc:
            best, best_sc = c, sc
    if best is None:
        sys.exit("ERROR: no non-empty candidate in candidates.json")
    return best.get("persona_id", "best"), best["persona_text"]


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


def analyze(rp: Path) -> dict:
    data = json.loads(rp.read_text())
    sims = data.get("simulations", [])
    rewards, infra = [], 0
    transfer = 0
    term_counter = Counter()
    per_task: dict[str, list[float]] = {}
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        term_counter[s.get("termination_reason", "")] += 1
        ms_str = json.dumps(s.get("messages") or [], default=str)
        if "transfer_to_human_agents" in ms_str:
            transfer += 1
        if r is None:
            infra += 1
            continue
        rewards.append(float(r))
        per_task.setdefault(s["task_id"], []).append(float(r))
    n = len(rewards)
    avg = sum(rewards) / n if n else 0.0
    task_means = [sum(v) / len(v) for v in per_task.values()]
    task_avg = sum(task_means) / len(task_means) if task_means else 0.0
    n_sims = len(sims)
    return {
        "n_sims": n_sims,
        "n_rewarded": n,
        "n_infra_none": infra,
        "avg_reward_per_sim": round(avg, 4),
        "avg_reward_per_task": round(task_avg, 4),
        "n_tasks": len(per_task),
        "transfer_to_human_count": transfer,
        "transfer_to_human_rate": round(transfer / n_sims, 4) if n_sims else 0.0,
        "termination_reasons": dict(term_counter),
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

    np_cmd = build_cmd(tids, "cert_telecom_glm52/no_persona/telecom", None)
    np_rp = run(np_cmd, "NO-PERSONA baseline")
    np_m = analyze(np_rp)
    print(f"[NO-PERSONA] {np_m}", flush=True)

    bp_cmd = build_cmd(tids, "cert_telecom_glm52/best/telecom", ptext)
    bp_rp = run(bp_cmd, "BEST persona")
    bp_m = analyze(bp_rp)
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
    print(
        f"transfer rate: no_persona={np_m['transfer_to_human_rate']}  best={bp_m['transfer_to_human_rate']}"
    )


if __name__ == "__main__":
    main()
