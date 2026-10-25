#!/usr/bin/env python3
"""Telecom no-persona 验证：gpt-4.1 agent (OpenAI 直连, 非 reasoning 模型)。

目的：对比 gpt-5.6-luna(high) 62% 转人工率，看 gpt-4.1 是否不同。
gpt-4.1 非 reasoning 模型 → agent-llm-args 不传 reasoning_effort，走 chat/completions。
口径同 cert_telecom_gpt56luna.py 的 no-persona phase：test split=54, 3 trials, conc=8。
额外分析转人工率 (transfer_to_human_agents 调用比例)。
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
SPLIT = REPO / "experiments/persona_search/configs/telecom_island_split.json"
SAVE_TO = "cert_telecom_gpt41_no_persona/telecom"
RESULTS = REPO / "data" / "simulations" / SAVE_TO / "results.json"
REPORT = (
    REPO
    / "data"
    / "persona_opt"
    / "pgepa_runs"
    / "gpt41_nopersona_check"
    / "nopersona_report.json"
)

AGENT = "openai/gpt-4.1"
AGENT_ARGS = "{}"  # gpt-4.1 非 reasoning，不传 reasoning_effort
USER = "user_simulator"
USER_LLM = "gpt-5.4"
EFFORT = "high"  # user 仍是 reasoning model
NUM_TRIALS = 3
CONC = 8
MAX_STEPS = 120
TIMEOUT = 600


def test_task_ids() -> list[str]:
    sp = json.loads(SPLIT.read_text())
    return sp["test"]["telecom"]


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tids = test_task_ids()
    print(
        f"agent={AGENT} (non-reasoning)  test_tasks={len(tids)} x{NUM_TRIALS} conc={CONC}",
        flush=True,
    )

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
        SAVE_TO,
        "--auto-resume",
        "--max-steps",
        str(MAX_STEPS),
        "--timeout",
        str(TIMEOUT),
        "--task-ids",
        *tids,
    ]
    print(
        f"\nrunning {' '.join(cmd[:6])}... ({len(tids)} tasks x{NUM_TRIALS})",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(REPO))

    if not RESULTS.exists():
        sys.exit(f"ERROR: no results.json at {RESULTS}")
    data = json.loads(RESULTS.read_text())
    sims = data.get("simulations", [])

    rewards, infra = [], 0
    transfer = 0
    term_counter = Counter()
    per_task: dict[str, list[float]] = {}
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        term = s.get("termination_reason", "")
        term_counter[term] += 1
        # 转人工检测：扫 messages 找 transfer_to_human_agents tool call
        ms_str = json.dumps(s.get("messages") or [], default=str)
        is_transfer = "transfer_to_human_agents" in ms_str
        if is_transfer:
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
    transfer_rate = round(transfer / n_sims, 4) if n_sims else 0.0

    report = {
        "domain": "telecom",
        "agent": AGENT,
        "agent_args": AGENT_ARGS,
        "user": USER,
        "user_llm": USER_LLM,
        "user_effort": EFFORT,
        "num_trials": NUM_TRIALS,
        "concurrency": CONC,
        "n_test_tasks": len(tids),
        "n_sims": n_sims,
        "n_rewarded": n,
        "n_infra_none": infra,
        "avg_reward_per_sim": round(avg, 4),
        "avg_reward_per_task": round(task_avg, 4),
        "transfer_to_human_count": transfer,
        "transfer_to_human_rate": transfer_rate,
        "termination_reasons": dict(term_counter),
        "compare": "gpt-5.6-luna(high) no-persona: transfer_rate=0.62, avg_reward_per_task=?",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n{'=' * 70}\nREPORT -> {REPORT}")
    print(json.dumps(report, indent=2))
    print(
        f"\n★ gpt-4.1 转人工率 = {transfer_rate} ({transfer}/{n_sims})  vs  gpt-5.6-luna = 0.62"
    )
    print(f"★ avg_reward_per_task = {round(task_avg, 4)}")


if __name__ == "__main__":
    main()
