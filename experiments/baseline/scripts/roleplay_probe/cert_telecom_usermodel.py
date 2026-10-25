#!/usr/bin/env python3
"""Telecom 统一-persona × 换-user-model 认证 (reward-only).

固定 agent = gpt-5.4 (high),给定一个 persona(top-1 全局最难),对指定 user
model 跑 no-persona vs best-persona @N trials on 完整 telecom test split (54).
输出 cert_report.json(no_persona / best_persona / delta_reward).

路由(user 侧,agent 始终 gpt-5.4 标准 OpenAI):
  claude-*  -> litellm anthropic provider, 须设 env ANTHROPIC_API_BASE=:4112
  fireworks -> user-llm-args 带 api_base :4114
  gpt-5.x   -> 标准 OpenAI

用法见 run_unified_persona_usermodel.sh。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
TAU2 = REPO / ".venv" / "bin" / "tau2"
SPLIT = REPO / "data" / "persona_opt" / "telecom_island_split.json"

AGENT = "openai/gpt-5.4"
AGENT_ARGS = '{"reasoning_effort":"high"}'
USER = "user_simulator"

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def test_task_ids() -> list[str]:
    sp = json.loads(SPLIT.read_text())
    return sp["test"]["telecom"]


def build_cmd(
    task_ids,
    run_name,
    persona_text,
    user_llm,
    user_args,
    conc,
    trials,
    max_steps,
    timeout,
):
    cmd = [
        str(TAU2),
        "run",
        "--domain",
        "telecom",
        "--user",
        USER,
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_args),
        "--agent-llm",
        AGENT,
        "--agent-llm-args",
        AGENT_ARGS,
        "--num-trials",
        str(trials),
        "--max-concurrency",
        str(conc),
        "--save-to",
        run_name,
        "--auto-resume",
        "--max-steps",
        str(max_steps),
        "--timeout",
        str(timeout),
    ]
    if persona_text:
        cmd += [
            "--user-persona",
            json.dumps({"custom_guidelines": ROLEPLAY_PREAMBLE + persona_text}),
        ]
    cmd += ["--task-ids", *task_ids]
    return cmd


def run(cmd, label):
    print(
        f"\n{'=' * 70}\n[{label}] {len(cmd)} args, conc={cmd[cmd.index('--max-concurrency') + 1]}\n{'=' * 70}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(REPO))
    save_to = cmd[cmd.index("--save-to") + 1]
    rp = REPO / "data" / "simulations" / save_to / "results.json"
    if not rp.exists():
        sys.exit(f"ERROR: no results.json at {rp}")
    return rp


def avg_reward(rp):
    sims = json.loads(rp.read_text()).get("simulations", [])
    rewards, per_task = [], {}
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        rewards.append(float(r))
        per_task.setdefault(s["task_id"], []).append(float(r))
    n = len(rewards)
    avg = sum(rewards) / n if n else 0.0
    tm = [sum(v) / len(v) for v in per_task.values()]
    return {
        "n_sims": len(sims),
        "n_rewarded": n,
        "avg_reward_per_sim": round(avg, 4),
        "avg_reward_per_task": round(sum(tm) / len(tm), 4) if tm else 0.0,
        "n_tasks": len(per_task),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-llm", required=True)
    ap.add_argument(
        "--user-llm-args",
        default="{}",
        help="JSON string merged over {reasoning_effort:high}",
    )
    ap.add_argument("--persona-file", required=True)
    ap.add_argument(
        "--out-name", required=True, help="save-to prefix, e.g. unified_um/sonnet5"
    )
    ap.add_argument("--report", required=True, help="cert_report.json output path")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    user_args = {"reasoning_effort": "high"}
    user_args.update(json.loads(args.user_llm_args) if args.user_llm_args else {})
    tids = test_task_ids()
    ptext = Path(args.persona_file).read_text().strip()
    print(
        f"telecom unified: user_llm={args.user_llm} user_args={user_args} "
        f"persona_chars={len(ptext)} test_tasks={len(tids)} trials={args.trials}",
        flush=True,
    )

    np_cmd = build_cmd(
        tids,
        f"{args.out_name}/no_persona/telecom",
        None,
        args.user_llm,
        user_args,
        args.conc,
        args.trials,
        args.max_steps,
        args.timeout,
    )
    np_m = avg_reward(run(np_cmd, "NO-PERSONA"))
    print(f"[NO-PERSONA] {np_m}", flush=True)

    bp_cmd = build_cmd(
        tids,
        f"{args.out_name}/best/telecom",
        ptext,
        args.user_llm,
        user_args,
        args.conc,
        args.trials,
        args.max_steps,
        args.timeout,
    )
    bp_m = avg_reward(run(bp_cmd, "BEST persona"))
    print(f"[BEST] {bp_m}", flush=True)

    delta = round(np_m["avg_reward_per_task"] - bp_m["avg_reward_per_task"], 4)
    report = {
        "domain": "telecom",
        "agent": AGENT,
        "agent_args": AGENT_ARGS,
        "user_llm": args.user_llm,
        "user_args": user_args,
        "num_trials": args.trials,
        "concurrency": args.conc,
        "n_test_tasks": len(tids),
        "persona_chars": len(ptext),
        "no_persona": np_m,
        "best_persona": bp_m,
        "delta_reward": delta,
        "interpretation": "delta>0 => persona lowers agent success (harder)",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nCERT REPORT -> {args.report}")
    print(
        f"Δ (no_persona - best) = {delta}  ({'persona HARDER' if delta > 0 else 'no effect / easier'})"
    )


if __name__ == "__main__":
    main()
