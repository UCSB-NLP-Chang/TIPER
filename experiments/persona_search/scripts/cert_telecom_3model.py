#!/usr/bin/env python3
"""Telecom 认证：no-persona vs best persona (参数化 agent model)。

cert_telecom_glm52.py 的参数化版：agent model / candidates.json / 输出路径都
可配，用于 3 个新 Fireworks model。口径与 GLM 版一致：3 trials, conc=8, test
split=54, 额外报转人工率。

agent = openai/<canonical> (high) 经 litellm openai provider -> :4114 shim ->
fireworks /v1/messages; user = gpt-5.4 (high) 经标准 OpenAI。

Usage:
  python cert_telecom_3model.py --agent-llm openai/gpt-oss-120b \
      --candidates data/persona_opt/pgepa_runs/gpt-oss-120b_4domain/telecom/isl_gpt-oss-120b_telecom_g0/candidates.json \
      --out-dir data/persona_opt/pgepa_runs/gpt-oss-120b_4domain/telecom/cert
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
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
TAU2 = REPO / ".venv" / "bin" / "tau2"
SPLIT = REPO / "experiments/persona_search/configs/telecom_island_split.json"

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def best_persona_text(cand_path: Path) -> tuple[str, str]:
    cands = json.loads(cand_path.read_text())
    best, best_sc = None, -1.0
    for c in cands:
        t = (c.get("persona_text") or "").strip()
        if len(t) < 50:
            continue
        sc = float(c.get("score", c.get("exploration_score", 0.0)))
        if sc > best_sc:
            best, best_sc = c, sc
    if best is None:
        sys.exit(f"ERROR: no non-empty candidate in {cand_path}")
    return best.get("persona_id", "best"), best["persona_text"]


def test_task_ids() -> list[str]:
    sp = json.loads(SPLIT.read_text())
    return sp["test"]["telecom"]


def build_cmd(
    task_ids,
    run_name,
    persona_text,
    agent_llm,
    agent_args,
    user_llm,
    effort,
    num_trials,
    conc,
    max_steps,
    timeout,
    user_llm_extra_args=None,
):
    user_args = {"reasoning_effort": effort}
    if user_llm_extra_args:
        extra = (
            json.loads(user_llm_extra_args)
            if isinstance(user_llm_extra_args, str)
            else user_llm_extra_args
        )
        user_args.update(extra)
    cmd = [
        str(TAU2),
        "run",
        "--domain",
        "telecom",
        "--user",
        "user_simulator",
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_args),
        "--agent-llm",
        agent_llm,
        "--agent-llm-args",
        agent_args,
        "--num-trials",
        str(num_trials),
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
        f"\n{'=' * 70}\n[{label}] running {len(cmd) - 2} tasks x3 conc=8\n{'=' * 70}",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-llm", required=True, help="openai/<canonical>")
    ap.add_argument(
        "--agent-llm-args",
        default='{"api_base":"http://127.0.0.1:4114/v1","api_key":"dummy","reasoning_effort":"high"}',
    )
    ap.add_argument("--candidates", required=True, help="candidates.json from optim")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument(
        "--user-llm-args",
        default=None,
        help="JSON dict extra kwargs for user sim LLM (e.g. api_base for fireworks models via :4114)",
    )
    ap.add_argument("--effort", default="high")
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--save-prefix",
        default=None,
        help=str(
            DATA_ROOT
            / "data/simulations/<prefix>/...; default headroom_3model/<model>/cert"
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.agent_llm.replace("openai/", "").replace("/", "_")
    save_prefix = args.save_prefix or f"cert_3model/{model_tag}/telecom"

    tids = test_task_ids()
    pid, ptext = best_persona_text(Path(args.candidates))
    print(
        f"best persona: id={pid} chars={len(ptext)} test_tasks={len(tids)} agent={args.agent_llm}",
        flush=True,
    )

    np_cmd = build_cmd(
        tids,
        f"{save_prefix}/no_persona/telecom",
        None,
        args.agent_llm,
        args.agent_llm_args,
        args.user_llm,
        args.effort,
        args.num_trials,
        args.conc,
        args.max_steps,
        args.timeout,
        args.user_llm_args,
    )
    np_rp = run(np_cmd, "NO-PERSONA baseline")
    np_m = analyze(np_rp)
    print(f"[NO-PERSONA] {np_m}", flush=True)

    bp_cmd = build_cmd(
        tids,
        f"{save_prefix}/best/telecom",
        ptext,
        args.agent_llm,
        args.agent_llm_args,
        args.user_llm,
        args.effort,
        args.num_trials,
        args.conc,
        args.max_steps,
        args.timeout,
        args.user_llm_args,
    )
    bp_rp = run(bp_cmd, "BEST persona")
    bp_m = analyze(bp_rp)
    print(f"[BEST] {bp_m}", flush=True)

    delta = round(np_m["avg_reward_per_task"] - bp_m["avg_reward_per_task"], 4)
    report = {
        "domain": "telecom",
        "agent": args.agent_llm,
        "agent_args": args.agent_llm_args,
        "user_llm": args.user_llm,
        "effort": args.effort,
        "num_trials": args.num_trials,
        "concurrency": args.conc,
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
    rp = out_dir / "cert_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nCERT REPORT -> {rp}")
    print(
        f"Δ (no_persona - best) = {delta}  ({'persona HARDER' if delta > 0 else 'no effect / easier'})"
    )
    print(
        f"transfer rate: no_persona={np_m['transfer_to_human_rate']}  best={bp_m['transfer_to_human_rate']}"
    )


if __name__ == "__main__":
    main()
