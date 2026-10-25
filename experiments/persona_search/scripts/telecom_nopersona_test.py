#!/usr/bin/env python3
"""No-persona baseline test on telecom test split (headroom check).

Parametrized sister of cert_telecom_glm52.py: runs ONLY the no-persona
condition (no --user-persona) on the telecom test split, @3 trials, and
reports per-task avg reward + transfer-to-human rate. Used to measure a new
agent model's baseline BEFORE optim.

Agent is a fireworks model: --agent-llm openai/<canonical> + --agent-llm-args
{api_base:4114, api_key:dummy, reasoning_effort:high} -> litellm openai
provider -> :4114 shim -> fireworks /v1/messages. user = gpt-5.4 (standard OpenAI).

Usage:
  python telecom_nopersona_test.py --agent-llm openai/gpt-oss-120b \
      --out-dir data/persona_opt/headroom_3model/gpt-oss-120b/telecom
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
SPLIT = REPO / "data" / "persona_opt" / "telecom_island_split.json"


def test_task_ids() -> list[str]:
    sp = json.loads(SPLIT.read_text())
    return sp["test"]["telecom"]


def build_cmd(
    task_ids,
    run_name,
    agent_llm,
    agent_args,
    user_llm,
    effort,
    num_trials,
    conc,
    max_steps,
    timeout,
):
    user_args = {"reasoning_effort": effort}
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
        "--task-ids",
        *task_ids,
    ]
    return cmd


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
    ap.add_argument(
        "--agent-llm", required=True, help="openai/<canonical> e.g. openai/gpt-oss-120b"
    )
    ap.add_argument(
        "--agent-llm-args",
        default='{"api_base":"http://127.0.0.1:4114/v1","api_key":"dummy","reasoning_effort":"high"}',
    )
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--save-to",
        default=None,
        help=str(
            DATA_ROOT
            / "data/simulations/<name>; default headroom_3model/<model>/telecom_no_persona"
        ),
    )
    args = ap.parse_args()

    model_tag = args.agent_llm.replace("openai/", "").replace("/", "_")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (REPO / "data" / "persona_opt" / "headroom_3model" / model_tag / "telecom")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    save_to = args.save_to or f"headroom_3model/{model_tag}/telecom_no_persona/telecom"
    tids = test_task_ids()
    print(
        f"telecom no-persona: {len(tids)} test tasks x{args.num_trials} conc={args.conc} "
        f"agent={args.agent_llm}/{args.effort} user={args.user_llm}/{args.effort}",
        flush=True,
    )

    cmd = build_cmd(
        tids,
        save_to,
        args.agent_llm,
        args.agent_llm_args,
        args.user_llm,
        args.effort,
        args.num_trials,
        args.conc,
        args.max_steps,
        args.timeout,
    )
    print(
        f"\n{'=' * 70}\n[NO-PERSONA baseline] running {len(tids)} tasks x{args.num_trials}\n{'=' * 70}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(REPO))
    rp = REPO / "data" / "simulations" / save_to / "results.json"
    if not rp.exists():
        sys.exit(f"ERROR: no results.json at {rp}")
    m = analyze(rp)
    print(f"[NO-PERSONA] {m}", flush=True)

    avg = m["avg_reward_per_task"]
    diff = round(1.0 - avg, 4)
    print(
        f"\n  baseline per_task={avg:.3f} difficulty={diff:.3f} transfer_rate={m['transfer_to_human_rate']}  "
        f"({'HIGH CEILING — headroom small' if avg >= 0.90 else 'headroom plausible'})",
        flush=True,
    )

    report = {
        "domain": "telecom",
        "n_test": len(tids),
        "trials": args.num_trials,
        "agent_llm": args.agent_llm,
        "agent_llm_args": args.agent_llm_args,
        "user_llm": args.user_llm,
        "effort": args.effort,
        "concurrency": args.conc,
        "no_persona": m,
        "baseline_avg_reward": avg,
        "baseline_difficulty": diff,
        "headroom_note": ("high_ceiling" if avg >= 0.90 else "plausible"),
    }
    rpf = out_dir / "nopersona_report.json"
    with open(rpf, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report -> {rpf}", flush=True)


if __name__ == "__main__":
    main()
