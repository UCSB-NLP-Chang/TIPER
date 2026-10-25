#!/usr/bin/env python3
"""Cert: best persona vs no-persona on ColBench code test split (sweet_rl).

agent = minimax-m3 (high, via :4114), human = gpt-5.4 (high). Reuses
build_task_lists + SweetRLPersonaAdapter + evaluate_reward from
gepa_runner_sweetrl so the eval path is identical to the search's test-set
phase. Split seed = 42 (same as search) -> same 50 test tasks.

Usage:
  python cert_sweetrl_test.py --best-persona sweet_rl/data/persona_opt/gepa_runs/minimax_code/best_persona.md \
      --trials 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
DATA_ROOT = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
_SWEET_RL_REPO = _REPO / "vendor" / "sweet_rl"
import sys

_BASELINE_SCRIPTS = _REPO / "experiments" / "baseline" / "scripts"
for _p in (str(_SWEET_RL_REPO), str(_HERE), str(_REPO), str(_BASELINE_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# minimax routing must patch openai.OpenAI before importing the adapter
# (which binds `from openai import OpenAI` at module load).
import minimax_routing  # noqa: E402

minimax_routing.patch_openai()


from gepa_runner_sweetrl import build_task_lists, evaluate_reward  # noqa: E402
from sweetrl_persona_adapter import SweetRLPersonaAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-persona", required=True)
    ap.add_argument(
        "--data",
        default=str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl"),
    )
    ap.add_argument("--agent-model", default="accounts/fireworks/models/minimax-m3")
    ap.add_argument("--agent-effort", default="high")
    ap.add_argument("--human-model", default="gpt-5.4")
    ap.add_argument("--human-effort", default="high")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--agent-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "llm_agent_code_prompt.txt"),
    )
    ap.add_argument(
        "--human-prompt",
        default=str(_SWEET_RL_REPO / "prompts" / "human_simulator_code_prompt.txt"),
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            DATA_ROOT
            / "data/persona_opt/pgepa_runs/minimax_3domain/cert"
            / "colbench_code"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    best_persona = Path(args.best_persona).read_text()
    # split: only test is used; train/val unused but build_task_lists returns all three.
    _train, _val, test = build_task_lists(args.data, 1, 1, args.test_n, args.split_seed)
    print(
        f"ColBench code cert: {len(test)} test tasks, {args.trials} trials, "
        f"agent={args.agent_model}/{args.agent_effort} human={args.human_model}/{args.human_effort}",
        flush=True,
    )

    adapter = SweetRLPersonaAdapter(
        agent_model=args.agent_model,
        agent_effort=args.agent_effort,
        human_model=args.human_model,
        human_effort=args.human_effort,
        agent_prompt_path=args.agent_prompt,
        human_prompt_path=args.human_prompt,
        max_steps=args.max_steps,
        num_trials=args.trials,
        max_concurrency=args.max_concurrency,
        agent_round_budget=args.max_steps,
        # raw-reward measurement: no judge, no gating (mirrors gepa_runner_sweetrl
        # evaluate_reward; scores here must stay comparable to published deltas)
        online_judge=False,
        severe_gate_rate=None,
        timeout_attack_threshold=None,
    )

    print("\n[1/2] no-persona baseline:", flush=True)
    no_p = evaluate_reward(adapter, test, "", "test/no-persona")
    print("\n[2/2] best persona:", flush=True)
    best = evaluate_reward(adapter, test, best_persona, "test/best-persona")

    delta = no_p["avg_reward"] - best["avg_reward"]
    print(
        f"\n  Δ reward (no-persona - best) = {delta:+.3f}  "
        f"({'persona HARDER ✓' if delta > 0 else 'no improvement ✗'})",
        flush=True,
    )

    report = {
        "domain": "colbench_code",
        "n_test": len(test),
        "trials": args.trials,
        "agent_model": args.agent_model,
        "human_model": args.human_model,
        "no_persona": no_p,
        "best_persona": best,
        "delta_reward": delta,
        "best_persona_path": str(args.best_persona),
    }
    rp = out_dir / "cert_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report -> {rp}", flush=True)


if __name__ == "__main__":
    main()
