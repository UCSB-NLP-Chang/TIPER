#!/usr/bin/env python3
"""No-persona baseline test on ColBench `code` test split (headroom check).

Sister of gepa_runner_sweetrl.py but runs ONLY the no-persona condition on the
TEST split (persona_text="") — no GEPA search. Used to measure a model's
baseline test-case pass rate BEFORE optim, so we can skip ceiling models.

Reuses build_task_lists + evaluate_reward + SweetRLPersonaAdapter so the eval
path is identical to the optim's final test-set eval.

Usage:
  python sweetrl_nopersona_test.py --agent-model gpt-oss-120b --agent-effort high \
      --test-trials 3 --out-dir data/persona_opt/headroom_3model/gpt-oss-120b/colbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SWEET_RL_REPO = Path(__file__).resolve().parents[3] / "vendor" / "sweet_rl"
_REPO_TOP = _SWEET_RL_REPO.parent
for _p in (str(_SWEET_RL_REPO), str(_HERE), str(_REPO_TOP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# patch openai.OpenAI BEFORE importing the adapter (routes fireworks models to :4114)
import minimax_routing  # noqa: E402

minimax_routing.patch_openai()

from gepa_runner_sweetrl import build_task_lists, evaluate_reward  # noqa: E402
from sweetrl_persona_adapter import (  # noqa: E402
    SweetRLPersonaAdapter,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(_SWEET_RL_REPO / "data" / "backend_tasks" / "test_110_hard.jsonl"),
    )
    ap.add_argument("--train-n", type=int, default=30)
    ap.add_argument("--val-n", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--agent-model", default="gpt-5.4")
    ap.add_argument("--agent-effort", default="high")
    ap.add_argument("--human-model", default="gpt-5.4")
    ap.add_argument("--human-effort", default="high")
    ap.add_argument("--test-trials", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--best-persona",
        default=None,
        help="if set, also evaluate this persona file on test split and write test_report.json "
        "(no_persona vs best_persona @test-trials) — used to rerun ONLY the test phase "
        "when the search persona is already saved but the auto test_report is bogus/missing",
    )
    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            _REPO_TOP
            / "data"
            / "persona_opt"
            / "headroom_3model"
            / args.agent_model.replace("/", "_")
            / "colbench"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _train, _val, test = build_task_lists(
        args.data, args.train_n, args.val_n, args.test_n, args.split_seed
    )
    print(
        f"ColBench code no-persona: test={len(test)} tasks x {args.test_trials} trials, "
        f"agent={args.agent_model}/{args.agent_effort} human={args.human_model}/{args.human_effort}",
        flush=True,
    )

    adapter = SweetRLPersonaAdapter(
        agent_model=args.agent_model,
        agent_effort=args.agent_effort,
        human_model=args.human_model,
        human_effort=args.human_effort,
        max_steps=args.max_steps,
        num_trials=args.test_trials,
        max_concurrency=args.max_concurrency,
        agent_round_budget=args.max_steps,
        # raw-reward baseline measurement: no judge, no gating
        online_judge=False,
        severe_gate_rate=None,
        timeout_attack_threshold=None,
    )

    print("\n[no-persona baseline] (persona_text=''):", flush=True)
    base = evaluate_reward(adapter, test, "", "test/no-persona")

    avg = base["avg_reward"]
    diff = 1.0 - avg
    print(
        f"\n  baseline avg_reward={avg:.3f} difficulty={diff:.3f} "
        f"fully_correct={base['fully_correct']}/{len(test)}  "
        f"({'HIGH CEILING — headroom small' if avg >= 0.80 else 'headroom plausible'})",
        flush=True,
    )

    report = {
        "domain": "colbench_code",
        "n_test": len(test),
        "trials": args.test_trials,
        "agent_model": args.agent_model,
        "agent_effort": args.agent_effort,
        "human_model": args.human_model,
        "human_effort": args.human_effort,
        "split_seed": args.split_seed,
        "no_persona": base,
        "baseline_avg_reward": avg,
        "baseline_difficulty": diff,
        "headroom_note": ("high_ceiling" if avg >= 0.80 else "plausible"),
    }
    rp = out_dir / "nopersona_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report -> {rp}", flush=True)

    # optional: also evaluate a saved best persona -> test_report.json (test-only rerun)
    if args.best_persona:
        bp = Path(args.best_persona)
        persona_text = bp.read_text().strip()
        print(f"\n[best-persona] ({bp.name}, {len(persona_text)} chars):", flush=True)
        best = evaluate_reward(adapter, test, persona_text, "test/best-persona")
        test_report = {
            "domain": "colbench_code",
            "n_test": len(test),
            "trials": args.test_trials,
            "agent_model": args.agent_model,
            "agent_effort": args.agent_effort,
            "no_persona": base,
            "best_persona": best,
            "best_persona_path": str(bp),
            "delta_reward": base["avg_reward"] - best["avg_reward"],
        }
        tp = out_dir / "test_report.json"
        with open(tp, "w") as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        print(
            f"  test_report -> {tp}  (nop={base['avg_reward']:.3f} best={best['avg_reward']:.3f} "
            f"Δ={test_report['delta_reward']:+.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
