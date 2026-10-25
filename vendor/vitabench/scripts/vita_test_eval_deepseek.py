#!/usr/bin/env python3
"""Test-only evaluation: re-run the 4-domain VitaBench test set with a new
agent model, reusing the EXISTING best personas from run_<domain>_v2/best_persona.md
(no GEPA search). Outputs per-domain run_<domain>_v2/test_report_deepseek.json
and a combined deepseek_test_summary.json.

Default agent = deepseek-v4-flash-0731 (via the openai_to_fireworks shim + Cresta
dev-proxy). user simulator / evaluator stay at the original V2 setting
(gpt-5.4 high / gpt-4.1) so only the agent under test changes.

Usage:
    python scripts/vita_test_eval_deepseek.py
    python scripts/vita_test_eval_deepseek.py --domains delivery --test-size 2 --test-trials 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vita_pgepa import split_tasks, to_vitatasks, evaluate_reward  # noqa: E402
from vita_persona_adapter import VitaPersonaAdapter  # noqa: E402

RUN_BASE = Path("data/persona_opt/vita_pgepa_runs")
DIR_KEY = {"cross_domain": "crossdomain"}  # run dir naming


def main():
    ap = argparse.ArgumentParser(description="VitaBench test-only eval with a swapped agent model.")
    ap.add_argument("--agent-llm", default="deepseek-v4-flash-0731")
    ap.add_argument("--agent-llm-args", default="{}")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--user-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--evaluator-llm", default="gpt-4.1")
    ap.add_argument("--evaluator-llm-args", default="{}")
    ap.add_argument("--test-size", type=int, default=40)
    ap.add_argument("--test-trials", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--domains", default="delivery,instore,ota,cross_domain")
    ap.add_argument("--enable-think", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--label-suffix", default="ds",
                    help="conversation dir suffix (test_<label>); avoids overwriting V2 traces.")
    args = ap.parse_args()

    agent_args = json.loads(args.agent_llm_args) if args.agent_llm_args else {}
    user_args = json.loads(args.user_llm_args) if args.user_llm_args else {}
    eval_args = json.loads(args.evaluator_llm_args) if args.evaluator_llm_args else {}
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    suf = args.label_suffix

    print(f"vita_test_eval: agent={args.agent_llm} user={args.user_llm} eval={args.evaluator_llm} "
          f"trials={args.test_trials} test_size={args.test_size} domains={domains}")

    summary = {}
    for d in domains:
        run_dir = RUN_BASE / f"run_{DIR_KEY.get(d, d)}_v2"
        persona_path = run_dir / "best_persona.md"
        if not persona_path.exists():
            print(f"[skip] {d}: no best_persona.md at {persona_path}")
            continue
        best_persona = persona_path.read_text()

        # Same test split as V2 (seed=300, 30/30/40).
        _tr, _va, test_ids = split_tasks(d, 30, 30, args.test_size, args.language, args.seed)
        test_tasks = to_vitatasks(test_ids, d)
        print(f"\n{'=' * 60}\n  DOMAIN {d}  agent={args.agent_llm}  n_test={len(test_tasks)} "
              f"trials={args.test_trials}\n{'=' * 60}")

        adapter = VitaPersonaAdapter(
            persona_id=f"vita_test_{suf}_{d}",
            agent_llm=args.agent_llm, agent_llm_args=agent_args,
            user_llm=args.user_llm, user_llm_args=user_args,
            max_steps=args.max_steps, num_trials=args.test_trials,
            evaluator_llm=args.evaluator_llm, evaluator_llm_args=eval_args,
            enable_think=args.enable_think, language=args.language, domain=d,
            max_concurrency=args.max_concurrency, seed_base=args.seed,
            online_judge=False,
        )
        adapter._num_trials = args.test_trials

        print("  No-persona baseline:")
        base = evaluate_reward(adapter, test_tasks, "", f"no_persona_{suf}", run_dir)
        print("  Best persona:")
        best = evaluate_reward(adapter, test_tasks, best_persona, f"best_persona_{suf}", run_dir)
        delta = base["avg_reward"] - best["avg_reward"]
        print(f"  => no={base['avg_reward']:.3f} best={best['avg_reward']:.3f} "
              f"Δ(no-best)={delta:+.3f}  solved {base['fully_correct']}/{best['fully_correct']}")

        report = {
            "agent_llm": args.agent_llm, "user_llm": args.user_llm,
            "evaluator_llm": args.evaluator_llm, "test_trials": args.test_trials,
            "test_size": args.test_size, "domain": d,
            "no_persona": base, "best_persona": best,
            "delta_reward_no_minus_best": delta,
            "best_persona_path": str(persona_path),
        }
        out = run_dir / f"test_report_{suf}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"  report -> {out}")
        summary[d] = {
            "no": base["avg_reward"], "best": best["avg_reward"], "delta": delta,
            "no_correct": base["fully_correct"], "best_correct": best["fully_correct"],
            "n": base["n"],
        }

    print(f"\n{'=' * 60}\n  SUMMARY (agent={args.agent_llm})\n{'=' * 60}")
    print(f"  {'domain':14s} {'no_reward':>10s} {'best_reward':>11s} {'Δ(no-best)':>11s} "
          f"{'solved no/best':>16s}")
    for d, s in summary.items():
        print(f"  {d:14s} {s['no']:10.3f} {s['best']:11.3f} {s['delta']:+11.3f} "
              f"{s['no_correct']:>5d}/{s['best_correct']:<10d}")
    (RUN_BASE / f"test_summary_{suf}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n  summary -> {RUN_BASE / f'test_summary_{suf}.json'}")


if __name__ == "__main__":
    main()
