#!/usr/bin/env python3
"""Test-set certification for enriched 3-seed GEPA runs.

Evaluates seed + val-top2 personas on the test split (54 tasks × 3 trials).
Resume-safe: skips any persona whose results.json already has the expected
number of sims (n_tasks × num_trials).

    .venv/bin/python experiments/persona_search/scripts/run_test_enriched_3seeds.py
    .venv/bin/python experiments/persona_search/scripts/run_test_enriched_3seeds.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SIMULATIONS_DIR  # noqa: E402, DATA_ROOT
from eval_persona import _run_domain_with_retry  # noqa: E402

SEEDS = {
    "00": str(
        DATA_ROOT / "data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md"
    ),
    "01": str(
        DATA_ROOT / "data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md"
    ),
    "02": str(
        DATA_ROOT
        / "data/persona_opt/seeds_brinks/brinks_02_front_loading_terse/gen_00.md"
    ),
    "03": str(
        DATA_ROOT / "data/persona_opt/seeds_brinks/brinks_03_verbose_polite/gen_00.md"
    ),
    "04": str(
        DATA_ROOT
        / "data/persona_opt/seeds_brinks/brinks_04_confused_drip_feeding/gen_00.md"
    ),
}

STATES = {
    "00": str(
        DATA_ROOT
        / "data/persona_opt/pgepa_runs/enriched_seed0_gpt54user/enriched_brinks_00/pgepa_state.json"
    ),
    "01": str(
        DATA_ROOT
        / "data/persona_opt/pgepa_runs/enriched_seed1_gpt54user/enriched_brinks_01/pgepa_state.json"
    ),
    "02": str(
        DATA_ROOT
        / "data/persona_opt/pgepa_runs/enriched_brinks_02_gpt54user/enriched_brinks_02/pgepa_state.json"
    ),
    "03": str(
        DATA_ROOT
        / "data/persona_opt/pgepa_runs/enriched_brinks_03_gpt54user/enriched_brinks_03/pgepa_state.json"
    ),
    "04": str(
        DATA_ROOT
        / "data/persona_opt/pgepa_runs/enriched_brinks_04_gpt54user/enriched_brinks_04/pgepa_state.json"
    ),
}


def load_test_tasks(split_path: str) -> list[str]:
    return json.loads(Path(split_path).read_text())["test"]["telecom"]


def _persona_title(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("# Persona:"):
            return ln.split("# Persona:")[1].split("(")[0].strip()
    return "?"


def build_eval_plan(seed_num: str, top_k: int = 2) -> list[dict]:
    """Return list of {label, persona_text, source, val_agg} for seed + val top-k."""
    seed_text = Path(SEEDS[seed_num]).read_text()
    state = json.loads(Path(STATES[seed_num]).read_text())
    cands = state.get("candidates", [])

    ranked = sorted(
        [c for c in cands if c.get("agg") is not None and c.get("feasible", True)],
        key=lambda c: c["agg"],
        reverse=True,
    )

    plan = []
    seen_idx = set()

    # Always include seed (idx=0)
    seed_cand = next((c for c in cands if c["idx"] == 0), None)
    plan.append(
        {
            "label": f"brinks_{seed_num}_seed",
            "persona_text": seed_text,
            "source": "seed",
            "val_agg": seed_cand["agg"] if seed_cand else None,
            "idx": 0,
        }
    )
    seen_idx.add(0)

    # Top-k by val agg (skip seed if already included)
    for c in ranked:
        if c["idx"] in seen_idx:
            continue
        if len(seen_idx) >= top_k + 1:  # seed + top_k
            break
        plan.append(
            {
                "label": f"brinks_{seed_num}_idx{c['idx']}",
                "persona_text": c["persona_text"],
                "source": f"optimized (gen={c.get('gen')}, val_agg={c['agg']:.3f})",
                "val_agg": c["agg"],
                "idx": c["idx"],
            }
        )
        seen_idx.add(c["idx"])

    return plan


def is_complete(out_dir: Path, n_expected: int) -> bool:
    rp = out_dir / "results.json"
    if not rp.exists():
        return False
    data = json.loads(rp.read_text())
    sims = data.get("simulations", [])
    good = [s for s in sims if (s.get("reward_info") or {}).get("reward") is not None]
    return len(good) >= n_expected


def score_results(results_path: Path) -> dict:
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])

    by_task: dict[str, list[float]] = defaultdict(list)
    for s in sims:
        tid = s.get("task_id", "")
        reward = (s.get("reward_info") or {}).get("reward")
        if reward is not None:
            by_task[tid].append(float(reward))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0

    avg_scores, pass_scores, pass3_scores, fail3_scores = [], [], [], []
    for tid, rewards in by_task.items():
        avg_scores.append(_mean(rewards))
        pass_scores.append(1.0 if any(r > 0 for r in rewards) else 0.0)
        pass3_scores.append(1.0 if all(r > 0 for r in rewards) else 0.0)
        fail3_scores.append(1.0 if all(r == 0 for r in rewards) else 0.0)

    return {
        "n_tasks": len(by_task),
        "n_sims": len(sims),
        "avg_at_3": _mean(avg_scores),
        "pass_at_3": _mean(pass_scores),
        "pass_cubed": _mean(pass3_scores),
        "fail_cubed": _mean(fail3_scores),
        "difficulty_avg": 1.0 - _mean(avg_scores),
        "difficulty_pass_cubed": 1.0 - _mean(pass3_scores),
    }


def make_eval_args(persona_text: str, num_trials: int, args) -> SimpleNamespace:
    return SimpleNamespace(
        user_type=args.user_type,
        user_llm=args.user_llm,
        effort=args.effort,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        num_trials=num_trials,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        sim_timeout=args.sim_timeout,
        max_infra_retries=args.max_infra_retries,
        use_bedrock=False,
        persona_text=persona_text,
        retrieval_config="bm25",
    )


def main():
    ap = argparse.ArgumentParser(description="Test-set cert for enriched 3-seed GEPA")
    ap.add_argument(
        "--split",
        default="experiments/persona_search/configs/telecom_island_split_fail2.json",
    )
    ap.add_argument("--out-base", default="gepa_opt/test_enriched_3seeds")
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="number of val-top personas per seed (in addition to seed itself)",
    )
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-concurrency", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--seeds",
        nargs="+",
        default=["02", "03", "04"],
        choices=["00", "01", "02", "03", "04"],
    )
    args = ap.parse_args()

    test_tasks = load_test_tasks(args.split)
    n_expected = len(test_tasks) * args.num_trials
    print(
        f"Test tasks: {len(test_tasks)}, trials: {args.num_trials}, "
        f"expected sims per persona: {n_expected}"
    )

    all_scores: dict[str, dict] = {}
    all_plan: list[dict] = []

    for seed_num in args.seeds:
        plan = build_eval_plan(seed_num, top_k=args.top_k)
        all_plan.extend(plan)
        print(f"\n{'=' * 60}")
        print(f"  brinks_{seed_num}: {len(plan)} personas to evaluate")
        for p in plan:
            title = _persona_title(p["persona_text"])
            print(
                f'    {p["label"]}: "{title}" ({p["source"]}, val_agg={p["val_agg"]})'
            )

    print(f"\n{'=' * 60}")
    print(
        f"Total: {len(all_plan)} personas × {len(test_tasks)} tasks × "
        f"{args.num_trials} trials = {len(all_plan) * n_expected} sims"
    )

    if args.dry_run:
        print("\n[dry-run] Would run the above. Exiting.")
        return

    for p in all_plan:
        label = p["label"]
        out_dir = SIMULATIONS_DIR / args.out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_complete(out_dir, n_expected):
            rp = out_dir / "results.json"
            scores = score_results(rp)
            print(f"\n[{label}] SKIP (already complete: {scores['n_sims']} sims)")
            print(
                f"  avg@3={scores['avg_at_3']:.3f}  pass@3={scores['pass_at_3']:.3f}  "
                f"pass^3={scores['pass_cubed']:.3f}  fail^3={scores['fail_cubed']:.3f}"
            )
            all_scores[label] = {
                **scores,
                "val_agg": p["val_agg"],
                "source": p["source"],
            }
            continue

        save_to = f"{args.out_base}/{label}"
        eval_args = make_eval_args(p["persona_text"], args.num_trials, args)
        title = _persona_title(p["persona_text"])
        print(
            f'\n[{label}] Running "{title}" — {args.num_trials} trials × '
            f"{len(test_tasks)} tasks ..."
        )
        _run_domain_with_retry(eval_args, "telecom", test_tasks, save_to)

        rp = out_dir / "results.json"
        if rp.exists():
            scores = score_results(rp)
            print(
                f"  [{label}] avg@3={scores['avg_at_3']:.3f}  "
                f"pass@3={scores['pass_at_3']:.3f}  "
                f"pass^3={scores['pass_cubed']:.3f}  "
                f"fail^3={scores['fail_cubed']:.3f}  "
                f"({scores['n_sims']} sims / {scores['n_tasks']} tasks)"
            )
            all_scores[label] = {
                **scores,
                "val_agg": p["val_agg"],
                "source": p["source"],
            }

    # Summary table
    print(f"\n{'=' * 80}")
    print(
        f"{'Persona':<32s} {'src':>5s} {'val':>5s} {'avg@3':>6s} {'p@3':>5s} "
        f"{'p^3':>5s} {'f^3':>5s} {'d_avg':>6s}"
    )
    print("-" * 80)
    for name, sc in all_scores.items():
        src = "seed" if "seed" in sc.get("source", "") else "opt"
        val = f"{sc['val_agg']:.3f}" if sc.get("val_agg") is not None else "?"
        print(
            f"{name:<32s} {src:>5s} {val:>5s} {sc['avg_at_3']:6.3f} "
            f"{sc['pass_at_3']:5.3f} {sc['pass_cubed']:5.3f} "
            f"{sc['fail_cubed']:5.3f} {sc['difficulty_avg']:6.3f}"
        )

    summary_path = SIMULATIONS_DIR / args.out_base / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
