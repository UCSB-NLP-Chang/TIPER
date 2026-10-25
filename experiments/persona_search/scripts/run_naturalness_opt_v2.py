#!/usr/bin/env python3
"""Naturalness optimization using ev2v4 best GEPA personas as seeds.

Runs all 5 seeds in parallel (one process per seed). Each seed goes through
the same optimization loop as run_naturalness_opt.py: mutate → minibatch
screen → full val eval → accept if JSD improves and difficulty ≥ floor.

Usage:
  .venv/bin/python experiments/persona_search/scripts/run_naturalness_opt_v2.py
  .venv/bin/python experiments/persona_search/scripts/run_naturalness_opt_v2.py --seeds b00 b03
  .venv/bin/python experiments/persona_search/scripts/run_naturalness_opt_v2.py --test-only --test-trials 3
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
DATA_ROOT = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments/persona_search/scripts"))

EV2V4_ROOT = DATA_ROOT / "data/persona_opt/pgepa_runs/ev2_3islands_v4"
OUT_TAG = "naturalness_opt_v2"
SIM_TAG = "naturalness_opt_v2"

SEEDS = {
    "b00": {
        "island": "brinks_00",
        "strategy_name": "Temporary-Test Snapback",
        "strategy_summary": "Cooperates with troubleshooting but treats steps as short-lived tests, snapping back to outdated assumptions. Freezes expectations from before the fix.",
        "protected_keywords": ["snapback", "temporary", "test", "freeze", "outdated"],
    },
    "b02": {
        "island": "brinks_02",
        "strategy_name": "Script-Weary Specialist-Seeker",
        "strategy_summary": "Dismisses standard troubleshooting as already-tried scripts, demands escalation or specialist. Misroutes after partial fixes.",
        "protected_keywords": [
            "script-weary",
            "escalation",
            "specialist",
            "already tried",
            "misroute",
        ],
    },
    "b03": {
        "island": "brinks_03",
        "strategy_name": "Baseline Restorer",
        "strategy_summary": "Wants the problem fixed without losing intentional custom setup. Willing to run tests but resists permanent changes to configuration baseline.",
        "protected_keywords": [
            "baseline",
            "custom setup",
            "restore",
            "intentional",
            "permanent",
        ],
    },
    "b04": {
        "island": "brinks_04",
        "strategy_name": "Meaning-First Translator",
        "strategy_summary": "Translates technical details into meaning/impact, dropping the qualifier that matters. Agent gets the gist but misses the critical detail.",
        "protected_keywords": [
            "meaning-first",
            "translate",
            "qualifier",
            "drops",
            "impact",
        ],
    },
    "p03": {
        "island": "p0003",
        "strategy_name": "Proxy Clears the Category",
        "strategy_summary": "Judges a whole area by the single concrete example they checked. One working line = 'calls are fine'. Performs checks but draws overly broad conclusions.",
        "protected_keywords": [
            "proxy",
            "category",
            "concrete example",
            "overly broad",
            "one working",
        ],
    },
}


def load_best_persona(island: str) -> str:
    state_file = EV2V4_ROOT / f"ev2v4_{island}" / "pgepa_state.json"
    state = json.loads(state_file.read_text())
    feasible = sorted(
        [c for c in state["candidates"] if c.get("feasible", True)],
        key=lambda c: c.get("agg", 0),
        reverse=True,
    )
    return feasible[0]["persona_text"]


def _run_one_seed(seed_key: str, args_dict: dict) -> dict:
    """Run optimization for one seed (called in subprocess)."""
    import argparse as _ap

    args = _ap.Namespace(**args_dict)

    import numpy as np
    from naturalness_reference import (
        load_human_ref,
        load_human_vecs,
        sample_human_conversations,
    )
    from run_naturalness_opt import (
        build_mutation_prompt,
        call_mutation_llm,
        run_eval,
        score_results,
    )

    cfg = SEEDS[seed_key]
    persona_text = load_best_persona(cfg["island"])

    human_ref = load_human_ref()
    human_vecs = load_human_vecs()
    human_samples = sample_human_conversations(n=3)

    out_dir = DATA_ROOT / "data" / "persona_opt" / OUT_TAG / seed_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{seed_key}] START — {cfg['strategy_name']} ({len(persona_text)} chars)")

    # Eval seed on val (skip if results already exist)
    seed_val_path = (
        DATA_ROOT
        / "data"
        / "simulations"
        / SIM_TAG
        / seed_key
        / "seed_val"
        / "results.json"
    )
    if seed_val_path.exists():
        print(f"[{seed_key}] Reusing existing seed_val results")
        seed_results = json.loads(seed_val_path.read_text())
    else:
        print(f"[{seed_key}] Evaluating seed on val set...")
        seed_results = run_eval(
            persona_text,
            args.val_tasks,
            save_to=f"{SIM_TAG}/{seed_key}/seed_val",
            num_trials=args.num_trials,
            max_concurrency=args.concurrency,
        )
    if not seed_results:
        return {"seed_key": seed_key, "status": "seed_eval_failed"}

    seed_scores = score_results(seed_results, human_vecs, human_ref)
    difficulty_floor = max(seed_scores["difficulty"] - args.difficulty_tolerance, 0)
    print(
        f"[{seed_key}] Seed: diff={seed_scores['difficulty']:.3f}, "
        f"JSD={seed_scores['jsd']:.4f}, floor={difficulty_floor:.3f}"
    )

    best_persona = persona_text
    best_jsd = seed_scores["jsd"]
    best_difficulty = seed_scores["difficulty"]
    best_profile = seed_scores["profile"]
    best_results = seed_results
    history = [{"iter": 0, "type": "seed", **seed_scores}]

    # Resume from previous run if history exists
    start_iter = 1
    history_path = out_dir / "history.json"
    best_path = out_dir / "best_persona.json"
    if history_path.exists():
        prev_history = json.loads(history_path.read_text())
        start_iter = max(e.get("iter", 0) for e in prev_history) + 1
        history = prev_history
        if best_path.exists():
            best_data = json.loads(best_path.read_text())
            best_persona = best_data["persona_text"]
            best_jsd = best_data.get("jsd", best_jsd)
            best_difficulty = best_data.get("difficulty", best_difficulty)
            # Re-score to get profile and results for traces
            best_val_dir = DATA_ROOT / "data" / "simulations" / SIM_TAG / seed_key
            # Find the latest accepted val results
            for e in reversed(prev_history):
                if e.get("type") == "accepted":
                    val_path = (
                        best_val_dir / f"iter{e['iter']:02d}_val" / "results.json"
                    )
                    if val_path.exists():
                        best_results = json.loads(val_path.read_text())
                        best_scores = score_results(best_results, human_vecs, human_ref)
                        best_profile = best_scores["profile"]
                        break
        print(f"[{seed_key}] Resuming from iter {start_iter} (best_jsd={best_jsd:.4f})")

    rng = np.random.default_rng(args.rng_seed)
    # Advance rng to match where we left off
    for _ in range(start_iter - 1):
        rng.choice(
            args.train_tasks, min(args.minibatch, len(args.train_tasks)), replace=False
        )

    from run_naturalness_opt import get_full_traces

    for iteration in range(start_iter, args.max_iters + 1):
        print(f"[{seed_key}] --- iter {iteration}/{args.max_iters} ---")

        mb_size = min(args.minibatch, len(args.train_tasks))
        mb_tasks = list(rng.choice(args.train_tasks, mb_size, replace=False))

        traces = get_full_traces(best_results, n=5)
        prompt = build_mutation_prompt(
            persona_text=best_persona,
            difficulty=best_difficulty,
            jsd=best_jsd,
            profile=best_profile,
            human_ref=human_ref,
            traces=traces,
            human_samples=human_samples,
            difficulty_floor=difficulty_floor,
            cfg=cfg,
        )

        mutation = call_mutation_llm(prompt, model=args.mutation_model)
        if not mutation:
            print(f"[{seed_key}]   mutation failed")
            history.append({"iter": iteration, "type": "mutation_failed"})
            continue

        new_persona = mutation["persona_text"]
        direction = mutation.get("mutation_direction", "")[:100]
        print(f"[{seed_key}]   mutation: {direction}")

        # Minibatch screen
        mb_results = run_eval(
            new_persona,
            mb_tasks,
            save_to=f"{SIM_TAG}/{seed_key}/iter{iteration:02d}_mb",
            num_trials=args.num_trials,
            max_concurrency=args.concurrency,
        )
        if not mb_results:
            history.append({"iter": iteration, "type": "mb_eval_failed"})
            continue

        mb_scores = score_results(mb_results, human_vecs, human_ref)
        if mb_scores["difficulty"] < difficulty_floor:
            print(
                f"[{seed_key}]   SCREENED: mb diff={mb_scores['difficulty']:.3f} < floor={difficulty_floor:.3f}"
            )
            history.append({"iter": iteration, "type": "screened", **mb_scores})
            continue

        # Full val eval
        val_results = run_eval(
            new_persona,
            args.val_tasks,
            save_to=f"{SIM_TAG}/{seed_key}/iter{iteration:02d}_val",
            num_trials=args.num_trials,
            max_concurrency=args.concurrency,
        )
        if not val_results:
            history.append({"iter": iteration, "type": "val_eval_failed"})
            continue

        val_scores = score_results(val_results, human_vecs, human_ref)
        print(
            f"[{seed_key}]   val: diff={val_scores['difficulty']:.3f}, "
            f"JSD={val_scores['jsd']:.4f} (best={best_jsd:.4f})"
        )

        if val_scores["difficulty"] < difficulty_floor:
            print(f"[{seed_key}]   REJECTED: val diff < floor")
            history.append(
                {"iter": iteration, "type": "rejected_difficulty", **val_scores}
            )
            continue

        if val_scores["jsd"] < best_jsd:
            print(
                f"[{seed_key}]   ACCEPTED: JSD {best_jsd:.4f} -> {val_scores['jsd']:.4f}"
            )
            best_persona = new_persona
            best_jsd = val_scores["jsd"]
            best_difficulty = val_scores["difficulty"]
            best_profile = val_scores["profile"]
            best_results = val_results

            best_out = {
                "persona_text": best_persona,
                "mutation_direction": direction,
                "jsd": best_jsd,
                "difficulty": best_difficulty,
                "iteration": iteration,
            }
            (out_dir / "best_persona.json").write_text(
                json.dumps(best_out, indent=2, ensure_ascii=False)
            )
            history.append({"iter": iteration, "type": "accepted", **val_scores})
        else:
            print(f"[{seed_key}]   REJECTED: JSD not improved")
            history.append({"iter": iteration, "type": "rejected_jsd", **val_scores})

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not (out_dir / "best_persona.json").exists():
        best_out = {
            "persona_text": best_persona,
            "mutation_direction": "seed (no improvement found)",
            "jsd": best_jsd,
            "difficulty": best_difficulty,
        }
        (out_dir / "best_persona.json").write_text(
            json.dumps(best_out, indent=2, ensure_ascii=False)
        )

    result = {
        "seed_key": seed_key,
        "seed_jsd": seed_scores["jsd"],
        "best_jsd": best_jsd,
        "best_difficulty": best_difficulty,
        "iterations": len(history) - 1,
        "accepted": sum(1 for h in history if h.get("type") == "accepted"),
    }
    print(
        f"[{seed_key}] DONE: JSD {seed_scores['jsd']:.4f} -> {best_jsd:.4f} "
        f"(Δ={best_jsd - seed_scores['jsd']:.4f}), accepted={result['accepted']}/{result['iterations']}"
    )
    return result


def _run_one_test(seed_key: str, args_dict: dict) -> dict:
    """Run test eval for one seed (called in subprocess)."""
    import argparse as _ap

    args = _ap.Namespace(**args_dict)

    from naturalness_reference import load_human_ref, load_human_vecs
    from run_naturalness_opt import run_eval, score_results

    cfg = SEEDS[seed_key]
    human_ref = load_human_ref()
    human_vecs = load_human_vecs()

    out_dir = DATA_ROOT / "data" / "persona_opt" / OUT_TAG / seed_key
    best_path = out_dir / "best_persona.json"

    results = {}

    # Test optimized persona
    if best_path.exists():
        best = json.loads(best_path.read_text())
        persona_text = best["persona_text"]
        print(f"[{seed_key}] Testing optimized (JSD={best.get('jsd', '?')})")
    else:
        persona_text = load_best_persona(cfg["island"])
        print(f"[{seed_key}] No optimized found, testing GEPA seed")

    r = run_eval(
        persona_text,
        args.test_tasks,
        save_to=f"{SIM_TAG}/{seed_key}/test",
        num_trials=args.test_trials,
        max_concurrency=args.concurrency,
    )
    if r:
        results["opt"] = score_results(r, human_vecs, human_ref)
        print(
            f"[{seed_key}] opt: diff={results['opt']['difficulty']:.3f}, JSD={results['opt']['jsd']:.4f}"
        )

    # Test GEPA seed for comparison
    seed_text = load_best_persona(cfg["island"])
    r = run_eval(
        seed_text,
        args.test_tasks,
        save_to=f"{SIM_TAG}/{seed_key}/test_seed",
        num_trials=args.test_trials,
        max_concurrency=args.concurrency,
    )
    if r:
        results["seed"] = score_results(r, human_vecs, human_ref)
        print(
            f"[{seed_key}] seed: diff={results['seed']['difficulty']:.3f}, JSD={results['seed']['jsd']:.4f}"
        )

    return {"seed_key": seed_key, **results}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seeds", nargs="+", default=list(SEEDS.keys()), choices=list(SEEDS.keys())
    )
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--difficulty-tolerance", type=float, default=0.05)
    ap.add_argument("--mutation-model", default="gpt-5.4")
    ap.add_argument("--seed", type=int, default=42, dest="rng_seed")
    ap.add_argument(
        "--split",
        default="experiments/persona_search/configs/telecom_island_split_fail2.json",
    )
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--test-trials", type=int, default=3)
    args = ap.parse_args()

    split = json.loads(
        (
            Path(args.split)
            if Path(args.split).is_absolute()
            else REPO_ROOT / args.split
        ).read_text()
    )
    train_tasks = split["train"]["telecom"]
    val_tasks = split["val"]["telecom"]
    test_tasks = split["test"]["telecom"]
    print(
        f"Split: train={len(train_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}"
    )
    print(f"Seeds: {args.seeds}")
    print(f"Concurrency per seed: {args.concurrency}")

    # Verify all state files exist
    for key in args.seeds:
        island = SEEDS[key]["island"]
        state_file = EV2V4_ROOT / f"ev2v4_{island}" / "pgepa_state.json"
        assert state_file.exists(), f"Missing: {state_file}"
        text = load_best_persona(island)
        print(f"  {key} ({island}): top1 persona {len(text)} chars")

    args_dict = {
        "num_trials": args.num_trials,
        "concurrency": args.concurrency,
        "max_iters": args.max_iters,
        "minibatch": args.minibatch,
        "difficulty_tolerance": args.difficulty_tolerance,
        "mutation_model": args.mutation_model,
        "rng_seed": args.rng_seed,
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "test_tasks": test_tasks,
        "test_trials": args.test_trials,
    }

    if args.test_only:
        print(
            f"\nTEST MODE: {len(args.seeds)} seeds × {args.test_trials} trials × {len(test_tasks)} tasks"
        )
        with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
            futures = {pool.submit(_run_one_test, k, args_dict): k for k in args.seeds}
            results = []
            for f in as_completed(futures):
                results.append(f.result())

        print(f"\n{'=' * 70}")
        print(
            f"{'Persona':<8} {'Seed Diff':>10} {'Seed JSD':>10} {'Opt Diff':>10} {'Opt JSD':>10} {'ΔJSD':>8}"
        )
        print("-" * 70)
        for r in sorted(results, key=lambda x: x["seed_key"]):
            s = r.get("seed", {})
            o = r.get("opt", {})
            djsd = o.get("jsd", 0) - s.get("jsd", 0) if s and o else float("nan")
            print(
                f"{r['seed_key']:<8} {s.get('difficulty', 0):>10.3f} {s.get('jsd', 0):>10.4f} "
                f"{o.get('difficulty', 0):>10.3f} {o.get('jsd', 0):>10.4f} {djsd:>+8.4f}"
            )
        return

    # Optimization: run all seeds in parallel
    print(
        f"\nOPTIMIZATION: {len(args.seeds)} seeds × {args.max_iters} iters (parallel)"
    )
    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        futures = {pool.submit(_run_one_seed, k, args_dict): k for k in args.seeds}
        all_results = []
        for f in as_completed(futures):
            all_results.append(f.result())

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    for r in sorted(all_results, key=lambda x: x.get("seed_key", "")):
        if "seed_jsd" in r:
            print(
                f"  {r['seed_key']}: JSD {r['seed_jsd']:.4f} -> {r['best_jsd']:.4f} "
                f"(Δ={r['best_jsd'] - r['seed_jsd']:.4f}), "
                f"accepted={r['accepted']}/{r['iterations']}"
            )
        else:
            print(f"  {r['seed_key']}: {r.get('status', 'unknown')}")

    summary_path = DATA_ROOT / "data" / "persona_opt" / OUT_TAG / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
