#!/usr/bin/env python3
"""Run NCUser non-collaborative behavior presets on the VitaBench delivery domain.

Programmatic runner (the VitaBench CLI's ``--user-llm-args`` is ``type=dict`` and
cannot parse nested JSON, so we build RunConfig directly and call run_domain).

Defaults mirror the telecom NCUser test setup:
  agent gpt-5.4 (high, via models.yaml) / user gpt-5.4 (high)
  --language english (so the English behavior rewrite prompts match the base
  utterance; the delivery domain's tool/DB content is Chinese, so an explicit
  English anchor is injected inside NCUserUserSimulator to stop gpt-5.4 drifting
  back to Chinese).
  5 presets (none/emotional/tangential/fragment/unavailable), 3 trials, seed 300.

Usage (run under the VitaBench venv):
  vitabench/.venv/bin/python vitabench/run_ncuser_delivery.py --mode pilot
  vitabench/.venv/bin/python vitabench/run_ncuser_delivery.py --mode full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vita.config import models
from vita.data_model.simulation import RunConfig
from vita.run import run_domain
from vita.utils.utils import DATA_DIR

DOMAIN = "delivery"
TASK_SET = "delivery"
PRESETS = {
    "none": {},
    "emotional": {"emotional_acts": True},
    "tangential": {"tangential": True},
    "fragment": {"fragment_dumping": True},
    "unavailable": {"unavailable_service": True},
}
# Behavior presets run with an EMPTY user persona (user_persona_override="") to
# match pgepa's no-persona condition, so the only difference from the pgepa
# no_persona baseline is the NCUser behavior overlay -> clean Delta. The `none`
# preset is skipped by default because pgepa already ran no_persona on the same
# 40 test tasks / same settings; its numbers are loaded as the baseline.
BEHAVIOR_PRESETS = ["emotional", "tangential", "fragment", "unavailable"]
PGEPA_TEST_EVAL = (
    Path(__file__).resolve().parent
    / "data"
    / "persona_opt"
    / "vita_pgepa_runs"
    / "run_delivery"
    / "test_eval_3trials.json"
)


def pgepa_no_persona_baseline() -> dict | None:
    """Load pgepa's no_persona test-eval numbers (same 40 tasks / 3 trials /
    gpt-5.4 high / chinese) as the Delta baseline so we don't re-run `none`."""
    if not PGEPA_TEST_EVAL.exists():
        return None
    data = json.loads(PGEPA_TEST_EVAL.read_text())
    np = data.get("no_persona", {})
    return {
        "avg_at_n": np.get("avg_at_n"),
        "pass_at_n": np.get("pass_at_n"),  # pgepa pass_at_n = fully_solved/n = Pass^3
        "n_fully_solved": np.get("n_fully_solved"),
    }


def load_task_ids() -> list[str]:
    from vita.run import load_tasks

    return [t.id for t in load_tasks(TASK_SET, language="english")]


def pgepa_test_ids(task_ids: list[str], seed: int = 300) -> list[str]:
    """Replicate the pgepa delivery test split: random.Random(seed).shuffle(all
    ids), test = ids[train+val : train+val+test] = ids[60:100] (40 tasks).
    Matches scripts/vita_pgepa.py split_tasks defaults (30/30/40, seed 300)."""
    import random

    ids = list(task_ids)
    random.Random(seed).shuffle(ids)
    return ids[60:100]


def select_pilot_ids(task_ids: list[str], n: int) -> list[str]:
    """Even-ish sample across the task id range (ids are numeric, sorted)."""
    if n >= len(task_ids):
        return task_ids
    step = len(task_ids) / n
    return [task_ids[int(i * step)] for i in range(n)]


def run_preset(preset: str, task_ids: list[str], args) -> dict:
    save_name = f"ncuser_delivery/{preset}"
    cfg = RunConfig(
        domain=DOMAIN,
        task_set_name=TASK_SET,
        task_ids=task_ids,
        agent="llm_agent",
        llm_agent=args.agent_llm,
        llm_args_agent={},
        user="ncuser",
        llm_user=args.user_llm,
        llm_args_user={"ncuser_config": PRESETS[preset]},
        user_persona_override="",  # match pgepa no_persona (empty persona) -> clean Delta
        llm_evaluator=args.evaluator_llm,
        llm_args_evaluator=dict(models[args.evaluator_llm]),
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=100000,  # match pgepa (disable tool-error cap; default 10 would
        #                    abort fragment/tangential sims that make the agent retry)
        max_duration=0,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        language=args.language,
        enable_think=True,
        save_to=save_name + ".json",
    )
    print(f"[{preset}] {len(task_ids)} tasks x {args.num_trials} -> {save_name}")
    run_domain(cfg)

    rp = DATA_DIR / "simulations" / (save_name + ".json")
    return aggregate(preset, rp)


def aggregate(preset: str, rp: Path) -> dict:
    if not rp.exists():
        return {"preset": preset, "n": 0, "reward_mean": None}
    data = json.loads(rp.read_text())
    sims = data.get("simulations", [])
    rewards = [
        (s.get("reward_info") or {}).get("reward") for s in sims
    ]
    rewards = [r for r in rewards if r is not None]
    n = len(rewards)
    mean = sum(rewards) / n if n else None
    return {
        "preset": preset,
        "n": n,
        "reward_mean": round(mean, 4) if mean is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pilot", "full", "custom"], default="custom")
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--evaluator-llm", default="gpt-4.1")
    ap.add_argument(
        "--language",
        default="chinese",
        choices=["chinese", "english"],
        help="chinese = match pgepa delivery (directly comparable Δ vs your "
        "chinese no_persona/best_persona numbers; none preset reproduces pgepa "
        "no_persona). english = NCUser-native (behavior prompts are English).",
    )
    ap.add_argument("--log-level", default="WARNING")
    ap.add_argument(
        "--presets",
        nargs="+",
        default=BEHAVIOR_PRESETS,
        choices=list(PRESETS.keys()),
        help="Default: the 4 behavior presets (none skipped — pgepa no_persona is "
        "used as the Delta baseline). Pass 'none' to also run it.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "pilot":
        args.num_tasks = args.num_tasks or 10
        args.num_trials = args.num_trials or 1
    elif args.mode == "full":
        args.num_tasks = args.num_tasks or 0  # 0 = pgepa 40-task test split
        args.num_trials = args.num_trials or 3
    if args.num_trials is None:
        args.num_trials = 1

    all_ids = load_task_ids()
    if args.mode == "full" and not args.num_tasks:
        # pgepa delivery test split (40 tasks, seed 300) — same set pgepa used
        # for its test_eval_3trials, so NCUser Δ is comparable to pgepa's.
        task_ids = pgepa_test_ids(all_ids, seed=args.seed)
    elif args.num_tasks and args.num_tasks > 0:
        task_ids = (
            select_pilot_ids(all_ids, args.num_tasks)
            if args.mode == "pilot"
            else all_ids[: args.num_tasks]
        )
    else:
        task_ids = all_ids
    print(
        f"delivery tasks: {len(task_ids)} | trials: {args.num_trials} | "
        f"presets: {args.presets}"
    )

    results = {}
    for preset in args.presets:
        results[preset] = run_preset(preset, task_ids, args)
        print(f"[{preset}] {results[preset]}")

    # Delta baseline: prefer a freshly-run `none` preset; otherwise fall back to
    # pgepa's no_persona test-eval numbers (same 40 tasks / 3 trials / settings),
    # so we don't waste rollouts re-running the vanilla baseline.
    baseline = results.get("none", {}).get("reward_mean")
    baseline_src = "none (this run)"
    if baseline is None:
        pg = pgepa_no_persona_baseline()
        if pg and pg.get("avg_at_n") is not None:
            baseline = pg["avg_at_n"]
            baseline_src = f"pgepa no_persona (avg_at_n={baseline:.4f}, pass^3={pg.get('pass_at_n')})"
    print(f"  baseline: {baseline_src}")
    for preset, r in results.items():
        if preset == "none" or baseline is None:
            r["delta_vs_none"] = None
            continue
        if r["reward_mean"] is not None:
            r["delta_vs_none"] = round(r["reward_mean"] - baseline, 4)
        else:
            r["delta_vs_none"] = None

    out_path = (
        Path(args.out)
        if args.out
        else DATA_DIR / "simulations" / "ncuser_delivery" / "summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n===== SUMMARY =====")
    for preset, r in results.items():
        delta = r.get("delta_vs_none")
        delta_s = f" d={delta:+.4f}" if delta is not None else ""
        print(f"  {preset:12} n={r['n']:>3} reward={r['reward_mean']}{delta_s}")
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
