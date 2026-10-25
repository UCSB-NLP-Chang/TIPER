"""Run NCUser non-collaborative behavior presets on the retail val set.

Mirrors run_ncuser_airline.py but for the retail domain. Uses the
retail_guitar_center_split_v4.json val split (30 tasks) as the test set, and the
retail unavailable-service OOR generator (ncuser_domain="retail").

Settings align with the airline/telecom runs: agent/user = gpt-5.4 (high),
3 trials, seed 300, max-concurrency 30, max-steps 120, sim-timeout 600.

Usage:
  # pilot: 10 tasks × 1 trial × 5 configs
  uv run python experiments/persona_search/scripts/run_ncuser_retail.py --mode pilot
  # full: 30 val tasks × 3 trials × 5 configs
  uv run python experiments/persona_search/scripts/run_ncuser_retail.py --mode full
  # custom
  uv run python experiments/persona_search/scripts/run_ncuser_retail.py --num-trials 3 --max-concurrency 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA_ROOT,
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    read_json,
    write_json,
)

TAU2 = REPO_ROOT / ".venv" / "bin" / "tau2"
SPLIT = DATA_ROOT / "data" / "persona_opt" / "retail_guitar_center_split_v4.json"

# Behavior presets: name -> ncuser_config flags.
PRESETS = {
    "none": {},
    "emotional": {"emotional_acts": True},
    "tangential": {"tangential": True},
    "fragment": {"fragment_dumping": True},
    "unavailable": {"unavailable_service": True},
}


def load_test_ids(split: str = "test") -> list[str]:
    """Retail split used as the NCUser test set. Default 'test' (54 tasks);
    'val' (30) for quick smoke runs."""
    data = read_json(SPLIT)
    return data[split]["retail"]


def select_pilot_ids(test_ids: list[str], n: int) -> list[str]:
    """Retail task ids are plain numeric strings; take the first n by int
    order (deterministic, simplest tasks for a smoke check)."""
    return sorted(test_ids, key=lambda t: int(t))[:n]


def build_cmd(
    preset: str,
    task_ids: list[str],
    run_name: str,
    agent_llm: str,
    user_llm: str,
    agent_args: dict,
    user_args: dict,
    num_trials: int,
    max_concurrency: int,
    max_steps: int,
    timeout: int | None,
    seed: int,
) -> list[str]:
    ncuser_config = PRESETS[preset]
    user_args_with_cfg = {
        **user_args,
        "ncuser_config": ncuser_config,
        "ncuser_domain": "retail",
    }
    cmd = [
        str(TAU2),
        "run",
        "--domain",
        "retail",
        "--user",
        "ncuser",
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_args_with_cfg),
        "--agent-llm",
        agent_llm,
        "--agent-llm-args",
        json.dumps(agent_args),
        "--num-trials",
        str(num_trials),
        "--seed",
        str(seed),
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        run_name,
        "--auto-resume",
        "--max-steps",
        str(max_steps),
    ]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    cmd += ["--task-ids", *task_ids]
    return cmd


def run_with_infra_retry(args, preset: str, task_ids: list[str], run_name: str) -> Path:
    """Run one preset, retrying sims that come back as infra artifacts."""
    rp = SIMULATIONS_DIR / run_name / "results.json"
    common = dict(
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        agent_args=json.loads(args.agent_llm_args) if args.agent_llm_args else {},
        user_args=json.loads(args.user_llm_args) if args.user_llm_args else {},
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        timeout=args.sim_timeout,
        seed=args.seed,
    )
    cmd = build_cmd(preset, task_ids, run_name, **common)
    print(
        f"[{preset}] running {len(task_ids)} task(s) × {args.num_trials} -> {run_name}"
    )
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    for attempt in range(1, args.max_infra_retries + 1):
        if not rp.exists():
            break
        data = read_json(rp)
        sims = data.get("simulations", [])
        broken = [s for s in sims if is_infra_broken(s)]
        if not broken:
            break
        broken_ids = sorted({s["task_id"] for s in broken})
        print(
            f"[{preset}] infra-retry {attempt}/{args.max_infra_retries}: "
            f"{len(broken)} infra sim(s) over {len(broken_ids)} task(s)"
        )
        data["simulations"] = [s for s in sims if not is_infra_broken(s)]
        write_json(rp, data)
        retry_cmd = build_cmd(preset, task_ids, run_name, **common)
        subprocess.run(retry_cmd, check=True, cwd=str(REPO_ROOT))

    return rp


def aggregate(preset: str, rp: Path) -> dict:
    if not rp.exists():
        return {"preset": preset, "n": 0, "reward_mean": None}
    data = read_json(rp)
    sims = [s for s in data.get("simulations", []) if not is_infra_broken(s)]
    rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
    rewards = [r for r in rewards if r is not None]
    n = len(rewards)
    mean = sum(rewards) / n if n else None
    return {
        "preset": preset,
        "n": n,
        "reward_mean": round(mean, 4) if mean is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["pilot", "full", "custom"], default="custom")
    ap.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="override task count (pilot default 10)",
    )
    ap.add_argument("--num-trials", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-infra-retries", type=int, default=3)
    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="which split of retail_guitar_center_split_v4 to run (default test=54)",
    )
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument(
        "--presets",
        nargs="+",
        default=list(PRESETS.keys()),
        choices=list(PRESETS.keys()),
    )
    ap.add_argument("--run-tag", default="ncuser_retail")
    ap.add_argument("--out", default=None, help="summary json path")
    args = ap.parse_args()

    if args.mode == "pilot":
        args.num_tasks = args.num_tasks or 10
        args.num_trials = args.num_trials or 1
    elif args.mode == "full":
        args.num_tasks = args.num_tasks or 0  # 0 = all val
        args.num_trials = args.num_trials or 3
    if args.num_trials is None:
        args.num_trials = 1

    test_ids = load_test_ids(args.split)
    if args.num_tasks and args.num_tasks > 0:
        if args.mode == "pilot":
            test_ids = select_pilot_ids(test_ids, args.num_tasks)
        else:
            test_ids = test_ids[: args.num_tasks]
    print(
        f"test tasks: {len(test_ids)} | trials: {args.num_trials} | presets: {args.presets}"
    )

    results = {}
    for preset in args.presets:
        run_name = f"{args.run_tag}/{preset}"
        rp = run_with_infra_retry(args, preset, test_ids, run_name)
        results[preset] = aggregate(preset, rp)
        print(f"[{preset}] {results[preset]}")

    # Δ vs none.
    baseline = results.get("none", {}).get("reward_mean")
    if baseline is not None:
        for preset, r in results.items():
            if r["reward_mean"] is not None:
                r["delta_vs_none"] = round(r["reward_mean"] - baseline, 4)
            else:
                r["delta_vs_none"] = None

    out_path = (
        Path(args.out) if args.out else SIMULATIONS_DIR / args.run_tag / "summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, results)
    print("\n===== SUMMARY =====")
    for preset, r in results.items():
        delta = r.get("delta_vs_none")
        delta_s = f" Δ={delta:+.4f}" if delta is not None else ""
        print(f"  {preset:12} n={r['n']:>3} reward={r['reward_mean']}{delta_s}")
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
