#!/usr/bin/env python3
"""Run the persona-optimization experiment over the 3 original seed personas.

Launches one GEPA exploration run per seed (p0001, p0002, p0003) IN PARALLEL,
each in its own namespace (distinct .skill file + simulation dir + run dir) so
they never collide. After all three finish, optionally runs a single pooled
certification across all candidates to pick one certified-best.

    # Full experiment (3 seeds in parallel, then certify the pooled candidates)
    python experiments/persona_search/scripts/run_3seed_experiment.py \
        --max-metric-calls 500 --minibatch-size 10

    # Cheap smoke (2 telecom tasks/seed, tiny budget, no certification)
    python experiments/persona_search/scripts/run_3seed_experiment.py \
        --max-metric-calls 8 --minibatch-size 2 --max-tasks 2 \
        --domains telecom --max-concurrency 3 --no-certify

Each seed's live log streams to  <exp-dir>/<pid>/run.log .
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from _common import PERSONA_OPT_DIR  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "gepa_runner.py"
CERTIFIER = SCRIPT_DIR / "certify_gepa_candidates.py"
LINEAGE_DIR = PERSONA_OPT_DIR / "lineages"

# The 3 original seed personas.
SEEDS = ["p0001", "p0002", "p0003"]


def seed_path(pid: str) -> Path:
    return LINEAGE_DIR / pid / "gen_00.md"


def build_runner_cmd(pid: str, run_dir: Path, args) -> list[str]:
    """Construct the gepa_runner.py command for one seed."""
    cmd = [
        sys.executable,
        str(RUNNER),
        "--seed-persona",
        str(seed_path(pid)),
        "--persona-id",
        f"gepa_{pid}",  # unique .skill + sim namespace
        "--run-dir",
        str(run_dir),
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--minibatch-size",
        str(args.minibatch_size),
        "--seed",
        str(args.seed),
        # models / infra
        "--agent-llm",
        args.agent_llm,
        "--agent-llm-args",
        args.agent_llm_args,
        "--user-type",
        args.user_type,
        "--user-llm",
        args.user_llm,
        "--effort",
        args.effort,
        "--mutation-model",
        args.mutation_model,
        "--max-concurrency",
        str(args.max_concurrency),
        "--sim-timeout",
        str(args.sim_timeout),
        "--max-steps",
        str(args.max_steps),
        "--num-trials",
        str(args.num_trials),
        "--max-infra-retries",
        str(args.max_infra_retries),
        "--retrieval-config",
        args.retrieval_config,
        "--domains",
        *args.domains,
        # judge
        "--judge-model",
        args.judge_model,
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--judge-maj",
        str(args.judge_maj),
        # GEPA tuning
        "--frontier-type",
        args.frontier_type,
        "--selection-strategy",
        args.selection_strategy,
        # mutation
        "--mutation-edit-pct",
        args.mutation_edit_pct,
        # convergence / early-stop
        "--patience",
        str(args.patience),
        "--converge-threshold",
        str(args.converge_threshold),
    ]
    if args.max_tasks:
        cmd += ["--max-tasks", str(args.max_tasks)]
    if not args.use_bedrock:
        cmd += ["--no-bedrock"]
    if args.bedrock_profile:
        cmd += ["--bedrock-profile", args.bedrock_profile]
    return cmd


def run_parallel_exploration(args, exp_dir: Path) -> dict[str, dict]:
    """Launch all seeds concurrently; stream each to its own log. Returns status."""
    procs: dict[str, dict] = {}
    for pid in SEEDS:
        run_dir = exp_dir / pid
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        cmd = build_runner_cmd(pid, run_dir, args)
        log_f = open(log_path, "w")
        print(f"  [{pid}] launching -> {run_dir}  (log: {log_path})")
        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR.parent.parent),
        )
        procs[pid] = {"proc": p, "log": log_f, "run_dir": run_dir, "start": time.time()}

    print(f"\n  All {len(procs)} seeds launched. Waiting for completion...\n")

    # Poll until all done, printing a heartbeat.
    remaining = set(procs)
    while remaining:
        time.sleep(15)
        done_now = []
        for pid in list(remaining):
            rc = procs[pid]["proc"].poll()
            if rc is not None:
                elapsed = time.time() - procs[pid]["start"]
                procs[pid]["returncode"] = rc
                procs[pid]["log"].close()
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  [{pid}] finished: {status}  ({elapsed / 60:.1f} min)")
                done_now.append(pid)
        for pid in done_now:
            remaining.discard(pid)
        if remaining:
            mins = (time.time() - min(procs[p]["start"] for p in remaining)) / 60
            print(f"  ... still running: {sorted(remaining)}  ({mins:.1f} min elapsed)")

    return procs


def run_certification(args, exp_dir: Path, procs: dict) -> None:
    """Pool candidates from all successful runs and certify against incumbent."""
    run_dirs = [
        str(procs[pid]["run_dir"])
        for pid in SEEDS
        if procs[pid].get("returncode") == 0
        and (procs[pid]["run_dir"] / "candidates.json").exists()
    ]
    if not run_dirs:
        print("  No successful runs with candidates.json — skipping certification.")
        return

    incumbent = args.incumbent or str(seed_path(SEEDS[0]))
    cert_out = exp_dir / "certification"
    cmd = [
        sys.executable,
        str(CERTIFIER),
        "--run-dirs",
        *run_dirs,
        "--incumbent",
        incumbent,
        "--top-k",
        str(args.top_k),
        "--out-dir",
        str(cert_out),
        # models / infra (mirror exploration, but certification trials)
        "--agent-llm",
        args.agent_llm,
        "--agent-llm-args",
        args.agent_llm_args,
        "--user-type",
        args.user_type,
        "--user-llm",
        args.user_llm,
        "--effort",
        args.effort,
        "--max-concurrency",
        str(args.max_concurrency),
        "--sim-timeout",
        str(args.sim_timeout),
        "--max-steps",
        str(args.max_steps),
        "--num-trials",
        str(args.cert_num_trials),
        "--max-infra-retries",
        str(args.max_infra_retries),
        "--retrieval-config",
        args.retrieval_config,
        "--domains",
        *args.domains,
        "--judge-model",
        args.judge_model,
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--judge-maj",
        str(args.judge_maj),
        # gates
        "--min-delta",
        str(args.min_delta),
        "--min-net-wins",
        str(args.min_net_wins),
        "--max-severe-rate",
        str(args.max_severe_rate),
        "--max-timeout-rate",
        str(args.max_timeout_rate),
        "--max-infra-rate",
        str(args.max_infra_rate),
        "--domain-collapse-asr",
        str(args.domain_collapse_asr),
    ]
    if args.cert_max_tasks:
        cmd += ["--max-tasks", str(args.cert_max_tasks)]
    if not args.use_bedrock:
        cmd += ["--no-bedrock"]
    if args.bedrock_profile:
        cmd += ["--bedrock-profile", args.bedrock_profile]

    print(f"\n  Certifying pooled candidates from {len(run_dirs)} run(s)")
    print(f"  Incumbent: {incumbent}")
    print(f"  Output:    {cert_out}\n")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent.parent))


def main():
    ap = argparse.ArgumentParser(
        description="Optimize 3 seed personas in parallel via GEPA."
    )
    ap.add_argument(
        "--exp-dir",
        default=None,
        help="experiment root (default: data/persona_opt/experiments/exp_<ts>)",
    )

    # Budget / search (shared by all seeds)
    ap.add_argument("--max-metric-calls", type=int, default=500)
    ap.add_argument("--minibatch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="cap tasks PER DOMAIN during exploration (0 = all reprep)",
    )
    ap.add_argument(
        "--frontier-type",
        default="instance",
        choices=["instance", "objective", "hybrid"],
    )
    ap.add_argument(
        "--selection-strategy",
        default="pareto",
        choices=["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
    )

    # Mutation
    ap.add_argument(
        "--mutation-edit-pct",
        default="10-15%",
        help="how much of the persona text to change per mutation",
    )

    # Convergence / early-stop (per seed; OR-composed with max-metric-calls)
    ap.add_argument(
        "--patience",
        type=int,
        default=8,
        help="stop a seed after N proposal rounds with no new full-valset "
        "best difficulty (0 = disabled; budget stays the hard cap)",
    )
    ap.add_argument(
        "--converge-threshold",
        type=float,
        default=0.0,
        help="stop a seed early if full-valset best difficulty >= this (0 = disabled)",
    )

    # Models / infra (mirror gepa_runner defaults)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument("--mutation-model", default="bedrock/us.anthropic.claude-opus-4-7")
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="PER-SEED concurrency; total load = 3x this. Lower it if "
        "running all 3 at once strains the machine/API.",
    )
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument(
        "--num-trials", type=int, default=1, help="trials during exploration"
    )
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])

    # Judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-maj", type=int, default=3)

    # Certification stage
    ap.add_argument(
        "--certify",
        dest="certify",
        action="store_true",
        default=True,
        help="run pooled certification after exploration (default on)",
    )
    ap.add_argument("--no-certify", dest="certify", action="store_false")
    ap.add_argument(
        "--incumbent",
        default=None,
        help="baseline persona for certification (default: p0001 seed)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="top candidates (pooled across all seeds) to certify",
    )
    ap.add_argument("--cert-num-trials", type=int, default=2)
    ap.add_argument(
        "--cert-max-tasks",
        type=int,
        default=0,
        help="cap tasks per domain during certification (0 = all reprep)",
    )
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--min-net-wins", type=int, default=5)
    ap.add_argument("--max-severe-rate", type=float, default=0.05)
    ap.add_argument("--max-timeout-rate", type=float, default=0.25)
    ap.add_argument("--max-infra-rate", type=float, default=0.10)
    ap.add_argument("--domain-collapse-asr", type=float, default=0.99)

    args = ap.parse_args()

    # Validate seeds exist
    missing = [pid for pid in SEEDS if not seed_path(pid).exists()]
    if missing:
        print(
            f"ERROR: missing seed persona files: {[str(seed_path(p)) for p in missing]}"
        )
        sys.exit(1)

    if args.exp_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.exp_dir = str(PERSONA_OPT_DIR / "experiments" / f"exp_{ts}")
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  3-SEED PARALLEL PERSONA OPTIMIZATION")
    print("=" * 64)
    print(f"  Seeds:            {SEEDS}")
    print(f"  Exp dir:          {exp_dir}")
    print(f"  max_metric_calls: {args.max_metric_calls} (per seed, hard cap)")
    print(
        f"  patience:         {args.patience} rounds "
        f"({'convergence on' if args.patience else 'disabled'})"
    )
    print(f"  minibatch_size:   {args.minibatch_size}")
    print(f"  domains:          {args.domains}")
    print(
        f"  concurrency:      {args.max_concurrency}/seed -> "
        f"~{args.max_concurrency * len(SEEDS)} total sims at peak"
    )
    print(f"  certify after:    {args.certify}")
    print("=" * 64 + "\n")

    t0 = time.time()
    procs = run_parallel_exploration(args, exp_dir)

    print(f"\n{'=' * 64}")
    print("  EXPLORATION COMPLETE")
    print("=" * 64)
    for pid in SEEDS:
        rc = procs[pid].get("returncode")
        cj = procs[pid]["run_dir"] / "candidates.json"
        best = procs[pid]["run_dir"] / "exploration_best_persona.md"
        print(
            f"  {pid}: rc={rc}  candidates.json={'Y' if cj.exists() else 'N'}  "
            f"exploration_best={'Y' if best.exists() else 'N'}"
        )
    print(f"  Total exploration wall time: {(time.time() - t0) / 60:.1f} min")

    if args.certify:
        print(f"\n{'=' * 64}")
        print("  CERTIFICATION (pooled across all seeds)")
        print("=" * 64)
        run_certification(args, exp_dir, procs)

    print(f"\n  Experiment dir: {exp_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
