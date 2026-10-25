#!/usr/bin/env python3
"""Prepare or execute a reproducible persona-search matrix.

The generated experiment directories live in the external persona-state
workspace. No API keys are copied into plans or shell launchers.

Examples:
    .venv/bin/python experiments/persona_search/setup_benchmark_experiments.py \
        --run-tag august_matrix

    .venv/bin/python experiments/persona_search/setup_benchmark_experiments.py \
        --run-tag smoke --domains telecom retail --iterations 1 \
        --execute dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cresta.paths import PERSONA_STATE_DIR, REPO_ROOT

DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "persona_search"
    / "configs"
    / "benchmark_domain_matrix.json"
)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema in {path}")
    if not isinstance(config.get("domains"), dict) or not config["domains"]:
        raise ValueError(f"config has no domains: {path}")
    return config


def _validate_profile(name: str, profile: dict[str, Any]) -> None:
    for key in ("runtime_domain", "split_file", "seed_persona", "minibatch"):
        if key not in profile:
            raise ValueError(f"{name}: missing required field {key!r}")
    for key in ("split_file", "seed_persona"):
        path = _resolve_repo_path(profile[key])
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {key} does not exist: {path}")

    split_path = _resolve_repo_path(profile["split_file"])
    split = json.loads(split_path.read_text())
    runtime_domain = profile["runtime_domain"]
    partitions: dict[str, set[str]] = {}
    for partition in ("train", "val"):
        if runtime_domain not in split.get(partition, {}):
            raise ValueError(
                f"{name}: split {partition!r} has no {runtime_domain!r} tasks"
            )
    for partition in ("train", "val", "test"):
        task_ids = split.get(partition, {}).get(runtime_domain, [])
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{name}: duplicate task IDs in {partition!r}")
        partitions[partition] = set(task_ids)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = partitions[left] & partitions[right]
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(
                f"{name}: split partitions {left!r}/{right!r} overlap: {examples}"
            )

    tau2_domain = profile.get("tau2_domain", runtime_domain)
    tasks_path = (
        REPO_ROOT / "datasets" / "tau2" / "domains" / tau2_domain / "tasks.json"
    )
    if not tasks_path.is_file():
        raise FileNotFoundError(f"{name}: task universe does not exist: {tasks_path}")
    known_ids = {str(task["id"]) for task in json.loads(tasks_path.read_text())}
    configured_ids = set().union(*partitions.values())
    missing_ids = sorted(configured_ids - known_ids)
    if missing_ids:
        raise ValueError(
            f"{name}: {len(missing_ids)} split task IDs are absent from "
            f"{tasks_path}: {missing_ids[:5]}"
        )


def _build_command(
    *,
    args: argparse.Namespace,
    name: str,
    profile: dict[str, Any],
    experiment_dir: Path,
    island: int,
) -> list[str]:
    island_name = f"{name}_g{island}"
    run_dir = experiment_dir / island_name
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "cresta.persona.search.parallel_gepa",
        "--seed-persona",
        str(_resolve_repo_path(profile["seed_persona"])),
        "--run-dir",
        str(run_dir),
        "--persona-id-base",
        island_name,
        "--domains",
        profile["runtime_domain"],
        "--split-file",
        str(_resolve_repo_path(profile["split_file"])),
        "--iterations",
        str(args.iterations),
        "--minibatch",
        str(profile["minibatch"] if args.minibatch is None else args.minibatch),
        "--mutation-edit-pct",
        profile.get("mutation_edit_pct", "40-50%"),
        "--num-trials",
        str(args.num_trials),
        "--val-num-trials",
        str(args.num_trials),
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-steps",
        str(args.max_steps),
        "--sim-timeout",
        str(args.sim_timeout),
        "--agent-llm",
        args.agent_model,
        "--agent-llm-args",
        json.dumps({"reasoning_effort": args.reasoning_effort}),
        "--user-type",
        "user_simulator",
        "--user-llm",
        args.user_model,
        "--effort",
        args.reasoning_effort,
        "--mutation-model",
        args.mutation_model,
        "--mutation-effort",
        args.reasoning_effort,
        "--judge-model",
        args.judge_model,
        "--judge-maj",
        "1",
        "--shared-seen-pool",
        str(experiment_dir / "shared_seen_pool.jsonl"),
        "--island-name",
        island_name,
        "--seed",
        str(island),
        "--no-bedrock",
        "--print-pareto",
    ]
    if profile.get("retrieval_config"):
        command += ["--retrieval-config", profile["retrieval_config"]]
    command += ["--lint-persona" if args.with_lint else "--no-lint-persona"]
    if args.with_embedding:
        command += ["--embedding-model", args.embedding_model]
    else:
        command += ["--no-embedding"]
    if args.judge_valset:
        command += ["--judge-valset"]
    if args.execute == "dry-run":
        command += ["--dry-run"]
    return command


def _write_launcher(path: Path, command: list[str]) -> None:
    api_guard = (
        ""
        if "--dry-run" in command
        else ': "${OPENAI_API_KEY:?Export OPENAI_API_KEY before launching}"\n'
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"{api_guard}"
        f"exec {shlex.join(command)}\n"
    )
    path.chmod(0o755)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up a multi-domain tau2 persona-search experiment matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--islands", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sim-timeout", type=int, default=600)
    parser.add_argument("--agent-model", default="gpt-5-mini")
    parser.add_argument("--user-model", default="gpt-5-mini")
    parser.add_argument("--mutation-model", default="openai/gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--with-embedding", action="store_true")
    parser.add_argument("--with-lint", action="store_true")
    parser.add_argument("--judge-valset", action="store_true")
    parser.add_argument(
        "--execute",
        choices=("none", "dry-run", "live"),
        default="none",
        help="none prepares launchers; dry-run validates all commands; live uses APIs",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.islands < 1:
        raise ValueError("--islands must be at least 1")
    if args.execute == "live" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --execute live")

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    available = config["domains"]
    selected = args.domains or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"unknown domains {unknown}; available domains: {sorted(available)}"
        )

    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.expanduser()
        if args.output_root
        else PERSONA_STATE_DIR / "benchmark_setups" / run_tag
    )
    plan_path = output_root / "plan.json"
    if plan_path.exists() and not args.force:
        raise FileExistsError(
            f"{plan_path} already exists; use a new --run-tag or pass --force"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = []
    for name in selected:
        profile = available[name]
        _validate_profile(name, profile)
        experiment_dir = output_root / config["benchmark"] / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for island in range(args.islands):
            command = _build_command(
                args=args,
                name=name,
                profile=profile,
                experiment_dir=experiment_dir,
                island=island,
            )
            launcher = experiment_dir / f"launch_g{island}.sh"
            _write_launcher(launcher, command)
            experiments.append(
                {
                    "benchmark": config["benchmark"],
                    "domain": name,
                    "runtime_domain": profile["runtime_domain"],
                    "island": island,
                    "run_dir": str(experiment_dir / f"{name}_g{island}"),
                    "launcher": str(launcher),
                    "command": command,
                }
            )

    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "config": str(config_path),
        "run_tag": run_tag,
        "execute": args.execute,
        "experiments": experiments,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"prepared {len(experiments)} experiment(s) -> {plan_path}")

    if args.execute == "none":
        return
    for experiment in experiments:
        print(
            f"\n=== {experiment['benchmark']} / {experiment['domain']} "
            f"/ island {experiment['island']} ===",
            flush=True,
        )
        subprocess.run(
            experiment["command"],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
