"""Evaluate ONE persona across both representative domains (plan §4).

Runs two ``tau2 run`` subprocesses (telecom + banking_knowledge, the latter with
``--retrieval-config terminal_use``), then aggregates the 30 task rewards into one
``persona_score = mean(reward over all evaluated tasks)``. Infrastructure-error
sims are dropped (mirrors ``compute_metrics``). num_trials defaults to 1.

Models are configurable; persona text is injected into the canonical tau2
``user_simulator`` through ``custom_guidelines``.

    .venv/bin/python -m cresta.persona.eval.runner --persona-id p0001 --round 1
    # smoke (1 task per domain):
    .venv/bin/python -m cresta.persona.eval.runner --persona-id p0001 --round 0 --max-tasks 1
    # wiring only (no API calls):
    .venv/bin/python -m cresta.persona.eval.runner --persona-id p0001 --round 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from cresta.persona.core._common import (
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    load_rep_tasks,
    read_json,
    round_dir,
    skill_name_for,
    write_json,
)

ALL_DOMAINS = {
    "telecom": "telecom",
    "banking": "banking_knowledge",
    "airline": "airline",
    "retail": "retail",
}


def _run_name(round_no: int, persona_id: str, domain_key: str) -> str:
    return f"persona_search/round_{round_no:02d}/{persona_id}/{domain_key}"


def _persona_config(persona_text: str) -> str:
    roleplay_preamble = (
        "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
        "as described below — every single turn, no exceptions. Never break character "
        "to be helpful, cooperative, or efficient beyond what your persona would do. "
        "If your persona is terse, give minimal information even if you know more. "
        "If your persona withholds details, do NOT volunteer them unprompted. "
        "The agent must EARN every piece of information through the right questions.\n\n"
    )
    return json.dumps({"custom_guidelines": roleplay_preamble + persona_text})


def _build_cmd(args, domain_key: str, task_ids: list[str], run_name: str) -> list[str]:
    domain = ALL_DOMAINS[domain_key]

    if getattr(args, "user_type", "user_simulator") != "user_simulator":
        raise ValueError("Persona evaluation supports only user_simulator")
    user_args = {}
    if hasattr(args, "user_llm_extra_args") and args.user_llm_extra_args:
        user_args = (
            json.loads(args.user_llm_extra_args)
            if isinstance(args.user_llm_extra_args, str)
            else args.user_llm_extra_args
        )
    if args.effort and "reasoning_effort" not in user_args:
        user_args["reasoning_effort"] = args.effort

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "tau2"),
        "run",
        "--domain",
        domain,
        "--user",
        args.user_type,
        "--user-llm",
        args.user_llm,
        "--user-llm-args",
        json.dumps(user_args),
        "--agent-llm",
        args.agent_llm,
        "--agent-llm-args",
        args.agent_llm_args,
        "--num-trials",
        str(args.num_trials),
        "--max-concurrency",
        str(args.max_concurrency),
        "--save-to",
        run_name,
        "--auto-resume",
    ]

    persona_text = getattr(args, "persona_text", None)
    if persona_text:
        cmd += ["--user-persona", _persona_config(persona_text)]

    if domain == "banking_knowledge":
        cmd += ["--retrieval-config", args.retrieval_config]
    cmd += ["--max-steps", str(args.max_steps)]
    if args.sim_timeout:
        cmd += ["--timeout", str(args.sim_timeout)]
    cmd += ["--task-ids", *task_ids]
    return cmd


def _run_domain_with_retry(
    args, domain_key: str, ids: list[str], run_name: str
) -> Path:
    """Run one domain, then retry sims that came back as infrastructure artifacts.

    A transient user-model error can leak into a sim (``is_infra_broken``).
    We drop those sims from results.json and re-invoke tau2 with ``--auto-resume``
    so it regenerates ONLY the dropped (task_id, trial) pairs. Repeat up to
    ``--max-infra-retries`` times; whatever is still broken afterwards is dropped
    at aggregation time.
    """
    rp = SIMULATIONS_DIR / run_name / "results.json"
    cmd = _build_cmd(args, domain_key, ids, run_name)
    print(f"[{domain_key}] running {len(ids)} task(s) -> save_to={run_name}")
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
            f"[{domain_key}] infra-retry {attempt}/{args.max_infra_retries}: "
            f"{len(broken)} infra sim(s) over {len(broken_ids)} task(s) -> {broken_ids}"
        )
        # Drop broken sims so --auto-resume re-runs exactly those tasks.
        # Pass ALL original task IDs (not just broken) because tau2's checkpoint
        # validation rejects a changed task set.
        data["simulations"] = [s for s in sims if not is_infra_broken(s)]
        write_json(rp, data)
        retry_cmd = _build_cmd(args, domain_key, ids, run_name)
        subprocess.run(retry_cmd, check=True, cwd=str(REPO_ROOT))

    # Report any still-broken sims after retries are exhausted.
    if rp.exists():
        still = [s for s in read_json(rp).get("simulations", []) if is_infra_broken(s)]
        if still:
            ids_still = sorted({s["task_id"] for s in still})
            print(
                f"[{domain_key}] WARNING: {len(still)} sim(s) still infra after "
                f"{args.max_infra_retries} retries (dropped from score): {ids_still}"
            )
    return rp


def _aggregate_results(
    results_path: Path, domain_key: str
) -> tuple[dict, int, int, dict]:
    """Return (per_task mean reward, n_infra_error, n_tasks_seen, timeout_stats).

    Timeout and max_steps sims are EXCLUDED from scoring — the persona_score
    should reflect genuine agent failure, not conversations that ran out of
    time.  Timeout stats are reported separately for validity checking.
    """
    if not results_path.exists():
        return {}, 0, 0, {}
    data = read_json(results_path)
    rewards = defaultdict(list)
    infra = 0
    n_timeout = 0
    n_maxsteps = 0
    n_scored = 0
    for sim in data.get("simulations", []):
        if is_infra_broken(sim):
            infra += 1
            continue
        reward = (sim.get("reward_info") or {}).get("reward")
        if reward is None:
            continue
        tr = sim.get("termination_reason", "")
        if tr in ("timeout", "max_steps"):
            if tr == "timeout":
                n_timeout += 1
            else:
                n_maxsteps += 1
            continue
        n_scored += 1
        rewards[sim["task_id"]].append(float(reward))
    per_task = {f"{domain_key}::{tid}": sum(v) / len(v) for tid, v in rewards.items()}
    total_non_infra = n_scored + n_timeout + n_maxsteps
    timeout_stats = {
        "n_timeout": n_timeout,
        "n_maxsteps": n_maxsteps,
        "n_scored": n_scored,
        "timeout_rate": (n_timeout + n_maxsteps) / total_non_infra
        if total_non_infra
        else 0,
    }
    return per_task, infra, len(rewards), timeout_stats


def _fabricate(args, round_no: int) -> dict:
    """--dry-run: write fake results.json files and aggregate them, no API calls."""
    rep = load_rep_tasks()
    per_task, paths = {}, {}
    for dk in args.domains:
        ids = rep[dk][: args.max_tasks] if args.max_tasks else rep[dk]
        run_name = _run_name(round_no, args.persona_id, dk)
        rp = SIMULATIONS_DIR / run_name / "results.json"
        sims = [
            {
                "task_id": t,
                "trial": 0,
                "termination_reason": "user_stop",
                "reward_info": {"reward": 1.0 if i % 2 == 0 else 0.0},
            }
            for i, t in enumerate(ids)
        ]
        write_json(rp, {"tasks": [{"id": t} for t in ids], "simulations": sims})
        pt, _, _, _ = _aggregate_results(rp, dk)
        per_task.update(pt)
        paths[dk] = str(rp)
    score = sum(per_task.values()) / len(per_task) if per_task else None
    return {
        "per_task": per_task,
        "results_paths": paths,
        "persona_score": score,
        "n_tasks": len(per_task),
        "dry_run": True,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one persona over both domains (plan §4)."
    )
    ap.add_argument("--persona-id", required=True)
    ap.add_argument("--skill-name", default=None, help="defaults to opt_<persona_id>")
    ap.add_argument(
        "--persona-file",
        type=Path,
        default=None,
        help="persona markdown/skill file injected as user_simulator custom_guidelines",
    )
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--out", default=None, help="defaults to rounds/round_NN/eval/<id>/"
    )
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="cap tasks PER DOMAIN (smoke)"
    )
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-concurrency", type=int, default=3)
    ap.add_argument(
        "--max-steps", type=int, default=120, help="max conversation steps per sim"
    )
    ap.add_argument(
        "--sim-timeout",
        type=int,
        default=600,
        help="per-sim wall-clock timeout in seconds",
    )
    ap.add_argument(
        "--user-type",
        default="user_simulator",
        choices=["user_simulator"],
    )
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["telecom", "banking"],
        choices=list(ALL_DOMAINS.keys()),
        help="which domains to evaluate (default: telecom + banking)",
    )
    ap.add_argument(
        "--max-infra-retries",
        type=int,
        default=2,
        help="re-run sims that came back as infrastructure artifacts (0 = drop only)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.skill_name = args.skill_name or skill_name_for(args.persona_id)
    args.persona_text = (
        args.persona_file.expanduser().read_text() if args.persona_file else None
    )

    out = (
        Path(args.out) if args.out else round_dir(args.round) / "eval" / args.persona_id
    )
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = _fabricate(args, args.round)
        summary["persona_id"] = args.persona_id
        write_json(out / "persona_eval.json", summary)
        print(
            f"[dry-run] persona_score={summary['persona_score']} over {summary['n_tasks']} tasks"
        )
        print(f"eval -> {out / 'persona_eval.json'}")
        return

    rep = load_rep_tasks()
    per_task, paths, infra_total = {}, {}, 0
    agg_timeout = {"n_timeout": 0, "n_maxsteps": 0, "n_scored": 0}
    for dk in args.domains:
        ids = rep[dk][: args.max_tasks] if args.max_tasks else rep[dk]
        if not ids:
            continue
        run_name = _run_name(args.round, args.persona_id, dk)
        rp = _run_domain_with_retry(args, dk, ids, run_name)
        pt, infra, n_seen, ts = _aggregate_results(rp, dk)
        per_task.update(pt)
        paths[dk] = str(rp)
        infra_total += infra
        for k in ("n_timeout", "n_maxsteps", "n_scored"):
            agg_timeout[k] += ts[k]
        print(
            f"[{dk}] {n_seen} task(s) scored, {infra} infra-error(s), "
            f"{ts['n_timeout']} timeout, {ts['n_maxsteps']} max-steps"
        )

    total = (
        agg_timeout["n_scored"] + agg_timeout["n_timeout"] + agg_timeout["n_maxsteps"]
    )
    timeout_rate = (
        (agg_timeout["n_timeout"] + agg_timeout["n_maxsteps"]) / total if total else 0
    )
    score = sum(per_task.values()) / len(per_task) if per_task else None
    summary = {
        "persona_id": args.persona_id,
        "skill_name": args.skill_name,
        "round": args.round,
        "per_task": per_task,
        "n_tasks": len(per_task),
        "infra_error_count": infra_total,
        "persona_score": score,
        "timeout_rate": timeout_rate,
        "timeout_stats": agg_timeout,
        "results_paths": paths,
        "models": {
            "agent": f"{args.agent_llm} {args.agent_llm_args}",
            "user": f"{args.user_type} {args.user_llm} effort={args.effort} bedrock={args.use_bedrock}",
        },
    }
    write_json(out / "persona_eval.json", summary)
    print(
        f"\npersona_score = {score} over {len(per_task)} tasks "
        f"({infra_total} infra-errors, {agg_timeout['n_timeout']} timeout, "
        f"{agg_timeout['n_maxsteps']} max-steps excluded | timeout_rate={timeout_rate:.0%})"
    )
    print(f"eval -> {out / 'persona_eval.json'}")


if __name__ == "__main__":
    main()
