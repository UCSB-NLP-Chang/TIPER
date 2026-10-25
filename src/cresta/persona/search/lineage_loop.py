#!/usr/bin/env python3
"""Lineage-based persona evolution loop.

3 seed personas (p0001, p0002, p0003) evolve independently via in-place
reflective mutation. No pool, no selection, no dedup.

Each round per lineage:
  1. Evaluate current persona on representative tasks
  2. Judge for validity
  3. If valid + agent succeeded on some tasks: reflective mutation in-place
  4. If illegal: revert to previous valid version
  5. Track score history, check convergence

    .venv/bin/python -m cresta.persona.search.lineage_loop --reset
    .venv/bin/python -m cresta.persona.search.lineage_loop --max-rounds 15
    .venv/bin/python -m cresta.persona.search.lineage_loop --dry-run --reset
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from cresta.paths import PERSONA_SEEDS_DIR, TAU2_DATA_DIR
from cresta.persona.core._common import (  # noqa: E402
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    load_rep_tasks,
    materialize,
    now_iso,
    read_json,
    round_dir,
    skill_name_for,
    write_json,
)
from cresta.persona.eval.runner import _persona_config
from cresta.persona.mutation.prompt import (  # noqa: E402
    MIN_IMPROVEMENT_DELTA,
    TIMEOUT_RATE_THRESHOLD,
    _build_mutation_prompt,
    _format_msg,
)
from cresta.persona.mutation.proposer import call_mutation_llm  # noqa: E402

LINEAGES = ["p0001", "p0002", "p0003"]
STATE_PATH = PERSONA_OPT_DIR / "lineage_state.json"
LINEAGE_DIR = PERSONA_OPT_DIR / "lineages"


def _propose_mutation(prompt: str, persona_text: str, args) -> dict:
    if args.dry_run:
        return {
            "persona_text": persona_text.rstrip()
            + "\n\n<!-- offline lineage smoke mutation -->\n",
            "mutation_direction": "dry_run",
            "reasoning": "Deterministic mutation used for offline wiring validation.",
        }
    return call_mutation_llm(prompt, args.mutation_model)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def init_state() -> dict:
    state = {"current_round": 0, "lineages": {}}
    for pid in LINEAGES:
        skill_path = PERSONAS_DIR / f"{skill_name_for(pid)}.skill"
        seed_path = PERSONA_SEEDS_DIR / "lineages" / pid / "gen_00.md"
        if skill_path.exists():
            text = skill_path.read_text()
        elif seed_path.exists():
            text = seed_path.read_text()
        else:
            raise FileNotFoundError(
                f"missing initial persona for {pid}: expected {skill_path} or {seed_path}"
            )
        gen_dir = LINEAGE_DIR / pid
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "gen_00.md").write_text(text)
        state["lineages"][pid] = {
            "generation": 0,
            "history": [],
            "best_score": None,
            "best_gen": 0,
            "stalled_rounds": 0,
            "status": "active",
        }
    write_json(STATE_PATH, state)
    print(f"initialized lineage state -> {STATE_PATH}")
    return state


def load_state() -> dict:
    if STATE_PATH.exists():
        return read_json(STATE_PATH)
    return init_state()


def current_text(pid: str, state: dict) -> str:
    gen = state["lineages"][pid]["generation"]
    p = LINEAGE_DIR / pid / f"gen_{gen:02d}.md"
    return p.read_text()


def save_generation(pid: str, gen: int, text: str) -> None:
    d = LINEAGE_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"gen_{gen:02d}.md").write_text(text)


def next_gen_number(pid: str) -> int:
    """Return max existing gen number + 1, avoiding file overwrites."""
    d = LINEAGE_DIR / pid
    if not d.exists():
        return 0
    existing = [int(f.stem.split("_")[1]) for f in d.glob("gen_*.md")]
    return max(existing) + 1 if existing else 0


# ---------------------------------------------------------------------------
# Eval + Judge (delegate to canonical package entrypoints)
# ---------------------------------------------------------------------------


def run_eval(pid: str, round_no: int, args) -> dict:
    persona_file = PERSONAS_DIR / f"{skill_name_for(pid)}.skill"
    cmd = [
        sys.executable,
        "-m",
        "cresta.persona.eval.runner",
        "--persona-id",
        pid,
        "--persona-file",
        str(persona_file),
        "--round",
        str(round_no),
        "--num-trials",
        str(args.num_trials),
        "--max-concurrency",
        str(args.max_concurrency),
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
        "--sim-timeout",
        str(args.sim_timeout),
        "--domains",
        *args.domains,
        "--max-infra-retries",
        str(args.max_infra_retries),
    ]
    if args.use_bedrock:
        cmd.append("--use-bedrock")
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"  [{pid}] running eval...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return read_json(round_dir(round_no) / "eval" / pid / "persona_eval.json")


def run_judge(pid: str, round_no: int, args) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "cresta.persona.eval.judge",
        "--persona-id",
        pid,
        "--round",
        str(round_no),
        "--mode",
        "sync",
        "--concurrency",
        str(args.judge_concurrency),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"  [{pid}] running judge...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return read_json(round_dir(round_no) / "judge" / pid / "judge_status.json")


# ---------------------------------------------------------------------------
# Severe failure collection (for feeding into reflective mutation)
# ---------------------------------------------------------------------------


def collect_severe_details(round_no: int, pid: str) -> list[dict]:
    """Collect severe failure details from judge output for one persona."""
    details = []
    for domain in ["telecom", "banking"]:
        psim = round_dir(round_no) / "judge" / pid / domain / "per_sim.jsonl"
        if not psim.exists():
            continue
        with open(psim) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("verdict") != "severe_failure":
                    continue
                # Get conversation snippet for context
                snippet = ""
                sim_msgs = row.get("messages", [])
                if sim_msgs:
                    snippet = "\n".join(
                        f"[{m.get('role', '?')}] {(m.get('content', '') if isinstance(m.get('content', ''), str) else '')[:150]}"
                        for m in sim_msgs[:6]
                    )
                details.append(
                    {
                        "task_id": row.get("task_id", "?"),
                        "domain": domain,
                        "violations": [
                            {
                                "type": v.get("type"),
                                "evidence": v.get("evidence", "")[:300],
                            }
                            for v in row.get("violations", [])
                        ],
                        "conversation_snippet": snippet,
                    }
                )
    return details


# ---------------------------------------------------------------------------
# Trace collection
# ---------------------------------------------------------------------------


def collect_success_traces(
    eval_summary: dict, max_traces: int = 8, sampling: str = "shortest"
) -> list[dict]:
    import random as _rnd

    traces = []
    for dk, rpath in (eval_summary.get("results_paths") or {}).items():
        if not Path(rpath).exists():
            continue
        data = read_json(rpath)
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward != 1.0:
                continue
            msgs = sim.get("messages", [])
            snippet = [_format_msg(m) for m in msgs]
            traces.append(
                {
                    "task_id": sim.get("task_id", "?"),
                    "n_turns": len(msgs),
                    "conversation": "\n".join(snippet),
                }
            )
    if sampling == "random":
        _rnd.shuffle(traces)
    else:
        traces.sort(key=lambda t: t.get("n_turns", 0))
    return traces[:max_traces]


def collect_failure_traces(
    eval_summary: dict, max_traces: int = 4, sampling: str = "shortest"
) -> list[dict]:
    """Collect traces where the agent FAILED (reward==0) — persona was effective."""
    import random as _rnd

    traces = []
    for dk, rpath in (eval_summary.get("results_paths") or {}).items():
        if not Path(rpath).exists():
            continue
        data = read_json(rpath)
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward is None or reward != 0.0:
                continue
            msgs = sim.get("messages", [])
            snippet = [_format_msg(m) for m in msgs]
            traces.append(
                {
                    "task_id": sim.get("task_id", "?"),
                    "n_turns": len(msgs),
                    "conversation": "\n".join(snippet),
                }
            )
    if sampling == "random":
        _rnd.shuffle(traces)
    else:
        traces.sort(key=lambda t: t.get("n_turns", 0))
    return traces[:max_traces]


# ---------------------------------------------------------------------------
# Main round logic
# ---------------------------------------------------------------------------


def _eval_and_judge(
    pid: str, round_no: int, args, suffix: str = ""
) -> tuple[dict, dict]:
    """Run eval + judge for one persona. suffix differentiates retries."""
    eval_summary = run_eval(pid, round_no, args)
    score = eval_summary.get("persona_score")
    n_tasks = eval_summary.get("n_tasks", 0)
    print(f"  [{pid}]{suffix} score={score:.3f} over {n_tasks} tasks")

    judge_result = run_judge(pid, round_no, args)
    status = judge_result.get("status")
    severe_rate = judge_result.get("severe_failure_rate")
    print(f"  [{pid}]{suffix} judge={status} severe_rate={severe_rate}")
    return eval_summary, judge_result


def run_one_lineage(pid: str, state: dict, round_no: int, args) -> None:
    lineage = state["lineages"][pid]
    if lineage["status"] == "converged":
        print(f"\n[{pid}] SKIPPED (converged at gen {lineage['best_gen']})")
        return

    gen = lineage["generation"]
    text = current_text(pid, state)
    print(f"\n{'─' * 50}")
    print(f"  {pid}  gen={gen}  best={lineage['best_score']}")
    print(f"{'─' * 50}")

    # 1. Materialize + eval + judge
    materialize(pid, text)
    eval_summary, judge_result = _eval_and_judge(pid, round_no, args)
    score = eval_summary.get("persona_score")
    status = judge_result.get("status")
    severe_rate = judge_result.get("severe_failure_rate")

    timeout_rate = eval_summary.get("timeout_rate", 0)

    gen_record = {
        "gen": gen,
        "round": round_no,
        "score": score,
        "status": status,
        "severe_rate": severe_rate,
        "timeout_rate": timeout_rate,
        "at": now_iso(),
    }

    # 1b. Check timeout_rate — if valid but too many timeouts, treat as timeout_attack
    if status == "valid" and timeout_rate > TIMEOUT_RATE_THRESHOLD:
        print(
            f"  [{pid}] TIMEOUT ATTACK (timeout_rate={timeout_rate:.0%} > {TIMEOUT_RATE_THRESHOLD:.0%}) "
            f"— score {score:.3f} is inflated by timeouts, reverting"
        )
        status = "timeout_attack"
        gen_record["status"] = "timeout_attack"

    # 2. If illegal or timeout_attack: collect severe details → revert → re-mutate with feedback → retry
    accumulated_severe = lineage.get("severe_failures", [])
    if status in ("illegal", "timeout_attack"):
        if status == "illegal":
            new_severe = collect_severe_details(round_no, pid)
            accumulated_severe.extend(new_severe)
            lineage["severe_failures"] = accumulated_severe
            print(
                f"  [{pid}] ILLEGAL (severe_rate={severe_rate}) — "
                f"collected {len(new_severe)} severe cases, reverting + re-mutating"
            )
        else:
            print(
                f"  [{pid}] TIMEOUT_ATTACK (timeout_rate={timeout_rate:.0%}) — "
                f"reverting + re-mutating away from friction"
            )

        # Find last valid gen to revert to
        valid_gens = [h for h in lineage["history"] if h.get("status") == "valid"]
        if not valid_gens:
            print(f"  [{pid}] no valid gen to revert to — skipping")
            lineage["history"].append(gen_record)
            _locked_write_json(STATE_PATH, state)
            return

        last_valid = valid_gens[-1]
        revert_gen = last_valid["gen"]
        revert_text = (LINEAGE_DIR / pid / f"gen_{revert_gen:02d}.md").read_text()
        save_generation(pid, gen, text)
        gen_record["reverted_to"] = revert_gen
        lineage["history"].append(gen_record)

        # Retry loop: re-mutate from reverted gen with severe feedback
        base_text = revert_text
        base_score = last_valid.get("score", score)
        for attempt in range(1, args.max_severe_retries + 1):
            print(
                f"  [{pid}] retry {attempt}/{args.max_severe_retries}: "
                f"re-mutating from gen {revert_gen} with {len(accumulated_severe)} severe cases as anti-patterns"
            )

            traces = collect_success_traces(eval_summary)
            t0 = time.time()
            prompt = _build_mutation_prompt(
                base_text,
                base_score,
                traces,
                lineage["history"],
                pid,
                severe_failures=accumulated_severe,
                timeout_rate=timeout_rate,
            )
            mutation = _propose_mutation(prompt, base_text, args)
            elapsed = time.time() - t0
            new_text = mutation["persona_text"]
            direction = mutation.get("mutation_direction", "unknown")
            reasoning = mutation.get("reasoning", "")
            print(f"  [{pid}] retry mutation: {direction} ({elapsed:.0f}s)")

            # Save, materialize, eval, judge the retry
            retry_gen = next_gen_number(pid)
            save_generation(pid, retry_gen, new_text)
            lineage["generation"] = retry_gen
            materialize(pid, new_text)

            # Clean old sim/judge data for this round so --auto-resume doesn't conflict
            import shutil

            for sub in ["eval", "judge"]:
                d = round_dir(round_no) / sub / pid
                if d.exists():
                    shutil.rmtree(d)
            sim_dir = SIMULATIONS_DIR / "persona_search" / f"round_{round_no:02d}" / pid
            if sim_dir.exists():
                shutil.rmtree(sim_dir)

            eval_summary, judge_result = _eval_and_judge(
                pid, round_no, args, suffix=f" [retry {attempt}]"
            )
            retry_status = judge_result.get("status")
            retry_severe = judge_result.get("severe_failure_rate")
            retry_score = eval_summary.get("persona_score")
            retry_timeout_rate = eval_summary.get("timeout_rate", 0)

            # Check timeout_attack on retry too
            if retry_status == "valid" and retry_timeout_rate > TIMEOUT_RATE_THRESHOLD:
                retry_status = "timeout_attack"

            retry_record = {
                "gen": retry_gen,
                "round": round_no,
                "score": retry_score,
                "status": retry_status,
                "severe_rate": retry_severe,
                "timeout_rate": retry_timeout_rate,
                "mutation_direction": direction,
                "mutation_reasoning": reasoning,
                "retry_attempt": attempt,
                "at": now_iso(),
            }

            if retry_status == "valid":
                print(
                    f"  [{pid}] retry {attempt} PASSED — score={retry_score:.3f} "
                    f"timeout_rate={retry_timeout_rate:.0%}"
                )
                retry_record["improved"] = lineage["best_score"] is None or (
                    retry_score is not None
                    and lineage["best_score"] - retry_score >= MIN_IMPROVEMENT_DELTA
                )
                if retry_record["improved"]:
                    lineage["best_score"] = retry_score
                    lineage["best_gen"] = retry_gen
                    lineage["stalled_rounds"] = 0
                else:
                    lineage["stalled_rounds"] += 1
                lineage["history"].append(retry_record)
                # Update timeout_rate for next mutation prompt
                timeout_rate = retry_timeout_rate
                _locked_write_json(STATE_PATH, state)
                return

            # Still illegal or timeout_attack — collect more severe details and try again
            if retry_status == "illegal":
                new_severe = collect_severe_details(round_no, pid)
                accumulated_severe.extend(new_severe)
                lineage["severe_failures"] = accumulated_severe
            retry_record["reverted_to"] = revert_gen
            lineage["generation"] = revert_gen
            lineage["history"].append(retry_record)
            timeout_rate = retry_timeout_rate
            print(
                f"  [{pid}] retry {attempt} still {retry_status} "
                f"(severe_rate={retry_severe}, timeout_rate={retry_timeout_rate:.0%}) — reverting"
            )

        print(
            f"  [{pid}] all {args.max_severe_retries} retries failed — "
            f"keeping gen {revert_gen}, moving on"
        )
        lineage["stalled_rounds"] += 1
        _locked_write_json(STATE_PATH, state)
        return

    # 3. Valid — update best (require MIN_IMPROVEMENT_DELTA to filter noise)
    improved = False
    if lineage["best_score"] is None or (
        score is not None and lineage["best_score"] - score >= MIN_IMPROVEMENT_DELTA
    ):
        lineage["best_score"] = score
        lineage["best_gen"] = gen
        lineage["stalled_rounds"] = 0
        improved = True
    else:
        lineage["stalled_rounds"] += 1

    gen_record["improved"] = improved

    # 4. Collect traces + mutate (include any accumulated severe failures)
    traces = collect_success_traces(eval_summary)
    if not traces:
        print(f"  [{pid}] no success traces — persona defeated agent on all tasks!")
        lineage["history"].append(gen_record)
        _locked_write_json(STATE_PATH, state)
        return

    print(f"  [{pid}] {len(traces)} success traces → reflective mutation...")
    t0 = time.time()
    prompt = _build_mutation_prompt(
        text,
        score,
        traces,
        lineage["history"],
        pid,
        severe_failures=accumulated_severe or None,
        timeout_rate=timeout_rate,
    )
    mutation = _propose_mutation(prompt, text, args)
    elapsed = time.time() - t0
    new_text = mutation["persona_text"]
    direction = mutation.get("mutation_direction", "unknown")
    reasoning = mutation.get("reasoning", "")

    print(f"  [{pid}] mutation: {direction} ({elapsed:.0f}s)")
    print(f"  [{pid}] reasoning: {reasoning}")

    new_gen = next_gen_number(pid)
    save_generation(pid, new_gen, new_text)
    lineage["generation"] = new_gen
    gen_record["mutation_direction"] = direction
    gen_record["mutation_reasoning"] = reasoning

    # 5. Convergence check
    if lineage["stalled_rounds"] >= args.patience:
        lineage["status"] = "converged"
        print(
            f"  [{pid}] CONVERGED at gen {lineage['best_gen']} "
            f"score={lineage['best_score']:.3f}"
        )

    lineage["history"].append(gen_record)
    _locked_write_json(STATE_PATH, state)


_state_lock = threading.Lock()
_original_write_json = write_json


def _locked_write_json(path, data):
    with _state_lock:
        _original_write_json(path, data)


# ---------------------------------------------------------------------------
# Generalization eval (full task set, 2 trials + retry)
# ---------------------------------------------------------------------------

FULL_TASKS = {
    "telecom": "base",
    "banking": "all",
}
GEN_CHECK_INTERVAL = 10


def _run_gen_eval(pid: str, gen: int, state: dict, args) -> dict:
    """Evaluate one persona on the FULL task set (2 trials, retry if Δ>5%)."""
    lineage = state["lineages"][pid]
    lineage_round = lineage.get("lineage_round", 0)
    text = (LINEAGE_DIR / pid / f"gen_{gen:02d}.md").read_text()
    materialize(pid, text)

    gen_dir = PERSONA_OPT_DIR / "gen_checks" / pid / f"gen_{gen:02d}"

    # Build full task set (all base tasks, not just reprep)
    full_tasks = _load_full_tasks(args.domains)

    summary = _run_multi_trial_eval(
        f"gen_check/{pid}/gen_{gen:02d}",
        text,
        gen_dir,
        args,
        task_ids_by_domain=full_tasks,
    )
    summary["pid"] = pid
    summary["gen"] = gen
    summary["lineage_round"] = lineage_round

    lineage.setdefault("gen_checks", []).append(summary)
    _locked_write_json(STATE_PATH, state)
    return summary


def _load_full_tasks(domains: list[str]) -> dict[str, list[str]]:
    """Load ALL tasks for each domain (not just reprep subset)."""
    full = {}
    domains_dir = TAU2_DATA_DIR / "tau2" / "domains"
    if "telecom" in domains:
        split = read_json(domains_dir / "telecom" / "split_tasks.json")
        full["telecom"] = split.get("base", [])
    if "banking" in domains:
        banking_dir = domains_dir / "banking_knowledge" / "tasks"
        full["banking"] = sorted(f.stem for f in banking_dir.glob("task_*.json"))
    return full


# ---------------------------------------------------------------------------
# Baseline eval (no-persona + seeds)
# ---------------------------------------------------------------------------


def _run_multi_trial_eval(
    label: str,
    persona_text: str | None,
    out_dir: Path,
    args,
    task_ids_by_domain: dict[str, list[str]] | None = None,
) -> dict:
    """Run 2 trials on a task set. If Δ>5%, run a 3rd. Return summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = task_ids_by_domain or load_rep_tasks()

    if args.dry_run:
        summary = {
            "label": label,
            "scores": [0.5, 0.5],
            "mean": 0.5,
            "n_trials": 2,
            "at": now_iso(),
            "dry_run": True,
        }
        write_json(out_dir / "summary.json", summary)
        print(f"  [{label}] dry-run scores=[0.5, 0.5] → mean=0.5")
        return summary

    scores = []
    for trial_no in range(1, 4):
        if trial_no == 3 and len(scores) == 2:
            if scores[0] is not None and scores[1] is not None:
                if abs(scores[0] - scores[1]) <= 0.05:
                    break
            else:
                break
            print(f"  [{label}] Δ={abs(scores[0] - scores[1]):.3f} > 5% → trial 3")

        user_args = {"reasoning_effort": args.effort}

        trial_rewards = {}
        for dk in args.domains:
            domain = {"telecom": "telecom", "banking": "banking_knowledge"}[dk]
            ids = rep.get(dk, [])
            if not ids:
                continue
            save_to = f"persona_search/{label}/trial_{trial_no}/{dk}"
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
                "1",
                "--max-concurrency",
                str(args.max_concurrency),
                "--save-to",
                save_to,
                "--auto-resume",
                "--max-steps",
                str(args.max_steps),
                "--timeout",
                str(args.sim_timeout),
                "--task-ids",
                *ids,
            ]
            if persona_text:
                cmd += ["--user-persona", _persona_config(persona_text)]
            if domain == "banking_knowledge":
                cmd += ["--retrieval-config", args.retrieval_config]
            print(f"  [{label}] trial {trial_no} {dk}...")
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

            rp = SIMULATIONS_DIR / save_to / "results.json"
            if rp.exists():
                from cresta.persona.eval.runner import _aggregate_results

                pt, _, _, _ = _aggregate_results(rp, dk)
                trial_rewards.update(pt)

        score = (
            sum(trial_rewards.values()) / len(trial_rewards) if trial_rewards else None
        )
        scores.append(score)

    valid = [s for s in scores if s is not None]
    mean = sum(valid) / len(valid) if valid else None
    summary = {
        "label": label,
        "scores": scores,
        "mean": mean,
        "n_trials": len(valid),
        "at": now_iso(),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"  [{label}] scores={scores} → mean={mean}")
    return summary


def run_baseline(state: dict, args) -> None:
    """Evaluate no-persona and seed personas on reprep set (2 trials + retry)."""
    baseline_dir = PERSONA_OPT_DIR / "baseline"

    configs: list[tuple[str, str | None]] = [("no_persona", None)]
    for pid in LINEAGES:
        seed_text = (LINEAGE_DIR / pid / "gen_00.md").read_text()
        materialize(pid, seed_text)
        configs.append((f"seed_{pid}", seed_text))

    def _run_one(cfg):
        label, skill = cfg
        return _run_multi_trial_eval(
            f"baseline/{label}",
            skill,
            baseline_dir / label,
            args,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as pool:
        results = list(pool.map(_run_one, configs))

    summary = {c[0]: r for c, r in zip(configs, results)}
    write_json(baseline_dir / "baseline_all.json", summary)
    print(f"\n  Baseline complete → {baseline_dir}")


# ---------------------------------------------------------------------------
# Per-lineage async loop
# ---------------------------------------------------------------------------


def run_lineage_loop(pid: str, state: dict, args) -> None:
    """Run one lineage's full optimization loop independently."""
    lineage = state["lineages"][pid]
    start_round = lineage.get("lineage_round", 0) + 1

    for lr in range(start_round, start_round + args.max_rounds):
        if lineage["status"] == "converged":
            print(f"\n[{pid}] CONVERGED at gen {lineage['best_gen']}")
            break

        lineage["lineage_round"] = lr
        # Map to a global round for file naming (avoids collisions with other lineages)
        global_round = lr
        print(f"\n[{pid}] === lineage_round {lr} ===")

        try:
            run_one_lineage(pid, state, global_round, args)
        except Exception as e:
            print(f"  [{pid}] ERROR in round {lr}: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Generalization check every GEN_CHECK_INTERVAL rounds
        if lr % GEN_CHECK_INTERVAL == 0 and lineage["best_gen"] is not None:
            print(f"  [{pid}] generalization check at round {lr}")
            try:
                _run_gen_eval(pid, lineage["best_gen"], state, args)
            except Exception as e:
                print(f"  [{pid}] gen_check ERROR: {e}")
                import traceback

                traceback.print_exc()
                raise

    # Final gen check on full set (always, regardless of round number)
    if lineage["best_gen"] is not None:
        last_check_gens = {gc["gen"] for gc in lineage.get("gen_checks", [])}
        if lineage["best_gen"] not in last_check_gens:
            print(f"  [{pid}] final generalization check (gen {lineage['best_gen']})")
            try:
                _run_gen_eval(pid, lineage["best_gen"], state, args)
            except Exception as e:
                print(f"  [{pid}] final gen_check ERROR: {e}")
                import traceback

                traceback.print_exc()
                raise

    # Final summary for this lineage
    lineage = state["lineages"][pid]
    print(
        f"\n[{pid}] DONE: best_score={lineage['best_score']} at gen {lineage['best_gen']} "
        f"({lineage.get('lineage_round', '?')} rounds, {len(lineage['history'])} evals)"
    )


def main():
    ap = argparse.ArgumentParser(description="Lineage-based persona evolution.")
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument(
        "--patience",
        type=int,
        default=3,
        help="stop lineage after N rounds without improvement",
    )
    ap.add_argument(
        "--max-severe-retries",
        type=int,
        default=3,
        help="retry mutations that trigger severe failures",
    )
    ap.add_argument("--reset", action="store_true", help="reset state, start fresh")
    ap.add_argument(
        "--baseline-only", action="store_true", help="run baseline eval only, then exit"
    )
    ap.add_argument(
        "--skip-baseline", action="store_true", help="skip baseline eval on fresh start"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise state, eval, judge, and mutation wiring without API calls",
    )
    ap.add_argument("--max-steps", type=int, default=120)

    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator", choices=["user_simulator"])
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--mutation-model", default="bedrock/us.anthropic.claude-opus-4-7")
    ap.add_argument("--retrieval-config", default="terminal_use")

    args = ap.parse_args()

    if args.reset or not STATE_PATH.exists():
        state = init_state()
    else:
        state = load_state()
        max_lr = max(
            lineage.get("lineage_round", 0) for lineage in state["lineages"].values()
        )
        print(f"resumed (max lineage_round={max_lr})")

    # Baseline: run once on fresh start or explicit flag
    baseline_done = (PERSONA_OPT_DIR / "baseline").exists()
    if args.baseline_only:
        run_baseline(state, args)
        return
    if args.reset and not args.skip_baseline:
        run_baseline(state, args)
    elif not baseline_done and not args.skip_baseline:
        run_baseline(state, args)

    # Launch all lineages in parallel — each runs its own loop independently
    def _run(pid):
        try:
            run_lineage_loop(pid, state, args)
        except Exception as e:
            print(f"  [{pid}] FATAL: {e}")
            import traceback

            traceback.print_exc()
            raise

    active = [p for p in LINEAGES if state["lineages"][p]["status"] != "converged"]
    if not active:
        print("All lineages already converged.")
        return

    print(f"\nStarting {len(active)} lineage(s): {active}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
        list(pool.map(_run, active))

    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    for pid in LINEAGES:
        lineage = state["lineages"][pid]
        gen_checks = lineage.get("gen_checks", [])
        last_gc = gen_checks[-1] if gen_checks else {}
        print(
            f"  {pid}: best_score={lineage['best_score']} at gen {lineage['best_gen']} "
            f"(rounds={lineage.get('lineage_round', '?')}, "
            f"evals={len(lineage['history'])})"
            f"  full_set={last_gc.get('mean', 'N/A')}"
        )


if __name__ == "__main__":
    main()
