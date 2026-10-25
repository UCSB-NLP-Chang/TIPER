#!/usr/bin/env python3
"""Post-GEPA certification: evaluate top-K candidates on reprep x 2 trials with
full judge, then select the certified best via paired comparison against incumbent.

GEPA's ``result.best_idx`` is the "exploration_best" — reprep-50 single-trial best.
This script runs proper multi-trial + judge + paired comparison to select a
"certified_best" that is reliably better than the incumbent.

    # Standard certification (top 3 candidates, 50 tasks x 2 trials)
    python experiments/persona_search/scripts/certify_gepa_candidates.py \
        --run-dirs data/persona_opt/gepa_runs/run_001 \
        --incumbent data/personas/opt_p0001.skill \
        --top-k 3

    # Quick certification (fewer tasks)
    python experiments/persona_search/scripts/certify_gepa_candidates.py \
        --run-dirs data/persona_opt/gepa_runs/run_001 \
        --incumbent data/personas/opt_p0001.skill \
        --top-k 3 --max-tasks 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    PERSONAS_DIR,
    REPO_ROOT,
    SIMULATIONS_DIR,
    load_rep_tasks,
    read_json,
    skill_name_for,
    write_json,
)
from eval_persona import _run_domain_with_retry  # noqa: E402
from lineage_loop import materialize  # noqa: E402
from tau_persona_adapter import Outcome, classify_outcome  # noqa: E402

JUDGE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "baseline"
    / "scripts"
    / "judge_user_intent_preservation.py"
)

DOMAIN_SPLIT_TASKS = {
    "telecom": REPO_ROOT
    / "datasets"
    / "tau2"
    / "domains"
    / "telecom"
    / "split_tasks.json",
    "banking_knowledge": REPO_ROOT
    / "experiments"
    / "persona_search"
    / "configs"
    / "banking_knowledge_split.json",
}


def load_full_base_tasks(domains: list[str]) -> dict[str, list[str]]:
    """Load the full evaluation task IDs for each domain.

    telecom: the 'base' split of datasets/tau2/domains/telecom/split_tasks.json.
    banking_knowledge: the 'test' split of configs/banking_knowledge_split.json
    (that domain has no split_tasks.json; the config file is its task split).
    """
    out: dict[str, list[str]] = {}
    for dk in domains:
        sp_path = DOMAIN_SPLIT_TASKS.get(dk)
        if not (sp_path and sp_path.exists()):
            out[dk] = []
            continue
        sp = read_json(sp_path)
        if dk == "banking_knowledge":
            out[dk] = list((sp.get("test") or {}).get("banking", []))
        else:
            out[dk] = list(sp.get("base", []))
    return out


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def load_candidates(run_dirs: list[str], top_k: int) -> list[dict]:
    """Load top-K candidates from one or more GEPA run directories.

    Each run_dir should contain candidates.json (written by gepa_runner.py).
    Candidates are ranked by exploration_score (descending) and deduplicated.
    """
    all_cands: list[dict] = []
    seen_texts: set[str] = set()

    for rd in run_dirs:
        cpath = Path(rd) / "candidates.json"
        if not cpath.exists():
            print(f"  WARNING: {cpath} not found, skipping")
            continue
        cands = read_json(cpath)
        for c in cands:
            text_hash = hash(c["persona_text"].strip())
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            c["source_run"] = rd
            all_cands.append(c)

    all_cands.sort(key=lambda c: c.get("exploration_score", 0), reverse=True)

    selected = all_cands[:top_k]
    print(f"  Loaded {len(all_cands)} unique candidates from {len(run_dirs)} run(s)")
    scores_str = [f"{c.get('exploration_score', 0):.3f}" for c in selected]
    print(f"  Selected top-{top_k}: exploration scores = {scores_str}")
    return selected


# ---------------------------------------------------------------------------
# Evaluation + judge
# ---------------------------------------------------------------------------


def evaluate_persona(
    persona_id: str,
    persona_text: str,
    args,
    domains: list[str],
    rep_tasks: dict[str, list[str]],
) -> dict:
    """Evaluate a single persona: run sims + judge, return per-task results."""
    materialize(persona_id, persona_text)
    eval_args = _make_eval_args(args, persona_id)

    results_by_task: dict[str, dict] = {}
    results_paths: dict[str, str] = {}

    for dk in domains:
        ids = rep_tasks.get(dk, [])
        if args.max_tasks:
            ids = ids[: args.max_tasks]
        if not ids:
            continue

        run_name = f"gepa_certify/{persona_id}/{dk}"
        try:
            rp = _run_domain_with_retry(eval_args, dk, ids, run_name)
        except Exception as e:
            print(f"  [certify {persona_id} {dk}] subprocess warning: {e}")
            rp = SIMULATIONS_DIR / run_name / "results.json"

        results_paths[dk] = str(rp)
        if rp.exists():
            data = read_json(rp)
            for sim in data.get("simulations", []):
                tid = sim["task_id"]
                trial = sim.get("trial", 0)
                sim["_domain"] = dk
                results_by_task[f"{tid}::t{trial}"] = sim

    judge_by_task = _run_judge(results_paths, f"gepa_certify/{persona_id}", args)

    task_results: list[dict] = []
    for key, sim in results_by_task.items():
        tid = sim["task_id"]
        jinfo = judge_by_task.get(tid)
        outcome = classify_outcome(sim, jinfo)
        reward = (sim.get("reward_info") or {}).get("reward")
        task_results.append(
            {
                "task_id": tid,
                "domain": sim.get("_domain", ""),
                "trial": sim.get("trial", 0),
                "reward": float(reward) if reward is not None else None,
                "termination_reason": sim.get("termination_reason", ""),
                "outcome": outcome.value,
            }
        )

    return {"persona_id": persona_id, "task_results": task_results}


def _make_eval_args(args, persona_id: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        skill_name=skill_name_for(persona_id),
        effort=args.effort,
        use_bedrock=args.use_bedrock,
        bedrock_profile=args.bedrock_profile,
        user_type=args.user_type,
        user_llm=args.user_llm,
        agent_llm=args.agent_llm,
        agent_llm_args=args.agent_llm_args,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        retrieval_config=args.retrieval_config,
        max_steps=args.max_steps,
        sim_timeout=args.sim_timeout,
        max_infra_retries=args.max_infra_retries,
    )


def _run_judge(results_paths: dict[str, str], run_label: str, args) -> dict[str, dict]:
    judge_by_task: dict[str, dict] = {}
    for dk, rpath in results_paths.items():
        if not Path(rpath).exists():
            continue
        judge_out = SIMULATIONS_DIR / run_label / "judge" / dk
        judge_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(JUDGE_SCRIPT),
            "sync",
            "--results",
            str(rpath),
            "--model",
            args.judge_model,
            "--maj",
            str(args.judge_maj),
            "--out",
            str(judge_out),
            "--concurrency",
            str(args.judge_concurrency),
        ]
        print(f"  [certify judge {dk}] judging -> {judge_out}")
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"  [certify judge {dk}] WARNING: judge failed: {e}")
            continue

        per_sim = judge_out / "per_sim.jsonl"
        if per_sim.exists():
            with open(per_sim) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    judge_by_task[row.get("task_id", "")] = {
                        "verdict": row.get("verdict"),
                        "violations": row.get("violations", []),
                    }
    return judge_by_task


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _agent_won(outcome: str) -> float | None:
    """Per-sim agent-success contribution.

    Returns None when the sim must be excluded from the success-rate
    denominator (infra/illegal/unknown — not valid persona evidence).
    Timeout and genuine agent failure both score 0.0 (agent did not succeed).
    """
    if outcome in (
        Outcome.INFRA_ERROR.value,
        Outcome.ILLEGAL_USER_BEHAVIOR.value,
        Outcome.UNKNOWN.value,
    ):
        return None
    if outcome in (
        Outcome.GENUINE_AGENT_FAILURE.value,
        Outcome.TIMEOUT_OR_MAX_STEPS.value,
    ):
        return 0.0
    return 1.0  # AGENT_SUCCESS


def compute_metrics(task_results: list[dict]) -> dict:
    """Compute certification metrics from a persona's task results."""
    n_total = len(task_results)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_scored": 0,
            "n_infra": 0,
            "agent_success_rate": 1.0,
            "severe_rate": 0,
            "timeout_rate": 0,
            "infra_rate": 0,
            "domain_asr": {},
        }

    outcomes = [r["outcome"] for r in task_results]
    n_success = outcomes.count(Outcome.AGENT_SUCCESS.value)
    n_genuine = outcomes.count(Outcome.GENUINE_AGENT_FAILURE.value)
    n_severe = outcomes.count(Outcome.ILLEGAL_USER_BEHAVIOR.value)
    n_timeout = outcomes.count(Outcome.TIMEOUT_OR_MAX_STEPS.value)
    n_infra = outcomes.count(Outcome.INFRA_ERROR.value)

    n_scored = n_total - n_infra  # excludes only infra noise

    # ASR denominator excludes infra/illegal/unknown; timeout counts as success.
    asr_vals = [v for r in task_results if (v := _agent_won(r["outcome"])) is not None]
    asr = sum(asr_vals) / len(asr_vals) if asr_vals else 1.0

    # Per-domain ASR (for domain-collapse detection).
    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in task_results:
        v = _agent_won(r["outcome"])
        if v is not None:
            by_domain[r.get("domain", "?")].append(v)
    domain_asr = {d: (sum(vs) / len(vs) if vs else 1.0) for d, vs in by_domain.items()}

    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_infra": n_infra,
        "n_agent_success": n_success,
        "n_genuine_failure": n_genuine,
        "n_severe": n_severe,
        "n_timeout": n_timeout,
        "agent_success_rate": asr,
        "severe_rate": n_severe / n_scored if n_scored else 0,
        "timeout_rate": n_timeout / n_scored if n_scored else 0,
        "infra_rate": n_infra / n_total,
        "domain_asr": domain_asr,
    }


def paired_comparison(
    candidate_results: list[dict],
    incumbent_results: list[dict],
    epsilon: float = 0.01,
) -> dict:
    """Paired per-task comparison: candidate vs incumbent (outcome-aware).

    Per task we compute each persona's agent-success-rate over its valid trials
    (infra/illegal/unknown excluded; timeout counts as agent-success so a
    candidate is NEVER credited for stalling the agent into a timeout).

    A 'win'  = candidate lowers agent-success vs incumbent (candidate is harder).
    A 'loss' = candidate raises it. Tie otherwise.
    """

    def _per_task_asr(results: list[dict]) -> dict[str, float]:
        bucket: dict[str, list[float]] = defaultdict(list)
        for r in results:
            v = _agent_won(r["outcome"])
            if v is not None:
                bucket[r["task_id"]].append(v)
        return {tid: sum(vs) / len(vs) for tid, vs in bucket.items() if vs}

    inc_asr = _per_task_asr(incumbent_results)
    cand_asr = _per_task_asr(candidate_results)

    wins, losses, ties = 0, 0, 0
    task_details: list[dict] = []
    for tid in set(inc_asr) & set(cand_asr):
        inc_mean = inc_asr[tid]
        cand_mean = cand_asr[tid]
        delta = inc_mean - cand_mean  # >0 => candidate harder (fewer agent successes)

        if delta > epsilon:
            wins += 1
            verdict = "win"
        elif delta < -epsilon:
            losses += 1
            verdict = "loss"
        else:
            ties += 1
            verdict = "tie"

        task_details.append(
            {
                "task_id": tid,
                "incumbent_agent_success": round(inc_mean, 3),
                "candidate_agent_success": round(cand_mean, 3),
                "delta": round(delta, 3),
                "verdict": verdict,
            }
        )

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "n_compared": wins + losses + ties,
        "task_details": sorted(task_details, key=lambda d: d["delta"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------


def check_gates(metrics: dict, paired: dict, gates: dict) -> tuple[bool, list[str]]:
    """Check acceptance gates. Returns (passed, [reasons for failure])."""
    failures: list[str] = []

    sr = metrics.get("severe_rate", 1)
    if sr > gates["max_severe_rate"]:
        failures.append(f"severe_rate={sr:.3f} > {gates['max_severe_rate']}")

    tr = metrics.get("timeout_rate", 1)
    if tr > gates["max_timeout_rate"]:
        failures.append(f"timeout_rate={tr:.3f} > {gates['max_timeout_rate']}")

    asr = metrics.get("agent_success_rate", 1)
    inc_asr = gates.get("incumbent_asr", asr)
    delta = inc_asr - asr
    if delta < gates["min_delta"]:
        failures.append(
            f"delta={delta:.3f} (inc_asr={inc_asr:.3f} - cand_asr={asr:.3f}) "
            f"< {gates['min_delta']}"
        )

    nw = paired.get("net_wins", 0)
    if nw < gates["min_net_wins"]:
        failures.append(f"net_wins={nw} < {gates['min_net_wins']}")

    ir = metrics.get("infra_rate", 0)
    if ir > gates["max_infra_rate"]:
        failures.append(f"infra_rate={ir:.3f} > {gates['max_infra_rate']}")

    # Domain collapse: the candidate created (almost) no difficulty in at least
    # one domain while being evaluated on multiple. Only meaningful with >1 domain.
    domain_asr = metrics.get("domain_asr", {})
    if len(domain_asr) > 1:
        collapsed = [
            d for d, a in domain_asr.items() if a >= gates["domain_collapse_asr"]
        ]
        if collapsed:
            asr_str = ", ".join(f"{d}={domain_asr[d]:.2f}" for d in collapsed)
            failures.append(
                f"domain_collapse: {asr_str} >= {gates['domain_collapse_asr']} "
                f"(no difficulty created)"
            )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Certify GEPA exploration-best candidates via multi-trial eval + judge."
    )
    ap.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="GEPA run directories containing candidates.json",
    )
    ap.add_argument(
        "--incumbent", required=True, help="Path to incumbent .skill/.md persona file"
    )
    ap.add_argument(
        "--top-k", type=int, default=3, help="Number of top candidates to certify"
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: first run-dir / certification)",
    )

    # Model / infra
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
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="Cap tasks per domain (0 = use all)"
    )
    ap.add_argument(
        "--use-full-tasks",
        action="store_true",
        default=False,
        help="Use the full base-split tasks (e.g. 114 telecom) "
        "instead of the reprep subset (35). Recommended for "
        "final certification.",
    )
    ap.add_argument(
        "--split-file",
        default=None,
        help="Load task IDs from a JSON split file (e.g. "
        "telecom_island_split_v20.json). Overrides --use-full-tasks.",
    )
    ap.add_argument(
        "--split-name",
        default="test",
        help="Which split to use from --split-file (default: test)",
    )

    # Judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-maj", type=int, default=3)

    # Gates
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.05,
        help="Min improvement in agent_success_rate vs incumbent",
    )
    ap.add_argument(
        "--min-net-wins", type=int, default=5, help="Min paired wins - losses"
    )
    ap.add_argument("--max-severe-rate", type=float, default=0.05)
    ap.add_argument("--max-timeout-rate", type=float, default=0.25)
    ap.add_argument(
        "--max-infra-rate",
        type=float,
        default=0.10,
        help="Max fraction of sims that may be infra errors",
    )
    ap.add_argument(
        "--domain-collapse-asr",
        type=float,
        default=0.99,
        help="Per-domain agent-success-rate at/above which a domain "
        "is considered collapsed (no difficulty created)",
    )

    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(args.run_dirs[0]) / "certification"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = {
        "min_delta": args.min_delta,
        "min_net_wins": args.min_net_wins,
        "max_severe_rate": args.max_severe_rate,
        "max_timeout_rate": args.max_timeout_rate,
        "max_infra_rate": args.max_infra_rate,
        "domain_collapse_asr": args.domain_collapse_asr,
    }

    if args.split_file:
        sf = read_json(Path(args.split_file))
        split_data = sf.get(args.split_name, {})
        task_pool = {}
        for dk in args.domains:
            ids = split_data.get(dk, [])
            if ids:
                task_pool[dk] = list(ids)
        task_label = f"split:{args.split_name}"
    elif args.use_full_tasks:
        task_pool = load_full_base_tasks(args.domains)
        task_label = "full-base"
    else:
        task_pool = load_rep_tasks()
        task_label = "reprep"
    task_counts = {dk: len(ids) for dk, ids in task_pool.items() if ids}
    print(f"  Task pool ({task_label}): {task_counts}")

    # 1. Load candidates
    print("=" * 60)
    print("  PHASE 1: Load candidates")
    print("=" * 60)
    candidates = load_candidates(args.run_dirs, args.top_k)
    if not candidates:
        print("  ERROR: No candidates found. Exiting.")
        return

    # 2. Evaluate incumbent
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Evaluate incumbent")
    print("=" * 60)
    inc_text = Path(args.incumbent).read_text()
    inc_pid = "gepa_certify_incumbent"
    print(f"  Incumbent: {args.incumbent} ({len(inc_text)} chars)")
    inc_eval = evaluate_persona(inc_pid, inc_text, args, args.domains, task_pool)
    inc_metrics = compute_metrics(inc_eval["task_results"])
    print(
        f"  Incumbent metrics: ASR={inc_metrics['agent_success_rate']:.3f}, "
        f"severe={inc_metrics['severe_rate']:.3f}, timeout={inc_metrics['timeout_rate']:.3f}, "
        f"infra={inc_metrics['infra_rate']:.3f}"
    )

    # 3. Evaluate candidates
    print(f"\n{'=' * 60}")
    print("  PHASE 3: Evaluate candidates")
    print("=" * 60)
    candidate_reports: list[dict] = []
    for i, cand in enumerate(candidates):
        pid = f"gepa_certify_c{i:02d}"
        print(
            f"\n  --- Candidate {i} (exploration_score={cand.get('exploration_score', '?')}) ---"
        )
        cand_eval = evaluate_persona(
            pid, cand["persona_text"], args, args.domains, task_pool
        )
        cand_metrics = compute_metrics(cand_eval["task_results"])
        paired = paired_comparison(cand_eval["task_results"], inc_eval["task_results"])

        gates_with_inc = {**gates, "incumbent_asr": inc_metrics["agent_success_rate"]}
        passed, reasons = check_gates(cand_metrics, paired, gates_with_inc)

        report = {
            "candidate_idx": i,
            "exploration_score": cand.get("exploration_score"),
            "source_run": cand.get("source_run"),
            "metrics": cand_metrics,
            "paired": {k: v for k, v in paired.items() if k != "task_details"},
            "paired_details": paired["task_details"],
            "gates_passed": passed,
            "gate_failures": reasons,
        }
        candidate_reports.append(report)

        status = "PASSED" if passed else "FAILED"
        print(
            f"  [{status}] ASR={cand_metrics['agent_success_rate']:.3f}, "
            f"severe={cand_metrics['severe_rate']:.3f}, "
            f"timeout={cand_metrics['timeout_rate']:.3f}, "
            f"infra={cand_metrics['infra_rate']:.3f}, "
            f"paired W/L/T={paired['wins']}/{paired['losses']}/{paired['ties']}"
        )
        if reasons:
            for r in reasons:
                print(f"    GATE FAIL: {r}")

    # 4. Select certified best
    print(f"\n{'=' * 60}")
    print("  PHASE 4: Select certified best")
    print("=" * 60)

    passed_reports = [r for r in candidate_reports if r["gates_passed"]]
    certified_best = None
    if passed_reports:
        passed_reports.sort(key=lambda r: r["paired"]["net_wins"], reverse=True)
        certified_best = passed_reports[0]
        ci = certified_best["candidate_idx"]
        print(f"  Certified best: candidate {ci}")
        print(
            f"    net_wins={certified_best['paired']['net_wins']}, "
            f"ASR={certified_best['metrics']['agent_success_rate']:.3f}, "
            f"severe={certified_best['metrics']['severe_rate']:.3f}"
        )

        best_text = candidates[ci]["persona_text"]
        best_path = out_dir / "certified_best_persona.md"
        best_path.write_text(best_text)
        materialize("gepa_certified_best", best_text)
        print(f"    Persona -> {best_path}")
        print(f"    Skill   -> {PERSONAS_DIR / 'opt_gepa_certified_best.skill'}")
    else:
        print("  No candidate passed all gates.")
        print("  Consider relaxing gates or running more GEPA iterations.")

    # 5. Write report
    report = {
        "incumbent": {
            "path": args.incumbent,
            "metrics": inc_metrics,
        },
        "gates": gates,
        "candidates": candidate_reports,
        "certified_best_idx": certified_best["candidate_idx"]
        if certified_best
        else None,
        "config": {
            "run_dirs": args.run_dirs,
            "top_k": args.top_k,
            "num_trials": args.num_trials,
            "agent_llm": args.agent_llm,
            "user_type": args.user_type,
            "domains": args.domains,
        },
    }
    report_path = out_dir / "certification_report.json"
    write_json(report_path, report)
    print(f"\n  Report -> {report_path}")


if __name__ == "__main__":
    main()
