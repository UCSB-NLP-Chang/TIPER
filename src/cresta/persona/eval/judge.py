"""Validate a tested persona via the existing LLM judge (plan §5).

Wraps the baseline intent-preservation judge (run mode, OpenAI Batch API)
and applies the validity rule:

    severe_rate = severe_failures / n_scored      (n_scored = ALL non-infra sims)
    status = "illegal" if severe_rate > THRESHOLD else "valid"   (THRESHOLD = 0.05)

Principled pruning (default): a *severe failure* (the persona cheated the agent
into failing) can only occur in a sim the agent actually FAILED — a fully
successful sim (reward == 1.0) is "ok" by construction. So we judge ONLY the
failed sims (reward < success-threshold), capped at ``--max-failures`` per persona
(worst reward first), and divide the severe count by the FULL scored-sim count so
the 0.05 threshold keeps its original meaning. Pass ``--judge-all`` to judge every
sim instead.

The judge never sees or alters the success score.

    .venv/bin/python -m cresta.persona.eval.judge --persona-id p0001 --round 1
    .venv/bin/python -m cresta.persona.eval.judge --persona-id p0001 --round 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cresta.persona.core._common import (
    REPO_ROOT,
    is_infra_broken,
    read_json,
    round_dir,
    write_json,
)

JUDGE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "baseline"
    / "scripts"
    / "judge_user_intent_preservation.py"
)


def _read_per_sim(out_dir: Path) -> list[dict]:
    path = out_dir / "per_sim.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge(rows: list[dict]) -> tuple[int, int]:
    """Return (severe, judged) across all sims (no_vote excluded from judged)."""
    judged = [r for r in rows if r.get("verdict") != "no_vote"]
    severe = sum(1 for r in judged if r.get("verdict") == "severe_failure")
    return severe, len(judged)


def _reward(sim: dict):
    return (sim.get("reward_info") or {}).get("reward")


def _scored_and_failed(results_path: str, success_thr: float):
    """(n_scored, [failed sims sorted worst-first]) for one domain's results.json.

    n_scored counts non-infra sims with a numeric reward (the denominator the
    success score itself uses). A sim is 'failed' if reward < success_thr.
    """
    data = read_json(results_path)
    scored, failed = 0, []
    for sim in data.get("simulations", []):
        if is_infra_broken(sim):
            continue
        r = _reward(sim)
        if r is None:
            continue
        scored += 1
        if float(r) < success_thr:
            failed.append(sim)
    failed.sort(key=lambda s: float(_reward(s)))
    return scored, failed, data


def main():
    ap = argparse.ArgumentParser(description="Judge a persona valid/illegal (plan §5).")
    ap.add_argument("--persona-id", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--eval-dir", default=None, help="defaults to rounds/round_NN/eval/<id>/"
    )
    ap.add_argument(
        "--out", default=None, help="defaults to rounds/round_NN/judge/<id>/"
    )
    ap.add_argument("--maj", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="cap judged failed sims per persona (worst reward first)",
    )
    ap.add_argument(
        "--success-threshold",
        type=float,
        default=0.999,
        help="reward >= this counts as a success (skipped by the judge)",
    )
    ap.add_argument(
        "--judge-all",
        action="store_true",
        help="judge every sim (disable the failed-only pruning)",
    )
    ap.add_argument("--max-wait", type=int, default=1800)
    ap.add_argument(
        "--mode",
        choices=["sync", "run"],
        default="sync",
        help="sync = fast concurrent chat-completions (default); run = OpenAI Batch API",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eval_dir = (
        Path(args.eval_dir)
        if args.eval_dir
        else round_dir(args.round) / "eval" / args.persona_id
    )
    out = (
        Path(args.out)
        if args.out
        else round_dir(args.round) / "judge" / args.persona_id
    )
    out.mkdir(parents=True, exist_ok=True)

    eval_summary = read_json(eval_dir / "persona_eval.json")
    results_paths = eval_summary.get("results_paths", {})

    # 1) Decide which sims to judge per domain.
    n_scored_total = 0
    per_domain_failed = {}  # dk -> (full_data, [failed sims])
    for dk, rpath in results_paths.items():
        n_scored, failed, data = _scored_and_failed(rpath, args.success_threshold)
        n_scored_total += n_scored
        per_domain_failed[dk] = (data, failed)

    n_failed_total = sum(len(v[1]) for v in per_domain_failed.values())

    if args.judge_all:
        # judge everything (legacy behavior) — pass original results.json through.
        selected = {dk: None for dk in results_paths}  # None => use original file
        truncated = False
    else:
        # Global cap: take the worst `max-failures` failed sims across domains.
        flat = [(dk, sim) for dk, (_, fs) in per_domain_failed.items() for sim in fs]
        flat.sort(key=lambda x: float(_reward(x[1])))
        capped = flat[: args.max_failures] if args.max_failures else flat
        truncated = bool(args.max_failures) and n_failed_total > args.max_failures
        selected = {}
        for dk in results_paths:
            sims = [sim for d, sim in capped if d == dk]
            selected[dk] = sims  # may be []

    # 2) Run the judge on the selected sims.
    all_rows = []
    n_judged_input = 0
    for dk, rpath in results_paths.items():
        dom_out = out / dk
        dom_out.mkdir(parents=True, exist_ok=True)

        if args.judge_all:
            judge_input = rpath
            sims_here = per_domain_failed[dk][0].get("simulations", [])
        else:
            sims_here = selected[dk]
            if not sims_here:
                continue  # nothing to judge in this domain
            data = per_domain_failed[dk][0]
            filtered = dict(data)
            filtered["simulations"] = sims_here
            fpath = dom_out / "filtered_results.json"
            write_json(fpath, filtered)
            judge_input = str(fpath)

        n_judged_input += len(sims_here)

        if args.dry_run:
            rows = [
                {
                    "task_id": s["task_id"],
                    "trial": s.get("trial", 0),
                    "verdict": "ok",
                    "intent_preserved": True,
                    "violations": [],
                }
                for s in sims_here
            ]
            with open(dom_out / "per_sim.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            all_rows.extend(rows)
            continue

        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(JUDGE_SCRIPT),
            args.mode,
            "--results",
            str(judge_input),
            "--model",
            args.judge_model,
            "--maj",
            str(args.maj),
            "--out",
            str(dom_out),
        ]
        if args.mode == "sync":
            cmd += ["--concurrency", str(args.concurrency)]
        else:
            cmd += ["--max-wait", str(args.max_wait)]
        print(f"[{dk}] judging {len(sims_here)} sim(s) [{args.mode}] -> {dom_out}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        all_rows.extend(_read_per_sim(dom_out))

    # 3) Apply the validity rule. Denominator = ALL scored sims, so successful
    #    sims (not judged) correctly count as non-severe.
    severe, judged = _merge(all_rows)
    denom = n_scored_total if not args.judge_all else (judged or n_scored_total)
    severe_rate = (severe / denom) if denom else None
    status = (
        "illegal"
        if (severe_rate is not None and severe_rate > args.threshold)
        else "valid"
    )

    status_obj = {
        "persona_id": args.persona_id,
        "round": args.round,
        "severe_failures": severe,
        "n_judged": judged,
        "n_scored": n_scored_total,
        "n_failed_total": n_failed_total,
        "max_failures": args.max_failures,
        "truncated": truncated,
        "judge_all": args.judge_all,
        "severe_failure_rate": severe_rate,
        "threshold": args.threshold,
        "status": status,
        "dry_run": args.dry_run,
    }
    write_json(out / "judge_status.json", status_obj)
    trunc = "  [TRUNCATED: >cap failures, some unjudged]" if truncated else ""
    print(
        f"\nsevere={severe}  judged_failed={judged}/{n_failed_total}  "
        f"scored={n_scored_total}  rate={severe_rate}  -> status={status}{trunc}"
    )
    print(f"judge -> {out / 'judge_status.json'}")


if __name__ == "__main__":
    main()
