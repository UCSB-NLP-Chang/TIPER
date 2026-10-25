#!/usr/bin/env python3
"""Audit whether a persona remains COOPERATIVE on test-set simulations.

Runs the existing judge_user_intent_preservation.py (sync mode) on a results.json
and produces a structured report of violation types, rates, and per-sim details.

This is INDEPENDENT of the GEPA optimization loop — it takes any results.json
(e.g. from a test-set cert run or fullset) and reports cooperative behavior stats.

Usage:
    # Judge a single results.json
    python experiments/persona_search/scripts/audit_cooperative_behavior.py \
        --results data/simulations/gepa_opt/.../results.json \
        --out /tmp/audit_telecom_best

    # Judge multiple results files (e.g. across domains)
    python experiments/persona_search/scripts/audit_cooperative_behavior.py \
        --results data/simulations/test_cert/telecom/results.json \
                  data/simulations/test_cert/airline/results.json \
        --out /tmp/audit_multi

    # Compare persona vs no-persona
    python experiments/persona_search/scripts/audit_cooperative_behavior.py \
        --results data/simulations/test_cert/telecom/persona/results.json \
        --baseline data/simulations/test_cert/telecom/no_persona/results.json \
        --out /tmp/audit_compare

    # Use existing judge results (skip re-judging)
    python experiments/persona_search/scripts/audit_cooperative_behavior.py \
        --judge-dir data/simulations/test_cert/telecom/judge \
        --results data/simulations/test_cert/telecom/results.json \
        --out /tmp/audit_reuse
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "baseline"
    / "scripts"
    / "judge_user_intent_preservation.py"
)


def _is_infra_broken(sim: dict) -> bool:
    ri = sim.get("reward_info") or {}
    if ri.get("error"):
        return True
    if sim.get("error"):
        return True
    if not sim.get("messages"):
        return True
    return False


def run_judge(
    results_path: str,
    out_dir: str,
    model: str = "gpt-5.2",
    maj: int = 1,
    concurrency: int = 16,
    failed_only: bool = False,
    max_sims: int | None = None,
) -> Path:
    """Run sync judge and return the output directory."""
    out = Path(out_dir)
    if (out / "summary.json").exists():
        print(f"  [judge] reusing existing results at {out}")
        return out

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        "sync",
        "--results",
        results_path,
        "--model",
        model,
        "--maj",
        str(maj),
        "--concurrency",
        str(concurrency),
        "--out",
        str(out),
    ]
    if max_sims:
        cmd += ["--max-sims", str(max_sims)]
    print(f"  [judge] {' '.join(cmd[-8:])}")
    subprocess.run(cmd, check=True)
    return out


def load_judge_results(judge_dir: Path) -> list[dict]:
    per_sim = judge_dir / "per_sim.jsonl"
    if not per_sim.exists():
        return []
    rows = []
    with open(per_sim) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results_meta(results_path: str) -> dict:
    """Load results.json and extract per-sim metadata."""
    data = json.loads(Path(results_path).read_text())
    sims = data.get("simulations", [])
    meta = {}
    for s in sims:
        if _is_infra_broken(s):
            continue
        key = (s.get("task_id", "?"), s.get("trial", 0))
        reward = (s.get("reward_info") or {}).get("reward")
        meta[key] = {
            "task_id": s.get("task_id"),
            "trial": s.get("trial", 0),
            "reward": reward,
            "n_messages": len(s.get("messages", [])),
            "termination": s.get("termination_reason", ""),
        }
    return meta


def analyze(judge_rows: list[dict], results_meta: dict | None = None) -> dict:
    """Produce a structured analysis from judge per_sim results."""
    total = len(judge_rows)
    verdicts = Counter(r.get("verdict", "unknown") for r in judge_rows)
    flagged = [
        r for r in judge_rows if r.get("verdict") in ("severe_failure", "violation")
    ]
    ok = [r for r in judge_rows if r.get("verdict") == "ok"]

    violation_types = Counter()
    violation_severity = Counter()
    violations_by_task = defaultdict(list)
    flagged_details = []
    critical_count = 0

    for r in flagged:
        task_id = r.get("task_id", "?")
        trial = r.get("trial", 0)
        has_critical = False
        for v in r.get("violations", []):
            vtype = v.get("type", "unknown")
            severity = v.get("severity", "unknown")
            violation_types[vtype] += 1
            violation_severity[severity] += 1
            if severity == "critical":
                has_critical = True
            violations_by_task[task_id].append(
                {
                    "trial": trial,
                    "type": vtype,
                    "subtype": v.get("subtype"),
                    "severity": severity,
                    "rule": v.get("rule", ""),
                    "evidence": v.get("evidence", ""),
                }
            )
        if has_critical:
            critical_count += 1
        detail = {
            "task_id": task_id,
            "trial": trial,
            "violations": r.get("violations", []),
            "intent_assessment": r.get("intent_assessment", ""),
        }
        if results_meta:
            key = (task_id, trial)
            if key in results_meta:
                detail["reward"] = results_meta[key]["reward"]
                detail["n_messages"] = results_meta[key]["n_messages"]
        flagged_details.append(detail)

    n_scored = total
    violation_rate = len(flagged) / n_scored if n_scored else 0.0
    critical_rate = critical_count / n_scored if n_scored else 0.0

    return {
        "total_sims": total,
        "verdicts": dict(verdicts),
        "violation_count": len(flagged),
        "violation_rate": violation_rate,
        "critical_count": critical_count,
        "critical_rate": critical_rate,
        "cooperative": critical_rate <= 0.05,
        "violation_types": dict(violation_types.most_common()),
        "violation_severity": dict(violation_severity),
        "tasks_with_violations": len(violations_by_task),
        "violations_by_task": dict(violations_by_task),
        "flagged_details": flagged_details,
    }


def print_report(analysis: dict, label: str = ""):
    header = f"COOPERATIVE BEHAVIOR AUDIT{f' — {label}' if label else ''}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    total = analysis["total_sims"]
    violations = analysis["violation_count"]
    v_rate = analysis["violation_rate"]
    critical = analysis["critical_count"]
    c_rate = analysis["critical_rate"]
    status = "COOPERATIVE" if analysis["cooperative"] else "NOT COOPERATIVE"

    print(f"  Sims judged:     {total}")
    print(f"  Violations:      {violations} ({v_rate:.1%})")
    print(f"  Critical:        {critical} ({c_rate:.1%})")
    print(f"  Status:          {status} (critical threshold: 5%)")
    print()

    if analysis["violation_types"]:
        print("  Violation types:")
        for vtype, count in analysis["violation_types"].items():
            print(f"    {vtype:40s} {count:3d}")
        print()

    if analysis.get("violation_severity"):
        print("  Severity breakdown:")
        for sev, count in analysis["violation_severity"].items():
            print(f"    {sev:40s} {count:3d}")
        print()

    if analysis["flagged_details"]:
        print(f"  Flagged sims (showing up to 10):")
        for d in analysis["flagged_details"][:10]:
            reward_str = f"  reward={d['reward']}" if "reward" in d else ""
            print(f"    {d['task_id']} trial={d['trial']}{reward_str}")
            for v in d.get("violations", []):
                sev = v.get("severity", "?")
                subtype = v.get("subtype", "")
                sub_str = f"/{subtype}" if subtype else ""
                print(f"      [{sev}] {v.get('type')}{sub_str}: {v.get('rule', '')}")
                ev = v.get("evidence", "")
                if ev:
                    print(f'        "{ev[:120]}"')
        print()

    print(f"{'=' * 60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Audit cooperative behavior on test-set simulations."
    )
    ap.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="path(s) to results.json files to judge",
    )
    ap.add_argument(
        "--baseline",
        nargs="*",
        default=None,
        help="optional no-persona baseline results.json for comparison",
    )
    ap.add_argument(
        "--judge-dir",
        default=None,
        help="reuse existing judge output dir (skip re-judging)",
    )
    ap.add_argument("--out", required=True, help="output directory for audit results")
    ap.add_argument("--model", default="gpt-5.2", help="judge model")
    ap.add_argument(
        "--maj",
        type=int,
        default=1,
        help="majority vote count (1=single vote, 3=maj@3)",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--failed-only",
        action="store_true",
        help="only judge agent-failed sims (cheaper but misses false positives)",
    )
    ap.add_argument("--max-sims", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_analyses = {}

    for rpath in args.results:
        label = Path(rpath).parent.name
        print(f"\n--- Judging: {rpath} (label={label}) ---")

        if args.judge_dir:
            judge_out = Path(args.judge_dir)
        else:
            judge_out = run_judge(
                rpath,
                str(out / f"judge_{label}"),
                model=args.model,
                maj=args.maj,
                concurrency=args.concurrency,
                max_sims=args.max_sims,
            )

        rows = load_judge_results(judge_out)
        if not rows:
            print(f"  WARNING: no judge results found at {judge_out}")
            continue

        meta = load_results_meta(rpath)
        analysis = analyze(rows, meta)
        print_report(analysis, label=label)
        all_analyses[label] = analysis

    if args.baseline:
        for bpath in args.baseline:
            label = f"baseline_{Path(bpath).parent.name}"
            print(f"\n--- Judging baseline: {bpath} (label={label}) ---")
            judge_out = run_judge(
                bpath,
                str(out / f"judge_{label}"),
                model=args.model,
                maj=args.maj,
                concurrency=args.concurrency,
                max_sims=args.max_sims,
            )
            rows = load_judge_results(judge_out)
            if rows:
                meta = load_results_meta(bpath)
                analysis = analyze(rows, meta)
                print_report(analysis, label=label)
                all_analyses[label] = analysis

    if len(all_analyses) > 1:
        print("\n" + "=" * 60)
        print("  COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"  {'Label':<30s} {'Sims':>5s} {'Viols':>6s} {'Crit':>5s} {'CritRate':>8s} {'Status':>15s}"
        )
        print(f"  {'-' * 30} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 8} {'-' * 15}")
        for label, a in all_analyses.items():
            status = "COOPERATIVE" if a["cooperative"] else "NOT COOPERATIVE"
            print(
                f"  {label:<30s} {a['total_sims']:>5d} {a['violation_count']:>6d} "
                f"{a['critical_count']:>5d} {a['critical_rate']:>7.1%} {status:>15s}"
            )
        print()

    report_path = out / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"Full report -> {report_path}")


if __name__ == "__main__":
    main()
