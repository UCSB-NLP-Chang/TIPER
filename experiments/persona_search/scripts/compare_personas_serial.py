#!/usr/bin/env python3
"""Serial difficulty certification across a persona progression.

Evaluates a fixed list of personas ONE AT A TIME (serial — avoids claude
subprocess oversubscription) on the same task pool with full judge, then prints
the agent-success-rate / difficulty progression. Intended to certify the climb:

    no_persona (cooperative floor) -> seed persona -> GEPA best candidate

Reuses the certification eval + scoring from certify_gepa_candidates.py so the
numbers are directly comparable to a real certification run (num_trials, judge,
outcome taxonomy, infra-drop all identical).

    python experiments/persona_search/scripts/compare_personas_serial.py \
        --run-dir data/persona_opt/gepa_runs/p0003_aggressive_20260606_145857 \
        --best-idx 3 --seed-idx 0 \
        --no-persona data/personas/opt_no_persona_baseline.skill \
        --domains telecom --num-trials 2 --max-concurrency 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_rep_tasks, read_json, write_json  # noqa: E402
from certify_gepa_candidates import (  # noqa: E402
    compute_metrics,
    evaluate_persona,
    load_full_base_tasks,
)

G, R, Y, C, BOLD, DIM, NC = (
    "\033[0;32m",
    "\033[0;31m",
    "\033[1;33m",
    "\033[0;36m",
    "\033[1m",
    "\033[2m",
    "\033[0m",
)


def main():
    ap = argparse.ArgumentParser(description="Serial persona difficulty comparison.")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="GEPA run dir with candidates.json (for seed/best texts)",
    )
    ap.add_argument(
        "--best-idx",
        type=int,
        default=3,
        help="candidates.json index of the GEPA best candidate",
    )
    ap.add_argument(
        "--seed-idx",
        type=int,
        default=0,
        help="candidates.json index of the seed (program 0)",
    )
    ap.add_argument(
        "--extra-idx",
        type=int,
        nargs="*",
        default=[],
        help="extra candidates.json indices to cert as intermediate "
        "rows (e.g. the best candidate's non-seed ancestors); "
        "inserted between seed and candidate, labeled idx<N>",
    )
    ap.add_argument(
        "--only-extra",
        action="store_true",
        default=False,
        help="cert ONLY the --extra-idx personas (skip the "
        "no_persona/seed/candidate trio, e.g. when already certified)",
    )
    ap.add_argument(
        "--no-persona-only",
        action="store_true",
        default=False,
        help="cert ONLY the cooperative no_persona baseline (e.g. to get a "
        "comparable difficulty anchor on the valset)",
    )
    ap.add_argument(
        "--no-persona",
        required=True,
        help="Path to the cooperative no_persona .skill file",
    )
    ap.add_argument("--out-dir", default=None)

    # Model / infra (defaults match the GEPA run + cert script)
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
    ap.add_argument("--domains", nargs="+", default=["telecom"])
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--use-full-tasks", action="store_true", default=False)

    # Judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-maj", type=int, default=3)

    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "progression_cert"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task pool
    if args.use_full_tasks:
        task_pool = load_full_base_tasks(args.domains)
        task_label = "full-base"
    else:
        task_pool = load_rep_tasks()
        task_label = "reprep"
    task_counts = {dk: len(ids) for dk, ids in task_pool.items() if ids}
    print(
        f"{BOLD}Task pool ({task_label}):{NC} {task_counts}  "
        f"trials={args.num_trials}  concurrency={args.max_concurrency}"
    )

    # Build the 3-point progression
    cands = read_json(Path(args.run_dir) / "candidates.json")
    if args.no_persona_only:
        personas = [("no_persona", Path(args.no_persona).read_text())]
    elif args.only_extra:
        personas = [(f"idx{i}", cands[i]["persona_text"]) for i in args.extra_idx]
    else:
        personas = [
            ("no_persona", Path(args.no_persona).read_text()),
            ("seed", cands[args.seed_idx]["persona_text"]),
        ]
        for i in args.extra_idx:
            personas.append((f"idx{i}", cands[i]["persona_text"]))
        personas.append(("candidate", cands[args.best_idx]["persona_text"]))
    print(
        f"{BOLD}Progression:{NC} "
        + " -> ".join(f"{lbl}({len(txt)}ch)" for lbl, txt in personas)
        + "\n"
    )

    rows = []
    for label, text in personas:
        print(f"{BOLD}{C}{'=' * 60}{NC}")
        print(f"{BOLD}Evaluating: {label}{NC}")
        print(f"{BOLD}{C}{'=' * 60}{NC}")
        pid = f"progcert_{label}"
        ev = evaluate_persona(pid, text, args, args.domains, task_pool)
        m = compute_metrics(ev["task_results"])
        m["label"] = label
        m["difficulty"] = 1.0 - m["agent_success_rate"]
        rows.append(m)
        print(
            f"  -> ASR={m['agent_success_rate']:.3f}  "
            f"difficulty={m['difficulty']:.3f}  "
            f"severe={m['severe_rate']:.3f}  timeout={m['timeout_rate']:.3f}  "
            f"infra={m['infra_rate']:.3f}  n_scored={m['n_scored']}\n"
        )

    # Progression table
    print(f"\n{BOLD}{C}{'=' * 72}{NC}")
    print(
        f"{BOLD}DIFFICULTY PROGRESSION{NC}  {DIM}(difficulty = 1 - agent_success_rate){NC}"
    )
    print(f"{BOLD}{C}{'=' * 72}{NC}")
    print(
        f"  {'persona':<12} {'difficulty':>11} {'ASR':>7} {'severe':>8} "
        f"{'timeout':>8} {'infra':>7} {'n_scored':>9}"
    )
    print(f"  {'-' * 12} {'-' * 11} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 9}")
    base_diff = rows[0]["difficulty"] if rows else 0
    for m in rows:
        d = m["difficulty"]
        dcol = G if d > 0.4 else (Y if d > 0.2 else DIM)
        scol = G if m["severe_rate"] <= 0.05 else R
        delta = (
            f"  {DIM}(+{(d - base_diff) * 100:.0f}pp){NC}"
            if m["label"] != "no_persona"
            else ""
        )
        print(
            f"  {m['label']:<12} {dcol}{d * 100:>10.1f}%{NC} "
            f"{m['agent_success_rate'] * 100:>6.1f}% "
            f"{scol}{m['severe_rate'] * 100:>7.1f}%{NC} "
            f"{m['timeout_rate'] * 100:>7.1f}% {m['infra_rate'] * 100:>6.1f}% "
            f"{m['n_scored']:>9}{delta}"
        )
    print()

    report = {
        "task_label": task_label,
        "task_counts": task_counts,
        "config": {
            "run_dir": args.run_dir,
            "best_idx": args.best_idx,
            "seed_idx": args.seed_idx,
            "num_trials": args.num_trials,
            "agent_llm": args.agent_llm,
            "user_type": args.user_type,
            "domains": args.domains,
            "max_concurrency": args.max_concurrency,
        },
        "progression": rows,
    }
    rp = out_dir / "progression_report.json"
    write_json(rp, report)
    print(f"{DIM}Report -> {rp}{NC}")


if __name__ == "__main__":
    main()
