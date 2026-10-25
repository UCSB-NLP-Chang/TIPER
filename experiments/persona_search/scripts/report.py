"""Final ranked output: hardest valid personas first (plan §10).

Emits valid personas sorted by ascending avg_success_rate (lowest = hardest for
the GPT-5.4 agent), with persona_id, persona_text, avg_success_rate,
num_evaluations, parent_id, mutation_direction, judge_status.

    .venv/bin/python experiments/persona_search/scripts/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LIB_PATH,
    PERSONA_OPT_DIR,
    now_iso,
    read_lib,
    write_json,
)


def _md(ranked: list[dict]) -> str:
    lines = [
        "# Persona Optimization — Final Report",
        "",
        f"_Generated {now_iso()}._ Hardest personas first (lowest agent success rate).",
        "",
        "| rank | persona_id | avg_success | evals | parent | direction |",
        "|-----:|------------|------------:|------:|--------|-----------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['persona_id']} | {r['avg_success_rate']:.3f} | "
            f"{r['num_evaluations']} | {r.get('parent_id') or '-'} | "
            f"{r.get('mutation_direction') or '-'} |"
        )
    lines.append("")
    for i, r in enumerate(ranked, 1):
        lines += [
            f"## {i}. {r['persona_id']} — success {r['avg_success_rate']:.3f}",
            "",
            f"- evaluations: {r['num_evaluations']}  | parent: {r.get('parent_id') or '-'}  "
            f"| direction: {r.get('mutation_direction') or '-'}  "
            f"| judge: {r.get('status')} (severe_rate={r.get('judge_severe_failure_rate')})",
            "",
            "```",
            r["persona_text"].rstrip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Final ranked persona report (plan §10).")
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--out", default=str(PERSONA_OPT_DIR / "final_report.json"))
    ap.add_argument("--md", default=str(PERSONA_OPT_DIR / "final_report.md"))
    args = ap.parse_args()

    records = read_lib(args.lib)
    valid = [
        r
        for r in records
        if r.get("status") == "valid" and r.get("avg_success_rate") is not None
    ]
    valid.sort(
        key=lambda r: (r["avg_success_rate"], r.get("created_at", ""), r["persona_id"])
    )

    ranked = [
        {
            "persona_id": r["persona_id"],
            "avg_success_rate": r["avg_success_rate"],
            "num_evaluations": r["num_evaluations"],
            "parent_id": r.get("parent_id"),
            "mutation_direction": r.get("mutation_direction"),
            "status": r.get("status"),
            "judge_severe_failure_rate": r.get("judge_severe_failure_rate"),
            "persona_text": r["persona_text"],
        }
        for r in valid
    ]
    write_json(
        args.out, {"generated_at": now_iso(), "n_valid": len(ranked), "ranked": ranked}
    )
    Path(args.md).write_text(_md(ranked), encoding="utf-8")

    print(f"{len(ranked)} valid personas (hardest first):")
    for i, r in enumerate(ranked[:10], 1):
        print(
            f"  {i:>2}. {r['persona_id']}  succ={r['avg_success_rate']:.3f}  "
            f"dir={r.get('mutation_direction') or '-'}"
        )
    print(f"\nreport -> {args.out}\nmarkdown -> {args.md}")


if __name__ == "__main__":
    main()
