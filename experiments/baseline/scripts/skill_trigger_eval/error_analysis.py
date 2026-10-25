#!/usr/bin/env python3
"""
Stage 4: Error Analysis — extract top FP/FN/boundary examples per skill.
"""

import json
import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/labeled_pairs.jsonl"
OUT_DIR = DATA_ROOT / "data/skill_eval/error_examples"

CANONICAL_SKILLS = [
    "frustration-escalation",
    "jargon-hit",
    "resolution",
    "transfer-pushback",
]


def canonicalize_skill(skill: str) -> str:
    if skill == "resolution-banking":
        return "resolution"
    return skill


def format_example(pair: dict) -> str:
    label = pair.get("label") or {}
    lines = [
        f"**Pair ID**: `{pair['pair_id']}`",
        f"**Skill**: {pair['skill']}",
        f"**Domain**: {pair['domain']} | **Preset**: {pair.get('preset_group', '?')}",
        f"**Capability**: {pair['capability']} | **Emotion**: {pair['emotion']}",
        f"**Turn**: {pair['turn_idx']}/{pair.get('n_user_turns', '?')}",
        f"**Model triggered**: {pair.get('model_triggered', '?')}",
        f"**Silver label**: should_trigger={label.get('should_trigger', '?')}",
        f"**Confidence**: {label.get('confidence', '?')}",
        f"**Method**: {pair.get('label_method', '?')}",
        "",
        f"**Agent message** (truncated):",
        f"> {pair.get('agent_message', '')[:500]}",
        "",
        f"**Condition evidence**: {label.get('condition_evidence', 'N/A')}",
        f"**Persona reason**: {label.get('persona_reason', 'N/A')}",
    ]
    if label.get("majority_vote"):
        lines.append(f"**Majority vote**: {label['majority_vote']}")
    return "\n".join(lines)


def main():
    labeled = []
    with open(IN_FILE) as f:
        for line in f:
            labeled.append(json.loads(line))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in labeled if p.get("sample_source") == "candidate"]

    print(f"Stage 4: Error Analysis")
    print(f"{'=' * 50}")

    for skill in CANONICAL_SKILLS:
        skill_pairs = [p for p in candidates if canonicalize_skill(p["skill"]) == skill]

        fps = []  # model triggered, silver says no
        fns = []  # model didn't trigger, silver says yes
        boundary_errors = []

        for p in skill_pairs:
            model = p.get("model_triggered", False)
            silver = (p.get("label") or {}).get("should_trigger", False)

            if model and not silver:
                fps.append(p)
            elif not model and silver:
                fns.append(p)

            if p.get("budget_tier") == "boundary" and model != silver:
                boundary_errors.append(p)

        # Sort by confidence (LOW first for more interesting examples)
        conf_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "FAILED": 3}
        fps.sort(
            key=lambda p: conf_order.get(
                (p.get("label") or {}).get("confidence", ""), 2
            )
        )
        fns.sort(
            key=lambda p: conf_order.get(
                (p.get("label") or {}).get("confidence", ""), 2
            )
        )

        # Write report
        report_lines = [f"# Error Analysis: {skill}\n"]
        report_lines.append(f"Total candidates: {len(skill_pairs)}")
        report_lines.append(f"False Positives: {len(fps)}")
        report_lines.append(f"False Negatives: {len(fns)}")
        report_lines.append(f"Boundary errors: {len(boundary_errors)}\n")

        report_lines.append("## False Positives (model triggered but shouldn't have)\n")
        for p in fps[:10]:
            report_lines.append(format_example(p))
            report_lines.append("\n---\n")

        report_lines.append(
            "\n## False Negatives (model didn't trigger but should have)\n"
        )
        for p in fns[:10]:
            report_lines.append(format_example(p))
            report_lines.append("\n---\n")

        report_lines.append("\n## Boundary Errors (±1 turn timing)\n")
        for p in boundary_errors[:10]:
            report_lines.append(format_example(p))
            report_lines.append("\n---\n")

        out_file = OUT_DIR / f"{skill.replace('-', '_')}.md"
        with open(out_file, "w") as f:
            f.write("\n".join(report_lines))

        print(
            f"  {skill}: {len(fps)} FP, {len(fns)} FN, {len(boundary_errors)} boundary → {out_file}"
        )

    # Summary
    summary_lines = ["# Error Analysis Summary\n"]
    for skill in CANONICAL_SKILLS:
        skill_pairs = [p for p in candidates if canonicalize_skill(p["skill"]) == skill]
        fps = sum(
            1
            for p in skill_pairs
            if p.get("model_triggered")
            and not (p.get("label") or {}).get("should_trigger", False)
        )
        fns = sum(
            1
            for p in skill_pairs
            if not p.get("model_triggered")
            and (p.get("label") or {}).get("should_trigger", False)
        )
        tps = sum(
            1
            for p in skill_pairs
            if p.get("model_triggered")
            and (p.get("label") or {}).get("should_trigger", False)
        )
        summary_lines.append(f"## {skill}")
        summary_lines.append(f"- TP: {tps}, FP: {fps}, FN: {fns}")
        p = tps / (tps + fps) if (tps + fps) > 0 else 0
        r = tps / (tps + fns) if (tps + fns) > 0 else 0
        summary_lines.append(f"- Precision: {p:.3f}, Recall: {r:.3f}\n")

    summary_file = OUT_DIR / "summary.md"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Summary → {summary_file}")


if __name__ == "__main__":
    main()
