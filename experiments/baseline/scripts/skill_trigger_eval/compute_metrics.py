#!/usr/bin/env python3
"""
Stage 3: Compute metrics from labeled pairs.

Two metric sets:
1. Candidate-set metrics (debug): raw TP/FP/FN/TN on candidates
2. Population-weighted metrics (report): adjusted using blind audit leakage

Also computes:
- Trigger rate calibration
- Boundary error rates
- Slice metrics by skill × {capability, emotion, domain, preset, turn_bucket}
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

IN_FILE = DATA_ROOT / "data/skill_eval/labeled_pairs.jsonl"
REPORT_CANDIDATE = DATA_ROOT / "data/skill_eval/report_candidate.json"
REPORT_POPULATION = DATA_ROOT / "data/skill_eval/report_population.json"

CANONICAL_SKILLS = [
    "frustration-escalation",
    "jargon-hit",
    "resolution",
    "transfer-pushback",
]

# Decision pairs for total counts
DP_FILE = DATA_ROOT / "data/skill_eval/decision_pairs.jsonl"


def canonicalize_skill(skill: str) -> str:
    if skill == "resolution-banking":
        return "resolution"
    return skill


def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def bootstrap_ci(values, stat_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(stat_fn(sample))
    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def main():
    labeled = []
    with open(IN_FILE) as f:
        for line in f:
            labeled.append(json.loads(line))

    if not labeled:
        print("No labeled pairs found.")
        return

    # Separate candidates and blind audit
    candidates = [p for p in labeled if p.get("sample_source") == "candidate"]
    blind_audit = [p for p in labeled if p.get("sample_source") == "blind_audit"]

    print(f"Stage 3: Compute Metrics")
    print(f"{'=' * 50}")
    print(f"Total labeled: {len(labeled)}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Blind audit: {len(blind_audit)}")
    print()

    # ------------------------------------------------------------------
    # 1. Candidate-set metrics (debug)
    # ------------------------------------------------------------------
    candidate_metrics = {}
    for skill in CANONICAL_SKILLS:
        skill_pairs = [p for p in candidates if canonicalize_skill(p["skill"]) == skill]
        tp = fp = fn = tn = 0
        for p in skill_pairs:
            model = p.get("model_triggered", False)
            silver = (p.get("label") or {}).get("should_trigger", False)
            if model and silver:
                tp += 1
            elif model and not silver:
                fp += 1
            elif not model and silver:
                fn += 1
            else:
                tn += 1

        precision, recall, f1 = compute_prf(tp, fp, fn)
        candidate_metrics[skill] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n": len(skill_pairs),
            "note": "NOT population estimates — candidate-set only",
        }

    print("=== Candidate-Set Metrics (DEBUG) ===")
    print(
        f"{'Skill':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'P':>6} {'R':>6} {'F1':>6}"
    )
    print("-" * 70)
    for skill in CANONICAL_SKILLS:
        m = candidate_metrics[skill]
        print(
            f"  {skill:<23} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
        )

    # Save candidate report
    with open(REPORT_CANDIDATE, "w") as f:
        json.dump(
            {"metrics": candidate_metrics, "n_candidates": len(candidates)}, f, indent=2
        )

    # ------------------------------------------------------------------
    # 2. Blind audit analysis
    # ------------------------------------------------------------------
    print("\n=== Blind Audit Results ===")
    blind_by_skill = defaultdict(list)
    for p in blind_audit:
        blind_by_skill[canonicalize_skill(p["skill"])].append(p)

    blind_positive_rates = {}
    for skill in CANONICAL_SKILLS:
        pool = blind_by_skill[skill]
        positives = sum(
            1 for p in pool if (p.get("label") or {}).get("should_trigger", False)
        )
        rate = positives / len(pool) if pool else 0
        blind_positive_rates[skill] = rate
        print(f"  {skill}: {positives}/{len(pool)} positive ({rate * 100:.1f}%)")
        if rate > 0.03:
            print(
                f"    ⚠️ High leakage rate! Rule screener may be missing cases. Consider Stage 1B."
            )

    # ------------------------------------------------------------------
    # 3. Population-weighted metrics (report)
    # ------------------------------------------------------------------
    # Load total pair counts from decision_pairs
    total_by_skill = defaultdict(int)
    model_triggered_by_skill = defaultdict(int)
    if DP_FILE.exists():
        with open(DP_FILE) as f:
            for line in f:
                p = json.loads(line)
                cs = canonicalize_skill(p["skill"])
                total_by_skill[cs] += 1
                if p["model_triggered"]:
                    model_triggered_by_skill[cs] += 1

    # Load screened negative counts
    screened_file = DATA_ROOT / "data/skill_eval/screened_pairs.jsonl"
    screened_negative_by_skill = defaultdict(int)
    if screened_file.exists():
        with open(screened_file) as f:
            for line in f:
                p = json.loads(line)
                if p["screen_score"] < 0.3:
                    screened_negative_by_skill[canonicalize_skill(p["skill"])] += 1

    print("\n=== Population-Weighted Metrics (REPORT) ===")
    population_metrics = {}
    for skill in CANONICAL_SKILLS:
        cm = candidate_metrics[skill]
        tp = cm["tp"]
        fp = cm["fp"]
        fn_candidate = cm["fn"]
        blind_rate = blind_positive_rates.get(skill, 0)
        n_screened_neg = screened_negative_by_skill.get(skill, 0)

        estimated_missed_fn = blind_rate * n_screened_neg
        adjusted_fn = fn_candidate + estimated_missed_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        adjusted_recall = tp / (tp + adjusted_fn) if (tp + adjusted_fn) > 0 else 0
        adjusted_f1 = (
            (2 * precision * adjusted_recall / (precision + adjusted_recall))
            if (precision + adjusted_recall) > 0
            else 0
        )

        model_rate = (
            model_triggered_by_skill[skill] / total_by_skill[skill]
            if total_by_skill[skill] > 0
            else 0
        )

        population_metrics[skill] = {
            "precision": round(precision, 4),
            "adjusted_recall": round(adjusted_recall, 4),
            "adjusted_f1": round(adjusted_f1, 4),
            "blind_positive_rate": round(blind_rate, 4),
            "estimated_missed_fn": round(estimated_missed_fn, 1),
            "screened_negatives": n_screened_neg,
            "model_trigger_rate": round(model_rate, 4),
            "total_pairs": total_by_skill[skill],
        }

        print(f"  {skill}:")
        print(f"    Precision: {precision:.3f}")
        print(
            f"    Adj. Recall: {adjusted_recall:.3f} (blind leak={blind_rate:.3f}, est missed FN={estimated_missed_fn:.0f})"
        )
        print(f"    Adj. F1: {adjusted_f1:.3f}")
        print(
            f"    Model trigger rate: {model_rate:.4f} ({model_triggered_by_skill[skill]}/{total_by_skill[skill]})"
        )

    with open(REPORT_POPULATION, "w") as f:
        json.dump(
            {
                "metrics": population_metrics,
                "blind_audit_results": {
                    skill: {
                        "positive_rate": blind_positive_rates.get(skill, 0),
                        "n": len(blind_by_skill[skill]),
                    }
                    for skill in CANONICAL_SKILLS
                },
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 4. Trigger rate calibration
    # ------------------------------------------------------------------
    print("\n=== Trigger Rate Calibration ===")
    print(f"{'Skill':<25} {'Model Rate':>10} {'Silver Rate':>12} {'Direction':>15}")
    print("-" * 65)
    for skill in CANONICAL_SKILLS:
        model_rate = population_metrics[skill]["model_trigger_rate"]
        silver_positives = sum(
            1
            for p in candidates + blind_audit
            if canonicalize_skill(p["skill"]) == skill
            and (p.get("label") or {}).get("should_trigger", False)
        )
        silver_total = sum(
            1
            for p in candidates + blind_audit
            if canonicalize_skill(p["skill"]) == skill
        )
        silver_rate = silver_positives / silver_total if silver_total > 0 else 0

        if silver_rate > 0 and model_rate < silver_rate * 0.5:
            direction = "severe under"
        elif model_rate < silver_rate:
            direction = "under-trigger"
        elif model_rate > silver_rate * 1.5:
            direction = "over-trigger"
        else:
            direction = "calibrated"

        print(f"  {skill:<23} {model_rate:>10.4f} {silver_rate:>12.4f} {direction:>15}")

    # ------------------------------------------------------------------
    # 5. Slice metrics
    # ------------------------------------------------------------------
    print("\n=== Slice Metrics (by skill × preset) ===")
    slices = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for p in candidates:
        skill = canonicalize_skill(p["skill"])
        preset = p.get("preset_group", "unknown")
        model = p.get("model_triggered", False)
        silver = (p.get("label") or {}).get("should_trigger", False)
        key = f"{skill}|{preset}"
        if model and silver:
            slices[key]["tp"] += 1
        elif model and not silver:
            slices[key]["fp"] += 1
        elif not model and silver:
            slices[key]["fn"] += 1
        else:
            slices[key]["tn"] += 1

    for key in sorted(slices.keys()):
        s = slices[key]
        p, r, f = compute_prf(s["tp"], s["fp"], s["fn"])
        if s["tp"] + s["fp"] + s["fn"] > 0:
            print(
                f"  {key:<45} TP={s['tp']:>3} FP={s['fp']:>3} FN={s['fn']:>3} P={p:.2f} R={r:.2f} F1={f:.2f}"
            )

    print(f"\nReports saved:")
    print(f"  Candidate: {REPORT_CANDIDATE}")
    print(f"  Population: {REPORT_POPULATION}")


if __name__ == "__main__":
    main()
