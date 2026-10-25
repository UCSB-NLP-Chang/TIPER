#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/.."  # repo root

echo "=== Stage 0: Extract Decision Points ==="
python3 scripts/skill_trigger_eval/extract_decision_points.py

echo ""
echo "=== Stage 0.5: Build Decision Pairs ==="
python3 scripts/skill_trigger_eval/build_decision_pairs.py

echo ""
echo "=== Stage 1A: Rule-Based Screening ==="
python3 scripts/skill_trigger_eval/apply_rule_screeners.py

echo ""
echo "=== Stage 1C: Merge Screen Results ==="
python3 scripts/skill_trigger_eval/merge_screen_results.py

echo ""
echo "=== Stage 1.5: Budget-Controlled Sampling ==="
python3 scripts/skill_trigger_eval/sample_eval_pairs.py

echo ""
echo "=== Stage 2: GPT-5.5 Labeling ==="
python3 scripts/skill_trigger_eval/label_pairs.py

echo ""
echo "=== Stage 3: Compute Metrics ==="
python3 scripts/skill_trigger_eval/compute_metrics.py

echo ""
echo "=== Stage 4: Error Analysis ==="
python3 scripts/skill_trigger_eval/error_analysis.py

echo ""
echo "=== Pipeline Complete ==="
echo "Reports:"
echo "  data/skill_eval/report_candidate.json"
echo "  data/skill_eval/report_population.json"
echo "  data/skill_eval/error_examples/"
