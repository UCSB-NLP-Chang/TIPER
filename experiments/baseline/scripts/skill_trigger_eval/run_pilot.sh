#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/.."  # repo root

echo "=== Pilot A: Stage 0-1A (no API cost) ==="
python3 scripts/skill_trigger_eval/extract_decision_points.py
python3 scripts/skill_trigger_eval/build_decision_pairs.py
python3 scripts/skill_trigger_eval/apply_rule_screeners.py
python3 scripts/skill_trigger_eval/merge_screen_results.py

echo ""
echo "=== Pilot B: Rule Screener Coverage Check ==="
echo "Checking: what % of model-triggered pairs were independently flagged by rules..."
python3 -c "
import json
from collections import defaultdict

dp = {}
with open('data/skill_eval/decision_pairs.jsonl') as f:
    for line in f:
        p = json.loads(line)
        if p['model_triggered']:
            dp[p['pair_id']] = True

screened = {}
with open('data/skill_eval/rule_screened_pairs.jsonl') as f:
    for line in f:
        p = json.loads(line)
        if p['pair_id'] in dp:
            screened[p['pair_id']] = p

by_skill = defaultdict(lambda: {'total': 0, 'rule_hit': 0})
for pid, p in screened.items():
    skill = p['skill']
    by_skill[skill]['total'] += 1
    if p['screen_source'] != 'auto_trigger' and p['screen_score'] >= 0.3:
        by_skill[skill]['rule_hit'] += 1

print('Rule coverage on model-triggered pairs:')
for skill, c in sorted(by_skill.items()):
    rate = c['rule_hit'] / c['total'] * 100 if c['total'] > 0 else 0
    print(f'  {skill}: {c[\"rule_hit\"]}/{c[\"total\"]} ({rate:.1f}%)')
"

echo ""
echo "Pilot A+B complete (no API cost)."
echo "Next: run Pilot C (blind audit) with: python3 scripts/skill_trigger_eval/label_pairs.py --pilot --n=800"
