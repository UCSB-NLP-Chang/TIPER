#!/usr/bin/env bash
# Wait for the pair-run certification to finish, then run naturalness + assemble
# the final difficulty×naturalness×validity table. Launched in background so the
# 5h cert doesn't need babysitting; emits one completion signal at the end.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
OUT=data/persona_opt/pgepa_runs/cert_pair_full
REPORT=$OUT/certification_report.json
LOG=/tmp/finish_cert_naturalness.log
: > "$LOG"

echo "[finish] waiting for cert report: $REPORT" | tee -a "$LOG"
# poll until cert process is gone AND the report file exists
while pgrep -f "cert_pair_full" | grep -qv grep; do
  sleep 120
done
echo "[finish] cert process exited @ $(date '+%H:%M')" | tee -a "$LOG"
sleep 10
if [ ! -f "$REPORT" ]; then
  echo "[finish] ERROR: cert finished but no report at $REPORT" | tee -a "$LOG"
  exit 1
fi

echo "[finish] running naturalness on all cert personas..." | tee -a "$LOG"
$PY experiments/persona_search/scripts/score_naturalness.py \
  --cert-report "$REPORT" --maj 3 --max-sims-per-persona 60 --workers 16 \
  >> "$LOG" 2>&1

echo "[finish] assembling final table..." | tee -a "$LOG"
$PY - "$REPORT" "$OUT/naturalness.json" "$OUT/FINAL_pair_cert.json" <<'PYEOF' 2>>"$LOG"
import json, sys
rep = json.load(open(sys.argv[1]))
nat = {r["pid"]: r for r in json.load(open(sys.argv[2]))}
rows = []
inc = rep["incumbent"]["metrics"]
rows.append({"pid":"gepa_certify_incumbent","role":"incumbent(seed p0003 gen00)",
             "source":"seed","exploration_score":None,
             "full_diff":round(1-inc["agent_success_rate"],4),
             "severe_rate":round(inc["severe_rate"],4),
             "valid":inc["severe_rate"]<=0.05,
             "timeout_rate":round(inc["timeout_rate"],4),
             "naturalness":(nat.get("gepa_certify_incumbent") or {}).get("naturalness")})
for i,c in enumerate(rep.get("candidates",[])):
    m=c["metrics"]; pid=f"gepa_certify_c{i:02d}"
    rows.append({"pid":pid,"role":"candidate",
                 "source":(c.get("source_run") or "").split("/")[-1],
                 "exploration_score":c.get("exploration_score"),
                 "full_diff":round(1-m["agent_success_rate"],4),
                 "severe_rate":round(m["severe_rate"],4),
                 "valid":m["severe_rate"]<=0.05,
                 "timeout_rate":round(m["timeout_rate"],4),
                 "naturalness":(nat.get(pid) or {}).get("naturalness")})
valid=[r for r in rows if r["valid"] and r["role"]=="candidate"]
valid.sort(key=lambda r:r["full_diff"],reverse=True)
best=valid[0] if valid else None
out={"rows":rows,"best_valid":best}
json.dump(out,open(sys.argv[3],"w"),indent=2)

def f(x): return "  -  " if x is None else f"{x}"
print("\n================ FINAL: pair-run cert (full-114 telecom × 2 trials) ================")
print(f"{'persona':<24}{'src':<18}{'expl':<7}{'full_diff':<10}{'severe':<8}{'valid':<7}{'nat':<5}")
print("-"*86)
for r in sorted(rows,key=lambda r:(-(r['full_diff'] or 0))):
    print(f"{r['pid']:<24}{r['source']:<18}{f(r['exploration_score']):<7}"
          f"{f(r['full_diff']):<10}{f(r['severe_rate']):<8}{('Y' if r['valid'] else 'GATED'):<7}"
          f"{f(r['naturalness']):<5}")
if best:
    print(f"\nBEST valid & hardest: {best['pid']} ({best['source']} expl={best['exploration_score']}) "
          f"full_diff={best['full_diff']} severe={best['severe_rate']} nat={best['naturalness']}")
print("wrote", sys.argv[3])
PYEOF

echo "[finish] DONE @ $(date '+%H:%M') -> $OUT/FINAL_pair_cert.json" | tee -a "$LOG"
