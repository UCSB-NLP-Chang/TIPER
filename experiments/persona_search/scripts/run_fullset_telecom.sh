#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 补跑 telecom fullset 的 opt / no_persona / p0003 cells (seed 4 岛已完成).
# 可断点续跑: assemble 会吸收自身已写输出, tau2 --auto-resume 只补缺 trial.
set -uo pipefail
cd "$REPO"
LOG=/tmp/fullset_telecom.log
echo "===== START $(date) =====" >>"$LOG"
.venv/bin/python experiments/persona_search/scripts/fullset_build.py --cells \
  telecom:brinks_00:opt telecom:brinks_02:opt telecom:brinks_03:opt telecom:brinks_04:opt \
  telecom:shared:no_persona \
  telecom_p0003:p0003:seed telecom_p0003:p0003:opt \
  >>"$LOG" 2>&1
echo "===== END $(date) rc=$? =====" >>"$LOG"
