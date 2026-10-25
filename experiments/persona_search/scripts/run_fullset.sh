#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 全量 fullset 补跑 wrapper：顺序跑所有 cell（cell 内 tau2 auto-resume）。
# 单进程串行，避免 OpenAI API 过订阅；崩溃后重跑会经 resume_path 吸收已完成 trial。
set -uo pipefail
cd "$REPO"
LOG=/tmp/fullset_run.log
echo "===== fullset 补跑 START $(date '+%F %T') =====" >> "$LOG"
.venv/bin/python experiments/persona_search/scripts/fullset_build.py >> "$LOG" 2>&1
rc=$?
echo "===== fullset 补跑 END rc=$rc $(date '+%F %T') =====" >> "$LOG"
exit $rc
