#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 补齐 telecom opt 四岛的 xhigh full(train/val)——464 slot,test 已 salvage。
# 正确 winner = ev2v4_cert/b0X_top1(a7020ec9a7/c655bad793/47c61c2629/e7405d0aa9)。
# 可断点续跑:assemble 吸收自身已写输出,tau2 --auto-resume 只补缺 trial。
set -uo pipefail
cd "$REPO"
LOG=/tmp/fullset_opt_xhigh.log
echo "===== START $(date) =====" >>"$LOG"
.venv/bin/python experiments/persona_search/scripts/fullset_build.py --cells \
  telecom:brinks_00:opt telecom:brinks_02:opt telecom:brinks_03:opt telecom:brinks_04:opt \
  >>"$LOG" 2>&1
echo "===== END $(date) rc=$? =====" >>"$LOG"
