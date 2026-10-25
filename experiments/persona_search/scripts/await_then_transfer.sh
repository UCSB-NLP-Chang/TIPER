#!/usr/bin/env bash
REPO="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Wait for the running top-delta test to finish, then launch the telecom
# cross-domain transfer test. Single blocking wait on the PID — no polling loop.
set -u
cd "$REPO"

TOPDELTA_PID="$(pgrep -f 'run_topdelta_test.py' | head -1 || true)"
echo "[await] top-delta pid=${TOPDELTA_PID:-none} @ $(date)"

# Block until that PID is gone (works for non-child via kill -0 loop with long sleeps).
if [ -n "${TOPDELTA_PID}" ]; then
  while kill -0 "${TOPDELTA_PID}" 2>/dev/null; do
    sleep 120
  done
fi
echo "[await] top-delta finished @ $(date); launching transfer test"

exec python3 experiments/persona_search/scripts/run_telecom_transfer_test.py \
  --max-concurrency 25 > /tmp/telecom_transfer_test.log 2>&1
