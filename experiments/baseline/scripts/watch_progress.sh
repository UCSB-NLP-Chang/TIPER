#!/usr/bin/env bash
# Real-time matrix progress. Reads from the experiment log (not results.json).
# Usage: bash scripts/watch_progress.sh
# Ctrl-C to stop watching; experiment keeps running.
LOG=/tmp/cccon_matrix_full.log

if [ ! -f "$LOG" ]; then
  echo "No experiment log found at $LOG"
  exit 1
fi

while true; do
  clear
  pgrep -f "run_cccon_agent_matrix" >/dev/null 2>&1 && ST="RUNNING" || ST="STOPPED"
  echo "Matrix: $ST    $(date '+%H:%M:%S')"
  echo ""

  echo "=== Cell history ==="
  grep -aE "^SKIP|^RERUN|^RUN |^EXIT\(" "$LOG" 2>/dev/null
  echo ""

  echo "=== Current cell ==="
  grep -aE "Status:" "$LOG" | tail -1
  echo ""

  echo "=== Errors this run ==="
  INFRA=$(grep -ac "infrastructure_error" "$LOG" 2>/dev/null)
  SSO=$(grep -ac "SSO session" "$LOG" 2>/dev/null)
  echo "infrastructure_error: $INFRA    SSO expired: $SSO"

  if [ "$ST" = "STOPPED" ]; then
    echo ""
    echo "Matrix process not running. This is the final state."
    break
  fi

  sleep 15
done
