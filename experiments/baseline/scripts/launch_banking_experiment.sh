#!/bin/bash
# Safe launcher for long-running banking experiment
# Uses nohup to survive SSH disconnection

echo "Launching banking all-presets experiment..."
echo "This will run for ~5-6 hours and survive SSH disconnection"
echo ""

# Check if already running
if pgrep -f "run_banking_all_presets" > /dev/null; then
    echo "ERROR: Banking experiment already running!"
    echo "PIDs:"
    pgrep -f "run_banking_all_presets"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Launch with nohup
nohup bash scripts/run_banking_all_presets.sh > logs/all_presets_experiment.log 2>&1 &
PID=$!

echo "✅ Experiment launched!"
echo "PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f logs/all_presets_experiment.log"
echo ""
echo "Check progress:"
echo "  bash scripts/check_banking_progress.sh"
echo ""
echo "The experiment will continue even if you disconnect SSH."
echo "Expected completion: ~5-6 hours (~23:30 tonight)"
