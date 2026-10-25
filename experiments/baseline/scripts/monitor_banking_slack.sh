#!/bin/bash
# Monitor banking experiment and send formatted Slack updates every 30 minutes

INTERVAL=1800  # 30 minutes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Banking 实验 Slack 监控已启动"
echo "更新频率: 每 30 分钟"
echo "频道: #all-yutaoxie"
echo ""
echo "按 Ctrl+C 停止监控"
echo ""

# Send initial notification
python3 scripts/send_banking_update.py

# Monitor loop
while true; do
    sleep $INTERVAL

    # Check if experiment is still running
    if ! pgrep -f "run_banking_all_presets" > /dev/null; then
        echo "$(date): 实验进程已停止，发送最终通知..."
        python3 scripts/send_banking_update.py
        echo "监控已停止"
        exit 0
    fi

    # Send periodic update
    echo "$(date): 发送进度更新..."
    python3 scripts/send_banking_update.py
done
