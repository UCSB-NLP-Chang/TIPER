#!/bin/bash
# Monitor banking experiment with AWS SSO check and Slack notifications

INTERVAL=1800  # 30 minutes
SSO_CHECK_INTERVAL=600  # Check SSO every 10 minutes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Banking 实验监控已启动"
echo "• 进度更新: 每 30 分钟"
echo "• AWS SSO 检查: 每 10 分钟"
echo "• 频道: #all-yutaoxie"
echo ""
echo "按 Ctrl+C 停止监控"
echo ""

# Send initial notification
python3 scripts/send_banking_update.py

# Initialize counters
last_progress_update=0
last_sso_check=0

# Monitor loop
while true; do
    current_time=$(date +%s)

    # Check AWS SSO status every 10 minutes
    if [ $((current_time - last_sso_check)) -ge $SSO_CHECK_INTERVAL ]; then
        echo "$(date): 检查 AWS SSO 状态..."
        python3 scripts/check_aws_sso.py
        sso_status=$?

        if [ $sso_status -ne 0 ]; then
            echo "$(date): ⚠️ AWS SSO 失败，已发送 Slack 通知并停止实验"
            exit 1
        fi

        last_sso_check=$current_time
    fi

    # Send progress update every 30 minutes
    if [ $((current_time - last_progress_update)) -ge $INTERVAL ]; then
        echo "$(date): 发送进度更新..."

        # Check if experiment is still running
        if ! pgrep -f "run_banking" > /dev/null; then
            echo "$(date): 实验进程已停止，发送最终通知..."
            python3 scripts/send_banking_update.py
            echo "监控已停止"
            exit 0
        fi

        python3 scripts/send_banking_update.py
        last_progress_update=$current_time
    fi

    sleep 60  # Sleep 1 minute between checks
done
