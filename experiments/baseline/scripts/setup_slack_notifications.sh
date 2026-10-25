#!/bin/bash
# Setup Slack notifications for banking experiment

echo "========================================================================"
echo "Banking Experiment - Slack Notification Setup"
echo "========================================================================"
echo ""
echo "你需要一个Slack Incoming Webhook URL。"
echo ""
echo "获取方式:"
echo "1. 访问 https://api.slack.com/apps"
echo "2. 创建或选择一个App"
echo "3. 启用 'Incoming Webhooks'"
echo "4. 添加Webhook到你想接收通知的频道"
echo "5. 复制Webhook URL (格式: https://hooks.slack.com/services/...)"
echo ""
read -p "请输入Slack Webhook URL: " WEBHOOK_URL

if [ -z "$WEBHOOK_URL" ]; then
    echo "错误: Webhook URL不能为空"
    exit 1
fi

# Save config
cat > .slack_config << CONF
SLACK_WEBHOOK_URL=$WEBHOOK_URL
CONF

chmod 600 .slack_config

echo ""
echo "✅ Slack配置已保存"
echo ""
echo "测试发送消息..."

# Test message
python3 scripts/send_slack_notification.py "测试消息" "Banking实验监控系统" "如果你在Slack看到这条消息，说明配置成功！✅" "good"

if [ $? -eq 0 ]; then
    echo "✅ 测试消息发送成功！检查你的Slack频道"
else
    echo "❌ 测试消息发送失败，请检查Webhook URL"
fi
