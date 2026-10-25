#!/bin/bash
# Setup Slack notifications using Bot Token

echo "========================================================================"
echo "Banking Experiment - Slack Bot Token Setup"
echo "========================================================================"
echo ""

BOT_TOKEN="REDACTED_SLACK_BOT_TOKEN"

echo "使用Bot Token: ${BOT_TOKEN:0:20}..."
echo ""
read -p "请输入Slack频道名称或ID (例如: #general 或 C1234567890): " CHANNEL

if [ -z "$CHANNEL" ]; then
    echo "使用默认频道: #general"
    CHANNEL="#general"
fi

# Save config
cat > .slack_config << CONF
SLACK_BOT_TOKEN=$BOT_TOKEN
SLACK_CHANNEL=$CHANNEL
CONF

chmod 600 .slack_config

echo ""
echo "✅ Slack Bot配置已保存"
echo ""
echo "测试发送消息..."

# Test message
python3 scripts/send_slack_notification_bot.py "测试消息" "Banking实验监控系统" "如果你在Slack看到这条消息，说明配置成功！✅" "good"

if [ $? -eq 0 ]; then
    echo "✅ 测试消息发送成功！检查你的Slack频道"
else
    echo "❌ 测试消息发送失败，请检查Bot Token和频道权限"
    echo ""
    echo "确保你的Bot有以下权限:"
    echo "  - chat:write"
    echo "  - chat:write.public (如果是公开频道)"
fi
