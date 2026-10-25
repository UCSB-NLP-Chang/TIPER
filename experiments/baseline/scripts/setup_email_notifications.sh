#!/bin/bash
# Setup email notifications for banking experiment
# This script will create a configuration file for email notifications

echo "========================================================================"
echo "Banking Experiment - Email Notification Setup"
echo "========================================================================"
echo ""

CONFIG_FILE=".email_config"

echo "请输入接收通知的邮箱地址:"
read EMAIL_TO

echo ""
echo "选择邮件发送方式:"
echo "1) Mac系统mail命令 (需要提前配置)"
echo "2) Gmail SMTP (需要Gmail账号和应用专用密码)"
echo "3) 其他SMTP服务 (需要SMTP配置)"
echo ""
read -p "请选择 (1-3): " EMAIL_METHOD

if [ "$EMAIL_METHOD" = "2" ]; then
    echo ""
    echo "Gmail SMTP配置:"
    read -p "Gmail账号 (例如 user@gmail.com): " GMAIL_USER
    read -sp "Gmail应用专用密码 (https://myaccount.google.com/apppasswords): " GMAIL_PASS
    echo ""
    
    cat > "$CONFIG_FILE" << CONF
EMAIL_METHOD=gmail
EMAIL_TO=$EMAIL_TO
GMAIL_USER=$GMAIL_USER
GMAIL_PASS=$GMAIL_PASS
CONF

elif [ "$EMAIL_METHOD" = "3" ]; then
    echo ""
    echo "SMTP配置:"
    read -p "SMTP服务器地址 (例如 smtp.example.com): " SMTP_HOST
    read -p "SMTP端口 (通常是465或587): " SMTP_PORT
    read -p "发件邮箱: " SMTP_USER
    read -sp "邮箱密码: " SMTP_PASS
    echo ""
    
    cat > "$CONFIG_FILE" << CONF
EMAIL_METHOD=smtp
EMAIL_TO=$EMAIL_TO
SMTP_HOST=$SMTP_HOST
SMTP_PORT=$SMTP_PORT
SMTP_USER=$SMTP_USER
SMTP_PASS=$SMTP_PASS
CONF

else
    cat > "$CONFIG_FILE" << CONF
EMAIL_METHOD=mail
EMAIL_TO=$EMAIL_TO
CONF
fi

chmod 600 "$CONFIG_FILE"

echo ""
echo "✅ 邮件配置已保存到 $CONFIG_FILE"
echo ""
echo "测试发送邮件..."

# Test email
python3 scripts/send_notification_email.py "测试邮件" "Banking实验监控系统测试" "如果你收到这封邮件，说明配置成功！"

if [ $? -eq 0 ]; then
    echo "✅ 测试邮件发送成功！"
else
    echo "❌ 测试邮件发送失败，请检查配置"
fi
