#!/usr/bin/env python3
"""Send Slack notification using Bot Token"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_config():
    """Load Slack configuration"""
    config_file = Path(".slack_config")
    if not config_file.exists():
        print("ERROR: .slack_config not found")
        sys.exit(1)

    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config


def send_via_bot_token(token, channel, text, title, details, color="good"):
    """Send message to Slack using Bot Token"""

    # Build blocks for rich formatting
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": title, "emoji": True},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": details}},
    ]

    payload = {"channel": channel, "text": text, "blocks": blocks}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("ok", False)
    except Exception as e:
        print(f"Error sending Slack message: {e}")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: send_slack_notification_bot.py <text> <title> <details> [color]")
        sys.exit(1)

    text = sys.argv[1]
    title = sys.argv[2]
    details = sys.argv[3]
    color = sys.argv[4] if len(sys.argv) > 4 else "good"

    config = load_config()
    token = config.get("SLACK_BOT_TOKEN")
    channel = config.get("SLACK_CHANNEL", "#general")

    if not token:
        print("ERROR: SLACK_BOT_TOKEN not found in config")
        sys.exit(1)

    success = send_via_bot_token(token, channel, text, title, details, color)

    if success:
        print("Slack message sent successfully")
        sys.exit(0)
    else:
        print("Failed to send Slack message")
        sys.exit(1)


if __name__ == "__main__":
    main()
