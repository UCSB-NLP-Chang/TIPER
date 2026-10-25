#!/usr/bin/env python3
"""Send Slack notification for banking experiment"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_config():
    """Load Slack configuration"""
    config_file = Path(".slack_config")
    if not config_file.exists():
        print("ERROR: .slack_config not found. Run setup_slack_notifications.sh first")
        sys.exit(1)

    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config


def send_slack_message(webhook_url, title, text, details, color="good"):
    """Send message to Slack via Incoming Webhook"""

    # Build attachment
    attachment = {
        "color": color,  # good (green), warning (yellow), danger (red)
        "title": title,
        "text": details,
        "footer": "Banking Experiment Monitor",
        "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
    }

    payload = {"text": text, "attachments": [attachment]}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            return response.status == 200
    except urllib.error.URLError as e:
        print(f"Error sending Slack message: {e}")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: send_slack_notification.py <text> <title> <details> [color]")
        sys.exit(1)

    text = sys.argv[1]
    title = sys.argv[2]
    details = sys.argv[3]
    color = sys.argv[4] if len(sys.argv) > 4 else "good"

    config = load_config()
    webhook_url = config.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not found in config")
        sys.exit(1)

    success = send_slack_message(webhook_url, title, text, details, color)

    if success:
        print("Slack message sent successfully")
        sys.exit(0)
    else:
        print("Failed to send Slack message")
        sys.exit(1)


if __name__ == "__main__":
    main()
