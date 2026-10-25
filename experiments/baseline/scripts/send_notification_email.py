#!/usr/bin/env python3
"""Send email notification for banking experiment"""

import sys
from pathlib import Path


def load_config():
    """Load email configuration"""
    config = {}
    config_file = Path(".email_config")
    if not config_file.exists():
        print("ERROR: .email_config not found. Run setup_email_notifications.sh first")
        sys.exit(1)

    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key] = value
    return config


def send_via_mail(to, subject, body):
    """Send email using system mail command"""
    import subprocess

    cmd = ["mail", "-s", subject, to]
    try:
        subprocess.run(cmd, input=body.encode(), check=True)
        return True
    except Exception as e:
        print(f"Error sending mail: {e}")
        return False


def send_via_smtp(to, subject, body, config):
    """Send email via SMTP"""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    method = config.get("EMAIL_METHOD")

    if method == "gmail":
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = config.get("GMAIL_USER")
        smtp_pass = config.get("GMAIL_PASS")
    else:
        smtp_host = config.get("SMTP_HOST")
        smtp_port = int(config.get("SMTP_PORT", 587))
        smtp_user = config.get("SMTP_USER")
        smtp_pass = config.get("SMTP_PASS")

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: send_notification_email.py <subject> <title> <body>")
        sys.exit(1)

    subject = sys.argv[1]
    title = sys.argv[2]
    body = sys.argv[3]

    config = load_config()
    email_to = config.get("EMAIL_TO")
    email_method = config.get("EMAIL_METHOD")

    full_body = f"{title}\n\n{body}"

    if email_method == "mail":
        success = send_via_mail(email_to, subject, full_body)
    else:
        success = send_via_smtp(email_to, subject, full_body, config)

    if success:
        print(f"Email sent to {email_to}")
        sys.exit(0)
    else:
        print("Failed to send email")
        sys.exit(1)


if __name__ == "__main__":
    main()
