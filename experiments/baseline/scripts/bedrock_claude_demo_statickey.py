#!/usr/bin/env python3
"""Demo: call Claude on Bedrock with credentials PASTED IN THE FILE — no SSO login.

⚠️  SECURITY
    - Hardcoding secrets is bad practice. Use this only for a throwaway local demo.
    - Do NOT commit this file with real keys filled in (add to .gitignore).
    - These placeholders are for YOU to paste your own keys. This script does NOT and
      must NOT read ~/.aws/credentials (break-glass, human-only).

Pick ONE of the two credential modes below and fill it in.

Run:
    .venv/bin/python scripts/bedrock_claude_demo_statickey.py "your prompt"
"""

import os
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
AWS_REGION = "us-west-2"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
MAX_TOKENS = 512

# ── MODE A: Bedrock API key (the cleanest "one token, no login") ───────────────
# AWS Bedrock supports a bearer token scoped to Bedrock only. Generate it in the
# console (Bedrock > API keys) or `aws bedrock create-api-key`. Paste here:
BEDROCK_API_KEY = ""  # e.g. "ABSKQmVkcm9ja0FQS...."  (leave "" to use Mode B)

# ── MODE B: classic AWS access key triplet (SigV4) ─────────────────────────────
# Long-term IAM user keys → only the first two are needed (no session token).
# Temporary STS creds      → all three required, and they expire (~1h).
AWS_ACCESS_KEY_ID = ""  # e.g. "AKIA..." or "ASIA..." (temp)
AWS_SECRET_ACCESS_KEY = ""
AWS_SESSION_TOKEN = ""
# ──────────────────────────────────────────────────────────────────────────────

import boto3
from botocore.exceptions import ClientError


def make_client():
    if BEDROCK_API_KEY:
        # Bedrock bearer-token path: boto3 reads AWS_BEARER_TOKEN_BEDROCK from env.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY
        print("[creds] using Bedrock API key (bearer token, no SSO)")
        return boto3.client("bedrock-runtime", region_name=AWS_REGION)

    if AWS_ACCESS_KEY_ID.startswith("PASTE"):
        sys.exit(
            "ERROR: fill BEDROCK_API_KEY (Mode A) or the access-key triplet (Mode B) first."
        )

    kwargs = dict(
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    if AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    print(
        f"[creds] using pasted access key {AWS_ACCESS_KEY_ID[:4]}… "
        f"({'temporary+session' if AWS_SESSION_TOKEN else 'long-term'})"
    )
    return boto3.client("bedrock-runtime", **kwargs)


def main():
    prompt = (
        sys.argv[1] if len(sys.argv) > 1 else "In one sentence, what is AWS Bedrock?"
    )
    client = make_client()
    try:
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 1.0},
        )
    except ClientError as e:
        err = e.response["Error"]
        print(f"\n[bedrock error] {err['Code']}: {err['Message']}")
        if err["Code"] in (
            "ExpiredTokenException",
            "UnrecognizedClientException",
            "InvalidSignatureException",
        ):
            print(
                "   -> your pasted credentials are expired or wrong (temp STS creds last ~1h)."
            )
        sys.exit(1)

    text = resp["output"]["message"]["content"][0]["text"]
    u = resp.get("usage", {})
    print(f"\nPROMPT : {prompt}")
    print(f"REPLY  : {text}")
    print(f"USAGE  : in={u.get('inputTokens')} out={u.get('outputTokens')}")


if __name__ == "__main__":
    main()
