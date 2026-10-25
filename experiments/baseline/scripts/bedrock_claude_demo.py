#!/usr/bin/env python3
"""Demo: call a Claude model on AWS Bedrock — with every credential/config it needs.

WHAT "凭证" ACTUALLY IS HERE (no single static key):

    ~/.aws/config  [profile us-west-2-staging_BedrockBatchInference]
        -> pure routing: sso_start_url / sso_account_id / sso_role_name / region. NO secret.

    ~/.aws/sso/cache/*.json
        -> the real secret: an SSO `accessToken` (~1 day TTL) + client registration.

    at call time the SDK does, automatically:
        SSO accessToken  --sso:GetRoleCredentials-->  temporary STS triplet
        (AccessKeyId / SecretAccessKey / SessionToken, ~1h TTL)  --SigV4-->  bedrock:InvokeModel

So you DON'T hardcode a key. You point at the PROFILE; boto3 resolves the chain.
The only thing that "expires" and breaks calls is the SSO accessToken — refresh with:
    aws sso login --profile us-west-2-staging_BedrockBatchInference

SECURITY: never read/use ~/.aws/credentials (break-glass, human-only). SSO is the path.

Run:
    .venv/bin/python scripts/bedrock_claude_demo.py
    .venv/bin/python scripts/bedrock_claude_demo.py "your prompt here"
"""

import os
import sys

# ── ALL CONFIG THE CALL NEEDS (explicit, self-contained) ──────────────────────
AWS_PROFILE = (
    "us-west-2-staging_BedrockBatchInference"  # SSO profile = the "credential"
)
AWS_REGION = "us-west-2"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"  # cross-region inference profile id
MAX_TOKENS = 512
# ── If you instead have EXPLICIT temporary STS creds (e.g. from another machine),
#    set these 3 env vars before running and they override the profile.
#    They expire (~1h) — do NOT paste long-lived secrets into this file.
#       export AWS_ACCESS_KEY_ID=...        # temporary
#       export AWS_SECRET_ACCESS_KEY=...    # temporary
#       export AWS_SESSION_TOKEN=...        # required for temporary creds
# ──────────────────────────────────────────────────────────────────────────────

import boto3
from botocore.exceptions import (
    ClientError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)


def make_client():
    """Resolve credentials from the SSO profile (or explicit env triplet if set)."""
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SESSION_TOKEN"):
        # explicit temporary creds path
        session = boto3.Session(region_name=AWS_REGION)
        print("[creds] using explicit AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN from env")
    else:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        print(f"[creds] using SSO profile '{AWS_PROFILE}' (region {AWS_REGION})")
    return session.client("bedrock-runtime")


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
    except (TokenRetrievalError, UnauthorizedSSOTokenError) as e:
        print(f"\n[SSO EXPIRED] {type(e).__name__}: {e}")
        print(f"   fix: aws sso login --profile {AWS_PROFILE}")
        sys.exit(1)
    except ClientError as e:
        print(
            f"\n[bedrock error] {e.response['Error']['Code']}: {e.response['Error']['Message']}"
        )
        sys.exit(1)

    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    print(f"\nPROMPT : {prompt}")
    print(f"MODEL  : {MODEL_ID}")
    print(f"REPLY  : {text}")
    print(f"USAGE  : in={usage.get('inputTokens')} out={usage.get('outputTokens')}")


if __name__ == "__main__":
    main()
