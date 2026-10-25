#!/usr/bin/env python3
"""Probe Cresta dev-proxy /proxy/fireworks for Qwen models.

Protocol: Anthropic Messages (the proxy speaks Anthropic over /proxy/fireworks).
Token: CRESTA_VERTEX_TOKEN (same Cresta token authorizes /proxy/vertex & /proxy/fireworks).
First hit glm-5p2 (known-good) to confirm token+protocol, then try Qwen ids.
"""

import os
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(line_buffering=True)

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

TOKEN = (
    os.environ.get("CRESTA_VERTEX_TOKEN")
    or os.environ.get("FIREWORKS_PROXY_TOKEN")
    or os.environ.get("VERTEX_PROXY_TOKEN", "")
)
BASE = os.environ.get(
    "CRESTA_FIREWORKS_PROXY_BASE",
    "https://dev-proxy.ops2.internal.cresta.ai/proxy/fireworks",
).rstrip("/")

MODELS = [
    "accounts/fireworks/models/glm-5p2",  # known-good control
    "accounts/fireworks/models/qwen3-235b-a22b-2507",
    "accounts/fireworks/models/qwen3.5-397b-a17b-think",
    "accounts/fireworks/models/qwen3-coder-480b-a35b",
    "accounts/fireworks/models/qwen3.7-plus",
    "accounts/fireworks/models/qwen3-next-80b-a3b",
    "accounts/fireworks/models/qwen3-32b",
    "accounts/fireworks/models/qwen3-coder-30b-a3b",
    "accounts/qwen/models/qwen3-235b-a22b-2507",  # alt namespace
]


def probe(mid):
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{BASE}/v1/messages",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "x-api-key": TOKEN,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": mid,
                    "max_tokens": 20,
                    "messages": [
                        {"role": "user", "content": "Reply with exactly one word: PONG"}
                    ],
                },
                timeout=60,
            )
        except Exception as e:
            return "ERR", f"{type(e).__name__}: {str(e)[:100]}"
        code = r.status_code
        body = r.text[:160].replace("\n", " ")
        if code == 200:
            try:
                txt = r.json()["content"][0]["text"][:30]
                return "YES", txt
            except Exception:
                return "YES", "(200 but parse)"
        if code == 429:
            print(f"    [{mid}] 429, backoff {25 * (attempt + 1)}s...", flush=True)
            time.sleep(25 * (attempt + 1))
            continue
        low = body.lower()
        if any(
            k in low
            for k in [
                "not found",
                "not_found",
                "no such",
                "unsupported model",
                "does not exist",
                "unknown model",
            ]
        ):
            return "NO", f"{code} {body}"
        return "ERR", f"{code} {body}"
    return "RATE", "exhausted"


print(
    f"base={BASE}  token={'set' if TOKEN else 'MISSING'} ({len(TOKEN)} chars)",
    flush=True,
)
print("=" * 80, flush=True)
for mid in MODELS:
    t0 = time.time()
    cls, info = probe(mid)
    dt = time.time() - t0
    flag = (
        "✅"
        if cls == "YES"
        else "❌"
        if cls == "NO"
        else "⏳"
        if cls == "RATE"
        else "⚠️"
    )
    print(f"  {flag} {mid:55s} {cls:5s} {dt:4.1f}s  {info}", flush=True)
    time.sleep(3.5)
