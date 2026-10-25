#!/usr/bin/env python3
"""Probe Vertex AI model ids through the Cresta dev-proxy.

Real 2026-08 model ids (Gemini naming from ai.google.dev/pricing;
Claude-on-Vertex from litellm vertex_ai/ conventions). For each: one
tiny completion, classify:
  YES  -> got a reply (exists + proxy allows)
  NO   -> 404 / not found / not supported / does not exist
  RATE -> 429 / RESOURCE_EXHAUSTED (retry backoff; counts as "exists")
  ERR  -> other

Throttled ~3.5s between calls (dev-proxy 20 req/min GLOBAL).
"""

import os
import sys
import time

from cresta.paths import REPO_ROOT

sys.stdout.reconfigure(line_buffering=True)  # flush every print when piped

env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["VERTEXAI_LOCATION"] = "us-central1"

import tau2.utils.cresta_vertex as cv  # noqa: E402

cv._apply_patch()
import litellm  # noqa: E402

GEMINI = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
]
CLAUDE = [
    "claude-3-5-sonnet-v2@002",
    "claude-3-5-haiku@001",
    "claude-3-7-sonnet@002",
    "claude-sonnet-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-haiku-4-5",
    "claude-fable-5",
]


def classify(exc):
    s = str(exc).lower()
    if any(
        k in s
        for k in ["429", "rate limit", "resource_exhausted", "rate_limit", "quota"]
    ):
        return "RATE"
    if any(
        k in s
        for k in [
            "404",
            "not found",
            "not supported",
            "does not exist",
            "not_available",
            "no such model",
            "unsupported",
        ]
    ):
        return "NO"
    return "ERR"


def probe(mid):
    full = f"vertex_ai/{mid}"
    for attempt in range(3):
        try:
            r = litellm.completion(
                model=full,
                messages=[{"role": "user", "content": "PONG"}],
                max_tokens=5,
            )
            txt = (r.choices[0].message.content or "").strip()
            return "YES", txt[:30]
        except Exception as e:
            cls = classify(e)
            if cls == "RATE" and attempt < 2:
                print(f"    [{mid}] RATE, backoff {30 * (attempt + 1)}s...", flush=True)
                time.sleep(30 * (attempt + 1))
                continue
            return cls, str(e)[:140].replace("\n", " ")
    return "RATE", "exhausted retries"


print(
    f"proxy={os.environ.get('CRESTA_VERTEX_PROXY_BASE')}  loc={os.environ['VERTEXAI_LOCATION']}",
    flush=True,
)
print("=" * 80, flush=True)
for group, models in [("GEMINI", GEMINI), ("CLAUDE-on-Vertex", CLAUDE)]:
    print(f"\n## {group}", flush=True)
    for mid in models:
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
        print(f"  {flag} {mid:35s} {cls:5s} {dt:4.1f}s  {info}", flush=True)
        time.sleep(3.5)
