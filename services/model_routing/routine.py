#!/usr/bin/env python3
"""Smoke test: call Vertex AI Gemini through the Cresta dev-proxy.

Loads .env, triggers the cresta_vertex patch (static bearer token +
httpx URL rewrite to dev-proxy /proxy/vertex), then fires one
litellm completion at a Gemini model and prints the reply + usage.
"""

import os
import sys

from cresta.paths import REPO_ROOT

# ── load .env (only if not already set) ──────────────────────────────
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Gemini lives in us-central1 (the .env default us-east5 is for Claude).
# gemini-3 would auto-switch to "global" inside llm_utils, but we call
# 2.5-flash here so set us-central1 explicitly.
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# ── trigger the Cresta Vertex patch (google.auth + httpx rewrite) ─────
import tau2.utils.cresta_vertex as cv  # noqa: E402

cv._apply_patch()

import litellm  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "vertex_ai/gemini-2.5-flash")
print(f"proxy  = {os.environ.get('CRESTA_VERTEX_PROXY_BASE')}")
print(
    f"project= {os.environ.get('VERTEXAI_PROJECT')}  location={os.environ.get('VERTEXAI_LOCATION')}"
)
print(f"model  = {MODEL}")
print("calling...")

try:
    resp = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly one word: PONG"}],
        max_tokens=20,
    )
    print("\n--- reply ---")
    print(resp.choices[0].message.content)
    u = resp.usage
    print(f"\n--- usage ---")
    print(
        f"prompt={u.prompt_tokens}  completion={u.completion_tokens}  total={u.total_tokens}"
    )
    print("\nSMOKE OK")
except Exception as e:
    print(f"\nSMOKE FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
