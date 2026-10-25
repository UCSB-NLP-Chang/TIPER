#!/usr/bin/env python3
"""扫 dev-proxy 各 path，看 minimax-m3 是否有别的 route（除 /proxy/fireworks 外）。

每个 path 试两种 model id：
  - claude-minimax-m3[1m]   (Claude Code alias, settings.json 用的)
  - accounts/fireworks/models/minimax-m3  (FQ fireworks id)

已知 /proxy/fireworks + 两者 → 502 Provider error (vendor down)。
看看是否还有别的 path 能 work。
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

TOKEN = os.environ.get("CRESTA_VERTEX_TOKEN") or os.environ.get(
    "FIREWORKS_PROXY_TOKEN", ""
)
HDR = {
    "Authorization": f"Bearer {TOKEN}",
    "x-api-key": TOKEN,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

MODELS = [
    "claude-minimax-m3[1m]",
    "accounts/fireworks/models/minimax-m3",
]
PATHS = [
    "/proxy/fireworks",  # 已知 502 vendor down
    "/proxy/vertex",  # 已知 8 个 Claude OK, minimax 不在
    "/proxy/minimax",
    "/proxy/minimax-m3",
    "/proxy/m3",
    "/proxy/anthropic",
    "/proxy/deepseek",
    "/proxy/claude",
    "/proxy/openai",
]

print(f"token={'set' if TOKEN else 'MISSING'} ({len(TOKEN)} chars)", flush=True)
print("=" * 90, flush=True)
for p in PATHS:
    for mid in MODELS:
        url = f"https://dev-proxy.ops2.internal.cresta.ai{p}/v1/messages"
        body = {
            "model": mid,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "PONG"}],
        }
        t0 = time.time()
        try:
            r = httpx.post(url, headers=HDR, json=body, timeout=20)
            dt = time.time() - t0
            code = r.status_code
            txt = r.text[:90].replace("\n", " ")
            if code == 200:
                flag = "✅"
            elif code == 404:
                flag = "❌"
            elif code in (502, 503, 504):
                flag = "🛑"
            elif code == 429:
                flag = "⏳"
            else:
                flag = "⚠️"
            print(
                f"  {flag} {code} {dt:4.1f}s {p:22s} {mid[:44]:44s} {txt}", flush=True
            )
        except Exception as e:
            print(
                f"  ⏱️ EXC {p:22s} {mid[:44]:44s} {type(e).__name__}: {str(e)[:60]}",
                flush=True,
            )
        time.sleep(2)
