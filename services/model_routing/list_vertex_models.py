#!/usr/bin/env python3
"""List available Vertex AI models through the Cresta dev-proxy.

Hits the publisher-models list endpoint for each publisher; the
cresta_vertex patch rewrites *-aiplatform.googleapis.com -> dev-proxy.
"""

import os

from cresta.paths import REPO_ROOT

# load .env
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")

import tau2.utils.cresta_vertex as cv  # noqa: E402

cv._apply_patch()

import httpx  # noqa: E402

PROJECT = os.environ.get("VERTEXAI_PROJECT", "cresta-ml")
REGIONS = os.environ.get("VERTEX_LIST_REGIONS", "us-central1,global").split(",")
PUBLISHERS = ["google", "anthropic", "mistral", "meta", "ai21", "claude"]

print(f"project={PROJECT}  proxy={os.environ.get('CRESTA_VERTEX_PROXY_BASE')}")
print(f"regions={REGIONS}  publishers={PUBLISHERS}")
print("=" * 80)

headers = {
    "Authorization": f"Bearer {os.environ.get('CRESTA_VERTEX_TOKEN')}",
    "Content-Type": "application/json",
}

for region in REGIONS:
    for pub in PUBLISHERS:
        host = (
            "aiplatform.googleapis.com"
            if region == "global"
            else f"{region}-aiplatform.googleapis.com"
        )
        url = f"https://{host}/v1/projects/{PROJECT}/locations/{region}/publishers/{pub}/models"
        try:
            r = httpx.get(url, headers=headers, timeout=30)
        except Exception as e:
            print(f"[{region}/{pub}] EXC {type(e).__name__}: {e}")
            continue
        if r.status_code != 200:
            # shorten error body
            body = r.text[:200].replace("\n", " ")
            print(f"[{region}/{pub}] HTTP {r.status_code}: {body}")
            continue
        data = r.json()
        models = data.get("models", data.get("publisherModels", []))
        ids = []
        for m in models:
            name = m.get("name", "")
            # name like "publishers/google/models/gemini-1.5-pro"
            mid = name.split("/")[-1] if name else m.get("version", "")
            ids.append(mid)
        ids = sorted(set(ids))
        print(f"[{region}/{pub}] {len(ids)} models:")
        for mid in ids:
            print(f"    {mid}")
