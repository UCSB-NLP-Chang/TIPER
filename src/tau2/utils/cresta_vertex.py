"""Cresta Vertex AI proxy integration for tau2-bench.

Two patches are applied when CRESTA_VERTEX_TOKEN is set:

1. google.auth.default → returns static credentials with the proxy bearer token
   so LiteLLM's vertex_ai provider doesn't try Google OAuth.

2. httpx.Client.send / httpx.AsyncClient.send → rewrites
       https://{location}-aiplatform.googleapis.com/...
   to
       {CRESTA_VERTEX_PROXY_BASE}/...
   so traffic goes to the internal proxy rather than GCP directly.

.env vars:
    CRESTA_VERTEX_TOKEN=<uuid>:<secret>
    CRESTA_VERTEX_PROXY_BASE=https://dev-proxy.ops2.internal.cresta.ai/proxy/vertex
    VERTEXAI_PROJECT=cresta-ml
    VERTEXAI_LOCATION=us-east5   # for Claude; us-central1 for Gemini

Model strings for --agent-llm / --user-llm:
    vertex_ai/claude-sonnet-4-5      (Claude via Anthropic on Vertex, us-east5)
    vertex_ai/gemini-1.5-pro         (Gemini, us-central1 — change VERTEXAI_LOCATION)
"""

import os
import re

_VERTEX_HOST_RE = re.compile(r"https://[^/]+-aiplatform\.googleapis\.com")


def _apply_patch() -> None:
    token = os.environ.get("CRESTA_VERTEX_TOKEN")
    if not token:
        return

    proxy_base = os.environ.get(
        "CRESTA_VERTEX_PROXY_BASE",
        "https://dev-proxy.ops2.internal.cresta.ai/proxy/vertex",
    ).rstrip("/")
    project_id = os.environ.get("VERTEXAI_PROJECT", "cresta-ml")

    # ── Patch 1: google.auth.default ────────────────────────────────────────
    try:
        import google.auth  # noqa: PLC0415
        import google.auth.credentials  # noqa: PLC0415
    except ImportError:
        from loguru import logger  # noqa: PLC0415

        logger.warning(
            "CRESTA_VERTEX_TOKEN is set, but google-auth is not installed; "
            "Cresta Vertex routing is disabled. Install the voice extra or "
            "google-auth to enable it."
        )
        return

    class _StaticTokenCredentials(google.auth.credentials.Credentials):
        def __init__(self) -> None:
            super().__init__()
            self.token = token

        @property
        def project_id(self) -> str:
            return project_id

        @property
        def expired(self) -> bool:
            return False

        @property
        def valid(self) -> bool:
            return True

        def refresh(self, request: object) -> None:
            pass  # static token, never refresh

    _static_creds = _StaticTokenCredentials()
    google.auth.default = lambda *args, **kwargs: (_static_creds, project_id)

    # ── Patch 2: httpx URL rewrite ───────────────────────────────────────────
    import httpx  # noqa: PLC0415

    _orig_client_send = httpx.Client.send
    _orig_async_send = httpx.AsyncClient.send

    def _rewrite(request: httpx.Request) -> httpx.Request:
        url_str = str(request.url)
        if "-aiplatform.googleapis.com" not in url_str:
            return request
        new_url = _VERTEX_HOST_RE.sub(proxy_base, url_str)
        # Drop the host header so httpx sets it correctly for the new URL.
        # Without this, the proxy receives the wrong Host and returns 404.
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=request.content,
        )

    def _patched_client_send(self, request, *args, **kwargs):
        return _orig_client_send(self, _rewrite(request), *args, **kwargs)

    async def _patched_async_send(self, request, *args, **kwargs):
        return await _orig_async_send(self, _rewrite(request), *args, **kwargs)

    httpx.Client.send = _patched_client_send
    httpx.AsyncClient.send = _patched_async_send

    from loguru import logger  # noqa: PLC0415

    logger.debug(
        "Cresta Vertex patch active: {} → {}",
        "*-aiplatform.googleapis.com",
        proxy_base,
    )


_apply_patch()
