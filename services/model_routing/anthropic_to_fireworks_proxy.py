#!/usr/bin/env python3
"""Local shim: inject `reasoning_effort` into Anthropic Messages requests and
forward to the Cresta fireworks proxy with a STATIC Bearer token, with
round-robin replica selection and exponential backoff on transient failures.

Why this exists:
    The fireworks proxy exposes GLM-5.2 (and minimax-m3, deepseek-v4-pro) over
    the Anthropic Messages protocol and enables thinking via a top-level
    `reasoning_effort:"high"` field. But litellm's `anthropic` provider rejects
    `reasoning_effort`/`thinking` for custom model ids like
    `accounts/fireworks/models/glm-5p2` (UnsupportedParamsError), and its
    `extra_body` is forwarded as a literal body field (HTTP 400). So to let tau2
    (which drives Claude models through litellm's anthropic path) run these
    Fireworks models with high reasoning, we run this tiny shim, point
    ANTHROPIC_API_BASE at it, and it injects the field + swaps the auth header.
    The proxy body is already standard Anthropic Messages, so nothing else is
    translated. Round-robin + exponential backoff (429/5xx/timeout) mirror
    openai_to_fireworks_proxy.py.

Run:
    FIREWORKS_PROXY_TOKEN=<uuid:secret> python anthropic_to_fireworks_proxy.py --port 4113

Point tau2 / litellm at it:
    ANTHROPIC_API_BASE=http://127.0.0.1:4113   ANTHROPIC_API_KEY=dummy
    agent/user model: anthropic/accounts/fireworks/models/<model>

Env:
    FIREWORKS_PROXY_TOKEN      <uuid:secret>  (falls back to VERTEX_PROXY_TOKEN)
    CRESTA_FIREWORKS_PROXY_BASE  default https://dev-proxy.ops2.internal.cresta.ai/proxy/fireworks
    GLM_REASONING_EFFORT       default "high"  ("" disables injection)
    FWPROXY_RETRY_BASE_S       backoff base seconds   (default 2)
    FWPROXY_RETRY_MAX_S        backoff cap seconds    (default 30)
    FWPROXY_MAX_ATTEMPTS       max attempts per request (default 4)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOKEN = os.environ.get("FIREWORKS_PROXY_TOKEN") or os.environ.get(
    "VERTEX_PROXY_TOKEN", ""
)
BASE = os.environ.get(
    "CRESTA_FIREWORKS_PROXY_BASE",
    "https://dev-proxy.ops2.internal.cresta.ai/proxy/fireworks",
).rstrip("/")

# --- Round-robin across DNS-resolved replicas ---
_HOSTNAME = "dev-proxy.ops2.internal.cresta.ai"
_RR_LOCK = threading.Lock()
_RR_CYCLE: itertools.cycle | None = None


def _resolve_replicas() -> list[str]:
    """Resolve all A records for the proxy hostname."""
    try:
        results = socket.getaddrinfo(_HOSTNAME, 443, socket.AF_INET, socket.SOCK_STREAM)
        ips = sorted({r[4][0] for r in results})
        return ips if ips else []
    except Exception:
        return []


def _get_next_ip() -> str | None:
    """Return next IP in round-robin order, refreshing DNS on first call."""
    global _RR_CYCLE
    with _RR_LOCK:
        if _RR_CYCLE is None:
            ips = _resolve_replicas()
            if not ips:
                return None
            _RR_CYCLE = itertools.cycle(ips)
            print(f"Round-robin replicas: {ips}")
        return next(_RR_CYCLE)


EFFORT = os.environ.get("GLM_REASONING_EFFORT", "high")

# --- Retry / exponential backoff (mirrors openai_to_fireworks_proxy) ---
_RETRY_BASE_S = float(os.environ.get("FWPROXY_RETRY_BASE_S", "2"))
_RETRY_MAX_S = float(os.environ.get("FWPROXY_RETRY_MAX_S", "30"))
_MAX_ATTEMPTS = int(os.environ.get("FWPROXY_MAX_ATTEMPTS", "4"))


def _is_transient(status: int | None, exc: BaseException | None) -> bool:
    """429 / 5xx / timeouts / connection errors are retryable."""
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    if exc is not None:
        name = type(exc).__name__
        if name in {
            "URLError",
            "timeout",
            "TimeoutError",
            "ConnectionError",
            "ConnectionResetError",
            "RemoteDisconnected",
        }:
            return True
        msg = str(exc).lower()
        if any(
            s in msg
            for s in (
                "timed out",
                "timeout",
                "overloaded",
                "rate limit",
                "too many requests",
                "bad gateway",
                "connection reset",
                "peer closed",
            )
        ):
            return True
    return False


def _deref_schema(schema: dict) -> dict:
    """Inline JSON-schema $ref/$defs in a tool input_schema.

    deepseek-v4-pro on the fireworks Anthropic endpoint fails to resolve
    `#/$defs/...` references (BadRequestError "'NoneType' has no attribute
    'lookup'"). Flattening the refs makes the schema self-contained. Harmless for
    models (GLM-5.2) that already handle refs. Cycle-guarded.
    """
    defs: dict = {}
    for key in ("$defs", "definitions"):
        d = schema.get(key)
        if isinstance(d, dict):
            defs.update(d)

    def resolve(node, stack):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                name = ref.split("/")[-1]
                if name in stack:  # circular ref -> break with a permissive object
                    return {"type": "object"}
                target = defs.get(name)
                extra = {k: resolve(v, stack) for k, v in node.items() if k != "$ref"}
                if target is None:
                    return extra
                out = dict(resolve(target, stack | {name}))
                out.update(extra)  # sibling keys override the resolved target
                return out
            return {k: resolve(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x, stack) for x in node]
        return node

    resolved = resolve(schema, frozenset())
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
        resolved.pop("definitions", None)
    return resolved


class Handler(BaseHTTPRequestHandler):
    def _proxy(self) -> None:
        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        # Inject reasoning_effort only if the caller specified neither it nor an
        # explicit thinking block, so an explicit request always wins.
        if EFFORT and "reasoning_effort" not in body and "thinking" not in body:
            body["reasoning_effort"] = EFFORT
        # Inline $ref/$defs in tool schemas so deepseek-v4-pro accepts them.
        tools = body.get("tools")
        if isinstance(tools, list):
            for t in tools:
                if isinstance(t, dict) and isinstance(t.get("input_schema"), dict):
                    t["input_schema"] = _deref_schema(t["input_schema"])
        data = json.dumps(body).encode()
        # Forward with round-robin replica selection + exponential backoff on
        # transient failures (429/5xx/timeout/conn). Streamed responses start
        # only after urlopen succeeds, so retries happen pre-stream.
        r, err = self._forward_with_retry(data)
        if r is None:
            if isinstance(err, tuple):  # (status, payload) from HTTPError
                status, payload = err
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(502)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(err)}).encode())
            return
        self.send_response(r.status)
        self.send_header(
            "content-type", r.headers.get("content-type", "application/json")
        )
        self.end_headers()
        while True:  # stream bytes straight back (handles SSE and plain JSON)
            chunk = r.read(8192)
            if not chunk:
                break
            self.wfile.write(chunk)
            try:
                self.wfile.flush()
            except Exception:
                pass

    def _forward_with_retry(self, data: bytes):
        """POST the Anthropic payload to the dev-proxy, retrying transient
        failures across round-robin replicas with exponential backoff.

        Returns (response, None) on success (caller streams the response) or
        (None, err) on terminal failure, where err is either an Exception or a
        (status, payload_bytes) tuple from the final HTTPError."""
        last_err = None
        for attempt in range(_MAX_ATTEMPTS):
            # Round-robin: pick a replica IP and connect directly via HTTPS.
            target_ip = _get_next_ip()
            if target_ip:
                url = f"https://{target_ip}/proxy/fireworks{self.path}"
                host_header = _HOSTNAME
            else:
                url = BASE + self.path
                host_header = None
            req = Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "content-type": "application/json",
                    "anthropic-version": self.headers.get(
                        "anthropic-version", "2023-06-01"
                    ),
                },
            )
            if host_header:
                req.add_header("Host", host_header)
            ctx = ssl.create_default_context()
            if target_ip:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            try:
                r = urlopen(req, timeout=600, context=ctx)
                return r, None  # success — caller streams
            except HTTPError as e:
                status = e.code
                payload = e.read()
                last_err = (status, payload)
                if not _is_transient(status, None) or attempt >= _MAX_ATTEMPTS - 1:
                    return None, last_err
            except URLError as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    return None, last_err
            except Exception as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    return None, last_err
            backoff = min(_RETRY_BASE_S * (2**attempt), _RETRY_MAX_S)
            time.sleep(backoff)
        return None, last_err

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def log_message(self, *a) -> None:  # silence per-request logging (no token leak)
        pass


class Server(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4113)
    args = ap.parse_args()
    if not TOKEN:
        print(
            "ERROR: FIREWORKS_PROXY_TOKEN (or VERTEX_PROXY_TOKEN) not set",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"anthropic->fireworks shim on http://127.0.0.1:{args.port}  ->  {BASE} "
        f"(reasoning_effort={EFFORT or 'none'}); token …{TOKEN[-6:]}"
    )
    Server(("127.0.0.1", args.port), Handler).serve_forever()
