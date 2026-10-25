#!/usr/bin/env python3
"""Local shim: translate OpenAI /v1/chat/completions requests into Anthropic
Messages and forward to the Cresta Fireworks dev-proxy with a STATIC Bearer token.

Why this exists:
    VitaBench's own llm_utils posts OpenAI chat-completions format directly to
    models.yaml `base_url` (no litellm). The Cresta dev-proxy Fireworks endpoint
    only speaks the Anthropic Messages protocol (/v1/messages). This shim sits
    between VitaBench and the dev-proxy: OpenAI request in -> Anthropic request
    -> dev-proxy -> Anthropic response -> OpenAI response back to VitaBench.

    Non-streaming only (VitaBench uses stream=False). Tool calls translated
    both ways: OpenAI `tools=[{type:function,function:{name,description,parameters}}]`
    <-> Anthropic `tools=[{name,description,input_schema}]`; OpenAI assistant
    `tool_calls` <-> Anthropic `tool_use` blocks; OpenAI `role:tool` result
    messages <-> Anthropic `tool_result` user blocks.

Run:
    FIREWORKS_PROXY_TOKEN=<uuid:secret> python openai_to_fireworks_proxy.py --port 4114

Point VitaBench at it (models.yaml entry):
    base_url: http://127.0.0.1:4114/v1/chat/completions
    model_id: accounts/fireworks/models/deepseek-v4-flash-0731
    headers: {Authorization: "Bearer dummy"}   # shim replaces auth with its own token

Env:
    FIREWORKS_PROXY_TOKEN        <uuid:secret>  (required)
    CRESTA_FIREWORKS_PROXY_BASE  default https://dev-proxy.ops2.internal.cresta.ai/proxy/fireworks
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

TOKEN = os.environ.get("FIREWORKS_PROXY_TOKEN", "")
BASE = os.environ.get(
    "CRESTA_FIREWORKS_PROXY_BASE",
    "https://dev-proxy.ops2.internal.cresta.ai/proxy/fireworks",
).rstrip("/")

# --- Round-robin across DNS-resolved replicas (mirrors anthropic_to_fireworks) ---
_HOSTNAME = "dev-proxy.ops2.internal.cresta.ai"
_RR_LOCK = threading.Lock()
_RR_CYCLE: itertools.cycle | None = None

# --- Retry / exponential backoff (mirrors persona_policies/llm_bedrock) ---
_RETRY_BASE_S = float(os.environ.get("FWPROXY_RETRY_BASE_S", "2"))
_RETRY_MAX_S = float(os.environ.get("FWPROXY_RETRY_MAX_S", "30"))
_MAX_ATTEMPTS = int(os.environ.get("FWPROXY_MAX_ATTEMPTS", "4"))

# --- Client-side rate limiting (dev-proxy caps ~20 req/min) ---
# Minimum interval between request *starts* across all threads. 3.1s ~= 19.3
# req/min, safely under the 20/min dev-proxy limit so 429s stay rare. Only
# gates this shim (deepseek path); gpt-4.1 re-judge calls go direct to OpenAI.
_RL_LOCK = threading.Lock()
_LAST_REQ_TIME = 0.0
_MIN_INTERVAL = float(os.environ.get("FWPROXY_MIN_INTERVAL_S", "3.1"))


def _rate_limit_acquire() -> None:
    """Block until at least _MIN_INTERVAL has passed since the last request."""
    global _LAST_REQ_TIME
    with _RL_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _LAST_REQ_TIME)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST_REQ_TIME = now


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
            print(f"Round-robin replicas: {ips}", file=sys.stderr)
        return next(_RR_CYCLE)


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


def _content_to_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


def oai_messages_to_anthropic(messages: list) -> tuple[list[str], list[dict]]:
    """Split OpenAI messages into (system_blocks, anthropic_messages)."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            c = m.get("content")
            if c:
                system_parts.append(_content_to_text(c))
            continue
        if role == "tool":
            # OpenAI tool result -> Anthropic user tool_result block
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id") or "toolu_unknown",
                            "content": _content_to_text(m.get("content")),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict] = []
            txt = _content_to_text(m.get("content"))
            if txt:
                blocks.append({"type": "text", "text": txt})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                arg_str = fn.get("arguments")
                try:
                    inp = json.loads(arg_str) if arg_str else {}
                except Exception:
                    inp = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or "toolu_0",
                        "name": fn.get("name", ""),
                        "input": inp,
                    }
                )
            out.append(
                {
                    "role": "assistant",
                    "content": blocks or [{"type": "text", "text": ""}],
                }
            )
            continue
        # user (default)
        c = m.get("content", "")
        if isinstance(c, list):
            blocks = []
            for p in c:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        blocks.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "image_url":
                        url = (p.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            media, _, b64 = url.partition(";base64,")
                            mt = media[5:] if media.startswith("data:") else "image/png"
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mt,
                                        "data": b64,
                                    },
                                }
                            )
                else:
                    blocks.append({"type": "text", "text": str(p)})
            out.append(
                {"role": "user", "content": blocks or [{"type": "text", "text": ""}]}
            )
        else:
            out.append({"role": "user", "content": c if isinstance(c, str) else str(c)})
    return system_parts, out


def oai_tools_to_anthropic(tools: list) -> list[dict]:
    out = []
    for t in tools or []:
        if isinstance(t, dict) and t.get("type") == "function":
            f = t.get("function", {}) or {}
            out.append(
                {
                    "name": f.get("name", ""),
                    "description": f.get("description", ""),
                    "input_schema": f.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        elif isinstance(t, dict) and "input_schema" in t:
            out.append(t)  # already anthropic-shaped
    return out


_STOP_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def anthropic_resp_to_oai(ant: dict, model: str) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, b in enumerate(ant.get("content", []) or []):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text_parts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": b.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                    },
                }
            )
    content = "".join(text_parts) if text_parts else None
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    stop = ant.get("stop_reason")
    finish_reason = _STOP_MAP.get(stop, "stop")
    usage = ant.get("usage", {}) or {}
    it = usage.get("input_tokens", 0) or 0
    ot = usage.get("output_tokens", 0) or 0
    return {
        "id": ant.get("id", "chatcmpl-proxy"),
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": it,
            "completion_tokens": ot,
            "total_tokens": it + ot,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _proxy(self) -> None:
        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        model = body.get("model", "")
        system_parts, anth_msgs = oai_messages_to_anthropic(body.get("messages", []))
        anth: dict = {
            "model": model,
            "messages": anth_msgs,
            "max_tokens": body.get("max_tokens")
            or body.get("max_completion_tokens")
            or 8192,
        }
        if system_parts:
            anth["system"] = "\n\n".join(system_parts)
        if body.get("temperature") is not None:
            anth["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            anth["top_p"] = body["top_p"]
        if body.get("tools"):
            anth["tools"] = oai_tools_to_anthropic(body["tools"])
        if body.get("reasoning_effort"):
            anth["reasoning_effort"] = body["reasoning_effort"]
        tc = body.get("tool_choice")
        if tc == "auto":
            anth["tool_choice"] = {"type": "auto"}
        elif tc == "none":
            anth["tool_choice"] = {"type": "none"}
        elif tc == "required":
            anth["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            anth["tool_choice"] = {
                "type": "tool",
                "name": (tc.get("function") or {}).get("name", ""),
            }
        data = json.dumps(anth).encode()
        # Forward to the dev-proxy with round-robin replica selection +
        # exponential backoff on transient failures (429/5xx/timeout/conn).
        ant = self._forward_with_retry(data)
        if isinstance(ant, Exception):
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(ant)}).encode())
            return
        oai = anthropic_resp_to_oai(ant, model)
        b = json.dumps(oai).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _forward_with_retry(self, data: bytes) -> dict | Exception:
        """POST the Anthropic payload to the dev-proxy, retrying transient
        failures across round-robin replicas with exponential backoff."""
        last_err: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            # Throttle to stay under the dev-proxy 20 req/min cap (skipped on
            # retries of the same attempt-chain would still serialize here,
            # which is fine — backoff already sleeps longer than this).
            _rate_limit_acquire()
            # Round-robin: pick a replica IP and connect directly via HTTPS.
            target_ip = _get_next_ip()
            if target_ip:
                url = f"https://{target_ip}/proxy/fireworks/v1/messages"
                host_header = _HOSTNAME
            else:
                url = BASE + "/v1/messages"
                host_header = None
            req = Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
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
                ant = json.loads(r.read())
                # Treat an empty/non-dict body as a transient failure too.
                if not isinstance(ant, dict) or not ant:
                    raise RuntimeError("empty response body from dev-proxy")
                return ant
            except HTTPError as e:
                status = e.code
                payload = e.read()
                last_err = RuntimeError(f"dev-proxy HTTP {status}: {payload[:200]!r}")
                if not _is_transient(status, None) or attempt >= _MAX_ATTEMPTS - 1:
                    return last_err
            except URLError as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    return last_err
            except Exception as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    return last_err
            backoff = min(_RETRY_BASE_S * (2**attempt), _RETRY_MAX_S)
            print(
                f"[shim retry] attempt {attempt + 1}/{_MAX_ATTEMPTS} "
                f"sleep={backoff:.1f}s err={last_err}",
                file=sys.stderr,
            )
            time.sleep(backoff)
        return last_err or RuntimeError("unknown forward failure")

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def log_message(self, *a) -> None:
        pass


class Server(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4114)
    args = ap.parse_args()
    if not TOKEN:
        print("ERROR: FIREWORKS_PROXY_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    print(
        f"openai->fireworks shim on http://127.0.0.1:{args.port}  ->  {BASE} "
        f"(token …{TOKEN[-6:]})"
    )
    Server(("127.0.0.1", args.port), Handler).serve_forever()
