"""Minimal proxy: Anthropic Messages API -> OpenAI Chat Completions API.

Translates /v1/messages requests (including streaming and tool use)
to OpenAI /v1/chat/completions, stripping Anthropic-specific params
and mapping responses back to Anthropic format.

Usage:
    OPENAI_API_KEY=sk-... python anthropic_to_openai_proxy.py [--port 4111]
"""

import json
import os
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
OPENAI_BASE = os.environ.get("OPENAI_BASE", "https://api.openai.com/v1")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "gpt-5.4")

_CC_BOILERPLATE_RE = re.compile(r"^x-anthropic-billing-header:.*$", re.MULTILINE)
_CC_AGENT_IDENTITY = "You are a Claude agent, built on Anthropic's Claude Agent SDK."

MODEL_ALIASES = {
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",
}


def resolve_model(model: str) -> str:
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if not model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return TARGET_MODEL
    return model


def strip_cc_boilerplate(system_blocks: list) -> list:
    """Remove Claude Code internal preamble blocks from the system prompt.

    CC prepends a billing header and an agent identity block that cause
    Claude models (via non-Anthropic backends) to ignore the actual
    application system prompt.
    """
    out = []
    for block in system_blocks:
        if isinstance(block, dict):
            text = block.get("text", "")
        else:
            text = str(block)

        if text.startswith("x-anthropic-billing-header:"):
            continue
        if text.strip() == _CC_AGENT_IDENTITY:
            continue
        out.append(block)
    return out


def anthropic_to_openai_messages(messages: list) -> list:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    parts.append(
                        f"[Tool call: {block.get('name', '?')}({json.dumps(block.get('input', {}))})]"
                    )
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    parts.append(f"[Tool result: {block.get('content', '')}]")
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts) if parts else ""
        out.append({"role": role, "content": content or ""})
    return out


def openai_to_anthropic_response(oai: dict, model: str) -> dict:
    choice = oai["choices"][0] if oai.get("choices") else {}
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    usage = oai.get("usage", {})
    stop = (
        "end_turn"
        if choice.get("finish_reason") == "stop"
        else choice.get("finish_reason", "end_turn")
    )
    return {
        "id": oai.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
    }


def build_sse_payload(oai_resp: dict, model: str) -> bytes:
    """Build the complete SSE byte payload from a non-streaming OpenAI response."""
    resp = openai_to_anthropic_response(oai_resp, model)
    text = resp["content"][0]["text"] if resp["content"] else ""
    msg_id = resp["id"]
    in_tok = resp["usage"]["input_tokens"]
    out_tok = resp["usage"]["output_tokens"]

    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": in_tok, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": resp["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": out_tok},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]

    lines = []
    for event_name, data in events:
        lines.append(f"event: {event_name}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines).encode()


MAX_COMPLETION_TOKENS_CAP = int(os.environ.get("MAX_COMPLETION_TOKENS_CAP", "16384"))


def call_openai(
    messages_out: list, max_tokens: int, model: str = "", temperature=None, top_p=None
) -> dict:
    oai_body = {
        "model": model or TARGET_MODEL,
        "messages": messages_out,
        "max_completion_tokens": min(max_tokens, MAX_COMPLETION_TOKENS_CAP),
    }
    if temperature is not None:
        oai_body["temperature"] = temperature
    if top_p is not None:
        oai_body["top_p"] = top_p

    req_data = json.dumps(oai_body).encode()
    sys.stderr.write(
        f"[proxy] -> {OPENAI_BASE}/chat/completions model={oai_body['model']} max_tokens={max_tokens} msgs={len(messages_out)}\n"
    )
    sys.stderr.flush()
    req = Request(
        f"{OPENAI_BASE}/chat/completions",
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[proxy] {format % args}\n")
        sys.stderr.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw)

        path = urlparse(self.path).path.rstrip("/")
        is_stream = body.get("stream", False)
        sys.stderr.write(
            f"[proxy] POST {self.path} model={body.get('model')} stream={is_stream}\n"
        )
        sys.stderr.flush()

        if path != "/v1/messages":
            self.send_error(404, f"Unknown path: {path}")
            return

        model = resolve_model(body.get("model", TARGET_MODEL))
        system_raw = body.get("system", "")
        if isinstance(system_raw, list):
            system_raw = strip_cc_boilerplate(system_raw)
            system_text = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in system_raw
            )
        else:
            system_text = _CC_BOILERPLATE_RE.sub("", str(system_raw))
            system_text = system_text.replace(_CC_AGENT_IDENTITY, "").strip()

        messages_out = anthropic_to_openai_messages(body.get("messages", []))
        if system_text:
            messages_out.insert(0, {"role": "system", "content": system_text})

        max_tokens = body.get("max_tokens", 4096)

        try:
            oai_resp = call_openai(
                messages_out,
                max_tokens,
                model=model,
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
            )
        except HTTPError as e:
            err_body = e.read().decode()
            sys.stderr.write(f"[proxy] OpenAI error {e.code}: {err_body[:300]}\n")
            sys.stderr.flush()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": err_body[:500]},
                    }
                ).encode()
            )
            return

        try:
            if is_stream:
                payload = build_sse_payload(oai_resp, model)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                self.close_connection = True
            else:
                anthropic_resp = openai_to_anthropic_response(oai_resp, model)
                resp_bytes = json.dumps(anthropic_resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)

            sys.stderr.write(f"[proxy] => 200 OK (stream={is_stream})\n")
            sys.stderr.flush()
        except BrokenPipeError:
            sys.stderr.write(f"[proxy] Client disconnected (broken pipe)\n")
            sys.stderr.flush()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path.rstrip("/") in ("/health", ""):
            resp = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)


class ThreadingHTTPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 4111
    print(
        f"Anthropic->OpenAI proxy on http://127.0.0.1:{port} (target: {TARGET_MODEL})"
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
