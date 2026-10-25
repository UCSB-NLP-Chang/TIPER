"""Logging proxy for the Anthropic Messages API.

Captures the EXACT `/v1/messages` request body + selected headers that a client
sends, appends it as one JSON line to $CAPTURE_JSONL, then returns a minimal valid
(optionally streamed) Anthropic response so the client does not error.

We only care about the *request* (system prompt, thinking, tools, temperature,
max_tokens, top_p, stop_sequences, anthropic-beta headers), not a real completion.

Point a client at it with:
    ANTHROPIC_BASE_URL=http://127.0.0.1:<port>  ANTHROPIC_API_KEY=dummy

Usage:
    CAPTURE_JSONL=/tmp/bb/captures.jsonl python scripts/backbone_capture/logging_proxy.py --port 4199
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn

CAPTURE_JSONL = os.environ.get("CAPTURE_JSONL", "/tmp/bb/captures.jsonl")
# headers worth capturing for the divergence audit
_HDR_KEEP = (
    "anthropic-version",
    "anthropic-beta",
    "x-api-key",
    "content-type",
    "user-agent",
)


def _record(label: str, body: dict, headers: dict) -> None:
    os.makedirs(os.path.dirname(CAPTURE_JSONL), exist_ok=True)
    kept = {k: v for k, v in headers.items() if k.lower() in _HDR_KEEP}
    # redact the api key value, keep only that it was present
    if "x-api-key" in {k.lower() for k in kept}:
        for k in list(kept):
            if k.lower() == "x-api-key":
                kept[k] = "<present>"
    entry = {"label": label, "ts": time.time(), "headers": kept, "body": body}
    with open(CAPTURE_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _json_response() -> bytes:
    return json.dumps(
        {
            "id": "msg_capture",
            "type": "message",
            "role": "assistant",
            "model": "capture",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()


def _sse_stream() -> bytes:
    """Minimal Anthropic streaming response (message_start ... message_stop)."""

    def ev(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    parts = [
        ev(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_capture",
                    "type": "message",
                    "role": "assistant",
                    "model": "capture",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        ev(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        ev(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ev(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ev("message_stop", {"type": "message_stop"}),
    ]
    return "".join(parts).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default logging
        pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except Exception:
            return {"_unparsed": raw.decode("utf-8", "replace")}

    def do_POST(self):  # noqa: N802
        body = self._read_body()
        # label comes from a query string ?label=... (set by the driver)
        label = "unknown"
        if "?" in self.path and "label=" in self.path:
            label = self.path.split("label=", 1)[1].split("&", 1)[0]
        _record(label, body, dict(self.headers))

        streaming = bool(body.get("stream"))
        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_sse_stream())
        else:
            payload = _json_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - health check
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


class ThreadingHTTPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4199)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"logging proxy on http://127.0.0.1:{args.port} -> {CAPTURE_JSONL}", flush=True
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
