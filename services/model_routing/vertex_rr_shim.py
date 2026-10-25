#!/usr/bin/env python3
"""Transparent RR + exponential-backoff shim for the Cresta dev-proxy
``/proxy/vertex`` route.

Why this exists:
    litellm's vertex_ai provider (via tau2's cresta_vertex.py host rewrite)
    connects to a single dev-proxy hostname. There is no round-robin across
    the 4 DNS A records and only litellm's coarse num_retries backoff. That
    caps effective throughput well below the dev-proxy's per-token 20 req/min
    ceiling because requests can't slip into rate-limit windows the way the
    fireworks shim's fast 4-attempt retry does.

    This shim sits between litellm and the dev-proxy:
        litellm -> http://127.0.0.1:PORT/<path>
                 -> https://{next IP}/proxy/vertex/<path>  (round-robin)
                 -> exponential backoff on 429/5xx/timeout (4 attempts, 2s->30s)
                 -> response passed straight back (no protocol translation)

    It is a TRANSPARENT passthrough: body, headers (incl. Authorization), and
    query string are forwarded unchanged; only the Host header is set to the
    dev-proxy hostname so TLS SNI / virtual-host routing on the replica works.

Run:
    python3 vertex_rr_shim.py --port 4115

Point tau2 at it (override the .env proxy base):
    export CRESTA_VERTEX_PROXY_BASE=http://127.0.0.1:4115
    # CRESTA_VERTEX_TOKEN stays as whatever authorizes /proxy/vertex;
    # litellm sends it as Authorization, this shim forwards it unchanged.

Env:
    RR_HOSTNAME         default dev-proxy.ops2.internal.cresta.ai
    RR_PATH_PREFIX      default /proxy/vertex
    FWPROXY_RETRY_BASE_S / _MAX_S / _MAX_ATTEMPTS  backoff tuning (defaults 2 / 30 / 4)
    FWPROXY_MIN_INTERVAL_S  client-side rate limit, 0 = off (default 0)
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

_HOSTNAME = os.environ.get("RR_HOSTNAME", "dev-proxy.ops2.internal.cresta.ai")
_PATH_PREFIX = os.environ.get("RR_PATH_PREFIX", "/proxy/vertex").rstrip("/")

_RR_LOCK = threading.Lock()
_RR_CYCLE: "itertools.cycle | None" = None

_RETRY_BASE_S = float(os.environ.get("FWPROXY_RETRY_BASE_S", "2"))
_RETRY_MAX_S = float(os.environ.get("FWPROXY_RETRY_MAX_S", "30"))
_MAX_ATTEMPTS = int(os.environ.get("FWPROXY_MAX_ATTEMPTS", "4"))

# Client-side rate limit; 0 = off (rely on dev-proxy 20/min + backoff).
_RL_LOCK = threading.Lock()
_LAST_REQ_TIME = 0.0
_MIN_INTERVAL = float(os.environ.get("FWPROXY_MIN_INTERVAL_S", "0"))


def _rate_limit_acquire() -> None:
    if _MIN_INTERVAL <= 0:
        return
    global _LAST_REQ_TIME
    with _RL_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _LAST_REQ_TIME)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST_REQ_TIME = now


def _resolve_replicas() -> list[str]:
    try:
        results = socket.getaddrinfo(_HOSTNAME, 443, socket.AF_INET, socket.SOCK_STREAM)
        ips = sorted({r[4][0] for r in results})
        return ips
    except Exception:
        return []


def _get_next_ip() -> str | None:
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


class Handler(BaseHTTPRequestHandler):
    def _proxy(self) -> None:
        length = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(length) if length else b""

        # Forward incoming headers unchanged except Host (set per-replica below)
        # and skip hop-by-hop / length (we recompute).
        fwd_headers = {}
        for k, v in self.headers.items():
            if k.lower() in (
                "host",
                "content-length",
                "accept-encoding",
                "transfer-encoding",
                "connection",
            ):
                continue
            fwd_headers[k] = v

        last_err: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            _rate_limit_acquire()
            target_ip = _get_next_ip()
            if target_ip:
                url = f"https://{target_ip}{_PATH_PREFIX}{self.path}"
                host_header = _HOSTNAME
            else:
                url = f"https://{_HOSTNAME}{_PATH_PREFIX}{self.path}"
                host_header = None
            req = Request(
                url,
                data=body if body else None,
                method=self.command,
                headers=fwd_headers,
            )
            if host_header:
                req.add_header("Host", host_header)
            ctx = ssl.create_default_context()
            if target_ip:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            try:
                r = urlopen(req, timeout=600, context=ctx)
                resp_body = r.read()
                self.send_response(r.status)
                # forward response headers (skip hop-by-hop / encoding)
                for k, v in r.headers.items():
                    if k.lower() in (
                        "transfer-encoding",
                        "connection",
                        "content-encoding",
                        "content-length",
                    ):
                        continue
                    self.send_header(k, v)
                self.send_header("content-length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return
            except HTTPError as e:
                status = e.code
                payload = e.read()
                last_err = RuntimeError(f"dev-proxy HTTP {status}: {payload[:200]!r}")
                if not _is_transient(status, None) or attempt >= _MAX_ATTEMPTS - 1:
                    # surface the upstream error verbatim to the client
                    self.send_response(status)
                    for k, v in e.headers.items():
                        if k.lower() in (
                            "transfer-encoding",
                            "connection",
                            "content-encoding",
                            "content-length",
                        ):
                            continue
                        self.send_header(k, v)
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
            except URLError as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    break
            except Exception as e:
                last_err = e
                if not _is_transient(None, e) or attempt >= _MAX_ATTEMPTS - 1:
                    break
            backoff = min(_RETRY_BASE_S * (2**attempt), _RETRY_MAX_S)
            print(
                f"[shim retry] attempt {attempt + 1}/{_MAX_ATTEMPTS} "
                f"sleep={backoff:.1f}s err={last_err}",
                file=sys.stderr,
            )
            time.sleep(backoff)

        self.send_response(502)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": str(last_err)}).encode())

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def log_message(self, *a) -> None:
        pass


class Server(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4115)
    args = ap.parse_args()
    print(
        f"vertex RR+backoff shim on http://127.0.0.1:{args.port} "
        f"-> {_HOSTNAME}{_PATH_PREFIX}"
    )
    Server(("127.0.0.1", args.port), Handler).serve_forever()
