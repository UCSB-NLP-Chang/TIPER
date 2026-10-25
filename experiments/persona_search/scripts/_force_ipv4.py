"""Monkey-patch socket.getaddrinfo to skip IPv6 (AF_INET6) results.

Import this module early (via PYTHONSTARTUP or sitecustomize) when the target
host returns AAAA records but IPv6 is unreachable, causing long SYN_SENT hangs.
"""

import socket

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(*args, **kwargs):
    results = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in results if r[0] == socket.AF_INET] or results


socket.getaddrinfo = _ipv4_only_getaddrinfo
