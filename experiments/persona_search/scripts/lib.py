"""Persona library inspector (read-only CLI).

The library lives at ``data/persona_opt/lib.jsonl``; every other component reads
and writes it through ``_common``. This CLI is for humans / Claude Code to
inspect state between steps.

    .venv/bin/python experiments/persona_search/scripts/lib.py show
    .venv/bin/python experiments/persona_search/scripts/lib.py show --status untested
    .venv/bin/python experiments/persona_search/scripts/lib.py show --ranked      # valid, hardest first
    .venv/bin/python experiments/persona_search/scripts/lib.py get --persona-id p0003
    .venv/bin/python experiments/persona_search/scripts/lib.py stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LIB_PATH, read_lib  # noqa: E402


def _fmt(r: dict) -> str:
    rate = r.get("avg_success_rate")
    rate_s = "  n/a" if rate is None else f"{rate:5.3f}"
    return (
        f"{r['persona_id']:>7} | {r['status']:<8} | succ={rate_s} | "
        f"evals={r.get('num_evaluations', 0)} | parent={r.get('parent_id') or '-':<7} | "
        f"dir={r.get('mutation_direction') or '-'}"
    )


def cmd_show(args):
    recs = read_lib(args.lib)
    if args.status:
        recs = [r for r in recs if r.get("status") == args.status]
    if args.ranked:
        recs = [
            r
            for r in recs
            if r.get("status") == "valid" and r.get("avg_success_rate") is not None
        ]
        recs.sort(
            key=lambda r: (
                r["avg_success_rate"],
                r.get("created_at", ""),
                r["persona_id"],
            )
        )
    for r in recs:
        print(_fmt(r))
    print(f"\n{len(recs)} persona(s)")


def cmd_get(args):
    recs = {r["persona_id"]: r for r in read_lib(args.lib)}
    r = recs.get(args.persona_id)
    if not r:
        print(f"no persona {args.persona_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_stats(args):
    recs = read_lib(args.lib)
    by_status = Counter(r.get("status") for r in recs)
    valid = [
        r
        for r in recs
        if r.get("status") == "valid" and r.get("avg_success_rate") is not None
    ]
    print(f"total personas : {len(recs)}")
    print(f"by status      : {dict(by_status)}")
    if valid:
        best = min(valid, key=lambda r: r["avg_success_rate"])
        print(f"valid scored   : {len(valid)}")
        print(f"global best    : {best['persona_id']} @ {best['avg_success_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Inspect the persona library.")
    ap.add_argument("cmd", choices=["show", "get", "stats"])
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--status", choices=["untested", "valid", "illegal"], default=None)
    ap.add_argument(
        "--ranked", action="store_true", help="valid personas, ascending success"
    )
    ap.add_argument("--persona-id")
    args = ap.parse_args()

    if args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "get":
        if not args.persona_id:
            ap.error("--persona-id required for get")
        cmd_get(args)
    else:
        cmd_stats(args)


if __name__ == "__main__":
    main()
