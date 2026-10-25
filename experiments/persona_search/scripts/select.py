"""Round selection (plan §3).

Pick exactly N personas (default 3) for a round:
  - n_explore (default 2): randomly sampled from ``untested`` pool
  - n_exploit (default 1): retest the best valid persona (lowest avg_success_rate)
  - if fewer untested than n_explore, fill with valid retests (ascending score)
  - if no valid to retest, fill with more untested
  - random sampling uses round number as seed for reproducibility

    .venv/bin/python experiments/persona_search/scripts/select.py --round 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LIB_PATH, read_lib, round_dir, write_json  # noqa: E402


def select(records: list[dict], n: int, n_explore: int, round_no: int) -> list[dict]:
    untested = [r for r in records if r.get("status") == "untested"]
    valid = [
        r
        for r in records
        if r.get("status") == "valid" and r.get("avg_success_rate") is not None
    ]
    valid.sort(
        key=lambda r: (
            r["avg_success_rate"],
            r.get("created_at", ""),
            r["persona_id"],
        )
    )

    n_exploit = n - n_explore
    chosen = []
    already = set()

    # 1) Explore: random sample from untested
    rng = random.Random(round_no)
    rng.shuffle(untested)
    for r in untested:
        if len(chosen) >= n_explore:
            break
        chosen.append(r)
        already.add(r["persona_id"])

    # 2) Exploit: retest best valid (lowest avg_success_rate)
    for r in valid:
        if len(chosen) >= n_explore + n_exploit:
            break
        if r["persona_id"] not in already:
            chosen.append(r)
            already.add(r["persona_id"])

    # 3) Fill shortfall from whichever pool has remaining
    if len(chosen) < n:
        for r in untested + valid:
            if len(chosen) >= n:
                break
            if r["persona_id"] not in already:
                chosen.append(r)
                already.add(r["persona_id"])

    return chosen


def main():
    ap = argparse.ArgumentParser(description="Select personas for a round (plan §3).")
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=3, help="total personas to select")
    ap.add_argument(
        "--n-explore",
        type=int,
        default=2,
        help="slots for random untested (rest = retest best valid)",
    )
    ap.add_argument(
        "--out", default=None, help="defaults to rounds/round_NN/selected.json"
    )
    args = ap.parse_args()

    records = read_lib(args.lib)
    chosen = select(records, args.n, args.n_explore, args.round)

    out = Path(args.out) if args.out else round_dir(args.round) / "selected.json"
    payload = {
        "round": args.round,
        "n_requested": args.n,
        "n_selected": len(chosen),
        "selected": [
            {
                "persona_id": r["persona_id"],
                "status": r["status"],
                "avg_success_rate": r.get("avg_success_rate"),
                "persona_text": r["persona_text"],
            }
            for r in chosen
        ],
    }
    write_json(out, payload)

    for r in chosen:
        rate = r.get("avg_success_rate")
        rate_s = "n/a" if rate is None else f"{rate:.3f}"
        print(f"  {r['persona_id']:>7} [{r['status']}] succ={rate_s}")
    if len(chosen) < args.n:
        print(f"  (only {len(chosen)}/{args.n} available)")
    print(f"selected -> {out}")


if __name__ == "__main__":
    main()
