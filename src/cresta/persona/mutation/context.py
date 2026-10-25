"""Dump library context so Claude Code can author mutation children (plan §6,§8).

Mutation is performed by Claude Code (me), not a scripted LLM. This component
gathers everything needed to author good, non-duplicate children:
  - valid_parents: personas tested THIS round with status=valid (the only eligible
    parents), with their scores;
  - the parent -> child allocation implied by plan §6;
  - illegal personas (so I avoid minor variants of them);
  - low-success valid personas and the directions already used vs the pool;
  - short snippets of every library persona for dedup awareness.

I read ``mutation_context.json``, write the children to ``children_in.json``
(list of {persona_text, parent_id, mutation_direction}), then run add_children.py.

    .venv/bin/python -m cresta.persona.mutation.context --round 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from cresta.persona.core._common import LIB_PATH, read_lib, round_dir, write_json

# Direction pool is no longer constrained — the mutation LLM freely authors
# its own mutation direction based on reflective analysis of conversation traces.
# Previous directions are tracked for diversity awareness only.
DIRECTION_POOL = []


def _allocation(n_parents: int) -> str:
    return {
        3: "3 parents -> 1 child each (3 total)",
        2: "2 parents -> one makes 2 variants, other makes 1 (3 total)",
        1: "1 parent -> 3 variants (3 total)",
        0: "0 parents -> NO mutations this round",
    }.get(min(n_parents, 3), "n/a")


def _snippet(text: str, n: int = 160) -> str:
    s = " ".join(text.split())
    return s[:n] + ("..." if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser(
        description="Dump mutation context for Claude Code (plan §6,§8)."
    )
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--out", default=None, help="defaults to rounds/round_NN/mutation_context.json"
    )
    args = ap.parse_args()

    records = read_lib(args.lib)
    valid_parents = [
        r
        for r in records
        if r.get("status") == "valid" and r.get("avg_success_rate") is not None
    ]
    valid_parents.sort(key=lambda r: (r.get("avg_success_rate", 1.0), r["persona_id"]))

    illegal = [r for r in records if r.get("status") == "illegal"]
    valid_all = [
        r
        for r in records
        if r.get("status") == "valid" and r.get("avg_success_rate") is not None
    ]
    valid_all.sort(key=lambda r: r["avg_success_rate"])
    dirs_used = Counter(
        r.get("mutation_direction") for r in records if r.get("mutation_direction")
    )
    underrepresented = [d for d in DIRECTION_POOL if dirs_used.get(d, 0) == 0]

    ctx = {
        "round": args.round,
        "n_valid_parents": len(valid_parents),
        "allocation": _allocation(len(valid_parents)),
        "direction_pool": DIRECTION_POOL,
        "directions_used": dict(dirs_used),
        "underrepresented_directions": underrepresented,
        "valid_parents": [
            {
                "persona_id": r["persona_id"],
                "avg_success_rate": r.get("avg_success_rate"),
                "mutation_direction": r.get("mutation_direction"),
                "persona_text": r["persona_text"],
            }
            for r in valid_parents
        ],
        "lowest_success_valid": [
            {
                "persona_id": r["persona_id"],
                "avg_success_rate": r["avg_success_rate"],
                "snippet": _snippet(r["persona_text"]),
            }
            for r in valid_all[:5]
        ],
        "illegal_personas_avoid_variants": [
            {
                "persona_id": r["persona_id"],
                "mutation_direction": r.get("mutation_direction"),
                "snippet": _snippet(r["persona_text"]),
            }
            for r in illegal
        ],
        "library_snippets": [
            {
                "persona_id": r["persona_id"],
                "status": r["status"],
                "snippet": _snippet(r["persona_text"]),
            }
            for r in records
        ],
        "constraints_reminder": [
            "edit ONLY persona text; rewrite ~30% of the parent",
            "preserve core identity, user intent, task relevance",
            "distinct mutation_direction per child",
            "must stay VALID for the judge: stay in character, never fabricate, "
            "give correct known info when asked, keep pursuing the original goal",
            "no contradictory/malicious/impossible behavior; don't make the user impossible to satisfy",
        ],
    }
    out = (
        Path(args.out) if args.out else round_dir(args.round) / "mutation_context.json"
    )
    write_json(out, ctx)
    print(f"valid_parents={len(valid_parents)}  allocation: {ctx['allocation']}")
    print(f"underrepresented directions: {underrepresented}")
    print(f"context -> {out}")
    print(
        "\nNow author children to "
        f"{round_dir(args.round) / 'children_in.json'} then run add_children.py"
    )


if __name__ == "__main__":
    main()
