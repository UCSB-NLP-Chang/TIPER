"""Dedup-guard and persist Claude-Code-authored mutation children (plan §8).

Reads children I authored (``children_in.json``: list of
{persona_text, parent_id, mutation_direction}) and, for each, checks similarity
against the WHOLE library (untested + valid + illegal):
  - reject if max cosine to ANY persona >= --cos-threshold (default 0.92)
    (exact / near / trivial / semantic duplicate);
  - reject if cosine to any ILLEGAL persona >= --illegal-threshold (default 0.85)
    (minor variant of an illegal persona);
  - otherwise accept; flag gray-zone (>= --gray-threshold, default 0.85) with the
    nearest neighbor so I can sanity-check.

Accepted children are appended to lib.jsonl as status=untested, with parent_id,
mutation_direction, avg_success_rate=null, num_evaluations=0, dedup_checked=true.
Rejections are reported (with reasons + nearest neighbors) so I can regenerate with
a different direction (plan: up to max_dedup_retries_per_child=5 across re-runs).

Similarity backend: ``--sim-method embed`` (OpenAI text-embedding-3-large, default)
or ``--sim-method difflib`` (offline, for smoke tests / no API key).

    .venv/bin/python experiments/persona_search/scripts/add_children.py --round 1
    .venv/bin/python experiments/persona_search/scripts/add_children.py --round 0 --sim-method difflib
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LIB_PATH,
    new_record,
    next_persona_id,
    read_json,
    read_lib,
    round_dir,
    upsert,
    write_json,
    write_lib,
)


def _cosine_sims(cand_texts, lib_texts):
    """(n_cand, n_lib) cosine matrix via OpenAI embeddings."""
    import numpy as np

    from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder

    emb = OpenAIEmbedder(model="text-embedding-3-large")
    cand = emb.embed(cand_texts)
    lib = emb.embed(lib_texts)
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
    lib = lib / (np.linalg.norm(lib, axis=1, keepdims=True) + 1e-12)
    return cand @ lib.T


def _difflib_sims(cand_texts, lib_texts):
    return [
        [SequenceMatcher(None, c, t).ratio() for t in lib_texts] for c in cand_texts
    ]


def main():
    ap = argparse.ArgumentParser(
        description="Dedup + persist authored children (plan §8)."
    )
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--children-in",
        default=None,
        help="defaults to rounds/round_NN/children_in.json",
    )
    ap.add_argument("--cos-threshold", type=float, default=0.92)
    ap.add_argument("--gray-threshold", type=float, default=0.85)
    ap.add_argument("--illegal-threshold", type=float, default=0.85)
    ap.add_argument("--sim-method", choices=["embed", "difflib"], default="embed")
    ap.add_argument(
        "--target", type=int, default=3, help="expected #children (for shortfall log)"
    )
    args = ap.parse_args()

    rd = round_dir(args.round)
    children_in_path = (
        Path(args.children_in) if args.children_in else rd / "children_in.json"
    )
    children_in = read_json(children_in_path)
    if isinstance(children_in, dict):
        children_in = children_in.get("children", [])

    records = read_lib(args.lib)
    lib_texts = [r["persona_text"] for r in records]
    lib_ids = [r["persona_id"] for r in records]
    lib_status = [r.get("status") for r in records]

    cand_texts = [c["persona_text"] for c in children_in]
    if cand_texts and lib_texts:
        sims = (_cosine_sims if args.sim_method == "embed" else _difflib_sims)(
            cand_texts, lib_texts
        )
    else:
        sims = [[0.0] * len(lib_texts) for _ in cand_texts]

    accepted, rejected = [], []
    for ci, child in enumerate(children_in):
        row = sims[ci]
        if lib_texts:
            j = max(range(len(row)), key=lambda k: row[k])
            max_cos, nn_id, nn_status = float(row[j]), lib_ids[j], lib_status[j]
            illegal_cos = float(
                max(
                    (row[k] for k in range(len(row)) if lib_status[k] == "illegal"),
                    default=0.0,
                )
            )
        else:
            max_cos, nn_id, nn_status, illegal_cos = 0.0, None, None, 0.0

        if max_cos >= args.cos_threshold:
            rejected.append(
                {
                    **_thin(child),
                    "reason": "near_duplicate",
                    "max_cos": round(max_cos, 4),
                    "nearest": nn_id,
                    "nearest_status": nn_status,
                }
            )
            continue
        if illegal_cos >= args.illegal_threshold:
            rejected.append(
                {
                    **_thin(child),
                    "reason": "variant_of_illegal",
                    "illegal_cos": round(illegal_cos, 4),
                }
            )
            continue

        pid = next_persona_id(records)
        rec = new_record(
            pid,
            child["persona_text"],
            status="untested",
            parent_id=child.get("parent_id"),
            mutation_direction=child.get("mutation_direction"),
            dedup_checked=True,
        )
        rec["last_round"] = args.round
        upsert(records, rec)  # adds to records so next_persona_id increments
        accepted.append(
            {
                "persona_id": pid,
                "parent_id": child.get("parent_id"),
                "mutation_direction": child.get("mutation_direction"),
                "max_cos": round(max_cos, 4),
                "nearest": nn_id,
                "gray_zone": bool(max_cos >= args.gray_threshold),
            }
        )

    write_lib(records, args.lib)
    log = {
        "round": args.round,
        "target": args.target,
        "n_authored": len(children_in),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "shortfall": max(0, args.target - len(accepted)),
        "accepted": accepted,
        "rejected": rejected,
        "sim_method": args.sim_method,
        "thresholds": {
            "cos": args.cos_threshold,
            "gray": args.gray_threshold,
            "illegal": args.illegal_threshold,
        },
    }
    write_json(rd / "children.json", {"accepted": accepted})
    write_json(rd / "mutation_log.json", log)

    for a in accepted:
        flag = "  [GRAY-ZONE: review]" if a["gray_zone"] else ""
        print(
            f"  + {a['persona_id']} <- {a['parent_id']} ({a['mutation_direction']}) "
            f"max_cos={a['max_cos']}{flag}"
        )
    for r in rejected:
        print(
            f"  - REJECT ({r['reason']}) parent={r.get('parent_id')} "
            f"dir={r.get('mutation_direction')} {r}"
        )
    if log["shortfall"]:
        print(
            f"\nSHORTFALL: {len(accepted)}/{args.target} unique children "
            f"— regenerate with different directions and re-run."
        )
    print(f"\nlibrary updated -> {args.lib}")
    print(f"mutation log -> {rd / 'mutation_log.json'}")


def _thin(child: dict) -> dict:
    return {
        "parent_id": child.get("parent_id"),
        "mutation_direction": child.get("mutation_direction"),
    }


if __name__ == "__main__":
    main()
