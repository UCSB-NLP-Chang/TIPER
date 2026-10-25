"""Convergence / stopping bookkeeping.

Strategy:
  - Run up to ``--max-rounds`` (default 20).
  - Early stop ONLY when the library has 0 untested personas (mutation has
    exhausted its ability to produce novel children — all dedup-rejected).
  - At the end, the winning persona is the valid one with the lowest
    ``avg_success_rate`` among those with >= ``--min-evals`` (default 3)
    evaluations.

No statistical significance test — just accumulate enough evals per persona
via the exploit-retest slot and let the running mean converge naturally.

    .venv/bin/python experiments/persona_search/scripts/converge.py --round 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CONVERGENCE_PATH,
    LIB_PATH,
    now_iso,
    read_json,
    read_lib,
    write_json,
)


def _best_persona(records, min_evals: int):
    """Find the valid persona with the lowest avg score (>= min_evals)."""
    candidates = [
        r
        for r in records
        if r.get("status") == "valid"
        and r.get("avg_success_rate") is not None
        and len(r.get("eval_scores", [])) >= min_evals
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["avg_success_rate"])
    return {
        "persona_id": best["persona_id"],
        "avg_success_rate": best["avg_success_rate"],
        "n_evals": len(best.get("eval_scores", [])),
        "eval_scores": best.get("eval_scores", []),
    }


def main():
    ap = argparse.ArgumentParser(description="Convergence bookkeeping.")
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--state", default=str(CONVERGENCE_PATH))
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument(
        "--min-evals",
        type=int,
        default=3,
        help="minimum evaluations for a persona to be the winner (default 3)",
    )
    args = ap.parse_args()

    records = read_lib(args.lib)
    best = _best_persona(records, args.min_evals)

    n_untested = sum(1 for r in records if r.get("status") == "untested")
    n_valid = sum(1 for r in records if r.get("status") == "valid")
    n_total = len(records)

    state_path = Path(args.state)
    state = read_json(state_path) if state_path.exists() else {"history": []}

    # Early stop: no untested personas remain (mutation exhausted)
    early_stop = n_untested == 0 and args.round > 1
    hit_max = args.round >= args.max_rounds
    should_stop = early_stop or hit_max

    if early_stop:
        stop_reason = "no untested personas remain (mutation exhausted)"
    elif hit_max:
        stop_reason = f"reached max rounds ({args.max_rounds})"
    else:
        stop_reason = None

    state["history"].append(
        {
            "round": args.round,
            "best_id": best["persona_id"] if best else None,
            "best_avg": best["avg_success_rate"] if best else None,
            "best_n_evals": best["n_evals"] if best else None,
            "n_untested": n_untested,
            "n_valid": n_valid,
            "n_total": n_total,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "at": now_iso(),
        }
    )
    state["should_stop"] = should_stop
    state["max_rounds"] = args.max_rounds
    state["min_evals"] = args.min_evals
    write_json(state_path, state)

    print(f"round {args.round}/{args.max_rounds}:")
    print(f"  library: {n_total} total, {n_valid} valid, {n_untested} untested")
    if best:
        print(
            f"  current best: {best['persona_id']} avg={best['avg_success_rate']:.3f}"
            f" ({best['n_evals']} evals, scores={[round(s, 3) for s in best['eval_scores']]})"
        )
    else:
        print(f"  no qualifying best yet (need >= {args.min_evals} evals)")
    print(f"  should_stop={should_stop}", end="")
    if stop_reason:
        print(f"  reason: {stop_reason}")
    else:
        print()

    if should_stop:
        print(f"\n>>> STOP: {stop_reason}. Run report.py.")


if __name__ == "__main__":
    main()
