"""Fold a round's eval + judge outputs back into the library (plan §1, §4, §5).

For every persona selected this round, reads ``eval/<id>/persona_eval.json`` and
``judge/<id>/judge_status.json`` and updates its library record:
  - avg_success_rate  -> running mean across num_evaluations
  - num_evaluations   -> +1
  - status            -> valid | illegal (from the judge)
  - judge_severe_failure_rate, last_round

Writes ``rounds/round_NN/round_record.json`` — the per-iteration record.
This is the ONLY component that writes tested-persona scores/status.

    .venv/bin/python experiments/persona_search/scripts/record_results.py --round 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LIB_PATH,
    lib_by_id,
    now_iso,
    read_json,
    read_lib,
    round_dir,
    upsert,
    write_json,
    write_lib,
)


def _running_mean(old_avg, old_n, new_val):
    if new_val is None:
        return old_avg
    if old_avg is None or not old_n:
        return new_val
    return (old_avg * old_n + new_val) / (old_n + 1)


def main():
    ap = argparse.ArgumentParser(
        description="Record eval+judge into the library (plan §4,§5)."
    )
    ap.add_argument("--lib", default=str(LIB_PATH))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--selected", default=None, help="defaults to rounds/round_NN/selected.json"
    )
    args = ap.parse_args()

    rd = round_dir(args.round)
    selected_path = Path(args.selected) if args.selected else rd / "selected.json"
    selected = read_json(selected_path)["selected"]

    records = read_lib(args.lib)
    by_id = lib_by_id(records)

    round_rows = []
    for sel in selected:
        pid = sel["persona_id"]
        rec = by_id.get(pid)
        if rec is None:
            print(f"  WARN: {pid} not in library, skipping")
            continue

        eval_summary = read_json(rd / "eval" / pid / "persona_eval.json")
        judge_status = read_json(rd / "judge" / pid / "judge_status.json")
        score = eval_summary.get("persona_score")

        old_n = rec.get("num_evaluations", 0)
        rec["avg_success_rate"] = _running_mean(
            rec.get("avg_success_rate"), old_n, score
        )
        rec["num_evaluations"] = old_n + 1
        rec["status"] = judge_status["status"]
        rec["judge_severe_failure_rate"] = judge_status.get("severe_failure_rate")
        rec["last_round"] = args.round
        if score is not None:
            rec.setdefault("eval_scores", []).append(score)
        upsert(records, rec)

        row = {
            "persona_id": pid,
            "persona_score": score,
            "avg_success_rate": rec["avg_success_rate"],
            "num_evaluations": rec["num_evaluations"],
            "status": rec["status"],
            "severe_failure_rate": judge_status.get("severe_failure_rate"),
            "n_tasks": eval_summary.get("n_tasks"),
            "infra_error_count": eval_summary.get("infra_error_count"),
        }
        round_rows.append(row)
        print(
            f"  {pid}: score={score} avg={rec['avg_success_rate']:.3f} "
            f"status={rec['status']} severe={judge_status.get('severe_failure_rate')}"
        )

    write_lib(records, args.lib)
    write_json(
        rd / "round_record.json",
        {
            "round": args.round,
            "recorded_at": now_iso(),
            "personas": round_rows,
        },
    )
    print(f"\nlibrary updated -> {args.lib}")
    print(f"round record -> {rd / 'round_record.json'}")


if __name__ == "__main__":
    main()
