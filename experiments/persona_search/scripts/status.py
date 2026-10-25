"""Live status dashboard for the persona optimization loop.

.venv/bin/python experiments/persona_search/scripts/status.py
.venv/bin/python experiments/persona_search/scripts/status.py --watch 30   # refresh every 30s
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CONVERGENCE_PATH,
    ROUNDS_DIR,
    SIMULATIONS_DIR,
    is_infra_broken,
    read_json,
    read_lib,
)

DOMAINS = {"telecom": 23, "banking": 7}


def _sim_counts(round_no: int, pid: str) -> dict[str, str]:
    """Per-domain sim completion counts for a persona in a round."""
    out = {}
    for dk, total in DOMAINS.items():
        rp = (
            SIMULATIONS_DIR
            / f"persona_opt/round_{round_no:02d}/{pid}/{dk}/results.json"
        )
        if not rp.exists():
            out[dk] = "-"
            continue
        try:
            data = json.load(open(rp))
            sims = data.get("simulations", [])
            n = len(sims)
            infra = sum(1 for s in sims if is_infra_broken(s))
            rewards = [
                (s.get("reward_info") or {}).get("reward")
                for s in sims
                if not is_infra_broken(s)
            ]
            valid_rewards = [r for r in rewards if r is not None]
            avg = sum(valid_rewards) / len(valid_rewards) if valid_rewards else None
            avg_s = f" avg={avg:.2f}" if avg is not None else ""
            infra_s = f" infra={infra}" if infra else ""
            out[dk] = f"{n}/{total}{avg_s}{infra_s}"
        except Exception:
            out[dk] = "err"
    return out


def _round_status(round_no: int) -> dict | None:
    rd = ROUNDS_DIR / f"round_{round_no:02d}"
    if not rd.exists():
        return None

    status = {"round": round_no, "phase": "unknown", "personas": []}

    sel_f = rd / "selected.json"
    if not sel_f.exists():
        status["phase"] = "selecting"
        return status

    selected = read_json(sel_f)
    pids = [s["persona_id"] for s in selected["selected"]]

    all_eval_done = True
    all_judge_done = True
    for pid in pids:
        p = {"persona_id": pid}
        eval_f = rd / "eval" / pid / "persona_eval.json"
        judge_f = rd / "judge" / pid / "judge_status.json"

        if eval_f.exists():
            ev = read_json(eval_f)
            p["score"] = ev.get("persona_score")
            p["n_tasks"] = ev.get("n_tasks")
            p["eval"] = "done"
        else:
            p["eval"] = "running"
            p["sims"] = _sim_counts(round_no, pid)
            all_eval_done = False

        if judge_f.exists():
            jd = read_json(judge_f)
            p["status"] = jd.get("status")
            p["severe_rate"] = jd.get("severe_failure_rate")
            p["judge"] = "done"
        else:
            p["judge"] = "pending" if not eval_f.exists() else "running"
            all_judge_done = False

        status["personas"].append(p)

    if not all_eval_done:
        status["phase"] = "evaluating"
    elif not all_judge_done:
        status["phase"] = "judging"
    elif not (rd / "round_record.json").exists():
        status["phase"] = "recording"
    elif (rd / "mutation_log.json").exists():
        status["phase"] = "complete"
    elif (rd / "logs" / "auto_mutate.log").exists():
        status["phase"] = "mutating"
    else:
        status["phase"] = "converging/mutating"

    rec_f = rd / "round_record.json"
    if rec_f.exists():
        rec = read_json(rec_f)
        for p in status["personas"]:
            for rr in rec.get("personas", []):
                if rr["persona_id"] == p["persona_id"]:
                    p["score"] = rr.get("persona_score")
                    p["status"] = rr.get("status")

    return status


def _process_alive() -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "run_loop.py|run_round.py"],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def render(clear: bool = False):
    if clear:
        os.system("clear")

    print("=" * 65)
    print("  PERSONA OPTIMIZATION — LIVE STATUS")
    print("=" * 65)

    # Process status
    alive = _process_alive()
    print(f"\nLoop process: {'RUNNING' if alive else 'STOPPED'}")

    # Library summary
    lib = read_lib()
    by_status = {}
    for r in lib:
        s = r.get("status", "?")
        by_status.setdefault(s, []).append(r)
    parts = []
    for s in ["valid", "untested", "illegal"]:
        if s in by_status:
            parts.append(f"{s}={len(by_status[s])}")
    print(f"Library: {len(lib)} personas ({', '.join(parts)})")

    # Convergence
    if CONVERGENCE_PATH.exists():
        state = read_json(CONVERGENCE_PATH)
        hist = state.get("history", [])
        latest = hist[-1] if hist else {}
        best_avg = latest.get("best_avg")
        best_id = latest.get("best_id", "?")
        best_str = f"{best_avg:.4f}" if best_avg is not None else "n/a"
        n_evals = latest.get("best_n_evals", 0)
        n_untested = latest.get("n_untested", "?")
        print(
            f"Best: {best_str} ({best_id}, {n_evals} evals)  "
            f"untested: {n_untested}  "
            f"stop: {state.get('should_stop')}"
        )

    # Top valid personas
    valid = sorted(
        [
            r
            for r in lib
            if r.get("status") == "valid" and r.get("avg_success_rate") is not None
        ],
        key=lambda r: r["avg_success_rate"],
    )
    if valid:
        print("\nTop valid personas:")
        for r in valid[:5]:
            print(
                f"  {r['persona_id']}: avg_success={r['avg_success_rate']:.4f}  "
                f"n_eval={r['num_evaluations']}  dir={r.get('mutation_direction', '-')}"
            )

    # Active/recent rounds
    print()
    for rnd in range(1, 20):
        rs = _round_status(rnd)
        if rs is None:
            break
        phase = rs["phase"]
        marker = " <<<" if phase not in ("complete",) else ""
        print(f"Round {rnd} [{phase}]{marker}")
        for p in rs.get("personas", []):
            parts = [p["persona_id"]]
            if "score" in p and p["score"] is not None:
                parts.append(f"score={p['score']:.3f}")
            if "status" in p:
                parts.append(p["status"])
            if p.get("eval") == "running" and "sims" in p:
                sim_parts = [f"{dk}:{v}" for dk, v in p["sims"].items() if v != "-"]
                if sim_parts:
                    parts.append(" ".join(sim_parts))
            elif p.get("eval") == "running":
                parts.append("eval:running")
            if p.get("judge") == "running":
                parts.append("judge:running")
            if "severe_rate" in p and p["severe_rate"] is not None:
                parts.append(f"severe={p['severe_rate']:.3f}")
            print(f"  {' | '.join(parts)}")

    # Log tail
    log = Path("/tmp/persona_opt_loop.log")
    if log.exists():
        print(f"\nLog tail ({log}):")
        lines = log.read_text().splitlines()
        for line in lines[-5:]:
            print(f"  {line}")

    print(f"\n[{time.strftime('%H:%M:%S')}]")


def main():
    ap = argparse.ArgumentParser(description="Live status of persona optimization.")
    ap.add_argument(
        "--watch",
        type=int,
        default=0,
        help="refresh interval in seconds (0 = once)",
    )
    args = ap.parse_args()

    if args.watch:
        while True:
            try:
                render(clear=True)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                break
    else:
        render()


if __name__ == "__main__":
    main()
