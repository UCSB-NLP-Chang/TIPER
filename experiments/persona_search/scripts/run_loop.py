"""Run the full optimization loop until convergence — zero human intervention.

Chains run_round.py repeatedly, each round including auto-mutation. Stops when
convergence_state.json says should_stop=true, or after --max-rounds.

    AWS_PROFILE=us-west-2-staging_dev \
      .venv/bin/python experiments/persona_search/scripts/run_loop.py --start-round 3

    # cap at 5 rounds total:
    .venv/bin/python experiments/persona_search/scripts/run_loop.py --start-round 3 --max-rounds 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CONVERGENCE_PATH, read_json  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")


def main():
    ap = argparse.ArgumentParser(description="Run optimization loop until convergence.")
    ap.add_argument("--start-round", type=int, required=True)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--n", type=int, default=3, help="personas per round")
    ap.add_argument("--eval-concurrency", type=int, default=3)
    ap.add_argument("--total-sim-concurrency", type=int, default=21)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--mutate-n-children", type=int, default=5)
    ap.add_argument("--bedrock-profile", default=None)
    args = ap.parse_args()

    for rnd in range(args.start_round, args.start_round + args.max_rounds):
        print(f"\n{'#' * 60}")
        print(f"# ROUND {rnd}")
        print(f"{'#' * 60}\n")

        cmd = [
            PY,
            str(HERE / "run_round.py"),
            "--round",
            str(rnd),
            "--n",
            str(args.n),
            "--eval-concurrency",
            str(args.eval_concurrency),
            "--total-sim-concurrency",
            str(args.total_sim_concurrency),
            "--sim-timeout",
            str(args.sim_timeout),
            "--mutate-n-children",
            str(args.mutate_n_children),
        ]
        if args.bedrock_profile:
            cmd += ["--bedrock-profile", args.bedrock_profile]

        rc = subprocess.run(cmd)
        if rc.returncode != 0:
            print(f"\nERROR: round {rnd} failed (exit {rc.returncode})")
            sys.exit(1)

        state = read_json(CONVERGENCE_PATH)
        if state.get("should_stop"):
            hist = state.get("history", [{}])
            latest = hist[-1] if hist else {}
            print(f"\n*** STOPPED after round {rnd} ***")
            print(f"reason: {latest.get('stop_reason', '?')}")
            print(f"best: {latest.get('best_id')} avg={latest.get('best_avg')}")
            break
    else:
        print(f"\nReached max rounds ({args.max_rounds}).")

    print("\nRun report.py for the final summary.")


if __name__ == "__main__":
    main()
