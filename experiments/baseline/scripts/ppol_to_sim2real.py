"""Convert PPol eval output (main json + trajectories.json) into the sim2real
results.json format so sim2real_usi_full.py / sim2real_full_table.py can score it.

sim2real expects:
  {"info": {"user_info": {"implementation": ..., "llm": ...}},
   "simulations": [{"task_id": ..., "reward_info": {"reward": ...}, "messages": [...]}, ...]}

PPol produces two sibling files:
  <stem>.json                       # results[] = [{task_id, persona_id, trial_idx, reward, status, ...}]
  <stem>.json_trajectories.json     # [{task_id, persona_id, trial_idx, messages: [{role, content}]}]

We join them on (task_id, persona_id, trial_idx) and emit one simulation row per
episode. reward is taken from the main json (NaN/error -> 0.0).

Usage:
  python ppol_to_sim2real.py \
      --main persona-policies/persona_policies/outputs/tau2_evals/tau2_airline_v5_val_n10_t3.json \
      --out  persona-policies/persona_policies/outputs/tau2_evals/tau2_airline_v5_val_sim2real.json \
      --label ppol-airline-v5val --llm gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _clean(rw):
    if rw is None or (isinstance(rw, float) and math.isnan(rw)):
        return 0.0
    return float(rw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="PPol main eval json (has results[])")
    ap.add_argument(
        "--trajectories",
        default=None,
        help="trajectories json; default = <main>_trajectories.json",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="ppol", help="user_info.implementation")
    ap.add_argument("--llm", default="gpt-5.4", help="user_info.llm")
    args = ap.parse_args()

    main_d = json.load(open(args.main))
    traj_path = args.trajectories or (args.main + "_trajectories.json")
    if not os.path.exists(traj_path):
        print(f"[err] trajectories not found: {traj_path}", file=sys.stderr)
        sys.exit(1)
    traj = json.load(open(traj_path))

    # index trajectories by (task_id, persona_id, trial_idx)
    tidx = {}
    for t in traj:
        k = (
            str(t.get("task_id")),
            str(t.get("persona_id")),
            int(t.get("trial_idx", 0)),
        )
        tidx[k] = t.get("messages", [])

    sims = []
    miss = 0
    for r in main_d.get("results", []):
        k = (
            str(r.get("task_id")),
            str(r.get("persona_id")),
            int(r.get("trial_idx", 0)),
        )
        msgs = tidx.get(k)
        if msgs is None:
            miss += 1
            msgs = []
        sims.append(
            {
                "task_id": str(r.get("task_id")),
                "persona_id": str(r.get("persona_id")),
                "trial_idx": int(r.get("trial_idx", 0)),
                "reward_info": {"reward": _clean(r.get("reward"))},
                "status": r.get("status"),
                "messages": msgs,
            }
        )

    out = {
        "info": {
            "user_info": {"implementation": args.label, "llm": args.llm},
            "domain": main_d.get("domain"),
            "agent_llm": main_d.get("agent_llm"),
            "user_llm": main_d.get("user_llm"),
            "reasoning_effort": main_d.get("reasoning_effort"),
            "split_file": main_d.get("split_file"),
            "num_trials": main_d.get("num_trials"),
        },
        "simulations": sims,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    n_msgs = sum(1 for s in sims if s["messages"])
    print(
        f"[write] {args.out}: {len(sims)} simulations ({n_msgs} with messages, {miss} missing traj)"
    )
    # task_id sample for alignment check
    tids = sorted({s["task_id"] for s in sims})
    print(f"[tasks] {len(tids)} unique task_ids; sample: {tids[:5]}")


if __name__ == "__main__":
    main()
