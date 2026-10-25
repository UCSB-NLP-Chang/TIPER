#!/usr/bin/env python3
"""Quick style eval: run N tasks with a persona, compute per-metric Dice vs human.

Usage:
  python experiments/persona_search/scripts/quick_style_eval.py --persona-file FILE --tag v0 --ntasks 10
  python experiments/persona_search/scripts/quick_style_eval.py --compare v0 v1 v2   # compare already-run tags
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = str(REPO / ".venv/bin/tau2")
SIM = REPO / "data/simulations"

sys.path.insert(0, str(REPO))
import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import sim2real_metrics as M
import sim2real_run_tau2 as R
from sim2real_full_table import DIMS_NOIDS, batch_metric_means, dice, load_human

M.set_mode("faithful")

SPLIT = json.loads(
    (
        REPO / "experiments/persona_search/configs/airline_alaska_split_v5.json"
    ).read_text()
)
VAL_IDS = SPLIT["val"]["airline"]

# Key metrics to display (the ones with biggest gaps)
KEY_METRICS = [
    ("Wds/trn", "D2"),
    ("Short%", "D1"),
    ("Formal%", "D1"),
    ("Ack%", "D1"),
    ("Polite%", "D1"),
    ("InfoQ%", "D3"),
    ("Pushbk%", "D3"),
    ("Pivot%", "D4"),
    ("FrntLoad%", "D2"),
    ("Open.Wds", "D2"),
    ("VerbCV", "D1"),
]


def get_human_means():
    rows, _ = load_human()
    for r in rows:
        if "_base" not in r:
            r["_base"] = r["instance_id"].replace("_ann2", "").replace("_ann3", "")
    airline_rows = [r for r in rows if r["_base"].startswith("airline")]
    return batch_metric_means(airline_rows)


def sim_metrics(path):
    d = json.load(open(path))
    sims = d.get("simulations", [])
    acc = {k: [] for k in M.METRIC_ORDER}
    rewards = []
    turn_counts = []
    for s in sims:
        msgs = s.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        turn_counts.append(len(user_msgs))
        mm = M.metrics_for_interaction(R.tau2_user_turns(msgs))
        if mm:
            for k in M.METRIC_ORDER:
                acc[k].append(mm[k])
        rewards.append(s.get("reward_info", {}).get("reward", 0))
    means = {k: st.mean(v) for k, v in acc.items() if v}
    return means, st.mean(rewards), st.mean(turn_counts), len(sims)


def run_and_eval(persona_file: str, tag: str, ntasks: int, ntrials: int = 3):
    persona_text = Path(persona_file).read_text()
    save_to = f"style_iter/{tag}"
    task_ids = VAL_IDS[:ntasks]

    persona_cfg = json.dumps(
        {
            "custom_guidelines": persona_text,
            "verbosity": "standard",
        }
    )

    cmd = [
        TAU2,
        "run",
        "--domain",
        "airline",
        "--agent-llm",
        "gpt-5.4",
        "--agent-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--user",
        "user_simulator",
        "--user-llm",
        "gpt-5.4",
        "--user-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--user-persona",
        persona_cfg,
        "--num-trials",
        str(ntrials),
        "--task-ids",
        *task_ids,
        "--max-steps",
        "120",
        "--max-concurrency",
        "30",
        "--save-to",
        save_to,
    ]

    print(f"Running {ntasks} tasks with tag={tag}...")
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
        return

    print_comparison([tag])


def print_comparison(tags: list[str]):
    human = get_human_means()

    # Reference columns: orig seed + orig opt
    refs = [
        ("orig_seed", SIM / "cert_alaska01_v5/seed/results.json"),
        ("orig_opt", SIM / "cert_alaska01_v5/optim/results.json"),
    ]
    ref_data = {}
    for rlabel, rpath in refs:
        ref_data[rlabel] = sim_metrics(rpath)

    all_labels = list(tags) + [r[0] for r in refs]

    print(f"\n{'Metric':<14} {'Dim':>3} {'Human':>7}", end="")
    for tag in tags:
        print(f"  {tag:>12}", end="")
    for rlabel, _ in refs:
        print(f"  {rlabel:>12}", end="")
    print()
    print("-" * (14 + 3 + 8 + 15 * len(all_labels)))

    all_sim = {}
    for tag in tags:
        path = SIM / f"style_iter/{tag}/results.json"
        if path.exists():
            all_sim[tag] = sim_metrics(path)
        else:
            print(f"  {tag}: no data")
    for rlabel, _ in refs:
        all_sim[rlabel] = ref_data[rlabel]

    for m, dim in KEY_METRICS:
        hv = human.get(m, 0)
        print(f"{m:<14} {dim:>3} {hv:>7.2f}", end="")
        for label in all_labels:
            if label in all_sim:
                sv = all_sim[label][0].get(m, 0)
                d = dice(sv, hv)
                print(f"  {sv:>6.2f}({d:>3.0f})", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    # Dim summary
    print()
    print(f"{'Dimension':<14}    {'':>7}", end="")
    for label in all_labels:
        print(f"  {label:>12}", end="")
    print()
    for dim_name, dim_metrics in DIMS_NOIDS.items():
        print(f"{dim_name:<14}    {'':>7}", end="")
        for label in all_labels:
            if label in all_sim:
                vals = [
                    dice(all_sim[label][0].get(m, 0), human.get(m, 0))
                    for m in dim_metrics
                ]
                print(f"  {st.mean(vals):>12.1f}", end="")
        print()

    # Summary stats
    print()
    for label in all_labels:
        if label in all_sim:
            _, r, t, n = all_sim[label]
            print(f"{label}: n={n} reward={r:.3f} avg_turns={t:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-file", help="Persona markdown file")
    ap.add_argument("--tag", help="Experiment tag (used as save dir)")
    ap.add_argument("--ntasks", type=int, default=10)
    ap.add_argument("--ntrials", type=int, default=3)
    ap.add_argument("--compare", nargs="*", help="Compare existing tags")
    args = ap.parse_args()

    if args.compare:
        print_comparison(args.compare)
    elif args.persona_file and args.tag:
        run_and_eval(args.persona_file, args.tag, args.ntasks, args.ntrials)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
