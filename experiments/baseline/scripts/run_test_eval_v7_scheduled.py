"""Scheduled test-set evaluation for GEPA v7.

Waits ~5h (by when v7 is expected to converge), then evaluates the SEED personas
plus the TOP-3 optimized personas (highest val difficulty) on the full test split
(telecom 56 + airline 33 + retail 45 = 134 tasks), 2 trials each, matching the
GEPA config (agent gpt-5.4 high, user user_simulator gpt-5.2).

To avoid API contention, after the delay it waits for the v7 GEPA process to
finish (so the pool is final and nothing competes), unless --no-wait-v7.

Selection is read from the v7 state AT TRIGGER TIME, so it reflects the latest
converged pool. Results -> data/simulations/gepa_opt/test_eval_v7/<label>/<domain>.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
VENV = REPO / ".venv" / "bin"
STATE = REPO / "data/persona_opt/pgepa_runs/xdom_v7_3domain/pgepa_state.json"
SPLIT = REPO / "experiments/persona_search/configs/crossdomain_variance_split_v3.json"
DOMAINS = ["telecom", "airline", "retail"]


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def v7_running():
    r = subprocess.run(
        ["pgrep", "-f", "crossdomain_gepa.py"], capture_output=True, text=True
    )
    return [p for p in r.stdout.split() if p]


def sanitize(name):
    return "".join(c if c.isalnum() else "_" for c in name)[:32].strip("_")


def select_personas(top):
    st = json.loads(STATE.read_text())
    cands = [c for c in st["candidates"] if c.get("feasible")]
    seeds = [c for c in cands if c["gen"] == 0]
    opt = sorted(
        [c for c in cands if c["gen"] > 0],
        key=lambda c: c.get("agg") or 0.0,
        reverse=True,
    )[:top]
    chosen = []
    for c in seeds:
        nm = (
            c["persona_text"]
            .splitlines()[0]
            .replace("# Persona:", "")
            .split("(effect")[0]
            .strip()
        )
        chosen.append(("seed", f"seed_i{c['idx']}_{sanitize(nm)}", c))
    for c in opt:
        nm = (
            c["persona_text"]
            .splitlines()[0]
            .replace("# Persona:", "")
            .split("(effect")[0]
            .strip()
        )
        chosen.append(("optim", f"opt_i{c['idx']}_g{c['gen']:02d}_{sanitize(nm)}", c))
    return chosen


def run(args):
    test = json.loads(SPLIT.read_text())["test"]
    chosen = select_personas(args.top)
    log(f"selected {len(chosen)} personas:")
    for grp, label, c in chosen:
        log(f"  [{grp}] {label}  (idx{c['idx']} gen{c['gen']} val_agg={c.get('agg')})")
    Path(REPO / "data/simulations/gepa_opt/test_eval_v7").mkdir(
        parents=True, exist_ok=True
    )
    (REPO / "data/simulations/gepa_opt/test_eval_v7/selection.json").write_text(
        json.dumps(
            [
                {
                    "group": g,
                    "label": l,
                    "idx": c["idx"],
                    "gen": c["gen"],
                    "val_agg": c.get("agg"),
                }
                for g, l, c in chosen
            ],
            indent=2,
        )
    )

    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "REDACTED_USE_OPENAI_API_KEY_ENV")

    for grp, label, c in chosen:
        persona_json = json.dumps({"custom_guidelines": c["persona_text"]})
        for domain in DOMAINS:
            tids = test.get(domain, [])
            if not tids:
                continue
            save_to = f"gepa_opt/test_eval_v7/{label}/{domain}"
            cmd = [
                str(VENV / "tau2"),
                "run",
                "--domain",
                domain,
                "--user",
                "user_simulator",
                "--user-llm",
                "gpt-5.2",
                "--agent-llm",
                "gpt-5.4",
                "--agent-llm-args",
                json.dumps({"reasoning_effort": "high"}),
                "--num-trials",
                str(args.trials),
                "--max-concurrency",
                str(args.concurrency),
                "--max-steps",
                "120",
                "--timeout",
                "600",
                "--seed",
                "300",
                "--save-to",
                save_to,
                "--auto-resume",
                "--user-persona",
                persona_json,
                "--task-ids",
                *tids,
            ]
            log(f"RUN {label} [{domain}] {len(tids)} tasks x{args.trials} -> {save_to}")
            if args.dry_run:
                log("  (dry-run, skip)")
                continue
            r = subprocess.run(cmd, env=env)
            if r.returncode != 0:
                log(f"  WARNING {label}/{domain} exited {r.returncode}")
    log("test_eval_v7 complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay-hours", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--no-wait-v7",
        action="store_true",
        help="don't wait for v7 to finish after the delay",
    )
    ap.add_argument(
        "--wait-v7-cap-hours",
        type=float,
        default=4.0,
        help="max extra wait for v7 to finish",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print selection+commands, no sleep, no run",
    )
    args = ap.parse_args()

    if args.dry_run:
        log("DRY RUN")
        run(args)
        return

    log(f"sleeping {args.delay_hours}h before test eval ...")
    time.sleep(args.delay_hours * 3600)

    if not args.no_wait_v7:
        cap = time.time() + args.wait_v7_cap_hours * 3600
        while v7_running() and time.time() < cap:
            log(
                f"v7 still running (pids {v7_running()}); waiting 5min to avoid contention ..."
            )
            time.sleep(300)
        if v7_running():
            log("v7 still running past cap; proceeding anyway.")
        else:
            log("v7 finished; starting test eval.")
    run(args)


if __name__ == "__main__":
    main()
