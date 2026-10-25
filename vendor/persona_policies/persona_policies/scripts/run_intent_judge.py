"""
Run `judge_user_intent_preservation.py` on tau2 eval results, filter violations,
then recompute per-task worst persona.

Reads the eval `_results.json` and `_trajectories.json` from `run_tau2_eval.py`,
embeds task instructions into each sim, runs the intent judge (sync mode, gpt-5.2,
maj@N), and writes `_intent_judged.json`.

Usage:
    .venv/bin/python persona_policies/scripts/run_intent_judge.py \\
        --results persona_policies/outputs/tau2_evals/tau2_airline_v5_val_n10_t3.json \\
        --trajectories ..._trajectories.json \\
        --domain airline --judge-model gpt-5.2 --maj 3 --concurrency 20 \\
        --env-file ../.env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TAUBENCH_ROOT = _REPO.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_TAUBENCH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_TAUBENCH_ROOT / "src"))

# Import the shared intent judge from the baseline compatibility path.
sys.path.insert(0, str(_TAUBENCH_ROOT / "experiments" / "baseline" / "scripts"))
from judge_user_intent_preservation import (  # noqa: E402
    _build_body,
    _exempt_control_markers,
    _instr_text,
    _render,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--domain", default="airline")
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--maj", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    from dotenv import load_dotenv
    load_dotenv(args.env_file or (str(_TAUBENCH_ROOT / ".env")))

    os.environ.setdefault("TAU2_DATA_DIR", str(_TAUBENCH_ROOT / "data"))
    from tau2.runner.helpers import get_tasks

    tasks = get_tasks(args.domain, task_split_name=None)
    task_map = {str(t.id): t for t in tasks}

    # Load results and trajectories
    results_data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    traj_data = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))

    # Index trajectories by (task_id, persona_id, trial_idx)
    traj_map = {}
    for t in traj_data:
        key = (str(t["task_id"]), str(t["persona_id"]), t.get("trial_idx", 0))
        traj_map[key] = t["messages"]

    episodes = [r for r in results_data["results"] if r["status"] == "ok"]
    print(f"[load] {len(episodes)} ok episodes, domain={args.domain}, "
          f"judge={args.judge_model} maj={args.maj}", flush=True)

    # Build sim list in the format the judge expects.
    # Pass raw tau2 objects — _instr_text handles both dict and StructuredUserInstructions.
    sims = []
    for ep in episodes:
        key = (str(ep["task_id"]), str(ep["persona_id"]), ep.get("trial_idx", 0))
        msgs = traj_map.get(key, [])
        task = task_map.get(str(ep["task_id"]))
        sims.append({
            "task_id": str(ep["task_id"]),
            "trial": ep.get("trial_idx", 0),
            "reward_info": {"reward": ep.get("reward")},
            "messages": msgs,
            "user_scenario": {
                "persona": getattr(task.user_scenario, "persona", None) if task and task.user_scenario else None,
                "instructions": getattr(task.user_scenario, "instructions", "") if task and task.user_scenario else "",
            },
            "_ep": ep,
        })

    # Run sync judging
    from openai import OpenAI
    client = OpenAI()
    n_req = len(sims) * args.maj
    print(f"[judge] {len(sims)} sims x maj={args.maj} = {n_req} requests "
          f"| concurrency={args.concurrency}", flush=True)

    jobs = []
    for i, s in enumerate(sims):
        instr = _instr_text(s)
        transcript = _render(s)
        body = _build_body(args.judge_model, instr, transcript)
        for v in range(args.maj):
            jobs.append((i, v, body))

    content_by_key = {}

    def _one(job):
        i, v, body = job
        cid = f"s{i}_v{v}"
        try:
            resp = client.chat.completions.create(**body)
            return cid, resp.choices[0].message.content, None
        except Exception as e:
            return cid, None, str(e)[:200]

    t0 = time.time()
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results_list = list(ex.map(_one, jobs))
    for cid, content, err in results_list:
        if err:
            n_err += 1
            continue
        content_by_key[cid] = content
    dt = time.time() - t0
    if n_err:
        print(f"[judge] {n_err}/{n_req} errors in {dt:.0f}s", flush=True)

    # Aggregate votes per sim
    judged = []
    n_violation = 0
    for i, s in enumerate(sims):
        verdicts, intent_votes, all_violations = [], [], []
        for v in range(args.maj):
            txt = content_by_key.get(f"s{i}_v{v}")
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            obj = _exempt_control_markers(obj)
            verdicts.append(obj.get("verdict", "ok"))
            intent_votes.append(bool(obj.get("intent_preserved", True)))
            for viol in obj.get("violations") or []:
                all_violations.append(viol)

        n_votes = len(verdicts)
        if n_votes == 0:
            ep_out = dict(s["_ep"])
            ep_out["intent_verdict"] = "no_vote"
            ep_out["intent_preserved"] = True
            ep_out["intent_violations"] = []
            judged.append(ep_out)
            continue

        severe = verdicts.count("severe_failure") + verdicts.count("violation")
        verdict = "violation" if severe > n_votes / 2 else "ok"
        intent_ok = intent_votes.count(True) >= intent_votes.count(False)

        if verdict == "violation":
            n_violation += 1

        ep_out = dict(s["_ep"])
        ep_out["intent_verdict"] = verdict
        ep_out["intent_preserved"] = intent_ok
        ep_out["intent_violations"] = all_violations if verdict == "violation" else []
        ep_out["intent_votes"] = dict(Counter(verdicts))
        judged.append(ep_out)

    n_total = len(judged)
    print(f"\n[judge done] {n_total} sims | "
          f"ok={n_total - n_violation} violation={n_violation} "
          f"({n_violation/n_total:.1%}) in {dt:.0f}s", flush=True)

    # Breakdown by violation type
    viol_types = Counter(
        v["type"] for ep in judged if ep["intent_verdict"] == "violation"
        for v in ep["intent_violations"]
    )
    viol_severity = Counter(
        v.get("severity", "?") for ep in judged if ep["intent_verdict"] == "violation"
        for v in ep["intent_violations"]
    )
    print(f"  Violation types: {dict(viol_types)}")
    print(f"  Violation severity: {dict(viol_severity)}")

    # Write judged file
    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".intent_judged.json")
    payload = dict(results_data)
    payload["intent_judge_model"] = args.judge_model
    payload["intent_judge_maj"] = args.maj
    payload["results"] = judged
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path}", flush=True)

    # ── Per-task worst persona (intent-ok only) ──
    clean = [r for r in judged if r["intent_verdict"] == "ok"]
    print(f"\n=== Per-task worst persona (intent-ok: {len(clean)}/{n_total}) ===")

    tp = defaultdict(list)
    for r in clean:
        tp[(r["task_id"], r["persona_id"])].append(r.get("reward", 0))
    pa = {k: sum(v)/len(v) for k, v in tp.items()}

    tw = defaultdict(list)
    for (tid, pid), avg_r in pa.items():
        tw[tid].append((pid, avg_r))

    worst = []
    for tid, entries in tw.items():
        w = min(entries, key=lambda x: x[1])
        worst.append((tid, w[0], w[1], len(entries)))
    worst.sort(key=lambda x: x[2])

    rew = [w[2] for w in worst]
    if rew:
        print(f"Tasks: {len(worst)}")
        print(f"Worst-persona mean reward: {sum(rew)/len(rew):.3f}")
        print(f"Worst-persona success rate: {sum(1 for r in rew if r>=1.0)/len(rew):.1%}")
        print()
        for lo, hi, tag in [(0,0.2,'0.0-0.2'),(0.2,0.4,'0.2-0.4'),(0.4,0.6,'0.4-0.6'),(0.6,0.8,'0.6-0.8'),(0.8,1.0,'0.8-1.0')]:
            n = sum(1 for r in rew if lo <= r < hi)
            print(f"  {tag}: {n:2d}")
        print()
        print("Worst 10:")
        for tid, pid, avg_r, n_p in worst[:10]:
            print(f"  {str(tid)[:60]} | {pid:<30} | avg={avg_r:.3f} (n={n_p})")


if __name__ == "__main__":
    main()
