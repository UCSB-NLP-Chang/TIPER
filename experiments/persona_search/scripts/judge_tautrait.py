"""Run the maj@1 offline user-behavior judge on τ-Trait's released GPT-5 trajectories.

τ-Trait ships 18 rollout files (4 traits × 4 domains + 2 none baselines, 910 sims
total) under ``tau-trait/trajectories/gpt5_trajectories/``. Each rollout stores the
AGENT-side transcript + the neutral task instruction the user-sim was given; the
trait (skeptical / confusion / impatience / incoherence) is a persona overlay that
is NOT recorded in the trace.

This script:
  1. converts those rollouts into the tau2 ``results.json`` shape the existing
     judge (``experiments/baseline/scripts/judge_user_intent_preservation.py``) consumes — the neutral
     ``info.task.instruction`` becomes the ROLE-PLAY INSTRUCTIONS the customer is
     bound to (the trait is deliberately NOT passed as persona, so the judge audits
     whether the trait-driven behavior crosses the "reasonable customer" line
     against the neutral task, which is the question we want answered);
  2. runs the judge in synchronous maj@1 mode (one chat-completion per sim);
  3. aggregates verdicts per (trait, domain) + overall: violation / critical /
     intent-lost rates and violation-type breakdowns.

Usage:
  uv run python experiments/persona_search/scripts/judge_tautrait.py convert \\
      --traj-dir tau-trait/trajectories/gpt5_trajectories \\
      --out data/intent_eval/tautrait
  uv run python experiments/persona_search/scripts/judge_tautrait.py judge \\
      --out data/intent_eval/tautrait --model gpt-5.2 --concurrency 16
  uv run python experiments/persona_search/scripts/judge_tautrait.py aggregate \\
      --out data/intent_eval/tautrait
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIT_DIR = Path(__file__).resolve().parent
JUDGE_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "baseline"
    / "scripts"
    / "judge_user_intent_preservation.py"
)

# filename -> (trait, domain). "none" baselines carry trait="none".
_FILES = {
    "confusion_airline.json": ("confusion", "airline"),
    "confusion_retail.json": ("confusion", "retail"),
    "confusion_telecom.json": ("confusion", "telecom"),
    "confusion_telehealth.json": ("confusion", "telehealth"),
    "impatience_airline.json": ("impatience", "airline"),
    "impatience_retail.json": ("impatience", "retail"),
    "impatience_telecom.json": ("impatience", "telecom"),
    "impatience_telehealth.json": ("impatience", "telehealth"),
    "incoherence_airline.json": ("incoherence", "airline"),
    "incoherence_retail.json": ("incoherence", "retail"),
    "incoherence_telecom.json": ("incoherence", "telecom"),
    "incoherence_telehealth.json": ("incoherence", "telehealth"),
    "skeptical_airline.json": ("skeptical", "airline"),
    "skeptical_retail.json": ("skeptical", "retail"),
    "skeptical_telecom.json": ("skeptical", "telecom"),
    "skeptical_telehealth.json": ("skeptical", "telehealth"),
    "telecom_none.json": ("none", "telecom"),
    "telehealth_none.json": ("none", "telehealth"),
}


def _convert_traj(traj: list[dict]) -> list[dict]:
    """τ-Trait {role,content} -> tau2 judge messages. Drop the agent system policy
    (the judge's _render ignores role=system anyway). Assistant tool-call
    precursors (content=null) are kept; _render skips null assistant text."""
    msgs = []
    for e in traj:
        role = e["role"]
        if role == "system":
            continue
        msgs.append({"role": role, "content": e["content"]})
    return msgs


def cmd_convert(args) -> None:
    traj_dir = Path(args.traj_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks, sims, order = [], [], []
    seen_task_ids = set()
    idx = 0
    for fname, (trait, domain) in _FILES.items():
        fpath = traj_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fpath} not found")
            continue
        rollouts = json.load(open(fpath))
        for r in rollouts:
            tid = r["task_id"]
            task_key = f"{domain}|{tid}"
            instr = (r.get("info") or {}).get("task", {}).get("instruction", "")
            if task_key not in seen_task_ids:
                seen_task_ids.add(task_key)
                tasks.append(
                    {
                        "id": task_key,
                        "user_scenario": {
                            "instructions": {"task_instructions": instr},
                        },
                    }
                )
            sims.append(
                {
                    "task_id": task_key,
                    "trial": r.get("trial", 0),
                    "reward_info": {"reward": r.get("reward")},
                    "messages": _convert_traj(r["traj"]),
                }
            )
            order.append(
                {
                    "idx": idx,
                    "trait": trait,
                    "domain": domain,
                    "task_id": tid,
                    "file": fname,
                    "reward": r.get("reward"),
                }
            )
            idx += 1

    results = {"tasks": tasks, "simulations": sims}
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    (out / "sim_order.json").write_text(json.dumps(order, ensure_ascii=False, indent=2))
    print(f"  wrote {out / 'results.json'}: {len(tasks)} tasks, {len(sims)} sims")
    print(f"  wrote {out / 'sim_order.json'}: {len(order)} sim-order rows")


def cmd_judge(args) -> None:
    out = Path(args.out)
    results = out / "results.json"
    if not results.exists():
        sys.exit(f"missing {results}; run `convert` first")
    cmd = [
        "uv",
        "run",
        "python",
        str(JUDGE_SCRIPT),
        "sync",
        "--results",
        str(results),
        "--model",
        args.model,
        "--maj",
        "1",
        "--concurrency",
        str(args.concurrency),
        "--out",
        str(out),
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def cmd_aggregate(args) -> None:
    out = Path(args.out)
    order = json.load(open(out / "sim_order.json"))
    rows = [
        json.loads(l)
        for l in (out / "per_sim.jsonl").read_text().splitlines()
        if l.strip()
    ]
    if len(rows) != len(order):
        print(f"  WARN: per_sim rows={len(rows)} != order={len(order)}")
    # The judge's _select_sims reorders sims by (task_id, trial) before judging,
    # and per_sim.jsonl is written in that post-sort (sim_meta) order. Replicate
    # the exact sort on our file-order sim list so the positional zip aligns.
    # task_id in results.json == f"{domain}|{tid}"; stable sort keeps file
    # (=trait) order within each (domain,tid) group.
    order_sorted = sorted(order, key=lambda m: (f"{m['domain']}|{m['task_id']}", 0))
    by_key = {}  # (trait,domain) -> list of row
    overall = []
    for i, row in enumerate(rows):
        meta = (
            order_sorted[i] if i < len(order_sorted) else {"trait": "?", "domain": "?"}
        )
        row = {
            **row,
            "trait": meta.get("trait"),
            "domain": meta.get("domain"),
            "orig_task_id": meta.get("task_id"),
        }
        overall.append(row)
        by_key.setdefault((meta["trait"], meta["domain"]), []).append(row)

    def _rates(rs):
        n = len([r for r in rs if r.get("verdict") != "no_vote"])
        if not n:
            return None
        viol = [r for r in rs if r.get("verdict") == "violation"]
        crit = [
            r
            for r in rs
            if r.get("verdict") == "violation"
            and any(
                v.get("severity") == "critical" for v in (r.get("violations") or [])
            )
        ]
        intent_lost = [r for r in rs if r.get("intent_preserved") is False]
        vtypes = Counter(v["type"] for r in viol for v in (r.get("violations") or []))
        vsubs = Counter(
            v.get("subtype") for r in viol for v in (r.get("violations") or [])
        )
        return {
            "n_judged": n,
            "n_total": len(rs),
            "violations": len(viol),
            "violation_rate": round(len(viol) / n, 3),
            "critical": len(crit),
            "critical_rate": round(len(crit) / n, 3),
            "intent_lost": len(intent_lost),
            "intent_lost_rate": round(len(intent_lost) / n, 3),
            "violation_types": dict(vtypes),
            "violation_subtypes": dict(vsubs),
            "reward_mean": round(sum(r.get("reward") or 0 for r in rs) / len(rs), 3),
        }

    per_group = {}
    for (trait, domain), rs in sorted(by_key.items()):
        per_group[f"{trait}|{domain}"] = _rates(rs)
    # per-trait rollup (across domains) and per-domain rollup
    per_trait, per_domain = {}, {}
    by_trait, by_dom = {}, {}
    for r in overall:
        by_trait.setdefault(r["trait"], []).append(r)
        by_dom.setdefault(r["domain"], []).append(r)
    for t, rs in sorted(by_trait.items()):
        per_trait[t] = _rates(rs)
    for d, rs in sorted(by_dom.items()):
        per_domain[d] = _rates(rs)

    summary = {
        "overall": _rates(overall),
        "per_trait": per_trait,
        "per_domain": per_domain,
        "per_trait_domain": per_group,
    }
    (out / "tautrait_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # pretty print
    def _fmt(d):
        if not d:
            return "n/a"
        return (
            f"n={d['n_judged']}/{d['n_total']} viol={d['violations']}({d['violation_rate']}) "
            f"crit={d['critical']}({d['critical_rate']}) intent_lost={d['intent_lost']}({d['intent_lost_rate']}) "
            f"rew={d['reward_mean']}"
        )

    print("\n===== OVERALL =====")
    print("  " + _fmt(summary["overall"]))
    if summary["overall"]:
        print("  types:", summary["overall"]["violation_types"])
        print("  subtypes:", summary["overall"]["violation_subtypes"])
    print("\n===== PER TRAIT =====")
    for t, d in summary["per_trait"].items():
        print(f"  {t:12} {_fmt(d)}")
    print("\n===== PER DOMAIN =====")
    for d_, d in summary["per_domain"].items():
        print(f"  {d_:12} {_fmt(d)}")
    print("\n===== PER TRAIT×DOMAIN =====")
    for k, d in summary["per_trait_domain"].items():
        print(f"  {k:26} {_fmt(d)}")
    print(f"\n  wrote {out / 'tautrait_judge_summary.json'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--traj-dir", default="tau-trait/trajectories/gpt5_trajectories")
    c.add_argument("--out", default=str(DATA_ROOT / "data/intent_eval/tautrait"))
    c.set_defaults(func=cmd_convert)
    j = sub.add_parser("judge")
    j.add_argument("--out", default=str(DATA_ROOT / "data/intent_eval/tautrait"))
    j.add_argument("--model", default="gpt-5.2")
    j.add_argument("--concurrency", type=int, default=16)
    j.set_defaults(func=cmd_judge)
    a = sub.add_parser("aggregate")
    a.add_argument("--out", default=str(DATA_ROOT / "data/intent_eval/tautrait"))
    a.set_defaults(func=cmd_aggregate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
