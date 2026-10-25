#!/usr/bin/env python3
"""No-persona baseline test on LiC test split (headroom check).

Sister of cert_lic_test.py but runs ONLY the no-persona condition
(wrap_persona("") = the search seed). Used to measure a model's baseline
difficulty BEFORE committing to a full GEPA optim — if the baseline is near
ceiling (e.g. GLM-5.2 LiC 0.837), optim is unlikely to find headroom.

Reuses LICPersonaAdapter.run_simulations so the materialize/wrap_persona +
run_demo subprocess + demo_runs.json parsing path is identical to the search.

Usage:
  python lic_nopersona_test.py --domain math \
      --split-file lost_in_conversation/data/lic_split_math.json \
      --agent-model gpt-oss-120b --effort high --trials 3 \
      --out-dir data/persona_opt/headroom_3model/gpt-oss-120b/lic_math
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
_LIC = _REPO / "lost_in_conversation"
for _p in (str(_LIC), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lic_persona_adapter import (  # noqa: E402
    LICPersonaAdapter,
    LICTask,
    Outcome,
    _aggregate_outcome,
    classify_outcome,
)


def load_test_tasks(split_file: str, domain: str) -> list[LICTask]:
    d = json.load(open(split_file))
    ids = d["test"][domain]
    return [LICTask(task_id=tid, domain=domain) for tid in ids]


def nopersona_condition(adapter, tasks, label):
    arts = adapter.run_simulations(tasks, {"persona_text": ""})
    per_task = []
    for t in tasks:
        trials = arts.runs_by_task.get(t.task_id) or []
        corrects = [bool(r.get("is_correct")) for r in trials if "error" not in r]
        per_task.append(sum(corrects) / len(corrects) if corrects else None)
    valid = [x for x in per_task if x is not None]
    avg_reward = sum(valid) / len(valid) if valid else float("nan")
    n_valid = len(valid)
    n_infra = sum(1 for x in per_task if x is None)
    n_illegal = 0
    for t in tasks:
        trials = arts.runs_by_task.get(t.task_id) or []
        outs = [classify_outcome(r) for r in trials]
        if _aggregate_outcome(outs) == Outcome.ILLEGAL_USER_BEHAVIOR:
            n_illegal += 1
    diff = 1.0 - avg_reward if avg_reward == avg_reward else float("nan")
    fully_correct = sum(1 for x in valid if x >= 1.0)
    pass3 = sum(1 for x in per_task if x is not None and x >= 1.0)
    print(
        f"  [{label}] n={len(tasks)} valid={n_valid} infra={n_infra} "
        f"illegal={n_illegal} avg_reward={avg_reward:.3f} difficulty={diff:.3f} "
        f"fully_correct={fully_correct}/{len(tasks)}",
        flush=True,
    )
    return {
        "avg_reward": avg_reward,
        "n_valid": n_valid,
        "n_infra": n_infra,
        "n_illegal": n_illegal,
        "per_task": per_task,
        "fully_correct": fully_correct,
        "pass3": pass3,
        "results_paths": arts.results_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain", required=True, choices=["database", "math", "code", "actions"]
    )
    ap.add_argument("--split-file", required=True)
    ap.add_argument(
        "--agent-model",
        default="accounts/fireworks/models/glm-5p2",
        help="canonical id for new fireworks models (e.g. gpt-oss-120b); "
        "accounts/fireworks/models/<x> for GLM/m3",
    )
    ap.add_argument("--user-model", default="gpt-5.4")
    ap.add_argument(
        "--effort", default="high", help="agent (assistant) reasoning effort"
    )
    ap.add_argument(
        "--user-effort",
        default="high",
        help="user-simulator reasoning effort (kept at high while agent can be max)",
    )
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subprocess-timeout", type=int, default=7200)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            _REPO
            / "data/persona_opt/headroom_3model"
            / args.agent_model.replace("/", "_")
            / f"lic_{args.domain}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_test_tasks(args.split_file, args.domain)
    print(
        f"LiC {args.domain} no-persona: {len(tasks)} test tasks, {args.trials} trials, "
        f"agent={args.agent_model}/{args.effort} user={args.user_model}/{args.user_effort}",
        flush=True,
    )

    adapter = LICPersonaAdapter(
        persona_id=f"nopersona_{args.domain}",
        agent_model=args.agent_model,
        user_model=args.user_model,
        reasoning_effort=args.effort,
        user_reasoning_effort=args.user_effort,
        domains=[args.domain],
        n_workers=args.workers,
        num_trials=args.trials,
        lic_dir=str(_LIC),
        eval_root=str(out_dir),
        subprocess_timeout=args.subprocess_timeout,
        mutation_context="basic",
        # rule-only legacy semantics for raw-reward baseline (no LLM judge)
        always_judge_minibatch=False,
    )

    print("\n[no-persona baseline] (wrap_persona('') = search seed):", flush=True)
    no_p = nopersona_condition(adapter, tasks, f"{args.domain}/no-persona")

    avg = no_p["avg_reward"]
    diff = 1.0 - avg if avg == avg else float("nan")
    print(
        f"\n  baseline avg_reward={avg:.3f} difficulty={diff:.3f}  "
        f"({'HIGH CEILING — optim headroom likely small' if avg >= 0.80 else 'headroom looks plausible'})",
        flush=True,
    )

    report = {
        "domain": args.domain,
        "n_test": len(tasks),
        "trials": args.trials,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "effort": args.effort,
        "user_effort": args.user_effort,
        "no_persona": {k: v for k, v in no_p.items() if k != "per_task"},
        "baseline_avg_reward": avg,
        "baseline_difficulty": diff,
        "headroom_note": (
            "high_ceiling" if (avg == avg and avg >= 0.80) else "plausible"
        ),
    }
    rp = out_dir / "nopersona_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report -> {rp}", flush=True)


if __name__ == "__main__":
    main()
