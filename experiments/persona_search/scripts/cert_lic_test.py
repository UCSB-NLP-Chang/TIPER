#!/usr/bin/env python3
"""Cert: best persona vs no-persona on LiC test split.

agent = minimax-m3 (high, via :4114 openai->fireworks shim), user/system = gpt-5.4
(high, standard OpenAI). Reuses LICPersonaAdapter.run_simulations so the
materialize/wrap_persona + run_demo subprocess + demo_runs.json parsing path is
identical to the search. no-persona = wrap_persona("") (the search seed).

Usage:
  python cert_lic_test.py --domain database --split-file lost_in_conversation/data/lic_split_database.json \
      --best-persona data/persona_opt/pgepa_runs/minimax_3domain/lic_database/isl_lic_database_g0/best_persona.md \
      --trials 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
DATA_ROOT = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
_LIC = _REPO / "vendor" / "lost_in_conversation"
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


def cert_condition(adapter, tasks, persona_text, label):
    arts = adapter.run_simulations(tasks, {"persona_text": persona_text})
    per_task = []
    for t in tasks:
        trials = arts.runs_by_task.get(t.task_id) or []
        corrects = [bool(r.get("is_correct")) for r in trials if "error" not in r]
        per_task.append(sum(corrects) / len(corrects) if corrects else None)
    valid = [x for x in per_task if x is not None]
    avg_reward = sum(valid) / len(valid) if valid else float("nan")
    n_valid = len(valid)
    n_infra = sum(1 for x in per_task if x is None)
    # illegal-user breakdown (validity health)
    n_illegal = 0
    for t in tasks:
        trials = arts.runs_by_task.get(t.task_id) or []
        outs = [classify_outcome(r) for r in trials]
        if _aggregate_outcome(outs) == Outcome.ILLEGAL_USER_BEHAVIOR:
            n_illegal += 1
    diff = 1.0 - avg_reward if avg_reward == avg_reward else float("nan")
    print(
        f"  [{label}] n={len(tasks)} valid={n_valid} infra={n_infra} "
        f"illegal={n_illegal} avg_reward={avg_reward:.3f} difficulty={diff:.3f}",
        flush=True,
    )
    return {
        "avg_reward": avg_reward,
        "n_valid": n_valid,
        "n_infra": n_infra,
        "n_illegal": n_illegal,
        "per_task": per_task,
        "results_paths": arts.results_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain", required=True, choices=["database", "math", "code", "actions"]
    )
    ap.add_argument("--split-file", required=True)
    ap.add_argument(
        "--best-persona", required=True, help="path to raw persona_text (search best)"
    )
    ap.add_argument("--agent-model", default="accounts/fireworks/models/minimax-m3")
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
            DATA_ROOT / "data/persona_opt/pgepa_runs/minimax_3domain/cert" / args.domain
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    best_persona_text = Path(args.best_persona).read_text()
    tasks = load_test_tasks(args.split_file, args.domain)
    print(
        f"LiC {args.domain} cert: {len(tasks)} test tasks, {args.trials} trials, "
        f"agent={args.agent_model}/{args.effort} user={args.user_model}/{args.user_effort}",
        flush=True,
    )

    adapter = LICPersonaAdapter(
        persona_id=f"cert_{args.domain}",
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
        # rule-only legacy semantics for raw-reward measurement (no LLM judge)
        always_judge_minibatch=False,
    )
    # SystemAgent (verify/extract) must use a CONSTANT reliable model (the agent
    # model, e.g. gpt-5.4), NOT the user_model under test. Otherwise (a) the
    # verifier varies with user_model (unfair reward comparison across user
    # models), and (b) claude user_model → system calls with temperature=0.0
    # → vertex shim rejects (thinking requires temp=1) → 100% infra errors.
    adapter._system_model = args.agent_model

    print("\n[1/2] no-persona baseline (wrap_persona('') = search seed):", flush=True)
    no_p = cert_condition(adapter, tasks, "", f"{args.domain}/no-persona")
    print("\n[2/2] best persona:", flush=True)
    best = cert_condition(
        adapter, tasks, best_persona_text, f"{args.domain}/best-persona"
    )

    delta = (
        (no_p["avg_reward"] - best["avg_reward"])
        if (
            no_p["avg_reward"] == no_p["avg_reward"]
            and best["avg_reward"] == best["avg_reward"]
        )
        else float("nan")
    )
    print(
        f"\n  Δ reward (no-persona - best) = {delta:+.3f}  "
        f"({'persona HARDER ✓' if delta > 0 else 'no improvement ✗'})",
        flush=True,
    )

    report = {
        "domain": args.domain,
        "n_test": len(tasks),
        "trials": args.trials,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "effort": args.effort,
        "no_persona": {k: v for k, v in no_p.items() if k != "per_task"},
        "best_persona": {k: v for k, v in best.items() if k != "per_task"},
        "delta_reward": delta,
        "best_persona_path": str(args.best_persona),
    }
    rp = out_dir / "cert_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report -> {rp}", flush=True)


if __name__ == "__main__":
    main()
