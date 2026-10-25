"""
Evaluate PPol-generated personas on vitabench by running full rollouts.

Persona injection: vitabench's user system prompt has a single <persona>{persona}
placeholder that defaults to user_profile (facts: job/address/diet/personality).
PGEPA replaces it wholesale with a full persona. PPol's expanded_instruction is
*behavior-only* and would drop the facts if substituted directly, so we compose:
    persona_text = str(user_profile) + "\n\n## Behavioral Style\n" + expanded_instruction
preserving the task facts while layering the behavioral policy on top.

Runs via the vitabench checkout's venv (which has `vita` importable):
    /path/to/vitabench/.venv/bin/python persona_policies/scripts/run_vitabench_eval.py \
        --persona-json .../vitabench_delivery_pgepa_test_n10.json \
        --n-tasks 1 --n-personas 2 --env-file ../.env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_VITABENCH_ROOT = _REPO.parent / "vitabench"
_DEFAULT_OUT_DIR = _REPO / "persona_policies/outputs/vitabench_evals"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run vitabench rollouts with PPol personas.")
    ap.add_argument("--persona-json", required=True, help="PPol generated persona JSON")
    ap.add_argument("--domain", default="delivery")
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--n-personas", type=int, default=None)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument(
        "--evaluator-llm",
        default="gpt-4.1",
        help="judge model for reward computation (PGEPA used gpt-4.1). Must "
        "exist as a key in vitabench's models.yaml; its dict entry is passed "
        "as llm_args_evaluator. The config.py default "
        "(anthropic.claude-3.7-sonnet) is NOT in models.yaml → KeyError, so "
        "this must always be set explicitly.",
    )
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-trials", type=int, default=1,
                    help="repeat each (task, persona) N times with seed+i")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="concurrent episodes (1 = serial)")
    ap.add_argument("--save-trajectories", action="store_true", default=True,
                    help="save rollout conversation messages (default on)")
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--vitabench-root", default=str(_DEFAULT_VITABENCH_ROOT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    vb_root = Path(args.vitabench_root)
    # Resolve paths against the current CWD before chdir'ing to vitabench root.
    args.persona_json = str(Path(args.persona_json).resolve())
    if args.out:
        args.out = str(Path(args.out).resolve())
    sys.path.insert(0, str(vb_root / "src"))
    os.chdir(vb_root)  # vitabench locates data/ relative to its repo root

    from vita.run import get_tasks, run_task  # noqa: E402
    from vita.config import models as vita_models  # noqa: E402

    # Resolve evaluator args from models.yaml (the judge's base_url/headers/
    # max_tokens/reasoning_effort live there). The config.py default evaluator
    # (anthropic.claude-3.7-sonnet) has no models.yaml entry → KeyError, so we
    # always pass both llm_evaluator and llm_args_evaluator together.
    if args.evaluator_llm not in vita_models:
        raise RuntimeError(
            f"evaluator-llm {args.evaluator_llm!r} not in vitabench models.yaml; "
            f"available: {sorted(vita_models)}"
        )
    evaluator_args = vita_models[args.evaluator_llm]

    tasks = get_tasks(args.domain, None, None, args.language)
    task_map = {str(t.id): t for t in tasks}
    print(f"[load] vitabench domain={args.domain} tasks={len(tasks)} | "
          f"agent={args.agent_llm}+{args.reasoning_effort} user={args.user_llm}+{args.reasoning_effort} "
          f"eval={args.evaluator_llm}",
          flush=True)

    # ── Build work items ─────────────────────────────────────────────────
    d = json.load(open(args.persona_json, encoding="utf-8"))
    task_entries = d["tasks"][: args.n_tasks] if args.n_tasks else d["tasks"]

    work_items: list[dict] = []
    for te in task_entries:
        tid = te["task_id"]
        task = task_map.get(tid)
        if task is None:
            print(f"[skip] task {tid} not in vitabench", flush=True)
            continue
        user_profile = str(task.user_scenario.user_profile)
        personas = te["personas"][: args.n_personas] if args.n_personas else te["personas"]
        for pi, p in enumerate(personas):
            expanded = p.get("expanded_instruction", "")
            persona_text = user_profile + "\n\n## Behavioral Style\n" + expanded
            for trial in range(args.num_trials):
                work_items.append({
                    "task": task,
                    "task_id": tid,
                    "persona": p,
                    "persona_idx": pi,
                    "persona_text": persona_text,
                    "trial": trial,
                    "seed": args.seed + trial,
                })

    print(f"[work] {len(work_items)} episodes ({len(task_entries)} tasks × "
          f"up to {args.n_personas or '?'} personas × {args.num_trials} trials) "
          f"workers={args.max_workers}",
          flush=True)

    # ── Run episodes ─────────────────────────────────────────────────────
    results = []
    trajectories: list[dict] = []

    from threading import Lock as _VBLock
    _print_lock = _VBLock()

    def _run_one(wi: dict) -> dict:
        p = wi["persona"]
        tid = wi["task_id"]
        pid = p.get("persona_id")
        t0 = time.time()
        try:
            sim = run_task(
                domain=args.domain,
                task=wi["task"],
                agent="llm_agent",
                user="user_simulator",
                llm_agent=args.agent_llm,
                llm_args_agent={"reasoning_effort": args.reasoning_effort},
                llm_user=args.user_llm,
                llm_args_user={"reasoning_effort": args.reasoning_effort},
                llm_evaluator=args.evaluator_llm,
                llm_args_evaluator=evaluator_args,
                max_steps=args.max_steps,
                language=args.language,
                user_persona_override=wi["persona_text"],
                seed=wi["seed"],
            )
            ri = getattr(sim, "reward_info", None)
            reward = getattr(ri, "reward", None) if ri is not None else getattr(sim, "reward", None)
            msgs = sim.get_messages() if hasattr(sim, "get_messages") else getattr(sim, "messages", [])
            traj = [
                {"role": str(getattr(m, "role", "")), "content": str(getattr(m, "content", ""))}
                for m in msgs
            ]
            status = "ok"
        except Exception as e:
            reward = None
            traj = []
            status = f"error: {type(e).__name__}: {str(e)[:200]}"
        dt = time.time() - t0
        with _print_lock:
            print(f"[{tid}] p{wi['persona_idx']} t{wi['trial']} ({pid}): "
                  f"reward={reward} n_msgs={len(traj)} {status} {dt:.0f}s", flush=True)
        return {
            "task_id": tid,
            "persona_id": pid,
            "axis_placement": p.get("axis_placement"),
            "trial_idx": wi["trial"],
            "seed": wi["seed"],
            "reward": reward,
            "status": status,
            "n_msgs": len(traj),
            "elapsed_s": round(dt, 1),
            "_trajectory": traj,
        }

    t_total = time.time()
    if args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            fut_map = {pool.submit(_run_one, wi): wi for wi in work_items}
            for fut in as_completed(fut_map):
                r = fut.result()
                if args.save_trajectories and r.get("_trajectory"):
                    trajectories.append({
                        "task_id": r["task_id"],
                        "persona_id": r["persona_id"],
                        "trial_idx": r["trial_idx"],
                        "messages": r.pop("_trajectory"),
                    })
                else:
                    r.pop("_trajectory", None)
                results.append(r)
    else:
        for wi in work_items:
            r = _run_one(wi)
            if args.save_trajectories and r.get("_trajectory"):
                trajectories.append({
                    "task_id": r["task_id"],
                    "persona_id": r["persona_id"],
                    "trial_idx": r["trial_idx"],
                    "messages": r.pop("_trajectory"),
                })
            else:
                r.pop("_trajectory", None)
            results.append(r)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    print(f"\n[done] {len(results)} episodes in {time.time()-t_total:.0f}s "
          f"(ok={n_ok} err={n_err})", flush=True)

    # ── Write ─────────────────────────────────────────────────────────────
    out_dir = _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        n_p = args.n_personas or len(task_entries[0]["personas"]) if task_entries else "?"
        out_path = out_dir / f"vitabench_{args.domain}_pgepa_test_n{n_p}_t{args.num_trials}.json"

    json.dump({
        "domain": args.domain,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
        "evaluator_llm": args.evaluator_llm,
        "reasoning_effort": args.reasoning_effort,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "persona_json": args.persona_json,
        "results": results,
    }, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[write] {out_path}", flush=True)

    if trajectories:
        traj_path = Path(str(out_path)).with_suffix("").with_suffix(
            Path(str(out_path)).suffix + "_trajectories.json"
        )
        traj_path.write_text(json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] trajectories → {traj_path} ({len(trajectories)} episodes)", flush=True)

    rewards = [r["reward"] for r in results if r["reward"] is not None]
    if rewards:
        sr = sum(1 for r in rewards if r is not None and r >= 1.0) / len(rewards)
        print(f"[summary] reward mean={sum(rewards)/len(rewards):.3f} "
              f"success_rate={sr:.2%} n={len(rewards)}", flush=True)


if __name__ == "__main__":
    main()
