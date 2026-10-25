"""
Run τ²-bench telecom episodes with PPol-generated personas.

Loads tasks from a PGEPA island-split JSON (telecom_island_split.json test
section, 54 tasks), pairs each task with its pre-generated personas, injects
the ``expanded_instruction`` into the user simulator's system prompt, and runs
one τ² episode per (task, persona) pair via ``run_simulation``.

Usage (run from repo root)::

    python persona_policies/scripts/run_tau2_eval.py \\
        --persona-json persona_policies/outputs/generated_personas/tau2_telecom_pgepa_test_n10.json \\
        --agent-llm gpt-5.4 --user-llm gpt-5.4 --reasoning-effort high \\
        --n-tasks 1 --n-personas 1 \\
        --env-file ../.env

Telecom reward is purely DB-based (all 2285 telecom tasks have no
nl_assertions), so no LLM judge is involved — EvaluationType.ALL only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_TAUBENCH_ROOT = _REPO.parents[1]
_DEFAULT_TELECOM_SPLIT = (
    _DEFAULT_TAUBENCH_ROOT
    / "cresta_persona/data/telecom_island_split.json"
)
_DEFAULT_OUT_DIR = _REPO / "persona_policies/outputs/tau2_evals"

# ── tau2 import helpers ──────────────────────────────────────────────────────


def _ensure_tau2_importable(taubench_root: Path) -> None:
    """Set TAU2_DATA_DIR and prepend tau2's src/ to sys.path."""
    src = taubench_root / "src"
    data = taubench_root / "data"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if data.is_dir():
        os.environ.setdefault("TAU2_DATA_DIR", str(data))


# ── Persona injection ────────────────────────────────────────────────────────

_PERSONA_INJECTION_TEMPLATE = """
## YOUR BEHAVIORAL STYLE FOR THIS CONVERSATION

In addition to your role and task above, you must embody the following behavioral
persona throughout this conversation. This persona affects HOW you communicate,
NOT WHAT you want or need. Your goal, preferences, and private information remain
exactly as described above.

--- PERSONA POLICY BEGIN ---
{persona_policy_text}
--- PERSONA POLICY END ---

CRITICAL RULES FOR PERSONA ADHERENCE:
1. Your GOAL and PRIVATE INFORMATION do not change. You still want to accomplish the same task.
2. Apply the behavioral style consistently across ALL your turns, not just the first one.
3. If the persona says you are terse, be terse even when you have a lot to say.
4. If the persona says you withhold information, do not give it unless the agent specifically asks.
5. If the persona involves emotional state (impatience, confusion), let it show gradually and realistically.
6. Do NOT break character. Do NOT mention that you have a persona or behavioral instructions.
7. The persona should feel like a natural human interaction style, not a caricature.
""".strip()


def _inject_into_system_prompt(base_prompt: str, persona_text: str) -> str:
    block = _PERSONA_INJECTION_TEMPLATE.format(persona_policy_text=persona_text.strip())
    return base_prompt.strip() + "\n\n" + block.strip()


def _build_persona_user_class():
    """Return a UserSimulator subclass that appends persona to system_prompt."""
    from tau2.user.user_simulator import UserSimulator

    class PPolUserSimulator(UserSimulator):
        _ppol_persona_text: str = ""

        @property
        def system_prompt(self) -> str:
            base = super().system_prompt
            if self._ppol_persona_text:
                return _inject_into_system_prompt(base, self._ppol_persona_text)
            return base

    return PPolUserSimulator


# ── Task loading ─────────────────────────────────────────────────────────────


def load_tau2_tasks_from_island_split(
    domain: str,
    split_file: Path,
    split_section: str = "test",
) -> list:
    """Load τ² tasks by id from a PGEPA island-split JSON {domain: [ids]}."""
    from tau2.runner.helpers import get_tasks

    raw = json.loads(split_file.read_text(encoding="utf-8"))
    section = raw.get(split_section, {})
    split_ids: set[str] = set()
    for _d, ids in section.items():
        split_ids.update(str(i) for i in ids)

    all_tasks = get_tasks(domain, task_split_name=None)  # all tasks, filter manually
    by_id = {str(t.id): t for t in all_tasks}
    tasks = [by_id[i] for i in split_ids if i in by_id]
    if len(tasks) != len(split_ids):
        missing = split_ids - {str(t.id) for t in tasks}
        print(f"[warn] {len(missing)} task ids from split not in tau2: {sorted(missing)[:10]}...",
              flush=True)
    return tasks


# ── Episode runner ───────────────────────────────────────────────────────────


def run_episode(
    task,
    domain: str,
    persona_text: str,
    agent_model: str,
    agent_args: dict,
    user_model: str,
    user_args: dict,
    seed: int,
    max_steps: int,
    max_turns: Optional[int] = None,
) -> dict:
    """Run one τ² episode with persona-injected user. Returns result dict."""
    import uuid

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.runner.build import _build_env_kwargs, build_agent, build_environment
    from tau2.runner.simulation import run_simulation

    cfg = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=agent_model,
        llm_user=user_model,
        llm_args_agent=agent_args,
        llm_args_user=user_args,
        max_steps=max_steps,
        seed=seed,
    )
    solo_mode = registry.get_agent_metadata(
        cfg.effective_agent, "solo_mode", default=False
    )
    env_kwargs = _build_env_kwargs(cfg, task)
    environment = build_environment(
        cfg.domain, solo_mode=solo_mode, env_kwargs=env_kwargs
    )
    agent = build_agent(
        cfg.effective_agent,
        environment,
        llm=cfg.llm_agent,
        llm_args=cfg.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
    )
    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None

    PPolUser = _build_persona_user_class()
    user = PPolUser(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=cfg.llm_user,
        llm_args=cfg.llm_args_user,
    )
    user._ppol_persona_text = persona_text

    orch = Orchestrator(
        domain=cfg.domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=cfg.effective_max_steps,
        max_errors=cfg.max_errors,
        seed=cfg.seed,
        solo_mode=solo_mode,
        simulation_id=str(uuid.uuid4()),
        validate_communication=cfg.enforce_communication_protocol,
        timeout=cfg.timeout,
    )
    if max_turns is not None and max_turns > 0:
        orch.max_steps = min(orch.max_steps, max_turns)

    # Telecom reward is pure DB check (no nl_assertions on any task) — no LLM judge needed.
    sim = run_simulation(orch, evaluation_type=EvaluationType.ALL)
    messages = sim.get_messages()
    n_msgs = len(messages)
    reward_info = getattr(sim, "reward_info", None)
    reward = float(reward_info.reward) if reward_info is not None else float("nan")
    trajectory = _serialize_messages(messages)
    return {
        "reward": reward,
        "n_msgs": n_msgs,
        "status": "ok",
        "trajectory": trajectory,
    }


def _serialize_messages(messages) -> list[dict]:
    """Convert τ² messages to serializable {role, content} dicts."""
    out = []
    for m in messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role is None:
            continue
        c = content
        if not isinstance(c, str):
            c = str(c) if c is not None else ""
        out.append({"role": str(role), "content": c})
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run τ²-bench telecom episodes with PPol-generated personas."
    )
    ap.add_argument(
        "--persona-json", required=True,
        help="PPol generated persona JSON (tau2_telecom_pgepa_test_n10.json)",
    )
    ap.add_argument("--domain", default="telecom")
    ap.add_argument(
        "--split-file",
        default=str(_DEFAULT_TELECOM_SPLIT),
        help="PGEPA island-split JSON (telecom_island_split.json)",
    )
    ap.add_argument("--split-section", default="test")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--n-personas", type=int, default=None)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--user-llm", default="gpt-5.4")
    ap.add_argument(
        "--reasoning-effort", default="high",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap orchestrator steps (null = only --max-steps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=1,
                    help="concurrent episodes (1 = serial)")
    ap.add_argument("--num-trials", type=int, default=1,
                    help="repeat each (task, persona) N times with seed+i")
    ap.add_argument("--save-trajectories", action="store_true", default=False,
                    help="save rollout conversation messages to _trajectories.json")
    ap.add_argument("--env-file", default=None,
                    help="load KEY=VALUE from .env file")
    ap.add_argument("--taubench-root", default=str(_DEFAULT_TAUBENCH_ROOT))
    ap.add_argument("--out", default=None, help="output JSON path")
    # gpt-5 reasoning override  (via litellm.completion monkeypatch)
    ap.add_argument(
        "--max-tokens", type=int, default=16000,
        help="max_tokens for gpt-5 reasoning models (thinking burns output budget)",
    )
    args = ap.parse_args()

    # ── Env ──────────────────────────────────────────────────────────────
    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    # When using OpenAI models, drop ANTHROPIC_* env so litellm doesn't route
    # through a Fireworks / Anthropic-compatible proxy.
    agent_is_openai = args.agent_llm.lower().startswith(("gpt-", "openai/"))
    user_is_openai = args.user_llm.lower().startswith(("gpt-", "openai/"))
    if agent_is_openai or user_is_openai:
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    # gpt-5 reasoning model compat: litellm needs reasoning_effort and
    # temperature=1. Patch litellm.completion for gpt-5 models.
    if args.reasoning_effort:
        import litellm as _litellm
        _orig = _litellm.completion
        _re = args.reasoning_effort
        _mt = args.max_tokens

        def _patched(*a, **kw):
            m = str(kw.get("model", "")).lower()
            if "gpt-5" in m:
                kw["reasoning_effort"] = _re
                kw["max_tokens"] = _mt
                kw["temperature"] = 1
            return _orig(*a, **kw)

        _litellm.completion = _patched

    # ── Tau2 setup ───────────────────────────────────────────────────────
    taubench_root = Path(args.taubench_root)
    _ensure_tau2_importable(taubench_root)
    split_file = Path(args.split_file)
    if not split_file.is_file():
        raise FileNotFoundError(f"split file not found: {split_file}")

    tasks = load_tau2_tasks_from_island_split(
        args.domain, split_file, args.split_section
    )
    task_map = {str(t.id): t for t in tasks}
    print(
        f"[load] tau2 domain={args.domain} tasks={len(tasks)} "
        f"split={split_file.name}::{args.split_section} "
        f"agent={args.agent_llm}+{args.reasoning_effort} "
        f"user={args.user_llm}+{args.reasoning_effort} "
        f"eval_type=ALL (DB-only, no NL judge)",
        flush=True,
    )

    # ── Persona JSON ─────────────────────────────────────────────────────
    d = json.loads(Path(args.persona_json).read_text(encoding="utf-8"))
    task_entries = d["tasks"][: args.n_tasks] if args.n_tasks else d["tasks"]

    agent_args = {"temperature": 1.0}
    user_args = {"temperature": 1.0}
    # reasoning_effort is injected by the litellm.completion monkeypatch above

    # ── Build work items ─────────────────────────────────────────────────
    work_items: list[dict] = []  # {task, persona, persona_idx, trial, seed}
    for te in task_entries:
        tid = str(te.get("task_id"))
        task = task_map.get(tid)
        if task is None:
            print(f"[skip] task {tid} not found", flush=True)
            continue
        personas = te.get("personas", [])[: args.n_personas] if args.n_personas else te.get("personas", [])
        for pi, p in enumerate(personas):
            for trial in range(args.num_trials):
                work_items.append({
                    "task": task,
                    "task_id": tid,
                    "persona": p,
                    "persona_idx": pi,
                    "trial": trial,
                    "seed": args.seed + trial,
                })

    print(f"[work] {len(work_items)} episodes ({len(task_entries)} tasks × "
          f"up to {args.n_personas or '?'} personas × {args.num_trials} trials) "
          f"workers={args.max_workers}",
          flush=True)

    # ── Run episodes ─────────────────────────────────────────────────────
    results: list[dict] = []
    trajectories: list[dict] = []
    t0_total = time.time()

    from threading import Lock as _Lock
    _print_lock = _Lock()

    def _run_one(wi: dict) -> dict:
        p = wi["persona"]
        expanded = p.get("expanded_instruction", "")
        tid = wi["task_id"]
        pid = p.get("persona_id")
        t0 = time.time()
        try:
            ep = run_episode(
                task=wi["task"],
                domain=args.domain,
                persona_text=expanded,
                agent_model=args.agent_llm,
                agent_args=agent_args,
                user_model=args.user_llm,
                user_args=user_args,
                seed=wi["seed"],
                max_steps=args.max_steps,
                max_turns=args.max_turns,
            )
            reward = ep["reward"]
            n_msgs = ep["n_msgs"]
            status = ep["status"]
            traj = ep.get("trajectory", [])
        except Exception as e:
            import traceback
            traceback.print_exc()
            reward = float("nan")
            n_msgs = 0
            status = f"error: {type(e).__name__}: {str(e)[:200]}"
            traj = []
        dt = time.time() - t0
        with _print_lock:
            print(
                f"[{tid}] p{wi['persona_idx']} t{wi['trial']} ({pid}): "
                f"reward={reward} n_msgs={n_msgs} {status} {dt:.0f}s",
                flush=True,
            )
        return {
            "task_id": tid,
            "persona_id": pid,
            "axis_placement": p.get("axis_placement"),
            "trial_idx": wi["trial"],
            "seed": wi["seed"],
            "reward": reward,
            "status": status,
            "n_msgs": n_msgs,
            "elapsed_s": round(dt, 1),
            "_trajectory": traj,
        }

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

    dt_total = time.time() - t0_total
    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = n_total - n_ok
    print(
        f"\n[done] {n_total} episodes in {dt_total:.0f}s "
        f"(ok={n_ok} err={n_err})",
        flush=True,
    )

    # ── Write ─────────────────────────────────────────────────────────────
    out_dir = _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        n_p = args.n_personas or len(task_entries[0]["personas"]) if task_entries else "?"
        out_path = out_dir / f"tau2_{args.domain}_pgepa_test_n{n_p}_t{args.num_trials}.json"

    payload = {
        "domain": args.domain,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
        "evaluation_type": "ALL",
        "reasoning_effort": args.reasoning_effort,
        "max_steps": args.max_steps,
        "max_turns": args.max_turns,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "persona_json": str(Path(args.persona_json).resolve()),
        "split_file": str(split_file.resolve()),
        "n_episodes": n_total,
        "elapsed_seconds": round(dt_total, 1),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path}", flush=True)

    if trajectories:
        traj_path = out_path.with_suffix("").with_suffix(
            out_path.suffix + "_trajectories.json"
        )
        traj_path.write_text(json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] trajectories → {traj_path} ({len(trajectories)} episodes)", flush=True)

    # ── Quick stats ──────────────────────────────────────────────────────
    rewards = [r["reward"] for r in results if r["reward"] == r["reward"]]  # not NaN
    if rewards:
        sr = sum(1 for r in rewards if r >= 1.0) / len(rewards)
        print(
            f"[summary] reward mean={sum(rewards)/len(rewards):.3f} "
            f"success_rate={sr:.2%} n={len(rewards)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
