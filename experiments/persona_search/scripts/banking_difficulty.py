"""Measure persona DIFFICULTY on the banking_knowledge domain over a 50-task
representative subset, for 3 personas: no_persona (cooperative floor), the seed
persona (gen_00 of a lineage), and the optimized persona (that lineage's GEPA
exploration-best).

Difficulty = 1 - mean(agent reward), aggregated exactly like the telecom runs:
infra-artifact sims are dropped+retried, timeout/max_steps sims are excluded from
scoring (see eval_persona._aggregate_results). Settings default to match the
telecom opus-4-7 pair runs (num_trials=2, max_steps=120, agent gpt-5.4
reasoning=high, user cc_user_con claude-opus-4-7 effort=high on Bedrock).

HARD CONSTRAINT honoured: this is a NEW file. It reuses the validated pipeline
(eval_persona, lineage_loop, _common) BY IMPORT with zero edits to them.

Examples
--------
    # show the plan (resolves personas + tasks, prints the tau2 cmds, no API):
    .venv/bin/python experiments/persona_search/scripts/banking_difficulty.py --dry-run

    # smoke (2 banking tasks, 1 trial):
    .venv/bin/python experiments/persona_search/scripts/banking_difficulty.py \
        --subset-size 2 --num-trials 1

    # full run, matching telecom (bedrock opus-4-7; needs `aws sso login` first):
    .venv/bin/python experiments/persona_search/scripts/banking_difficulty.py

    # via the Cresta vertex proxy with opus-4-8 (no Bedrock SSO; needs the
    # ambient CLAUDE_CODE_USE_VERTEX=1 env — see ccuser-vertex-native memory):
    .venv/bin/python experiments/persona_search/scripts/banking_difficulty.py \
        --no-bedrock --user-llm claude-opus-4-8
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    REPO_ROOT,
    SIMULATIONS_DIR,
    read_json,
    skill_name_for,
    write_json,
)
from eval_persona import (  # noqa: E402
    _aggregate_results,
    _build_cmd,
    _run_domain_with_retry,
)
from lineage_loop import materialize  # noqa: E402

BANKING_TASKS = (
    REPO_ROOT / "datasets" / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
)
DEFAULT_SUBSET = PERSONA_OPT_DIR / "representative_tasks_banking_50.json"

# Measured per-rollout cost (opus list price, ~67.5% cache-read on long convos);
# see ccuser-cache-and-banking-cost memory. Used only for the upfront estimate.
USD_PER_ROLLOUT_WITH_CACHE = 1.55
USD_PER_ROLLOUT_NO_CACHE = 3.0


# ---------------------------------------------------------------------------
# Banking 50-task subset
# ---------------------------------------------------------------------------


def _all_banking_task_ids() -> list[str]:
    data = read_json(BANKING_TASKS)
    tasks = data if isinstance(data, list) else data.get("tasks", data)
    return [t["id"] for t in tasks]


def load_or_build_subset(path: Path, size: int, seed: int) -> list[str]:
    """Return `size` banking task ids, persisting the choice to `path`.

    No banking IRT/discrimination data exists, so the subset is a deterministic
    seeded sample over all banking_knowledge tasks (reproducible, labelled as
    such). If `path` already exists its "banking" list is used verbatim.
    """
    if path.exists():
        ids = list(read_json(path).get("banking", []))
        if ids:
            print(f"[subset] loaded {len(ids)} banking task(s) from {path}")
            return ids
    allids = _all_banking_task_ids()
    size = min(size, len(allids))
    rng = random.Random(seed)
    ids = sorted(rng.sample(allids, size))
    write_json(
        path,
        {
            "banking": ids,
            "telecom": [],
            "metadata": {
                "source": "deterministic_seeded_sample_no_irt",
                "seed": seed,
                "target_size": size,
                "pool_size": len(allids),
            },
        },
    )
    print(
        f"[subset] built {len(ids)}/{len(allids)} banking task(s) (seed={seed}) -> {path}"
    )
    return ids


# ---------------------------------------------------------------------------
# Persona resolution -> materialized opt_<pid>.skill
# ---------------------------------------------------------------------------


def _best_candidate_text(candidates_path: Path) -> str:
    cands = read_json(candidates_path)
    best = next((c for c in cands if c.get("is_exploration_best")), None)
    if best is None:
        best = max(cands, key=lambda c: c.get("exploration_score", -1))
    return best["persona_text"]


def _materialize_one(label: str, pid: str, text: str) -> dict:
    materialize(pid, text)
    print(f"[persona] {label:18s} -> {skill_name_for(pid)}.skill ({len(text)} chars)")
    return {"label": label, "pid": pid, "skill_name": skill_name_for(pid)}


def resolve_personas(
    lineages: list[str], roles: list[str], include_floor: bool
) -> list[dict]:
    """Return [{label, pid, skill_name}] for the requested personas, each
    materialized to data/personas/opt_<pid>.skill under a bankdiff_ id so
    nothing existing is clobbered.

    no_persona (the shared cooperative floor) is emitted once if include_floor;
    then for each lineage, the requested roles ("seed" = gen_00, "optimized" =
    that lineage's GEPA exploration-best). Labels: "no_persona", "<lineage>_<role>".
    """
    out = []
    if include_floor:
        floor = PERSONAS_DIR / "opt_no_persona_baseline.skill"
        if not floor.exists():
            raise FileNotFoundError(f"no_persona source missing: {floor}")
        out.append(
            _materialize_one("no_persona", "bankdiff_no_persona", floor.read_text())
        )
    for lineage in lineages:
        for role in roles:
            if role == "seed":
                src = PERSONA_OPT_DIR / "lineages" / lineage / "gen_00.md"
                text = src.read_text() if src.exists() else None
            else:  # optimized
                src = (
                    PERSONA_OPT_DIR
                    / "pgepa_runs"
                    / f"pgepa_pair_{lineage}"
                    / "candidates.json"
                )
                text = _best_candidate_text(src) if src.exists() else None
            if text is None:
                raise FileNotFoundError(
                    f"persona source for '{lineage}/{role}' missing: {src}"
                )
            label = f"{lineage}_{role}"
            out.append(_materialize_one(label, f"bankdiff_{label}", text))
    return out


# ---------------------------------------------------------------------------
# Run + score
# ---------------------------------------------------------------------------


def make_args(a, skill_name: str) -> SimpleNamespace:
    """Build the namespace eval_persona._build_cmd / _run_domain_with_retry read."""
    return SimpleNamespace(
        skill_name=skill_name,
        effort=a.effort,
        use_bedrock=a.use_bedrock,
        bedrock_profile=a.bedrock_profile,
        user_type=a.user_type,
        user_llm=a.user_llm,
        agent_llm=a.agent_llm,
        agent_llm_args=a.agent_llm_args,
        num_trials=a.num_trials,
        max_concurrency=a.max_concurrency,
        retrieval_config=a.retrieval_config,
        max_steps=a.max_steps,
        sim_timeout=a.sim_timeout,
        max_infra_retries=a.max_infra_retries,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lineages",
        nargs="+",
        default=["p0003", "p0002"],
        help="lineages whose seed+optimized to test (default p0003 p0002)",
    )
    ap.add_argument(
        "--roles",
        nargs="+",
        default=["seed", "optimized"],
        choices=["seed", "optimized"],
        help="which roles per lineage",
    )
    ap.add_argument(
        "--floor",
        dest="floor",
        action="store_true",
        default=True,
        help="include the no_persona cooperative floor (default on)",
    )
    ap.add_argument("--no-floor", dest="floor", action="store_false")
    ap.add_argument(
        "--tasks-file", default=str(DEFAULT_SUBSET), help="banking 50-subset json"
    )
    ap.add_argument("--subset-size", type=int, default=50)
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument(
        "--run-tag",
        default="run",
        help="save under data/simulations/banking_difficulty/<tag>/",
    )
    # run settings (default = telecom opus-4-7 pair-run config)
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="sim fan-out WITHIN one persona's tau2 run",
    )
    ap.add_argument(
        "--persona-concurrency",
        type=int,
        default=1,
        help="how many personas to run in parallel (each its own tau2 run). "
        "global concurrency = persona-concurrency x max-concurrency",
    )
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument(
        "--retrieval-config",
        default="bm25",
        help="banking retrieval (bm25=offline/free)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="resolve+print tau2 cmds, no API calls"
    )
    a = ap.parse_args()

    ids = load_or_build_subset(Path(a.tasks_file), a.subset_size, a.subset_seed)
    ids = ids[: a.subset_size]
    personas = resolve_personas(a.lineages, a.roles, a.floor)

    n_roll = len(personas) * len(ids) * a.num_trials
    print(
        f"\n=== banking difficulty: {len(personas)} personas x {len(ids)} tasks "
        f"x {a.num_trials} trials = {n_roll} rollouts ==="
    )
    print(
        f"    user={a.user_type} {a.user_llm} effort={a.effort} bedrock={a.use_bedrock} | "
        f"agent={a.agent_llm} | retrieval={a.retrieval_config}"
    )
    pc = max(1, a.persona_concurrency)
    print(
        f"    concurrency: {pc} personas x {a.max_concurrency} sims = {pc * a.max_concurrency} global"
    )
    print(
        f"    est. cost: ~${n_roll * USD_PER_ROLLOUT_WITH_CACHE:.0f} (cache) .. "
        f"~${n_roll * USD_PER_ROLLOUT_NO_CACHE:.0f} (no cache)"
    )

    if a.dry_run:
        for p in personas:
            run_name = f"banking_difficulty/{a.run_tag}/{p['label']}"
            cmd = _build_cmd(make_args(a, p["skill_name"]), "banking", ids, run_name)
            print(f"\n--- {p['label']} ---\n" + " ".join(cmd))
        print("\n[dry-run] no simulations executed.")
        return

    def run_one(p: dict) -> dict:
        run_name = f"banking_difficulty/{a.run_tag}/{p['label']}"
        rp = _run_domain_with_retry(
            make_args(a, p["skill_name"]), "banking", ids, run_name
        )
        per_task, infra, n_seen, ts = _aggregate_results(rp, "banking")
        mean_reward = sum(per_task.values()) / len(per_task) if per_task else None
        difficulty = (1.0 - mean_reward) if mean_reward is not None else None
        row = {
            "label": p["label"],
            "pid": p["pid"],
            "skill_name": p["skill_name"],
            "n_tasks_scored": n_seen,
            "mean_reward": mean_reward,
            "difficulty": difficulty,
            "infra_error_count": infra,
            "timeout_stats": ts,
            "results_path": str(rp),
        }
        d = f"{difficulty:.4f}" if difficulty is not None else "n/a"
        print(
            f"[{p['label']}] difficulty={d}  (mean_reward={mean_reward}, "
            f"{n_seen} scored, {infra} infra, {ts['n_timeout']}+{ts['n_maxsteps']} timeout/maxsteps)"
        )
        return row

    rows = []
    if pc <= 1:
        rows = [run_one(p) for p in personas]
    else:
        # Each persona runs as its own tau2 subprocess writing to a distinct
        # save-to dir + its own materialized skill, so they're independent. Run
        # up to `pc` at once; global sim concurrency = pc x max_concurrency.
        with ThreadPoolExecutor(max_workers=pc) as ex:
            futs = {ex.submit(run_one, p): p for p in personas}
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: [p["label"] for p in personas].index(r["label"]))

    summary = {
        "domain": "banking_knowledge",
        "lineages": a.lineages,
        "n_tasks": len(ids),
        "num_trials": a.num_trials,
        "tasks_file": a.tasks_file,
        "models": {
            "agent": f"{a.agent_llm} {a.agent_llm_args}",
            "user": f"{a.user_type} {a.user_llm} effort={a.effort} bedrock={a.use_bedrock}",
            "retrieval": a.retrieval_config,
        },
        "personas": rows,
    }
    out = SIMULATIONS_DIR / "banking_difficulty" / a.run_tag / "summary.json"
    write_json(out, summary)
    print("\n=== difficulty (1 - mean reward) ===")
    for r in rows:
        d = f"{r['difficulty']:.4f}" if r["difficulty"] is not None else "n/a"
        print(f"  {r['label']:10s}  {d}   ({r['n_tasks_scored']} tasks)")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
