"""Probe a few NEW seed personas on the telecom representative set: how much do
they move agent reward vs the no_persona floor, and how natural are they (sim2real
approach-#2 P_geom precision, STRICT mode) vs the human reference manifold.

Personas (default): no_persona floor + two brinks seeds (terse_blunt HELPS,
verbose_polite HARMS). Tasks: the 35-telecom IRT rep set inside
representative_tasks_50mixed.json.

Reward aggregation is identical to the telecom pair runs (drop infra +
timeout/maxstep, dedup trials) via eval_persona._aggregate_results. Naturalness
reuses the VALIDATED sim2real_pr machinery by import (load_human / feature_matrix /
precision_recall), mirroring sim2real_naturalness.py's strict-mode computation.

HARD CONSTRAINT honoured: NEW file. Reuses eval_persona, lineage_loop, _common,
sim2real_pr, sim2real_metrics BY IMPORT with zero edits to them.

    # plan only (resolve personas + tasks, print cmds, no API, no human-load):
    .venv/bin/python experiments/persona_search/scripts/telecom_persona_probe.py --dry-run

    # full probe via vertex opus-4-7 (no Bedrock SSO):
    .venv/bin/python experiments/persona_search/scripts/telecom_persona_probe.py \
        --no-bedrock --user-llm claude-opus-4-7 --persona-concurrency 3 --max-concurrency 10
"""

from __future__ import annotations

import argparse
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

MIXED = PERSONA_OPT_DIR / "representative_tasks_50mixed.json"
SEEDS_BRINKS = PERSONA_OPT_DIR / "seeds_brinks"
NO_PERSONA_SKILL = PERSONAS_DIR / "opt_no_persona_baseline.skill"


def load_telecom_ids() -> list[str]:
    return list(read_json(MIXED).get("telecom", []))


def _materialize_one(label: str, pid: str, text: str) -> dict:
    materialize(pid, text)
    print(f"[persona] {label:18s} -> {skill_name_for(pid)}.skill ({len(text)} chars)")
    return {"label": label, "pid": pid, "skill_name": skill_name_for(pid)}


def resolve_personas(brinks: list[str], include_floor: bool) -> list[dict]:
    out = []
    if include_floor:
        if not NO_PERSONA_SKILL.exists():
            raise FileNotFoundError(
                f"no_persona floor skill missing: {NO_PERSONA_SKILL}"
            )
        out.append(
            _materialize_one(
                "no_persona", "teleprobe_no_persona", NO_PERSONA_SKILL.read_text()
            )
        )
    for name in brinks:
        src = SEEDS_BRINKS / name / "gen_00.md"
        if not src.exists():
            raise FileNotFoundError(f"brinks seed missing: {src}")
        out.append(_materialize_one(name, f"teleprobe_{name}", src.read_text()))
    return out


def make_args(a, skill_name: str) -> SimpleNamespace:
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
        retrieval_config="bm25",  # ignored for telecom
        max_steps=a.max_steps,
        sim_timeout=a.sim_timeout,
        max_infra_retries=a.max_infra_retries,
    )


def naturalness_strict(persona_results: dict, mode: str, k: int) -> dict:
    """persona_results: {label: results_path}. Returns {label: {P_geom, R, F1, n}}.
    Mirrors sim2real_naturalness.py main() for non-hybrid modes (feat=feature_matrix)."""
    sys.path.insert(0, str(REPO_ROOT))
    import sim2real_metrics as M  # noqa: E402
    from sim2real_pr import (  # noqa: E402
        all_spaces,
        feature_matrix,
        human_turn_lists,
        load_human,
        pca_fit,
        precision_recall,
        sim_turn_lists,
        standardizer,
    )

    M.set_mode(mode)
    rows, _ = load_human()
    Hx_raw = feature_matrix(human_turn_lists(rows))
    mu, sd = standardizer(Hx_raw)
    pca = pca_fit((Hx_raw - mu) / sd)
    pca_name = f"pca{pca[2]}"
    print(
        f"[naturalness] human ref: {len(Hx_raw)} interactions, space={pca_name}, mode={mode}, k={k}"
    )

    res = {}
    for label, rp in persona_results.items():
        if rp is None or not Path(rp).exists():
            res[label] = None
            continue
        tl = sim_turn_lists(str(rp))
        Sx_raw = feature_matrix(tl)
        sp = all_spaces(Hx_raw, Sx_raw, mu, sd, pca)
        Hp, Sp = sp[pca_name]
        P, Rc, F1 = precision_recall(Hp, Sp, k)
        res[label] = {
            "P_geom_naturalness": round(float(P), 4),
            "R_geom_coverage": round(float(Rc), 4),
            "F1": round(float(F1), 4),
            "n_interactions": len(tl),
        }
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--brinks",
        nargs="+",
        default=["brinks_00_terse_blunt", "brinks_03_verbose_polite"],
        help="brinks seed dir names under data/persona_opt/seeds_brinks/",
    )
    ap.add_argument("--floor", dest="floor", action="store_true", default=True)
    ap.add_argument("--no-floor", dest="floor", action="store_false")
    ap.add_argument("--run-tag", default="telecom_persona_probe")
    ap.add_argument("--max-tasks", type=int, default=0, help="0=all 35 telecom ids")
    # run settings (telecom pair-run config)
    ap.add_argument("--num-trials", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--persona-concurrency", type=int, default=3)
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
        "--nat-mode", default="strict", choices=["strict", "faithful", "hybrid"]
    )
    ap.add_argument("--nat-k", type=int, default=3)
    ap.add_argument(
        "--no-naturalness", dest="naturalness", action="store_false", default=True
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ids = load_telecom_ids()
    if a.max_tasks:
        ids = ids[: a.max_tasks]
    personas = resolve_personas(a.brinks, a.floor)
    pc = max(1, a.persona_concurrency)
    n_roll = len(personas) * len(ids) * a.num_trials
    print(
        f"\n=== telecom persona probe: {len(personas)} personas x {len(ids)} tasks "
        f"x {a.num_trials} trials = {n_roll} rollouts ==="
    )
    print(
        f"    user={a.user_type} {a.user_llm} effort={a.effort} bedrock={a.use_bedrock} | agent={a.agent_llm}"
    )
    print(
        f"    concurrency: {pc} personas x {a.max_concurrency} sims = {pc * a.max_concurrency} global "
        f"| naturalness={a.naturalness} (mode={a.nat_mode}, k={a.nat_k})"
    )

    if a.dry_run:
        for p in personas:
            run_name = f"{a.run_tag}/{p['label']}"
            cmd = _build_cmd(make_args(a, p["skill_name"]), "telecom", ids, run_name)
            print(f"\n--- {p['label']} ---\n" + " ".join(cmd))
        print("\n[dry-run] no simulations executed.")
        return

    def run_one(p: dict):
        run_name = f"{a.run_tag}/{p['label']}"
        rp = _run_domain_with_retry(
            make_args(a, p["skill_name"]), "telecom", ids, run_name
        )
        per_task, infra, n_seen, ts = _aggregate_results(rp, "telecom")
        mean_reward = sum(per_task.values()) / len(per_task) if per_task else None
        difficulty = (1.0 - mean_reward) if mean_reward is not None else None
        print(
            f"[{p['label']}] reward={mean_reward}  difficulty={difficulty}  "
            f"({n_seen} scored, {infra} infra, {ts['n_timeout']}+{ts['n_maxsteps']} timeout/maxsteps)"
        )
        return {
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

    rows = []
    if pc <= 1:
        rows = [run_one(p) for p in personas]
    else:
        with ThreadPoolExecutor(max_workers=pc) as ex:
            futs = {ex.submit(run_one, p): p for p in personas}
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: [p["label"] for p in personas].index(r["label"]))

    # naturalness (strict) on the produced telecom sims
    nat = {}
    if a.naturalness:
        nat = naturalness_strict(
            {r["label"]: r["results_path"] for r in rows}, a.nat_mode, a.nat_k
        )
        for r in rows:
            r["naturalness"] = nat.get(r["label"])

    base = next((r["difficulty"] for r in rows if r["label"] == "no_persona"), None)
    summary = {
        "domain": "telecom",
        "tasks_file": str(MIXED),
        "n_tasks": len(ids),
        "num_trials": a.num_trials,
        "nat_mode": a.nat_mode,
        "nat_k": a.nat_k,
        "models": {
            "agent": f"{a.agent_llm} {a.agent_llm_args}",
            "user": f"{a.user_type} {a.user_llm} effort={a.effort} bedrock={a.use_bedrock}",
        },
        "no_persona_difficulty": base,
        "personas": rows,
    }
    out = SIMULATIONS_DIR / a.run_tag / "summary.json"
    write_json(out, summary)

    print(
        "\n=== telecom probe: reward effect vs no_persona  +  naturalness (P_geom, strict) ==="
    )
    print(
        f"{'persona':22s} {'reward':>7s} {'difficulty':>10s} {'effect':>8s} {'nat(P)':>7s} {'cov(R)':>7s} {'F1':>6s}"
    )
    for r in rows:
        d = r["difficulty"]
        eff = (
            "(base)"
            if r["label"] == "no_persona" or base is None or d is None
            else f"{d - base:+.3f}"
        )
        n = r.get("naturalness") or {}
        print(
            f"{r['label']:22s} {str(round(r['mean_reward'], 3)):>7s} {str(round(d, 3)) if d is not None else 'n/a':>10s} "
            f"{eff:>8s} {str(n.get('P_geom_naturalness', '-')):>7s} {str(n.get('R_geom_coverage', '-')):>7s} "
            f"{str(n.get('F1', '-')):>6s}"
        )
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
