#!/usr/bin/env python3
"""Select representative task subsets from the MAIN AGENT MATRIX (no personas/presets).

Unlike scripts/select_representative_tasks.py (which discriminated 3 models x 6 user
personas), this version uses the canonical agent matrix:
  agents = {gpt-5.2, gpt-5.4, sonnet-4.5}   (system under test)
  users  = {gpt52user, sonnet46user, cc_con} (user simulator backbone)
  trials = 3 (stored as repeated sims per task_id within one results.json)

Discrimination axes (per user request, "agent + user double axis"):
  model_disc = std(per-agent mean rate)   over the 3 agents
  user_disc  = std(per-user  mean rate)   over the 3 user simulators
  total_disc = model_disc + user_disc

Banking is RAGGED (no single user covers all 3 agents): gpt-5.2 has 3 users,
gpt-5.4 has 2 (cc_con cell is broken / 2 tasks), sonnet-4.5 has only cc_con.
We pool over whatever cells exist per agent/user => agent x user are confounded
for banking. This caveat is recorded in the output metadata.

Output: data/tau2/representative_tasks_mainmatrix.json (does NOT overwrite the old set).
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

OUTPUT_PATH = DATA_ROOT / "data/tau2/representative_tasks_mainmatrix.json"
BANKING_TASK_DIR = DATA_ROOT / "data/tau2/domains/banking_knowledge/tasks"

BANKING_TARGET = 25
TELECOM_TARGET = 50
MIN_TASKS_FOR_VALID_CELL = (
    50  # skip broken/partial cells (e.g. gpt54 cc_con banking = 2)
)

AGENTS = ["gpt-5.2", "gpt-5.4", "sonnet-4.5"]
USERS = ["gpt52user", "sonnet46user", "cc_con"]

# (agent, user) -> results dir. Missing/broken cells simply absent or skipped by size.
TELECOM_CELLS = {
    ("gpt-5.2", "gpt52user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/telecom_gpt52_high_reasoning"
    ),
    ("gpt-5.2", "sonnet46user"): str(
        DATA_ROOT
        / "data/canonical_agent_matrix/telecom_gpt52_sonnet46user_high_reasoning"
    ),
    ("gpt-5.2", "cc_con"): str(
        DATA_ROOT / "data/simulations/telecom_gpt52_sonnet46cconuser_high_reasoning_v2"
    ),
    ("gpt-5.4", "gpt52user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/telecom_gpt54_gpt52user_high_reasoning"
    ),
    ("gpt-5.4", "sonnet46user"): str(
        DATA_ROOT
        / "data/canonical_agent_matrix/telecom_gpt54_sonnet46user_high_reasoning"
    ),
    ("gpt-5.4", "cc_con"): str(
        DATA_ROOT / "data/simulations/telecom_gpt54_sonnet46cconuser_high_reasoning_v2"
    ),
    ("sonnet-4.5", "gpt52user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/telecom_sonnet45_high_reasoning"
    ),
    ("sonnet-4.5", "sonnet46user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/telecom_sonnet45_high_sonnet46user"
    ),
    ("sonnet-4.5", "cc_con"): str(
        DATA_ROOT / "data/simulations/telecom_sonnet45_sonnet46cconuser_high_reasoning"
    ),
}

BANKING_CELLS = {
    ("gpt-5.2", "gpt52user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/banking_gpt52_gpt52user_high_reasoning"
    ),
    ("gpt-5.2", "sonnet46user"): str(
        DATA_ROOT / "data/simulations/banking_gpt52_sonnet46user_high_reasoning"
    ),
    ("gpt-5.2", "cc_con"): str(
        DATA_ROOT / "data/simulations/banking_gpt52_sonnet46cconuser_high_reasoning"
    ),
    ("gpt-5.4", "gpt52user"): str(
        DATA_ROOT / "data/canonical_agent_matrix/banking_gpt54_gpt52user_high_reasoning"
    ),
    ("gpt-5.4", "sonnet46user"): str(
        DATA_ROOT
        / "data/canonical_agent_matrix/banking_gpt54_sonnet46user_high_reasoning"
    ),
    ("gpt-5.4", "cc_con"): str(
        DATA_ROOT / "data/simulations/banking_gpt54_sonnet46cconuser_high_reasoning"
    ),  # broken (2 tasks) -> skipped by size
    ("sonnet-4.5", "cc_con"): str(
        DATA_ROOT / "data/simulations/banking_sonnet45_sonnet46cconuser_high_reasoning"
    ),
    # sonnet-4.5 x {gpt52user, sonnet46user} not run yet
}


def load_cell(path: str) -> dict[str, float]:
    """Return {task_id: mean_reward_over_trials} for one results.json, or {} if invalid."""
    f = Path(path) / "results.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    per_task = defaultdict(list)
    for sim in data.get("simulations", []):
        ri = sim.get("reward_info")
        if isinstance(ri, dict) and ri.get("reward") is not None:
            per_task[sim["task_id"]].append(float(ri["reward"]))
    if len(per_task) < MIN_TASKS_FOR_VALID_CELL:
        return {}
    return {tid: float(np.mean(rs)) for tid, rs in per_task.items()}


def compute_task_stats(cells: dict) -> tuple[dict, dict]:
    """Compute per-task pooled_rate, model_disc (agents), user_disc (users), total_disc.

    Returns (stats, coverage) where coverage logs which cells were used/skipped.
    """
    # rate[task][agent][user] = mean reward over trials
    rate = defaultdict(lambda: defaultdict(dict))
    coverage = {}
    all_tasks = set()
    for (agent, user), path in cells.items():
        cell = load_cell(path)
        coverage[f"{agent}|{user}"] = {
            "path": path,
            "n_tasks": len(cell),
            "used": bool(cell),
        }
        for tid, r in cell.items():
            rate[tid][agent][user] = r
            all_tasks.add(tid)

    stats = {}
    for tid in sorted(all_tasks):
        per_agent = {}  # agent -> mean over users present
        per_user = defaultdict(list)  # user -> list across agents
        all_cell_rates = []
        for agent in AGENTS:
            ur = rate[tid].get(agent, {})
            if not ur:
                continue
            per_agent[agent] = float(np.mean(list(ur.values())))
            for user, r in ur.items():
                per_user[user].append(r)
                all_cell_rates.append(r)
        if not all_cell_rates:
            continue
        per_user_mean = {u: float(np.mean(v)) for u, v in per_user.items()}

        pooled_rate = float(np.mean(all_cell_rates))
        model_disc = (
            float(np.std(list(per_agent.values()))) if len(per_agent) > 1 else 0.0
        )
        user_disc = (
            float(np.std(list(per_user_mean.values())))
            if len(per_user_mean) > 1
            else 0.0
        )
        total_disc = model_disc + user_disc

        stats[tid] = {
            "pooled_rate": pooled_rate,
            "model_disc": model_disc,
            "user_disc": user_disc,
            "total_disc": total_disc,
            "per_agent_rate": per_agent,
            "per_user_rate": per_user_mean,
            "n_agents": len(per_agent),
        }
    return stats, coverage


def get_difficulty_tier(rate: float) -> str:
    if rate < 0.20:
        return "hard"
    elif rate < 0.60:
        return "medium"
    return "easy"


def get_telecom_category(task_id: str) -> str:
    return task_id[1 : task_id.index("]")]


def correct_rate_deviation(
    pool: dict, selected: list, max_deviation: float = 0.05, max_iters: int = 300
) -> list:
    """Greedy swap so subset per-agent mean rate is within max_deviation of pool per-agent rate."""

    def per_agent_pool_rate(agent, tids):
        vals = [
            pool[t]["per_agent_rate"][agent]
            for t in tids
            if agent in pool[t]["per_agent_rate"]
        ]
        return np.mean(vals) if vals else 0.0

    def devs(sel):
        return {
            a: per_agent_pool_rate(a, sel) - per_agent_pool_rate(a, list(pool))
            for a in AGENTS
        }

    def maxdev(d):
        return max(abs(x) for x in d.values())

    selected = list(selected)
    sset = set(selected)
    unsel = [t for t in pool if t not in sset]

    it = 0
    for it in range(max_iters):
        d = devs(selected)
        if maxdev(d) <= max_deviation:
            break
        worst = max(d, key=lambda a: abs(d[a]))
        if (
            d[worst] > 0
        ):  # subset too high on worst agent -> drop a high task, add a low one
            outs = sorted(
                selected,
                key=lambda t: pool[t]["per_agent_rate"].get(worst, 0),
                reverse=True,
            )
            ins = sorted(unsel, key=lambda t: pool[t]["per_agent_rate"].get(worst, 0))
        else:
            outs = sorted(
                selected, key=lambda t: pool[t]["per_agent_rate"].get(worst, 0)
            )
            ins = sorted(
                unsel,
                key=lambda t: pool[t]["per_agent_rate"].get(worst, 0),
                reverse=True,
            )
        best, best_score = None, maxdev(d)
        for o in outs[:10]:
            for i in ins[:20]:
                trial = [t for t in selected if t != o] + [i]
                s = maxdev(devs(trial))
                if s < best_score:
                    best_score, best = s, (o, i)
        if best is None:
            break
        o, i = best
        selected = [t for t in selected if t != o] + [i]
        sset = set(selected)
        unsel = [t for t in pool if t not in sset]

    fd = devs(selected)
    status = "converged" if maxdev(fd) <= max_deviation else "DID NOT converge"
    print(
        f"  Rate correction {status} in {it + 1} iters (max per-agent dev {maxdev(fd):.1%})"
    )
    return selected


def stratified_select_banking(stats: dict, target: int) -> list:
    informative = {t: s for t, s in stats.items() if 0.0 < s["pooled_rate"] < 1.0}
    floor = sum(1 for s in stats.values() if s["pooled_rate"] == 0.0)
    ceil = sum(1 for s in stats.values() if s["pooled_rate"] == 1.0)
    print(
        f"\nBanking: {len(stats)} total, {floor} floor, {ceil} ceiling, {len(informative)} informative"
    )
    tiers = defaultdict(list)
    for t, s in informative.items():
        tiers[get_difficulty_tier(s["pooled_rate"])].append((t, s))
    total = len(informative)
    selected = []
    for name in ["hard", "medium", "easy"]:
        tt = tiers[name]
        quota = max(1, round(target * len(tt) / total)) if total else 0
        tt.sort(key=lambda x: x[1]["total_disc"], reverse=True)
        picked = [t for t, _ in tt[:quota]]
        selected.extend(picked)
        print(f"  {name}: {len(tt)} tasks -> picked {len(picked)} (quota {quota})")
    if len(selected) > target:
        selected.sort(key=lambda t: stats[t]["total_disc"], reverse=True)
        selected = selected[:target]
    selected = correct_rate_deviation(informative, selected)
    return selected, informative


def stratified_select_telecom(stats: dict, target: int) -> list:
    informative = {t: s for t, s in stats.items() if 0.0 < s["pooled_rate"] < 1.0}
    floor = sum(1 for s in stats.values() if s["pooled_rate"] == 0.0)
    ceil = sum(1 for s in stats.values() if s["pooled_rate"] == 1.0)
    print(
        f"\nTelecom: {len(stats)} total, {floor} floor, {ceil} ceiling, {len(informative)} informative"
    )
    cats = defaultdict(list)
    for t, s in informative.items():
        cats[get_telecom_category(t)].append((t, s))
    total = len(informative)
    selected = []
    for cat, ct in sorted(cats.items()):
        cquota = max(1, round(target * len(ct) / total))
        tiers = defaultdict(list)
        for t, s in ct:
            tiers[get_difficulty_tier(s["pooled_rate"])].append((t, s))
        csel = []
        for name in ["hard", "medium", "easy"]:
            tt = tiers[name]
            if not tt:
                continue
            tquota = max(1, round(cquota * len(tt) / len(ct)))
            tt.sort(key=lambda x: x[1]["total_disc"], reverse=True)
            csel.extend([t for t, _ in tt[:tquota]])
        if len(csel) > cquota:
            csel.sort(key=lambda t: stats[t]["total_disc"], reverse=True)
            csel = csel[:cquota]
        selected.extend(csel)
        print(f"  {cat}: {len(ct)} tasks -> picked {len(csel)} (quota {cquota})")
    if len(selected) > target:
        selected.sort(key=lambda t: stats[t]["total_disc"], reverse=True)
        selected = selected[:target]
    selected = correct_rate_deviation(informative, selected)
    return selected, informative


def validate(pool: dict, selected: list, domain: str):
    print(f"\n{domain} validation (vs informative pool of {len(pool)}):")
    print(f"  {'agent':<11}{'pool':>8}{'subset':>8}{'delta':>8}")
    pool_rank, sub_rank = {}, {}
    for a in AGENTS:
        pr = [
            pool[t]["per_agent_rate"][a] for t in pool if a in pool[t]["per_agent_rate"]
        ]
        sr = [
            pool[t]["per_agent_rate"][a]
            for t in selected
            if a in pool[t]["per_agent_rate"]
        ]
        p = np.mean(pr) if pr else 0.0
        s = np.mean(sr) if sr else 0.0
        pool_rank[a], sub_rank[a] = p, s
        flag = " !" if abs(s - p) > 0.05 else ""
        print(f"  {a:<11}{p:>7.1%}{s:>7.1%}{s - p:>+7.1%}{flag}")
    pr = sorted(AGENTS, key=lambda a: pool_rank[a], reverse=True)
    sr = sorted(AGENTS, key=lambda a: sub_rank[a], reverse=True)
    print(
        f"  agent ranking: pool={pr} subset={sr} -> {'OK' if pr == sr else 'CHANGED'}"
    )
    tiers = defaultdict(int)
    for t in selected:
        tiers[get_difficulty_tier(pool[t]["pooled_rate"])] += 1
    print(f"  difficulty coverage: {dict(tiers)}")


def main():
    print("=" * 64)
    print("Representative Subset Selection — MAIN AGENT MATRIX (agent+user axes)")
    print("=" * 64)

    print("\n--- Banking (ragged) ---")
    b_stats, b_cov = compute_task_stats(BANKING_CELLS)
    for k, v in b_cov.items():
        print(f"  cell {k:<24} n={v['n_tasks']:<4} {'USED' if v['used'] else 'SKIP'}")
    b_sel, b_pool = stratified_select_banking(b_stats, BANKING_TARGET)
    validate(b_pool, b_sel, "Banking")

    print("\n--- Telecom (full 3x3) ---")
    t_stats, t_cov = compute_task_stats(TELECOM_CELLS)
    for k, v in t_cov.items():
        print(f"  cell {k:<24} n={v['n_tasks']:<4} {'USED' if v['used'] else 'SKIP'}")
    t_sel, t_pool = stratified_select_telecom(t_stats, TELECOM_TARGET)
    validate(t_pool, t_sel, "Telecom")

    out = {
        "banking": sorted(b_sel),
        "telecom": sorted(t_sel),
        "metadata": {
            "selection_method": "irt_discrimination_stratified_agent+user",
            "source": "main_agent_matrix",
            "agents": AGENTS,
            "users": USERS,
            "discrimination": "total_disc = model_disc(agents) + user_disc(users)",
            "banking_caveat": "RAGGED: no user covers all 3 agents; gpt-5.4 cc_con cell broken, "
            "sonnet-4.5 only has cc_con. agent x user confounded for banking.",
            "banking_coverage": b_cov,
            "telecom_coverage": t_cov,
            "banking_stats": {t: b_stats[t] for t in b_sel},
            "telecom_stats": {t: t_stats[t] for t in t_sel},
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"\nSelected {len(b_sel)} banking + {len(t_sel)} telecom = {len(b_sel) + len(t_sel)}"
    )
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
