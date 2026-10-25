"""Data loaders for persona-opt visualizations.

Two data sources:
  1. gepa_state.bin  — candidate-level data (difficulty/validity/lineage/budget).
     Unpickling needs `experiments/persona_search/scripts` on sys.path (adapter class refs),
     handled below so plot scripts run from anywhere without PYTHONPATH.
  2. eval_NNNN/.../results.json — per-sim message/token/length/termination data.

Field notes (verified against live data 2026-06-07):
  - reward lives in sim["reward_info"]["reward"]  (sim["reward"] is None).
  - per-message tokens: msg["usage"]["completion_tokens"|"prompt_tokens"].
  - objective_scores keys: difficulty, validity, validity_known,
    timeout_safety, semantic, infra_error, feasible.
  - subscores: prog_candidate_val_subscores[cand] = {task_idx_str: score}.
  - lineage: parent_program_for_candidate[cand] = [parent_idx] (or [None] for root).
  - discovery budget: num_metric_calls_by_discovery[cand] = cumulative evals.
  - per-iter trace (run_log.json / full_program_trace): i, selected_program_candidate,
    subsample_ids, subsample_scores (parent), new_subsample_scores (child).
"""

from __future__ import annotations

import json
import pickle
import sys
from glob import glob
from pathlib import Path

_THIS = Path(__file__).resolve()
PERSONA_OPT_DIR = _THIS.parent.parent  # experiments/persona_search/scripts
REPO_ROOT = PERSONA_OPT_DIR.parent.parent  # repo root
if str(PERSONA_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(PERSONA_OPT_DIR))  # so pickle can resolve adapter classes

# Runtime data workspace (same root as src/cresta/paths.py DATA_WORKSPACE):
# legacy viz data keeps the ``data/`` sub-path component under this root.
import os  # noqa: E402

DATA_ROOT = Path(
    os.environ.get("TAU2BENCH_DATA_ROOT", str(REPO_ROOT.parent / "tau2bench-data"))
)

GEPA_RUNS = DATA_ROOT / "data" / "persona_opt" / "gepa_runs"
GEPA_OPT_SIMS = DATA_ROOT / "data" / "simulations" / "gepa_opt"

# run_id -> (gepa run dir suffix, eval sims dir name)
# NB: p0003 is an earlier *completed* reference run with a different naming
# scheme (aggressive, not aggressive50; eval sims gepa_p0003_v2, not *_v2_50t).
RUN_DIRS = {
    "p0001": ("p0001_aggressive50_20260606_151059", "gepa_p0001_v2_50t"),
    "p0002": ("p0002_aggressive50_20260606_151059", "gepa_p0002_v2_50t"),
    "p0003": ("p0003_aggressive_20260606_145857", "gepa_p0003_v2"),
}


# --------------------------------------------------------------------------
# gepa_state.bin  (candidate-level)
# --------------------------------------------------------------------------
def load_state(run_id: str) -> dict:
    rd = GEPA_RUNS / RUN_DIRS[run_id][0] / "gepa_state.bin"
    with open(rd, "rb") as f:
        return pickle.load(f)


def candidate_table(run_id: str) -> list[dict]:
    """One row per candidate with objective scores, parent, discovery budget."""
    s = load_state(run_id)
    obj = s["prog_candidate_objective_scores"]
    parents = s["parent_program_for_candidate"]
    disc = s["num_metric_calls_by_discovery"]
    rows = []
    for i, o in enumerate(obj):
        p = parents[i][0] if i < len(parents) and parents[i] else None
        rows.append(
            {
                "cand": i,
                "parent": p,
                "discovered_at": disc[i] if i < len(disc) else None,
                **{
                    k: o.get(k)
                    for k in (
                        "difficulty",
                        "validity",
                        "validity_known",
                        "timeout_safety",
                        "semantic",
                        "infra_error",
                        "feasible",
                    )
                },
            }
        )
    return rows


def subscores(run_id: str) -> list[list[float]]:
    """candidate x task matrix of per-task scores (difficulty, 0/0.5/1)."""
    s = load_state(run_id)
    out = []
    for d in s["prog_candidate_val_subscores"]:
        # keys are str task indices "0".."49"
        items = sorted(d.items(), key=lambda kv: int(kv[0]))
        out.append([v for _, v in items])
    return out


def iter_trace(run_id: str) -> list[dict]:
    """Per-iteration trace: selected parent + subsample scores old/new."""
    s = load_state(run_id)
    return s["full_program_trace"]


def best_candidate(run_id: str) -> int:
    rows = candidate_table(run_id)
    return max(range(len(rows)), key=lambda i: rows[i]["difficulty"] or 0)


# --------------------------------------------------------------------------
# eval results.json  (per-sim / per-message)
# --------------------------------------------------------------------------
def eval_result_files(run_id: str) -> list[Path]:
    base = GEPA_OPT_SIMS / RUN_DIRS[run_id][1]
    files = sorted(glob(str(base / "eval_*" / "*" / "results.json")))
    return [Path(f) for f in files]


def load_all_sims(run_id: str, max_rounds: int | None = None) -> list[dict]:
    """Flatten all simulations across eval rounds, tagging each with round idx."""
    sims = []
    files = eval_result_files(run_id)
    if max_rounds:
        files = files[-max_rounds:]
    for f in files:
        rnd = int(f.parent.parent.name.split("_")[-1])
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        for s in r.get("simulations", []):
            s["_round"] = rnd
            sims.append(s)
    return sims


def sim_reward(sim: dict):
    ri = sim.get("reward_info") or {}
    return ri.get("reward")


def per_turn_tokens(sims: list[dict]) -> dict[str, list[int]]:
    """completion_tokens grouped by role across all messages."""
    out: dict[str, list[int]] = {"user": [], "assistant": []}
    for s in sims:
        for m in s.get("messages", []):
            role = m.get("role")
            u = m.get("usage") or {}
            ct = u.get("completion_tokens")
            if role in out and ct is not None:
                out[role].append(ct)
    return out


def message_lengths(sims: list[dict]) -> dict[str, list[int]]:
    """len(content) grouped by role."""
    out: dict[str, list[int]] = {"user": [], "assistant": [], "tool": []}
    for s in sims:
        for m in s.get("messages", []):
            role = m.get("role")
            c = m.get("content")
            if role in out and isinstance(c, str):
                out[role].append(len(c))
    return out


def conversation_lengths(sims: list[dict]) -> list[tuple[int, float | None]]:
    """(num_messages, reward) per sim."""
    out = []
    for s in sims:
        out.append((len(s.get("messages", [])), sim_reward(s)))
    return out


def termination_counts(sims: list[dict]) -> dict[str, int]:
    from collections import Counter

    c = Counter(s.get("termination_reason", "?") for s in sims)
    return dict(c)
