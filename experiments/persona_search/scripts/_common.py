"""Shared helpers for the persona-optimization components.

Every component is a small, self-contained CLI that reads/writes JSON and the
persona library (``data/persona_opt/lib.jsonl``). This module centralises paths,
atomic I/O, and the library record schema so the components stay tiny.

Run a component directly, e.g.::

    .venv/bin/python experiments/persona_search/scripts/select.py --round 1
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# experiments/persona_search/scripts/_common.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Runtime data workspace: same root as src/cresta/paths.py DATA_WORKSPACE
# (TAU2BENCH_DATA_ROOT, default ../tau2bench-data). Legacy scripts keep the
# ``data/`` sub-path component under this root, e.g. data/persona_opt/.
DATA_ROOT = Path(
    os.environ.get("TAU2BENCH_DATA_ROOT", str(REPO_ROOT.parent / "tau2bench-data"))
)

PERSONA_OPT_DIR = DATA_ROOT / "data" / "persona_opt"
LIB_PATH = PERSONA_OPT_DIR / "lib.jsonl"
ROUNDS_DIR = PERSONA_OPT_DIR / "rounds"
GOAL_PATH = PERSONA_OPT_DIR / "goal.json"
CONVERGENCE_PATH = PERSONA_OPT_DIR / "convergence_state.json"
EMB_CACHE_DIR = PERSONA_OPT_DIR / ".persona_emb_cache"

# Materialized search personas (.skill text files written by lineage_loop.materialize)
PERSONAS_DIR = DATA_ROOT / "data" / "personas"
# Generated personas are namespaced so they never clobber the hand-written ones.
OPT_PREFIX = "opt_"

REP_TASKS_PATH = PERSONA_OPT_DIR / "representative_tasks_50telecom.json"

# tau2 writes runs to <DATA_ROOT>/data/simulations/<save_to>/results.json
SIMULATIONS_DIR = DATA_ROOT / "data" / "simulations"

VALID_STATUSES = ("untested", "valid", "illegal")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Atomic JSON / JSONL I/O
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: Path, obj: Any) -> None:
    _atomic_write(Path(path), json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def round_dir(round_no: int) -> Path:
    return ROUNDS_DIR / f"round_{round_no:02d}"


# ---------------------------------------------------------------------------
# Library (lib.jsonl): one persona record per line
# ---------------------------------------------------------------------------


def new_record(
    persona_id: str,
    persona_text: str,
    *,
    status: str = "untested",
    parent_id: Optional[str] = None,
    mutation_direction: Optional[str] = None,
    dedup_checked: bool = True,
) -> dict:
    return {
        "persona_id": persona_id,
        "created_at": now_iso(),
        "persona_text": persona_text,
        "status": status,
        "parent_id": parent_id,
        "mutation_direction": mutation_direction,
        "avg_success_rate": None,
        "num_evaluations": 0,
        "dedup_checked": dedup_checked,
        "judge_severe_failure_rate": None,
        "last_round": None,
    }


def read_lib(path: Path = LIB_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_lib(records: list[dict], path: Path = LIB_PATH) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    if text:
        text += "\n"
    _atomic_write(Path(path), text)


def lib_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["persona_id"]: r for r in records}


def upsert(records: list[dict], record: dict) -> list[dict]:
    """Insert or replace a record by persona_id, preserving order."""
    by_id = {r["persona_id"]: i for i, r in enumerate(records)}
    if record["persona_id"] in by_id:
        records[by_id[record["persona_id"]]] = record
    else:
        records.append(record)
    return records


def next_persona_id(records: list[dict]) -> str:
    """p0001, p0002, ... based on the highest existing numeric id."""
    mx = 0
    for r in records:
        pid = r.get("persona_id", "")
        if pid.startswith("p") and pid[1:].isdigit():
            mx = max(mx, int(pid[1:]))
    return f"p{mx + 1:04d}"


def skill_name_for(persona_id: str) -> str:
    """Name (no extension) the simulator loads: opt_<persona_id>."""
    return f"{OPT_PREFIX}{persona_id}"


# ---------------------------------------------------------------------------
# Representative task set
# ---------------------------------------------------------------------------


def load_rep_tasks(path: Path = REP_TASKS_PATH) -> dict[str, list[str]]:
    """Return {"banking": [...], "telecom": [...]}."""
    d = read_json(path)
    return {
        "banking": list(d.get("banking", [])),
        "telecom": list(d.get("telecom", [])),
    }


# ---------------------------------------------------------------------------
# Infrastructure-error detection
# ---------------------------------------------------------------------------

INFRA_ERROR = "infrastructure_error"

# Transient cc_user backbone failures occasionally leak the raw API error string
# into a USER message instead of terminating the sim as infrastructure_error.
# Such a sim is an infra artifact, not persona behavior: it must be dropped from
# scoring AND excluded from the judge (otherwise the leaked text reads as a
# break-character / fabrication severe failure and falsely marks a persona illegal).
_INFRA_CONTENT_MARKERS = (
    "API Error:",
    "Internal server error",
    "litellm.",
    "RateLimitError",
    "Connection error",
)


def is_infra_broken(sim: dict) -> bool:
    """True if a sim is an infrastructure artifact (drop from scoring + judging)."""
    if sim.get("termination_reason") == INFRA_ERROR:
        return True
    for m in sim.get("messages", []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and any(mk in c for mk in _INFRA_CONTENT_MARKERS):
            return True
    return False
