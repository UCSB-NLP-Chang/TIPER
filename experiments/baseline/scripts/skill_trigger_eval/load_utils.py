"""Utility for loading conversation history on demand from results.json files."""

import json
import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[5].parent / "tau2bench-data"),
    )
)

BASE = DATA_ROOT / "data" / "simulations"

_sim_cache: dict[str, dict] = {}


def _load_results(sim_dir: str) -> dict:
    """Load and cache results.json for a simulation directory."""
    if sim_dir not in _sim_cache:
        path = BASE / sim_dir / "results.json"
        with open(path) as f:
            data = json.load(f)
        sims_by_task = {}
        for sim in data.get("simulations", []):
            sims_by_task[sim["task_id"]] = sim
        _sim_cache[sim_dir] = sims_by_task
    return _sim_cache[sim_dir]


def get_messages(sim_dir: str, task_id: str) -> list[dict]:
    """Get the messages array for a specific simulation."""
    sims = _load_results(sim_dir)
    sim = sims.get(task_id)
    if sim is None:
        return []
    return sim.get("messages") or []


def get_conversation_up_to(sim_dir: str, task_id: str, msg_idx: int) -> list[dict]:
    """Get conversation history up to (not including) the given message index."""
    messages = get_messages(sim_dir, task_id)
    history = []
    for m in messages[:msg_idx]:
        entry = {"role": m.get("role", "unknown")}
        content = m.get("content")
        if content:
            entry["content"] = content
        tc = m.get("tool_calls")
        if tc:
            entry["tool_calls"] = [
                {"name": t.get("name", ""), "arguments": t.get("arguments", {})}
                for t in tc
            ]
        history.append(entry)
    return history


def clear_cache():
    """Clear the simulation cache to free memory."""
    _sim_cache.clear()


def preload_dirs(sim_dirs: set[str]):
    """Pre-load results.json for a batch of simulation directories."""
    for d in sim_dirs:
        if d not in _sim_cache:
            try:
                _load_results(d)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
