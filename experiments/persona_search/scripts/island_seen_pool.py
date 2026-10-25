"""Cross-island shared "seen persona" pool for parallel persona optimization.

Several parallel_gepa.py processes (one per seed = one island) each keep their
OWN candidate pool, but share ONE seen pool so the personas they generate stay
UNIQUE across islands. Concretely:

  * each island, before mutating, loads every other island's persona TEXTS and
    folds them into its local ``seen`` list -> the existing layered diversity gate
    (lexical Jaccard + embedding cosine) now rejects near-duplicates of ANY
    island's persona, not just its own (dedup, NOT distance-maximization).
  * each island, after sampling a child, appends {island, idx, summary, text} to
    the pool. The short SUMMARY (persona title + mutation_direction, ~2 lines) is
    what gets injected back into the mutation prompt as "already explored" so the
    mutator is steered away from rehashing them — kept short on purpose so the
    prompt context does not blow up as the pool grows.

Storage: one JSONL file, appended under an ``fcntl.flock`` exclusive lock so the
concurrent islands never interleave writes. Reads take a shared lock. No DB, no
server — minimal and crash-safe (append-only).
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path


class SeenPool:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # touch so shared-lock reads never fail on a missing file
        if not self.path.exists():
            self.path.touch()

    # -- reads (shared lock) ------------------------------------------------
    def _read_all(self) -> list[dict]:
        rows: list[dict] = []
        with open(self.path) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # tolerate a half-written tail line
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return rows

    def texts(self, exclude_island: str | None = None) -> list[str]:
        """All persona texts in the pool (optionally excluding one island's own,
        though including them is harmless — they are already in local seen)."""
        return [
            r["text"]
            for r in self._read_all()
            if r.get("text") and r.get("island") != exclude_island
        ]

    def summaries(
        self, limit: int | None = None, exclude_island: str | None = None
    ) -> list[str]:
        rows = [
            r
            for r in self._read_all()
            if r.get("summary") and r.get("island") != exclude_island
        ]
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]  # most recent N
        return [
            f"[{r.get('island', '?')}#{r.get('idx', '?')}] {r['summary']}" for r in rows
        ]

    def count(self) -> int:
        return len(self._read_all())

    # -- write (exclusive lock) --------------------------------------------
    def add(self, island: str, idx: int | str, summary: str, text: str) -> None:
        row = {
            "island": island,
            "idx": idx,
            "summary": summary.strip().replace("\n", " ")[:600],
            "text": text,
        }
        line = json.dumps(row, ensure_ascii=False)
        with open(self.path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def summarize_persona(text: str, direction: str | None = None) -> str:
    """Two-line summary for prompt injection: the persona's title line (the
    ``# Persona: ...`` header) + its mutation_direction label. No LLM call."""
    title = ""
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            title = ln.lstrip("#").strip()
            break
    d = f" | dir={direction}" if direction else ""
    return (title[:80] + d).strip()
