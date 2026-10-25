#!/usr/bin/env python3
"""Build deterministic metadata for local Cresta corpora."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_WORKSPACE = Path(
    os.environ.get("TAU2BENCH_DATA_ROOT", ROOT.parent / "tau2bench-data")
).expanduser()
DATA_DIR = Path(
    os.environ.get(
        "CRESTA_DATA_DIR",
        DATA_WORKSPACE / "private" / "cresta" / "raw",
    )
).expanduser()
OUT = ROOT / "datasets" / "cresta" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_count(path: Path) -> int | None:
    value = json.loads(path.read_text())
    return len(value) if isinstance(value, (dict, list)) else None


def main() -> None:
    files = []
    for path in sorted(DATA_DIR.glob("*.json")):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "records": record_count(path),
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "data_classification": "proprietary",
        "files": files,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
