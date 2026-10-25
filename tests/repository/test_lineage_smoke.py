from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_lineage_loop_offline_smoke(tmp_path: Path) -> None:
    state_dir = tmp_path / "persona-state"
    env = os.environ.copy()
    env.update(
        {
            "CRESTA_PERSONA_DATA_DIR": str(state_dir),
            "TAU2_SIMULATIONS_DIR": str(tmp_path / "simulations"),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
            ),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cresta.persona.search.lineage_loop",
            "--dry-run",
            "--reset",
            "--skip-baseline",
            "--max-rounds",
            "1",
            "--domains",
            "telecom",
            "--max-concurrency",
            "1",
            "--no-bedrock",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    state = json.loads((state_dir / "lineage_state.json").read_text())
    assert set(state["lineages"]) == {"p0001", "p0002", "p0003"}
    assert all(
        len(lineage["history"]) == 1 and lineage["history"][0]["status"] == "valid"
        for lineage in state["lineages"].values()
    )


def test_lineage_loop_has_no_legacy_entrypoint_or_data_paths() -> None:
    source = (
        ROOT / "src" / "cresta" / "persona" / "search" / "lineage_loop.py"
    ).read_text()
    assert 'REPO_ROOT / "scripts" / "persona_opt"' not in source
    assert 'REPO_ROOT / "data" / "tau2"' not in source
