from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = (
    ROOT / "experiments" / "persona_search" / "setup_benchmark_experiments.py"
)


def test_multi_domain_benchmark_setup_dry_run(tmp_path: Path) -> None:
    output_root = tmp_path / "matrix"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SETUP_SCRIPT),
            "--run-tag",
            "repository_smoke",
            "--output-root",
            str(output_root),
            "--domains",
            "telecom",
            "retail",
            "--iterations",
            "1",
            "--execute",
            "dry-run",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    plan = json.loads((output_root / "plan.json").read_text())
    assert [item["domain"] for item in plan["experiments"]] == [
        "telecom",
        "retail",
    ]
    for item in plan["experiments"]:
        launcher = Path(item["launcher"])
        assert launcher.is_file()
        assert os.access(launcher, os.X_OK)
        assert "--dry-run" in item["command"]
        assert "OPENAI_API_KEY" not in launcher.read_text()
