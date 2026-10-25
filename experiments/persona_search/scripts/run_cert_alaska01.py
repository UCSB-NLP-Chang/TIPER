#!/usr/bin/env python3
"""Cert of alaska_01_v5: optim (gen12 best, idx4 agg=0.233) vs seed (gen0), on the
FULL airline base split (50 tasks) x 4 canonical trials, agent+user gpt-5.4 high.

The persona custom_guidelines are read VERBATIM from the GEPA search's own on-disk
sims (already wrapped with the real method-actor prefix), so the cert is
byte-identical to what the search evaluated. Output ->
data/simulations/cert_alaska01_v5/{optim,seed}. tau2 --auto-resume = crash-safe.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = str(REPO / ".venv/bin/tau2")

# source sim dirs whose custom_guidelines we replay verbatim
SRC = {
    "seed": str(DATA_ROOT / "data/simulations/gepa_opt/alaska_01_v5_g00_seed"),
    "optim": str(
        DATA_ROOT / "data/simulations/gepa_opt/alaska_01_v5_g12_k00_full_diff"
    ),
}


def guidelines_from(dir_rel: str) -> str:
    for rj in glob.glob(str(REPO / dir_rel / "**/results.json"), recursive=True):
        info = json.load(open(rj)).get("info") or {}
        cg = ((info.get("user_info") or {}).get("persona_config") or {}).get(
            "custom_guidelines"
        )
        if cg:
            return cg
    raise RuntimeError(f"no custom_guidelines found under {dir_rel}")


def run_cell(name: str, guidelines: str) -> None:
    save_rel = f"cert_alaska01_v5/{name}"
    cmd = [
        TAU2,
        "run",
        "--domain",
        "airline",
        "--user",
        "user_simulator",
        "--user-llm",
        "gpt-5.4",
        "--user-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--agent-llm",
        "gpt-5.4",
        "--agent-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--num-trials",
        "4",
        "--seed",
        "300",
        "--max-concurrency",
        "15",
        "--max-steps",
        "120",
        "--save-to",
        save_rel,
        "--auto-resume",
        "--user-persona",
        json.dumps({"custom_guidelines": guidelines}, ensure_ascii=False),
    ]
    print(
        f"=== {name}: airline base 50t x4 -> {save_rel} "
        f"(guidelines {len(guidelines)} chars) ===",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(REPO), check=False)


def main() -> int:
    g = {k: guidelines_from(v) for k, v in SRC.items()}
    for name in ("optim", "seed"):
        run_cell(name, g[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
