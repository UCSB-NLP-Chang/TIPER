"""Canonical filesystem locations for Cresta extensions.

The code repository contains source, experiment definitions, and small fixtures.
Private corpora and generated run state live in the sibling data workspace.
Every location can be overridden for CI or remote execution.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_WORKSPACE = Path(
    os.environ.get("TAU2BENCH_DATA_ROOT", REPO_ROOT.parent / "tau2bench-data")
).expanduser()

TAU2_DATA_DIR = Path(
    os.environ.get("TAU2_DATA_DIR", REPO_ROOT / "datasets")
).expanduser()
CRESTA_DATA_DIR = Path(
    os.environ.get("CRESTA_DATA_DIR", DATA_WORKSPACE / "private" / "cresta" / "raw")
).expanduser()

PERSONA_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "persona_search"
PERSONA_CONFIG_DIR = PERSONA_EXPERIMENT_DIR / "configs"
PERSONA_SEEDS_DIR = PERSONA_EXPERIMENT_DIR / "seeds"
PERSONA_STATE_DIR = Path(
    os.environ.get(
        "CRESTA_PERSONA_DATA_DIR",
        DATA_WORKSPACE / "persona_search" / "state",
    )
).expanduser()

ARTIFACTS_DIR = Path(
    os.environ.get("TAU2_ARTIFACTS_DIR", DATA_WORKSPACE / "artifacts")
).expanduser()
SIMULATIONS_DIR = Path(
    os.environ.get("TAU2_SIMULATIONS_DIR", ARTIFACTS_DIR / "simulations")
).expanduser()

VENDOR_DIR = REPO_ROOT / "vendor"
VITABENCH_ROOT = Path(
    os.environ.get("VITABENCH_ROOT", VENDOR_DIR / "vitabench")
).expanduser()
PERSONA_POLICIES_ROOT = Path(
    os.environ.get("PERSONA_POLICIES_ROOT", VENDOR_DIR / "persona_policies")
).expanduser()
