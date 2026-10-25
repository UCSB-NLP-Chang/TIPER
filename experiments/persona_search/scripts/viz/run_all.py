"""Run every persona-opt plot for the given runs.

Usage:
    .venv/bin/python experiments/persona_search/scripts/viz/run_all.py
    .venv/bin/python experiments/persona_search/scripts/viz/run_all.py --runs p0001 --out-dir /tmp/figs

Each plot is a standalone module (plot_*.py) with its own build(run_id, out_dir)
and its own local STYLE block — edit one file to restyle one plot. Shared
global style/colors live in _style.py; data loaders in _data.py.
Outputs: <out-dir>/<run>/<run>_<plot>.png
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import _data

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

PLOTS = [
    "plot_optim_dynamics",
    "plot_difficulty_trajectory",
    "plot_pareto",
    "plot_objective_breakdown",
    "plot_lineage_tree",
    "plot_pertask_heatmap",
    "plot_mutation_acceptance",
    "plot_tokens_per_turn",
    "plot_message_length",
    "plot_conversation_length",
    "plot_termination",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["p0001", "p0002", "p0003"])
    ap.add_argument("--out-dir", default=str(_data.DATA_ROOT / "data/persona_opt/figs"))
    ap.add_argument(
        "--only", nargs="+", default=None, help="subset of plot module names to run"
    )
    a = ap.parse_args()

    plots = a.only or PLOTS
    for run_id in a.runs:
        out = Path(a.out_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        for name in plots:
            mod = importlib.import_module(name)
            try:
                path = mod.build(run_id, out)
                print(f"[ok]   {run_id}/{name} -> {path}")
            except Exception as e:
                print(f"[FAIL] {run_id}/{name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
