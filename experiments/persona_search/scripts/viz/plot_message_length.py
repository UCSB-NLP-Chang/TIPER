"""Plot 8: message length (characters) distribution by role.

Histogram of len(content) for user-sim, agent, and tool messages. Persona
style shows up here: a "concise" persona has short user messages; a
"withholding/verbose" persona shifts the user distribution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _data
import _style as st
import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

ROLES = ["user", "assistant", "tool"]


def build(run_id: str, out_dir: Path) -> str:
    sims = _data.load_all_sims(run_id)
    lens = _data.message_lengths(sims)

    fig, ax = st.new_fig()
    allvals = [v for r in ROLES for v in lens[r]]
    hi = np.percentile(allvals, 99) if allvals else 1000
    bins = np.linspace(0, hi, 40)
    for role in ROLES:
        if lens[role]:
            ax.hist(
                lens[role],
                bins=bins,
                alpha=0.55,
                color=st.PALETTE[role],
                label=f"{role} (μ={np.mean(lens[role]):.0f})",
            )
    ax.set_xlabel("message length (characters)")
    ax.set_ylabel("count")
    ax.set_title(f"{run_id}: message length by role")
    ax.legend()
    out = out_dir / f"{run_id}_message_length.png"
    return st.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["p0001", "p0002"])
    ap.add_argument("--out-dir", default=str(_data.DATA_ROOT / "data/persona_opt/figs"))
    a = ap.parse_args()
    for r in a.runs:
        d = Path(a.out_dir) / r
        d.mkdir(parents=True, exist_ok=True)
        print(build(r, d))


if __name__ == "__main__":
    main()
