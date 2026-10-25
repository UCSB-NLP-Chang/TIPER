"""Plot 10: termination-reason distribution over eval rounds.

Stacked-area of termination reasons (user_stop / max_steps / timeout / ...)
per eval round. Doubles as a DATA-QUALITY monitor: a spike of `timeout`
across a round = SSO/infra failure (poisoned data), not difficulty.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
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

# stable color per termination reason
TERM_COLORS = {
    "user_stop": "#2e7d32",
    "max_steps": "#ef8a17",
    "timeout": "#d1495b",
    "agent_stop": "#1f77b4",
    "?": "#9e9e9e",
}


def build(run_id: str, out_dir: Path) -> str:
    sims = _data.load_all_sims(run_id)
    by_round = defaultdict(Counter)
    for s in sims:
        by_round[s.get("_round")][s.get("termination_reason", "?")] += 1
    rounds = sorted(by_round)
    reasons = sorted({r for c in by_round.values() for r in c})

    mat = np.array([[by_round[rd][rs] for rd in rounds] for rs in reasons], float)
    colors = [TERM_COLORS.get(r, "#777") for r in reasons]

    fig, ax = st.new_fig()
    ax.stackplot(rounds, mat, labels=reasons, colors=colors, alpha=0.85)
    ax.set_xlabel("eval round")
    ax.set_ylabel("# simulations")
    ax.set_title(f"{run_id}: termination reasons over rounds")
    ax.legend(loc="upper left", ncol=2)
    # data-quality note
    tot = mat.sum()
    to = mat[reasons.index("timeout")].sum() if "timeout" in reasons else 0
    ax.text(
        0.99,
        0.97,
        f"timeout share: {to / tot:.1%}" if tot else "no data",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=st.PALETTE["difficulty"],
    )
    out = out_dir / f"{run_id}_termination.png"
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
