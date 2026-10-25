"""Plot 3: per-candidate objective breakdown.

Grouped bars of every objective component the optimizer tracks
(difficulty / validity / timeout_safety / semantic / feasible) for each
candidate. Lets you see, e.g., a high-difficulty candidate that cheated on
validity, or one that scored difficulty via timeouts (low timeout_safety).
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

COMPONENTS = ["difficulty", "validity", "timeout_safety", "semantic", "feasible"]


def build(run_id: str, out_dir: Path) -> str:
    rows = _data.candidate_table(run_id)
    cands = [f"c{r['cand']}" for r in rows]
    x = np.arange(len(rows))
    n = len(COMPONENTS)
    w = 0.8 / n

    fig, ax = st.new_fig(figsize=(max(8, len(rows) * 1.4), 5.5))
    for j, comp in enumerate(COMPONENTS):
        vals = [(r.get(comp) if r.get(comp) is not None else 0) for r in rows]
        ax.bar(
            x + (j - n / 2) * w + w / 2,
            vals,
            w,
            label=comp,
            color=st.PALETTE.get(comp, st.PALETTE["neutral"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cands)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("candidate")
    ax.set_ylabel("score (0–1)")
    ax.set_title(f"{run_id}: objective breakdown per candidate")
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    out = out_dir / f"{run_id}_objective_breakdown.png"
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
