"""Plot 7: average completion tokens per turn, by role.

Left: distribution (violin) of completion_tokens per message for user-sim vs
agent across all sims. Right: mean tokens/turn per eval round (trend over the
optimization). Long agent turns can signal the persona forcing more reasoning;
long user turns signal verbose personas.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
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


def build(run_id: str, out_dir: Path) -> str:
    sims = _data.load_all_sims(run_id)
    tok = _data.per_turn_tokens(sims)

    # per-round means
    by_round = defaultdict(lambda: {"user": [], "assistant": []})
    for s in sims:
        rnd = s.get("_round")
        for m in s.get("messages", []):
            u = (m.get("usage") or {}).get("completion_tokens")
            if m.get("role") in ("user", "assistant") and u is not None:
                by_round[rnd][m["role"]].append(u)
    rounds = sorted(by_round)

    import matplotlib.pyplot as plt

    st.apply_style()
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    # violin
    data = [tok["user"] or [0], tok["assistant"] or [0]]
    parts = axL.violinplot(data, showmeans=True, showextrema=False)
    for pc, role in zip(parts["bodies"], ("user", "assistant")):
        pc.set_facecolor(st.PALETTE[role])
        pc.set_alpha(0.6)
    axL.set_xticks([1, 2])
    axL.set_xticklabels(["user-sim", "agent"])
    axL.set_ylabel("completion tokens / message")
    axL.set_title("token distribution by role")

    # trend
    for role in ("user", "assistant"):
        means = [
            np.mean(by_round[r][role]) if by_round[r][role] else np.nan for r in rounds
        ]
        axR.plot(rounds, means, "-o", ms=3, color=st.PALETTE[role], label=role)
    axR.set_xlabel("eval round")
    axR.set_ylabel("mean completion tokens / turn")
    axR.set_title("tokens/turn over optimization")
    axR.legend()
    fig.suptitle(f"{run_id}: avg tokens per turn", fontweight="bold")
    out = out_dir / f"{run_id}_tokens_per_turn.png"
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
