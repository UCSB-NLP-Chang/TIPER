#!/usr/bin/env python3
"""Wait for the alaska_03 cert to finish, then summarize optim vs seed.

Polls the cert process (pid passed as argv[1]) every 30 min. When it exits,
loads both cells' results.json, computes per-cell avg reward + difficulty
(1 - avg_reward) over all 50 tasks x 3 trials, and writes a summary file.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
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
SUMMARY = Path("/tmp/cert_alaska03_logs/SUMMARY.txt")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cell_stats(name: str) -> dict:
    rewards = []
    for rj in glob.glob(
        str(f"{DATA_ROOT}/data/simulations/cert_alaska03_v5/{name}/**/results.json"),
        recursive=True,
    ):
        obj = json.load(open(rj))
        for sim in obj.get("simulations", []):
            r = sim.get("reward_info", {}).get("reward")
            if r is not None:
                rewards.append(float(r))
    n = len(rewards)
    avg = sum(rewards) / n if n else float("nan")
    return {"n": n, "avg_reward": avg, "difficulty": 1 - avg if n else float("nan")}


def main() -> int:
    pid = int(sys.argv[1])
    while alive(pid):
        time.sleep(1800)  # 30 min
    time.sleep(20)  # let final files flush
    lines = ["alaska_03 cert: full airline base 50t x3, agent+user gpt-5.4 high", ""]
    for name in ("optim", "seed"):
        s = cell_stats(name)
        lines.append(
            f"{name:6s}: n={s['n']:3d}  avg_reward={s['avg_reward']:.4f}  "
            f"difficulty={s['difficulty']:.4f}"
        )
    o = cell_stats("optim")
    sd = cell_stats("seed")
    if o["n"] and sd["n"]:
        lines.append("")
        lines.append(
            f"delta difficulty (optim - seed) = {o['difficulty'] - sd['difficulty']:+.4f}"
        )
        lines.append("(positive = optim harder = better persona)")
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
