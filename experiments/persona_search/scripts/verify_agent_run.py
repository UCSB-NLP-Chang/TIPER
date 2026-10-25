#!/usr/bin/env python3
"""Verify a fullset_<agent> run: per-cell task count, exactly-3-canonical-trials,
non-null rewards, seed purity. Usage: verify_agent_run.py <prefix>  (e.g. fullset_glm)."""

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

CANON = {626729, 373753, 361454}
TGT = {"airline": 50, "retail": 114, "banking": 97, "telecom": 114}


def main(prefix: str) -> None:
    lines = []
    grand = 0
    issues = 0
    for f in sorted(
        glob.glob(f"{DATA_ROOT}/data/simulations/{prefix}/*/*/results.json")
    ):
        d = json.load(open(f))
        sims = d["simulations"]
        rel = f.split(prefix + "/")[1].rsplit("/", 1)[0]
        dom = rel.split("/")[0]
        bytask = Counter()
        nullrew = 0
        badseed = 0
        for s in sims:
            tid = s.get("task_id") or (s.get("task") or {}).get("id")
            bytask[tid] += 1
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                nullrew += 1
            sd = s.get("seed")
            if sd is not None and sd not in CANON:
                badseed += 1
        n_task = len(bytask)
        exact3 = sum(1 for v in bytask.values() if v == 3)
        grand += len(sims)
        bad = (n_task != TGT[dom]) or (exact3 != n_task) or nullrew > 0 or badseed > 0
        if bad:
            issues += 1
        lines.append(
            "%-32s sims=%d tasks=%d/%d exact3=%d nullrew=%d badseed=%d %s"
            % (
                rel,
                len(sims),
                n_task,
                TGT[dom],
                exact3,
                nullrew,
                badseed,
                "<<ISSUE" if bad else "ok",
            )
        )
    lines.append("TOTAL sims=%d  cells_with_issue=%d" % (grand, issues))
    open(f"verify_{prefix}.out", "w").write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fullset_glm")
