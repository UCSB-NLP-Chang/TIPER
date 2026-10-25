#!/usr/bin/env python3
"""Fullset 汇总:按 split 出 avg@3 与 pass^k(k=1,2,3),外加 Δ(opt-seed)/Δ(opt-no_persona)。

- avg@3 : 每 task 取 3 个 canonical-seed trial 的平均 reward,再对 split 内 task 平均。
- pass^k: 一个 task 的 3 个 canonical trial 中至少 k 个 reward==1 才算通过 → 通过 task 占比。
  pass^1=至少 1/3 对, pass^2=至少 2/3 对, pass^3=3/3 全对。三个 pass^k 均只在
  3 trial 完整的 task 上统计(同分母),互相可比。
"""

import importlib.util
import json
import sys
from collections import defaultdict

spec = importlib.util.spec_from_file_location(
    "fb", "experiments/persona_search/scripts/fullset_build.py"
)
fb = importlib.util.module_from_spec(spec)
sys.modules["fb"] = fb
spec.loader.exec_module(fb)
CANON = set(fb.CANON_SEEDS)
SIM = fb.SIM
SPLITS = ("train", "val", "test")


def cell_metrics(save_rel: str, ids: dict) -> dict:
    """return {split: {avg3, pass3, n, ntrials}} for one cell."""
    p = SIM / save_rel / "results.json"
    by_task = defaultdict(dict)  # task -> {seed: reward}
    if p.exists():
        for s in json.load(open(p))["simulations"]:
            seed = s.get("seed")
            if seed not in CANON:
                continue
            r = (s.get("reward_info") or {}).get("reward")
            by_task[str(s["task_id"])][seed] = r
    out = {}
    for sp in SPLITS:
        avgs, p1, p2, p3, ntr = [], [], [], [], 0
        for t in ids[sp]:
            rs = [by_task[t].get(sd) for sd in CANON if by_task[t].get(sd) is not None]
            ntr += len(rs)
            if not rs:
                continue
            avgs.append(sum(rs) / len(rs))
            # pass^k 只在 3-trial 完整的 task 上统计(同分母),k=1/2/3 互相可比
            if len(rs) == 3:
                ones = sum(1 for x in rs if x == 1)
                p1.append(1.0 if ones >= 1 else 0.0)
                p2.append(1.0 if ones >= 2 else 0.0)
                p3.append(1.0 if ones >= 3 else 0.0)
        out[sp] = {
            "avg3": sum(avgs) / len(avgs) if avgs else float("nan"),
            "pass1": sum(p1) / len(p1) if p1 else float("nan"),
            "pass2": sum(p2) / len(p2) if p2 else float("nan"),
            "pass3": sum(p3) / len(p3) if p3 else float("nan"),
            "n": len(ids[sp]),
            "ntrials": ntr,
        }
    return out


def main() -> None:
    cells = fb.build_cells()
    # group by domain, index by (island,cond)
    by_dom = defaultdict(dict)
    for c in cells:
        ids = fb.split_ids(c.domain)
        by_dom[c.domain][(c.island, c.cond)] = (c, cell_metrics(c.save_rel, ids))

    for dom, cellmap in by_dom.items():
        print(f"\n{'=' * 110}\n### {dom}\n{'=' * 110}")
        # header
        print(
            f"{'cell':22s} | "
            + " || ".join(f"{sp} avg@3 / p^1 / p^2 / p^3" for sp in SPLITS)
        )
        for (isl, cond), (c, m) in cellmap.items():
            cols = []
            for sp in SPLITS:
                cols.append(
                    f"{m[sp]['avg3']:.3f}/{m[sp]['pass1']:.3f}/{m[sp]['pass2']:.3f}/{m[sp]['pass3']:.3f}"
                )
            print(f"{isl + '_' + cond:22s} | " + " || ".join(cols))

        # Δ tables: for each island with both seed & opt
        nop = cellmap.get(("shared", "no_persona"))
        islands = sorted({isl for (isl, cond) in cellmap if cond in ("seed", "opt")})
        print(
            f"\n  Δ(opt−seed)  &  Δ(opt−no_persona)   [avg@3 | p^1 | p^2 | p^3], test-split:"
        )
        for isl in islands:
            seed = cellmap.get((isl, "seed"))
            opt = cellmap.get((isl, "opt"))
            if not (seed and opt):
                continue
            for sp in ["test"]:
                o, s = opt[1][sp], seed[1][sp]
                line = (
                    f"    {isl:12s} opt−seed: {o['avg3'] - s['avg3']:+.3f} | "
                    f"{o['pass1'] - s['pass1']:+.3f} | {o['pass2'] - s['pass2']:+.3f} | {o['pass3'] - s['pass3']:+.3f}"
                )
                if nop:
                    n = nop[1][sp]
                    line += (
                        f"   opt−no_p: {o['avg3'] - n['avg3']:+.3f} | "
                        f"{o['pass1'] - n['pass1']:+.3f} | {o['pass2'] - n['pass2']:+.3f} | {o['pass3'] - n['pass3']:+.3f}"
                    )
                print(line)


if __name__ == "__main__":
    main()
