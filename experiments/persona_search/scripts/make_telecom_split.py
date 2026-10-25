"""Build a DISTRIBUTION-HOMOGENEOUS telecom train/val/test split.

Universe = crossdomain_variance_split_v3 telecom union (114 tasks). We stratify by
(issue-type x PERSONA-difficulty-tag) and split EACH stratum into train/val/test at
the same target ratio, so all three splits carry the same mix of issue types and
persona-difficulty tags. This replaces the earlier "val = representative_50telecom
(精选难任务) + train/test = 随机剩余" split, whose val<->test difficulty gap was
partly a distribution artifact rather than pure overfitting.

Sizes train=30 / val=30 / test=54: a deliberately LARGE held-out test (≈47%) so
the post-hoc generalization verdict is low-variance (the earlier overfitting was
only visible on a large test set); train/val kept small (minibatch still samples
10 from train; val drives selection at num_trials=1). Exact totals are enforced
with the largest-remainder method over per-stratum floor allocations.
Deterministic (seed=42).
"""

import json
import random
import re
from collections import defaultdict

SRC = "experiments/persona_search/configs/crossdomain_variance_split_v3.json"
OUT = "experiments/persona_search/configs/telecom_island_split.json"
SEED = 42
TARGET = {"train": 30, "val": 30, "test": 54}  # sums to 114 (large held-out test)


def issue(t: str) -> str:
    return t[1 : t.index("]")]


def ptag(t: str) -> str:
    m = re.search(r"PERSONA:(\w+)", t)
    return m.group(1) if m else "?"


def main():
    xd = json.load(open(SRC))
    universe = sorted(
        set().union(
            *(set(xd[s].get("telecom", [])) for s in xd if isinstance(xd[s], dict))
        )
    )
    assert len(universe) == sum(TARGET.values()), (
        f"universe {len(universe)} != target total {sum(TARGET.values())}"
    )

    rng = random.Random(SEED)
    # group by (issue, persona-tag) stratum
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in universe:
        strata[(issue(t), ptag(t))].append(t)

    p = {k: TARGET[k] / len(universe) for k in TARGET}
    out = {k: [] for k in TARGET}

    # floor allocation per split WITHIN each stratum (keeps the mix homogeneous)
    for sk in sorted(strata):
        items = strata[sk]
        rng.shuffle(items)
        alloc = {k: int(p[k] * len(items)) for k in TARGET}
        i = 0
        for k in ("train", "val", "test"):
            out[k] += items[i : i + alloc[k]]
            i += alloc[k]

    # globally top up the leftover (rounding) tail to exact targets, filling the
    # split with the largest remaining deficit first (largest-remainder method)
    assigned = set().union(*(set(v) for v in out.values()))
    leftovers = [t for t in universe if t not in assigned]
    rng.shuffle(leftovers)
    while leftovers:
        deficits = sorted(((TARGET[k] - len(out[k]), k) for k in TARGET), reverse=True)
        _, k = deficits[0]
        out[k].append(leftovers.pop())

    for k in TARGET:
        assert len(out[k]) == TARGET[k], f"{k}: {len(out[k])} != {TARGET[k]}"
    # disjoint check
    s = {k: set(out[k]) for k in TARGET}
    assert (
        not (s["train"] & s["val"])
        and not (s["train"] & s["test"])
        and not (s["val"] & s["test"])
    ), "splits overlap!"

    result = {
        "meta": {
            "domain": "telecom",
            "note": "distribution-homogeneous split: stratified by (issue-type x "
            "PERSONA-tag), each stratum split at train30/val30/test54 ratio "
            "via largest-remainder; seed=42. Large held-out test (~47%) for a "
            "low-variance generalization verdict. val NO LONGER = "
            "representative_50 — all three splits share the same issue/difficulty mix.",
            "universe": len(universe),
            "source_universe": "crossdomain_variance_split_v3 telecom union (114)",
            "seed": SEED,
            "strata": "issue-type x PERSONA-tag",
        },
        "train": {"telecom": sorted(out["train"])},
        "val": {"telecom": sorted(out["val"])},
        "test": {"telecom": sorted(out["test"])},
    }
    json.dump(result, open(OUT, "w"), indent=1)

    # report distributions
    def dist(ids, fn):
        from collections import Counter

        return dict(Counter(fn(t) for t in ids))

    print(
        f"wrote {OUT}  train={len(out['train'])} val={len(out['val'])} test={len(out['test'])}"
    )
    for k in ("train", "val", "test"):
        print(f"  {k:5s} issue={dist(out[k], issue)}  ptag={dist(out[k], ptag)}")


if __name__ == "__main__":
    main()
