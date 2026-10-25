#!/usr/bin/env python3
"""Compute Sim2Real D1-D4 USI metrics on evolved-persona trajectory files.

Reads the *_trajectories.json files (list of eps with `messages`), extracts
USER turns (the user simulator), computes M.metrics_for_interaction per ep,
averages per file, prints a comparison table.

Mode: strict (rule-based, free). For Chinese delivery, lexicon metrics
(Polite/Uncert/Certn/Emot/Accuse/Pivot/Pushbk/ClarfyQ/InfoQ) and whitespace-
split structural metrics (Wds/trn/Short%/VerbCV/FrntLoad%/Open.Wds) are
unreliable — flag in output. Use --mode hybrid for LLM-rubric semantic metrics.
"""

import argparse
import json
import os
import re
import statistics as st
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import sim2real_metrics as M

RE_CONTROL = re.compile(r"#{2,}[A-Z][A-Z\- ]*#{2,}")


def user_turns(messages):
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if c is None:
            continue
        c = RE_CONTROL.sub(" ", str(c)).strip()
        if c:
            out.append(c)
    return out


def cell(path, mode):
    d = json.load(open(path))
    eps = d if isinstance(d, list) else d.get("simulations", [])
    if mode == "hybrid":
        import sim2real_llm_metrics as L

        # warm cache in one pass
        all_t = [user_turns(s.get("messages", [])) for s in eps]
        all_t = [t for t in all_t if t]
        L.label_interactions(all_t)
        metric_fn = L.hybrid_metrics_for_interaction
    else:
        metric_fn = M.metrics_for_interaction
    per = {k: [] for k in M.METRIC_ORDER}
    n = 0
    nut = []
    for s in eps:
        t = user_turns(s.get("messages", []))
        if not t:
            continue
        mm = metric_fn(t)
        if mm is None:
            continue
        n += 1
        nut.append(len(t))
        for k in M.METRIC_ORDER:
            per[k].append(mm[k])
    means = {k: (st.mean(v) if v else float("nan")) for k, v in per.items()}
    return n, (st.mean(nut) if nut else 0.0), means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strict", "hybrid"], default="strict")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    rows = []
    for p in a.paths:
        n, nut, means = cell(p, a.mode)
        rows.append((p, n, nut, means))
    order = M.METRIC_ORDER
    # print table: metric x condition
    names = [
        os.path.basename(p)
        .replace("_n1_t3.json_trajectories.json", "")
        .replace("_trajectories.json", "")
        for p, *_ in rows
    ]
    print(f"=== USI D1-D4 ({a.mode} mode) ===")
    hdr = f"{'metric':<12}" + "".join(f"{nm[:14]:>16}" for nm in names)
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        line = f"{k:<12}"
        for _, _, _, means in rows:
            v = means[k]
            line += f"{(f'{v:.2f}' if v == v else 'nan'):>16}"
        print(line)
    print("-" * len(hdr))
    print(f"{'n_eps':<12}" + "".join(f"{n:>16}" for _, n, _, _ in rows))
    print(
        f"{'usr_turns/ep':<12}"
        + "".join(f"{f'{nut:.1f}':>16}" for _, _, nut, _ in rows)
    )


if __name__ == "__main__":
    main()
