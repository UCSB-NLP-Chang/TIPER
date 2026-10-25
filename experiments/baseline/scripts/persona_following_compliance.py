"""Does a LONGER persona get followed WORSE by the user simulator?

For each persona we (1) parse its declared, STYLE-MEASURABLE target behaviors by
keyword, mapping each to a sim2real metric + expected direction, then (2) measure
the persona's ACTUAL metric means over its rollouts, and (3) score compliance =
fraction of declared targets whose actual metric lands on the expected side of
the no_persona baseline. Finally we correlate compliance with persona LENGTH
(chars) and #declared-targets to test the "too long -> poor following" hypothesis.

CAVEAT: only the STYLE-MEASURABLE subset of instructions is checked here. Long
mutations carry many SEMANTIC instructions (e.g. "scoped assent", "front-end
labels not back-end ids") that no single metric captures -> those need an LLM
adherence judge (separate, costs API). Coverage (#targets matched) is reported so
you can see how much of each persona this objective test actually exercises.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from persona_behavior_jsd import (
    METRIC_ORDER,
    persona_samples,
)
from persona_dispersion import no_persona_samples
from persona_frontier_diversity import sim_dir

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

STATE = str(DATA_ROOT / "data/persona_opt/pgepa_runs/xdom_v7_3domain/pgepa_state.json")
SEED_TMPL = str(DATA_ROOT / "data/simulations/gepa_opt/xdom6_g00_seed{n}_val")
OPT_TMPL = str(DATA_ROOT / "data/simulations/gepa_opt/xdom7_g{gen:02d}_k00_full_diff")
NOPERS = str(DATA_ROOT / "data/simulations/haiku_ccuser_no_persona")
VALMAP = str(DATA_ROOT / "data/persona_opt/val25_taskids.json")

# keyword (regex, on lowered persona text) -> list of (metric, direction 'hi'/'lo')
RULES = [
    (r"front[- ]?load", [("FrntLoad%", "hi")]),
    (r"pack (many|more)?.*(id|number)|many account id", [("IDs/trn", "hi")]),
    (
        r"\bterse\b|few words|just a few words|one[- ]word|very short",
        [("Wds/trn", "lo"), ("Short%", "hi")],
    ),
    (
        r"long, detailed|write long|verbose|always elaborate|rarely send one[- ]word",
        [("Wds/trn", "hi"), ("Short%", "lo")],
    ),
    (r"polite|please/thanks|frequent please", [("Polite%", "hi")]),
    (r"skip pleasantries|blunt", [("Polite%", "lo")]),
    (r"unsure|hesitant|not sure|\bi think\b|\bmaybe\b", [("Uncert%", "hi")]),
    (r"strong certainty|definitely|for sure", [("Certn%", "hi")]),
    (r"change (your )?mind|switch topic|never mind|\bactually\b", [("Pivot%", "hi")]),
    (r"clarifying question|what do you mean", [("ClarfyQ%", "hi")]),
    (r"rarely ask questions|mostly state|do not ask", [("InfoQ%", "lo")]),
    (
        r"reveal.*gradually|little at a time|drip|few identifiers|give few id",
        [("IDs/trn", "lo"), ("FrntLoad%", "lo")],
    ),
]


def declared_targets(text):
    t = text.lower()
    targets = {}
    for rx, mds in RULES:
        if re.search(rx, t):
            for m, d in mds:
                targets[m] = d  # last rule wins on conflict (rare)
    return targets


def main():
    st = json.load(open(STATE))
    cands = [c for c in st["candidates"] if c.get("feasible")]

    # baseline: no_persona metric means (val-filtered)
    val_map = json.load(open(VALMAP))
    nop = no_persona_samples(NOPERS, val_map)
    base = {m: float(np.mean([v[m] for v in nop])) for m in METRIC_ORDER}

    rows = []
    for c in cands:
        d = sim_dir(c, SEED_TMPL, OPT_TMPL)
        samples, _ = persona_samples(d)
        if not samples:
            continue
        act = {m: float(np.mean([v[m] for v in samples])) for m in METRIC_ORDER}
        tgts = declared_targets(c["persona_text"])
        hits, checks = 0, []
        for m, dirn in tgts.items():
            diff = act[m] - base[m]
            ok = (diff > 0) if dirn == "hi" else (diff < 0)
            hits += ok
            checks.append((m, dirn, ok, diff))
        comp = hits / len(tgts) if tgts else float("nan")
        nm = (
            c["persona_text"]
            .splitlines()[0]
            .replace("# Persona:", "")
            .split("(effect")[0]
            .strip()
        )
        rows.append(
            dict(
                idx=c["idx"],
                gen=c["gen"],
                name=nm[:26],
                chars=len(c["persona_text"]),
                ntgt=len(tgts),
                comp=comp,
                checks=checks,
            )
        )

    rows.sort(key=lambda r: r["chars"])
    print(f"{'persona':30s}{'gen':>4s}{'chars':>7s}{'#tgt':>5s}{'compliance':>11s}")
    for r in rows:
        ct = (
            f"{r['comp'] * 100:.0f}% ({sum(1 for _, _, ok, _ in r['checks'] if ok)}/{r['ntgt']})"
            if r["ntgt"]
            else "n/a"
        )
        print(
            f"i{r['idx']:<2d} {r['name']:27s}{r['gen']:>4d}{r['chars']:>7d}{r['ntgt']:>5d}{ct:>11s}"
        )

    # correlations (only personas with >=2 measurable targets)
    use = [r for r in rows if r["ntgt"] >= 2 and not np.isnan(r["comp"])]
    if len(use) >= 3:
        ch = np.array([r["chars"] for r in use], float)
        cp = np.array([r["comp"] for r in use], float)
        nt = np.array([r["ntgt"] for r in use], float)

        def corr(a, b):
            return (
                float(np.corrcoef(a, b)[0, 1])
                if a.std() > 0 and b.std() > 0
                else float("nan")
            )

        print(f"\nn={len(use)} personas with >=2 measurable targets")
        print(
            f"  corr(length_chars, compliance) = {corr(ch, cp):+.2f}  (hypothesis: negative)"
        )
        print(f"  corr(#targets,    compliance) = {corr(nt, cp):+.2f}")
        # short vs long split at median
        med = np.median(ch)
        short = [r["comp"] for r in use if r["chars"] <= med]
        long = [r["comp"] for r in use if r["chars"] > med]
        print(
            f"  mean compliance  short(<= {med:.0f} chars) = {np.mean(short) * 100:.0f}%  "
            f"|  long(> {med:.0f}) = {np.mean(long) * 100:.0f}%"
        )

    # show per-check detail for the longest 3 and shortest 3
    print("\n=== per-target detail (✗ = declared but NOT followed) ===")
    for r in rows[:3] + rows[-3:]:
        det = "  ".join(
            f"{m}{'↑' if d == 'hi' else '↓'}{'✓' if ok else '✗'}"
            for m, d, ok, _ in r["checks"]
        )
        print(f"i{r['idx']:<2d} {r['name'][:22]:23s}({r['chars']:>4d}c) {det}")


if __name__ == "__main__":
    main()
