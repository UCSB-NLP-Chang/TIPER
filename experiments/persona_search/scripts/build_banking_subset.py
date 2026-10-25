"""Build a banking_knowledge representative task subset for the persona-difficulty
experiment, using existing banking simulation runs as the signal.

Method (option "C": headroom-focused, for a difficulty-INCREASING persona):
  1. Pool all banking runs under data/simulations/*bank*/ (skip .bak/smoke/test),
     dropping infrastructure-artifact sims (is_infra_broken).
  2. Restrict the signal to STRONG-agent settings (gpt-5.4 / gpt-5.5) — the agent
     the persona experiment actually uses. A "setting" is the run dir name with a
     trailing _run<N> stripped.
  3. Drop floor (strong agent never solves: max reward == 0) and ceiling (always
     solved: min reward == 1) tasks — neither can show a persona effect.
  4. headroom(task) = mean strong-agent reward. A harder persona can only depress
     tasks that the strong agent currently solves often, so rank by headroom
     DESCENDING and take the top N.

Writes data/persona_opt/representative_tasks_banking_<N>.json in the schema
load_rep_tasks() reads ({"banking":[...], "telecom":[], "metadata":{...}}), ready
for `banking_difficulty.py --tasks-file <that>`.

HARD CONSTRAINT honoured: NEW file, reuses _common by import, edits nothing.

    .venv/bin/python experiments/persona_search/scripts/build_banking_subset.py --n 30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402, DATA_ROOT
    PERSONA_OPT_DIR,
    is_infra_broken,
    write_json,
)

DEFAULT_STRONG = ("_g54", "_g55", "gpt54", "gpt55")


def _setting_of(results_path: str) -> str:
    return re.sub(r"_run\d+$", "", Path(results_path).parent.name)


def _keep_dir(d: str) -> bool:
    s = d.lower()
    return not any(
        x in s for x in (".bak_", ".bak/", "/smoke_", "smoke_banking", "_test/")
    )


def collect(strong_markers: tuple[str, ...]):
    """Return (strong_rewards, pooled_rewards, n_settings, n_strong_settings)."""
    dirs = [
        d
        for d in sorted(
            glob.glob(str(DATA_ROOT / "data/simulations/*bank*/results.json"))
        )
        if _keep_dir(d)
    ]
    strong: dict[str, list[float]] = {}
    pooled: dict[str, list[float]] = {}
    settings, strong_settings = set(), set()
    for d in dirs:
        st = _setting_of(d)
        settings.add(st)
        is_strong = any(m in st.lower() for m in strong_markers)
        if is_strong:
            strong_settings.add(st)
        try:
            data = json.load(open(d))
        except Exception:
            continue
        for sim in data.get("simulations", []):
            if is_infra_broken(sim):
                continue
            r = (sim.get("reward_info") or {}).get("reward")
            tid = sim.get("task_id")
            if r is None or tid is None:
                continue
            pooled.setdefault(tid, []).append(float(r))
            if is_strong:
                strong.setdefault(tid, []).append(float(r))
    return strong, pooled, len(settings), len(strong_settings)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=30, help="subset size (top-N by headroom)")
    ap.add_argument(
        "--strong-markers",
        nargs="+",
        default=list(DEFAULT_STRONG),
        help="dir-name substrings identifying the experiment's agent",
    )
    ap.add_argument(
        "--max-headroom",
        type=float,
        default=0.97,
        help="exclude near-trivial tasks above this strong-agent mean",
    )
    ap.add_argument(
        "--out", default=None, help="default representative_tasks_banking_<N>.json"
    )
    a = ap.parse_args()

    strong, pooled, n_set, n_strong = collect(tuple(a.strong_markers))
    cand = []
    for tid, rs in strong.items():
        mx, mn, mean = max(rs), min(rs), sum(rs) / len(rs)
        if mx == 0.0 or mn == 1.0:  # strong-agent floor / ceiling
            continue
        if mean > a.max_headroom:  # near-trivial, little persona signal
            continue
        cand.append((tid, mean, len(rs)))
    cand.sort(key=lambda x: (-x[1], x[0]))  # headroom desc, id tiebreak
    chosen = cand[: a.n]
    ids = sorted(t for t, _, _ in chosen)

    out = (
        Path(a.out)
        if a.out
        else PERSONA_OPT_DIR / f"representative_tasks_banking_{len(ids)}.json"
    )
    headrooms = [m for _, m, _ in chosen]
    write_json(
        out,
        {
            "banking": ids,
            "telecom": [],
            "metadata": {
                "method": "option_C_headroom_topN_strong_agent",
                "strong_markers": a.strong_markers,
                "n_settings_total": n_set,
                "n_strong_settings": n_strong,
                "n_strong_discriminating_pool": len(cand),
                "selected": len(ids),
                "headroom_range": [round(min(headrooms), 3), round(max(headrooms), 3)]
                if headrooms
                else None,
                "headroom_by_task": {t: round(m, 3) for t, m, _ in chosen},
                "max_headroom_filter": a.max_headroom,
            },
        },
    )
    print(
        f"strong settings: {n_strong}/{n_set} | discriminating pool: {len(cand)} | selected: {len(ids)}"
    )
    if headrooms:
        print(
            f"headroom (strong-agent mean reward) of selected: "
            f"min {min(headrooms):.2f} median {statistics.median(headrooms):.2f} max {max(headrooms):.2f}"
        )
    print(f"-> {out}")
    print(
        f"\nrun:  .venv/bin/python experiments/persona_search/scripts/banking_difficulty.py "
        f"--tasks-file {out} --subset-size {len(ids)}"
    )


if __name__ == "__main__":
    main()
