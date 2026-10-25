"""Render side-by-side case studies of raw sonnet46user vs cc_con for chosen tasks.

Picks representative judged pairs (from realism_judge/per_pair.jsonl) and dumps the
two full transcripts + the judge verdicts to a markdown file, so we can read the
ACTUAL behavioral divergence (grounding/fabrication, naturalness) turn-by-turn.
"""

import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_user_realism import (  # noqa: E402
    CC_RUN,
    RAW_RUN,
    _instr_text,
    _load,
    _render,
)


def main():
    rows = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in open(str(DATA_ROOT / "data/skill_eval/realism_judge/per_pair.jsonl"))
    }
    raw_top, cc_top = _load(RAW_RUN), _load(CC_RUN)
    tasks_by_id = {t["id"]: t for t in cc_top["tasks"]}
    raw_idx = {s["task_id"]: s for s in raw_top["simulations"] if s.get("trial") == 0}
    cc_idx = {s["task_id"]: s for s in cc_top["simulations"] if s.get("trial") == 0}

    # selection: (label, predicate) — pick first match
    def pick(pred, n=1):
        out = []
        for tid, r in rows.items():
            if pred(r) and tid in raw_idx and tid in cc_idx:
                out.append(tid)
                if len(out) >= n:
                    break
        return out

    selections = []
    # 1. cc fabricates: raw wins BOTH dims 5-0 (cleanest grounding failure)
    for tid in pick(
        lambda r: (
            r["instruction_tally"].get("raw", 0) == 5
            and r["realism_tally"].get("raw", 0) == 5
        ),
        2,
    ):
        selections.append(("CC LOSES BOTH (grounding + realism), raw 5-0/5-0", tid))
    # 2. cc more natural AND better IF: cc wins both 5-0
    for tid in pick(
        lambda r: (
            r["instruction_tally"].get("cc", 0) == 5
            and r["realism_tally"].get("cc", 0) == 5
        ),
        2,
    ):
        selections.append(("CC WINS BOTH 5-0", tid))
    # 3. classic tradeoff: cc more realistic but raw follows instructions better
    for tid in pick(
        lambda r: (
            r["realism_winner"] == "cc" and r["instruction_tally"].get("raw", 0) >= 4
        ),
        2,
    ):
        selections.append(("TRADEOFF: cc more natural, raw more grounded", tid))

    out = ["# Case studies: raw `sonnet46user` vs `cc_con` (telecom gpt54, trial 0)\n"]
    seen = set()
    for label, tid in selections:
        if tid in seen:
            continue
        seen.add(tid)
        r = rows[tid]
        out.append(f"\n---\n\n## {label}\n")
        out.append(f"**task**: `{tid}`\n")
        out.append(
            f"**verdict** — realism: {r['realism_winner']} {r['realism_tally']} | "
            f"instruction-following: {r['instruction_winner']} {r['instruction_tally']}\n"
        )
        # a couple judge reasons
        out.append("\n**judge reasons (sample):**\n")
        for c in r["calls"][:2]:
            v, mp = c["verdict"], c["mapping"]
            rw, iw = (
                mp.get(v.get("realism_winner")),
                mp.get(v.get("instruction_winner")),
            )
            out.append(f"- realism→{rw}: {v.get('realism_reason')}")
            out.append(f"- instr→{iw}: {v.get('instruction_reason')}")
        out.append(
            f"\n**ROLE-PLAY INSTRUCTIONS:**\n```\n{_instr_text(tasks_by_id[tid])}\n```\n"
        )
        out.append(
            "\n### RAW sonnet46user transcript\n```\n"
            + _render(raw_idx[tid])
            + "\n```\n"
        )
        out.append("\n### CC_CON transcript\n```\n" + _render(cc_idx[tid]) + "\n```\n")

    open("reports/case_study_sonnet46_vs_cccon.md", "w").write("\n".join(out))
    print(f"wrote reports/case_study_sonnet46_vs_cccon.md with {len(seen)} cases:")
    for label, tid in selections:
        print(f"  - {label}: {tid[:50]}")


if __name__ == "__main__":
    main()
