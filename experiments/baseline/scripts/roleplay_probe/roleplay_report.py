#!/usr/bin/env python3
"""Role-play probe report: aggregate judge scores + behavior signals + difficulty
佐证 + judge IAA into the 2-axis positioning table + per-dim table + signal table.

Reads (for a domain):
  <root>/judge_scores.jsonl      — per (record, judge) {D1..D6}
  <root>/behavior_agg.json        — per (user_model, persona) signal means
  <root>/<user_model>/<persona_id>/<task>.json — dump records (for Δreward baseline)

Produces <root>/roleplay_report.md + roleplay_report.json.

2-axis (independently reported, never collapsed to one total):
  Fidelity    = mean(D1, D2)
  Adversarial = normalize(D3 - D6)   # D3 high + D6 low = strong adversarial actor
                                 # D3 low  + D6 high = tame method actor (sonnet-5 hyp)
IAA: Spearman + Kendall per dim between the two judges (manual impl, no scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
DUMP_ROOT = REPO / "data/simulations/roleplay_probe"
DIMS = ("D1", "D2", "D3", "D4", "D5", "D6")


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    ra, rb = _ranks(a), _ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if da == 0 or db == 0 else round(num / (da * db), 3)


def kendall(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    conc = disc = tie = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = da * db
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                tie += 1
    tot = conc + disc + tie
    return None if tot == 0 else round((conc - disc) / tot, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dump-root", default=None)
    args = ap.parse_args()
    root = Path(args.dump_root) if args.dump_root else DUMP_ROOT / args.domain

    # ---- judge scores ----
    scores = defaultdict(lambda: defaultdict(list))  # (um,pid)[dim] -> [vals per judge]
    by_key = defaultdict(dict)  # (um,pid,task)[judge_id] -> {dims}
    for line in (root / "judge_scores.jsonl").read_text().splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        k = (o["user_model"], o["persona_id"], o["task_id"])
        by_key[k][o["judge_id"]] = o
    # per (um,pid) mean across tasks AND judges
    per_up = defaultdict(lambda: defaultdict(list))
    for k, judges in by_key.items():
        um, pid, tid = k
        for jid, o in judges.items():
            for d in DIMS:
                if o.get(d) is not None:
                    per_up[(um, pid)][d].append(o[d])

    user_models = sorted({k[0] for k in by_key})
    personas = sorted({k[1] for k in by_key})

    # ---- difficulty 佐证: persona vs no_persona Δreward (per um) ----
    rewards = defaultdict(lambda: defaultdict(list))  # um[pid] -> [reward]
    for p in root.rglob("*.json"):
        if p.name in (
            "judge_scores.jsonl",
            "judge_summary.json",
            "behavior_signals.jsonl",
            "behavior_agg.json",
            "roleplay_report.json",
            "roleplay_report.md",
        ):
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if "agent_reward" in rec and rec.get("agent_reward") is not None:
            rewards[rec["user_model"]][rec["persona_id"]].append(rec["agent_reward"])
    delta = {}
    for um in user_models:
        base = rewards[um].get("no_persona", [])
        base_avg = sum(base) / len(base) if base else None
        row = {}
        for pid in personas:
            if pid == "no_persona":
                continue
            pr = rewards[um].get(pid, [])
            if pr and base_avg is not None:
                row[pid] = round(base_avg - sum(pr) / len(pr), 3)
        delta[um] = {
            "baseline_avg": round(base_avg, 3) if base_avg is not None else None,
            "delta_no_minus_persona": row,
        }

    # ---- IAA per dim ----
    iaa = {}
    for d in DIMS:
        a, b = [], []
        for k, judges in by_key.items():
            if "gpt52" in judges and "dsv4pro" in judges:
                va = judges["gpt52"].get(d)
                vb = judges["dsv4pro"].get(d)
                if va is not None and vb is not None:
                    a.append(va)
                    b.append(vb)
        iaa[d] = {"n": len(a), "spearman": spearman(a, b), "kendall": kendall(a, b)}

    # ---- behavior signals ----
    bpath = root / "behavior_agg.json"
    beh = json.loads(bpath.read_text()) if bpath.exists() else {}

    # ---- build report ----
    def mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    lines = [f"# Role-Play Probe Report — {args.domain}\n"]
    lines.append("## Per-dimension scores (mean over persona × task × judge)\n")
    lines.append(
        "| user_model | " + " | ".join(DIMS) + " | Fidelity(D1+D2) | Adv(D3−D6) |"
    )
    lines.append("|" + "---|" * (len(DIMS) + 3))
    rep = {}
    for um in user_models:
        dvals = {d: [] for d in DIMS}
        for pid in personas:
            for d in DIMS:
                dvals[d].extend(per_up[(um, pid)][d])
        row = {d: mean(dvals[d]) for d in DIMS}
        fid = mean((row["D1"], row["D2"])) if row["D1"] is not None else None
        adv = (
            round(row["D3"] - row["D6"], 2)
            if row["D3"] is not None and row["D6"] is not None
            else None
        )
        lines.append(
            f"| {um} | " + " | ".join(str(row[d]) for d in DIMS) + f" | {fid} | {adv} |"
        )
        rep[um] = {"dims": row, "fidelity": fid, "adversarial": adv}
    lines.append("")

    lines.append(
        "## Difficulty 佐证 (no_persona − persona reward, >0 = persona harder)\n"
    )
    lines.append("| user_model | baseline | mean Δ over personas | per-persona Δ |")
    lines.append("|---|---|---|---|")
    for um in user_models:
        d = delta.get(um, {})
        bs = d.get("baseline_avg")
        dd = d.get("delta_no_minus_persona", {})
        m = round(sum(dd.values()) / len(dd), 3) if dd else None
        lines.append(
            f"| {um} | {bs} | {m} | "
            + ", ".join(f"{k}:{v}" for k, v in dd.items())
            + " |"
        )
    lines.append("")

    lines.append("## Judge IAA (gpt-5.2 vs deepseek-v4-pro)\n")
    lines.append("| dim | n | spearman | kendall |")
    lines.append("|---|---|---|---|")
    for d in DIMS:
        v = iaa[d]
        lines.append(f"| {d} | {v['n']} | {v['spearman']} | {v['kendall']} |")
    lines.append("")

    if beh:
        lines.append("## Behavior signals (mean over persona × task)\n")
        lines.append(
            "| user_model | turns | avgChars | ask | supp | comply | proactive | transfer |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for um in user_models:
            rows = list(beh.get(um, {}).values())
            if not rows:
                continue
            nc = len(rows)
            lines.append(
                f"| {um} | "
                f"{sum(r['n_user_turns'] for r in rows) / nc:.1f} | "
                f"{sum(r['avg_user_chars'] for r in rows) / nc:.0f} | "
                f"{sum(r['ask_markers'] for r in rows) / nc:.1f} | "
                f"{sum(r['supplement_markers'] for r in rows) / nc:.1f} | "
                f"{sum(r['comply_markers'] for r in rows) / nc:.1f} | "
                f"{sum(r['proactive_info_turns'] for r in rows) / nc:.1f} | "
                f"{sum(r['transfer_rate'] for r in rows) / nc:.3f} |"
            )
        lines.append("")

    md = "\n".join(lines)
    (root / "roleplay_report.md").write_text(md)
    (root / "roleplay_report.json").write_text(
        json.dumps(
            {"per_user_model": rep, "delta": delta, "iaa": iaa},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(md)
    print(f"\nwrote {root / 'roleplay_report.md'} + roleplay_report.json")


if __name__ == "__main__":
    main()
