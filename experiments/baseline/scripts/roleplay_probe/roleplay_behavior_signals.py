#!/usr/bin/env python3
"""Behavioral signals from user messages — regex佐证 for the judge's D3
(Adversarial Commitment) and D6 (Over-Cooperative), to keep the judge honest
against style bias.

Reads dump records (same as roleplay_judge.py) and emits per-record signals +
per (user_model, persona) aggregates. Signals are heuristic and noisy — they
SUPPORT the judge scores, never override them.

Signals (computed over CUSTOMER/user messages only):
  n_user_turns        — participation density (tau2 rootcause: user-channel attack)
  avg_user_chars      — terseness proxy (low = terse persona held)
  first_turn_chars    — info volunteered upfront (low = withholding persona)
  ask_markers         — "要不要/应该..吗/should I/can I/要不要我" count
                        (D3 ask-vs-snek; D6 proposing-next-step over-coop)
  supplement_markers  — "补充/另外/刚才..但/先别..我/等一下" count
                        (delivery-style withhold-then-supplement adversarial)
  comply_markers      — "好的/OK/行/没问题/听你的" immediate compliance (D3 low)
  transfer_markers    — "人工/转人工/manager/speak to a person" (escalation)
  proactive_info_turns — turns (after 1st) introducing new entities (D6 over-coop)

Output: data/simulations/roleplay_probe/<domain>/behavior_signals.jsonl
        + behavior_agg.json (per user_model x persona means)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
DUMP_ROOT = REPO / "data/simulations/roleplay_probe"

ASK_RE = re.compile(
    r"要不要|应不应该|是否|能不能|可以吗|要不要我|should i|can i|shall i|do you want me to",
    re.I,
)
SUPP_RE = re.compile(
    r"补充|另外|对了|顺便|刚才.{0,8}(但|不过|其实)|先别|等一下|wait|actually|one more thing",
    re.I,
)
COMPLY_RE = re.compile(
    r"^(好的|好|OK|行|可以|没问题|听你的|明白|收到|嗯|ok)[，。!！. ]*$", re.I
)
TRANSFER_RE = re.compile(
    r"人工|转人工|客服|manager|speak to a (person|human)|supervisor", re.I
)
# entity-ish tokens introduced later (loose: digits/units/proper nouns)
ENTITY_RE = re.compile(r"\d|元|份|号|码|[a-zA-Z]{4,}")


def signals_for(rec: dict) -> dict:
    ums = [m for m in rec.get("messages", []) if m.get("role") == "user"]
    contents = [m.get("content", "") or "" for m in ums]
    n = len(contents)
    chars = [len(c) for c in contents]
    ask = sum(len(ASK_RE.findall(c)) for c in contents)
    supp = sum(len(SUPP_RE.findall(c)) for c in contents)
    comply = sum(1 for c in contents if COMPLY_RE.match(c.strip()))
    transfer = 1 if any(TRANSFER_RE.search(c) for c in contents) else 0
    # proactive info: turns after the 1st that introduce a new entity
    seen = set()
    proactive = 0
    for i, c in enumerate(contents):
        toks = set(ENTITY_RE.findall(c))
        if i > 0 and (toks - seen):
            proactive += 1
        seen |= toks
    return {
        "domain": rec.get("domain"),
        "user_model": rec.get("user_model"),
        "persona_id": rec.get("persona_id"),
        "task_id": rec.get("task_id"),
        "agent_reward": rec.get("agent_reward"),
        "n_user_turns": n,
        "avg_user_chars": round(sum(chars) / n, 1) if n else 0,
        "first_turn_chars": chars[0] if chars else 0,
        "ask_markers": ask,
        "supplement_markers": supp,
        "comply_markers": comply,
        "transfer": transfer,
        "proactive_info_turns": proactive,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dump-root", default=None)
    args = ap.parse_args()
    root = Path(args.dump_root) if args.dump_root else DUMP_ROOT / args.domain
    records = []
    for p in sorted(root.rglob("*.json")):
        if p.name in (
            "behavior_signals.jsonl",
            "behavior_agg.json",
            "judge_scores.jsonl",
            "judge_summary.json",
        ):
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if "messages" in rec and "user_model" in rec:
            records.append(rec)
    print(f"{args.domain}: {len(records)} records")

    out = root / "behavior_signals.jsonl"
    agg = defaultdict(lambda: defaultdict(list))
    with open(out, "w") as f:
        for rec in records:
            s = signals_for(rec)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            key = (s["user_model"], s["persona_id"])
            for k in (
                "n_user_turns",
                "avg_user_chars",
                "first_turn_chars",
                "ask_markers",
                "supplement_markers",
                "comply_markers",
                "transfer",
                "proactive_info_turns",
            ):
                agg[key][k].append(s[k])

    # per user_model x persona means
    means = {}
    for (um, pid), kv in agg.items():
        means.setdefault(um, {})[pid] = {
            k: round(sum(v) / len(v), 2) for k, v in kv.items() if k != "transfer"
        }
        means[um][pid]["transfer_rate"] = round(
            sum(kv["transfer"]) / len(kv["transfer"]), 3
        )
    (root / "behavior_agg.json").write_text(
        json.dumps(means, ensure_ascii=False, indent=2)
    )
    print(f"wrote {out} + behavior_agg.json")
    # quick per-user_model rollup
    print("\nper user_model rollup (mean over persona×task):")
    for um in sorted(means):
        rows = list(means[um].values())
        nc = len(rows)
        print(
            f"  {um:<28} turns={sum(r['n_user_turns'] for r in rows) / nc:.1f} "
            f"avgCh={sum(r['avg_user_chars'] for r in rows) / nc:.0f} "
            f"ask={sum(r['ask_markers'] for r in rows) / nc:.1f} "
            f"supp={sum(r['supplement_markers'] for r in rows) / nc:.1f} "
            f"comply={sum(r['comply_markers'] for r in rows) / nc:.1f} "
            f"proactive={sum(r['proactive_info_turns'] for r in rows) / nc:.1f} "
            f"transfer={sum(r['transfer_rate'] for r in rows) / nc:.3f}"
        )


if __name__ == "__main__":
    main()
