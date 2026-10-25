#!/usr/bin/env python3
"""Extract user objectives from Brinks Home Security real conversations.

Reads user_simulator_dataset.json (1413 real customer-agent conversations),
extracts each conversation's user objective from early VISITOR turns
(skipping auth: account number, code word, name confirmation), and
outputs a CSV + JSON summary with topic classifications.

Usage:
  python extract_brinks_objectives.py
  python extract_brinks_objectives.py --input /path/to/user_simulator_dataset.json
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from cresta.paths import CRESTA_DATA_DIR, PERSONA_STATE_DIR

DEFAULT_INPUT = CRESTA_DATA_DIR / "user_simulator_dataset_v2.json"

_AUTH_PATTERNS = [
    re.compile(r"^[\d\s\-\(\)\.]+$"),  # phone/account numbers
    re.compile(
        r"^(yes|no|yeah|yep|nope|ok|okay|correct|that'?s? (right|correct))\.?$", re.I
    ),
    re.compile(r"^(code ?word|the code ?word is)\b", re.I),
    re.compile(r"^(my )?(code ?word|pass ?word|pass ?phrase)\b", re.I),
]


def _is_auth_turn(text: str) -> bool:
    """Return True if this visitor turn is part of the auth flow."""
    cleaned = text.strip().rstrip(".")
    if len(cleaned) < 3:
        return True
    for pat in _AUTH_PATTERNS:
        if pat.match(cleaned):
            return True
    # Short single-word responses (name, code word value, yes/no)
    words = cleaned.split()
    if len(words) <= 3 and len(cleaned) < 30:
        lower = cleaned.lower()
        if any(k in lower for k in ["code word", "codeword"]):
            return True
    return False


def extract_visitor_turns(messages: list[dict]) -> list[dict]:
    """Extract VISITOR turns with index, merging consecutive messages."""
    turns = []
    buf = []
    start_idx = -1
    for i, m in enumerate(messages):
        if m.get("speaker") == "VISITOR":
            text = m.get("text", "").strip()
            if text:
                if not buf:
                    start_idx = i
                buf.append(text)
        else:
            if buf:
                turns.append({"text": " ".join(buf), "msg_idx": start_idx})
                buf = []
    if buf:
        turns.append({"text": " ".join(buf), "msg_idx": start_idx})
    return turns


def extract_objective(messages: list[dict], max_scan: int = 8) -> str:
    """Extract user objective from first substantive VISITOR turn after auth."""
    visitor_turns = extract_visitor_turns(messages)
    objective_parts = []

    for turn in visitor_turns[:max_scan]:
        text = turn["text"].strip()
        if _is_auth_turn(text):
            continue

        objective_parts.append(text)

        reason_indicators = [
            "i need",
            "i want",
            "i'm calling",
            "i am calling",
            "calling about",
            "i'm having",
            "i am having",
            "having trouble",
            "having an issue",
            "help me",
            "help with",
            "assist me",
            "my camera",
            "my sensor",
            "my alarm",
            "my wifi",
            "my system",
            "my doorbell",
            "my panel",
            "my detector",
            "not working",
            "went offline",
            "going offline",
            "keeps going",
            "trouble with",
            "issue with",
            "problem with",
            "cancel",
            "schedule",
            "install",
            "billing",
            "payment",
            "flashing",
            "blinking",
            "no power",
            "no light",
        ]
        if any(ind in text.lower() for ind in reason_indicators):
            break

    return (
        " | ".join(objective_parts)
        if objective_parts
        else "[auth-only, no objective stated]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument(
        "--out-csv", type=Path, default=PERSONA_STATE_DIR / "brinks_objectives.csv"
    )
    ap.add_argument(
        "--out-json", type=Path, default=PERSONA_STATE_DIR / "brinks_objectives.json"
    )
    args = ap.parse_args()

    data = json.load(open(args.input))
    print(f"Loaded {len(data)} conversations from {args.input}")

    rows = []
    for entry in data:
        msgs = entry.get("messages", [])
        visitor_turns = extract_visitor_turns(msgs)
        objective = extract_objective(msgs)

        rows.append(
            {
                "conversation_id": entry.get("conversation_id", ""),
                "conv_type": entry.get("conv_type", ""),
                "topics": entry.get("topics", []),
                "resolution": entry.get("resolution", ""),
                "message_count": entry.get("message_count", len(msgs)),
                "visitor_turns": len(visitor_turns),
                "objective": objective,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "conv_type",
                "topics",
                "resolution",
                "message_count",
                "visitor_turns",
                "objective",
            ],
        )
        w.writeheader()
        for r in rows:
            row_copy = dict(r)
            row_copy["topics"] = (
                "|".join(r["topics"]) if isinstance(r["topics"], list) else r["topics"]
            )
            w.writerow(row_copy)

    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} rows to {args.out_csv} and {args.out_json}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    topic_counter = Counter()
    for r in rows:
        for t in r["topics"] if isinstance(r["topics"], list) else []:
            topic_counter[t] += 1
    print(f"\nTopics (multi-label, {len(topic_counter)} unique):")
    for k, v in topic_counter.most_common():
        print(f"  {k}: {v} ({v / len(rows) * 100:.1f}%)")

    type_counter = Counter(r["conv_type"] for r in rows)
    print(f"\nconv_type: {dict(type_counter)}")

    res_counter = Counter(r["resolution"] for r in rows)
    print(f"\nresolution:")
    for k, v in res_counter.most_common():
        print(f"  {k}: {v}")

    no_obj = [r for r in rows if r["objective"].startswith("[")]
    print(f"\nNo objective extracted: {len(no_obj)}/{len(rows)}")

    print(f"\n{'=' * 60}")
    print("SAMPLE OBJECTIVES BY TOPIC")
    print(f"{'=' * 60}")
    for topic, _ in topic_counter.most_common():
        topic_rows = [
            r
            for r in rows
            if topic in (r["topics"] if isinstance(r["topics"], list) else [])
        ]
        shown = 0
        print(f"\n--- {topic} ({len(topic_rows)} convos) ---")
        for r in topic_rows:
            if not r["objective"].startswith("[") and shown < 5:
                print(f"  {r['objective'][:180]}")
                shown += 1


if __name__ == "__main__":
    main()
