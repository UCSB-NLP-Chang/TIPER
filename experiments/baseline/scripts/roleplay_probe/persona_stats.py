#!/usr/bin/env python3
"""Complexity stats for the selected adversarial personas.

Reads personas_selected.json ({domain: [{persona_id, persona_text, ...}]}) and
reports per-persona + per-domain aggregates:
  chars, lines, words, sentences, rules (directive clauses), avg sentence len,
  distinct-directive-verb count, and a complexity flag.

"Rule" heuristic = count of directive clauses. A persona instruction is a list
of behavioral rules; we count clauses that impose behavior, approximated by:
  - bullet/numbered markers (-, •, *, ‣, 1., 1))
  - sentences containing a directive verb (must/should/always/never/要/必须/
    别/不要/总是/从不/倾向/尽量/拒绝/坚持/偷偷/主动/只/仅)
Whichever is larger (bullet style vs prose style), to avoid under-counting
prose personas that pack many rules into one paragraph.

Output: stdout table + writes personas_stats.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN = HERE / "personas_selected.json"
OUT = HERE / "personas_stats.json"

# Imperative directive verbs (bullet-style personas: "must never...", "要...").
DIRECTIVE_VERBS = [
    "must",
    "should",
    "refuse",
    "insist",
    "hide",
    "withhold",
    "proactively",
    "avoid",
    "tend to",
    "you tend",
    "要",
    "必须",
    "别",
    "不要",
    "总是",
    "从不",
    "倾向",
    "尽量",
    "拒绝",
    "坚持",
    "偷偷",
    "主动",
    "只",
    "仅",
    "绝不",
    "务必",
    "不得",
]
# Constraint markers — also catch descriptive/first-person prose personas
# ("you do not estimate", "I never swap", "only the copied wording", "instead of
#  cleaning up"). A persona sentence usually IS a rule if it carries one of these.
CONSTRAINT_MARKERS = [
    "do not",
    "don't",
    "never",
    "always",
    "only",
    "exactly",
    "instead of",
    "instead",
    "rather than",
    "rather",
    "without",
    "not ",
    "no ",
    "neither",
    "either",
    "unless",
    "if asked",
    "when asked",
]

BULLET_RE = re.compile(r"(?:^|\n)\s*(?:[-•*‣]|\d+[.)）]|[a-zA-Z][.)）])\s+")


def _sentences(t: str) -> list[str]:
    t = t.replace("\n", " ")
    parts = re.split(r"[。.!？?!?；;]+", t)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _words(t: str) -> int:
    # mixed zh/en: count CJK chars + latin word tokens
    cjk = len(re.findall(r"[一-鿿]", t))
    en = len(re.findall(r"[A-Za-z]+", t))
    return cjk + en


def persona_stats(text: str) -> dict:
    chars = len(text)
    lines = len([l for l in text.splitlines() if l.strip()])
    words = _words(text)
    sents = _sentences(text)
    n_sent = max(len(sents), 1)
    bullet_rules = len(BULLET_RE.findall(text))
    low_sents = [s.lower() for s in sents]
    verb_rules = sum(1 for s in low_sents if any(v in s for v in DIRECTIVE_VERBS))
    constraint_rules = sum(
        1 for s in low_sents if any(m in s for m in CONSTRAINT_MARKERS)
    )
    # prose rule count = sentences carrying a directive verb OR a constraint marker
    prose_rules = sum(
        1
        for s in low_sents
        if any(v in s for v in DIRECTIVE_VERBS)
        or any(m in s for m in CONSTRAINT_MARKERS)
    )
    rules = max(bullet_rules, prose_rules)
    avg_sent = round(words / n_sent, 1)
    # complexity flag: prose persona with >=8 rules and >=120 words, or
    # bullet persona with >=6 rules. Tuned to flag "rich" personas.
    complex_flag = (rules >= 8 and words >= 120) or (bullet_rules >= 6 and words >= 100)
    return {
        "chars": chars,
        "lines": lines,
        "words": words,
        "sentences": len(sents),
        "rules": rules,
        "bullet_rules": bullet_rules,
        "prose_rules": prose_rules,
        "verb_rules": verb_rules,
        "constraint_rules": constraint_rules,
        "avg_sent_words": avg_sent,
        "complex": complex_flag,
    }


def main():
    data = json.loads(IN.read_text())
    rows = []
    agg = {}
    for dom, personas in data.items():
        ds = []
        for p in personas:
            s = persona_stats(p["persona_text"])
            s.update(
                {
                    "domain": dom,
                    "persona_id": p["persona_id"],
                    "difficulty": p.get("difficulty"),
                }
            )
            rows.append(s)
            ds.append(s)
        agg[dom] = {
            "n": len(ds),
            "avg_chars": round(sum(r["chars"] for r in ds) / max(len(ds), 1), 0),
            "avg_words": round(sum(r["words"] for r in ds) / max(len(ds), 1), 0),
            "avg_rules": round(sum(r["rules"] for r in ds) / max(len(ds), 1), 1),
            "avg_lines": round(sum(r["lines"] for r in ds) / max(len(ds), 1), 1),
            "complex_count": sum(1 for r in ds if r["complex"]),
            "min_rules": min(r["rules"] for r in ds) if ds else 0,
            "max_rules": max(r["rules"] for r in ds) if ds else 0,
        }

    # per-persona table
    hdr = f"{'persona_id':<22}{'chars':>6}{'lines':>6}{'words':>6}{'sent':>5}{'rules':>6}{'avgS':>6}{'cmpx':>6}"
    print("=" * len(hdr))
    print("PER-PERSONA COMPLEXITY STATS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for dom, personas in data.items():
        print(f"  [{dom}]")
        for p in personas:
            s = persona_stats(p["persona_text"])
            flag = "yes" if s["complex"] else "no"
            print(
                f"  {p['persona_id']:<20}{s['chars']:>6}{s['lines']:>6}"
                f"{s['words']:>6}{s['sentences']:>5}{s['rules']:>6}"
                f"{s['avg_sent_words']:>6}{flag:>6}"
            )
        print()

    # per-domain aggregate
    print("=" * len(hdr))
    print("PER-DOMAIN AGGREGATE")
    print("=" * len(hdr))
    print(
        f"{'domain':<16}{'n':>4}{'avgCh':>7}{'avgW':>7}{'avgR':>7}{'cmpx':>6}{'minR':>6}{'maxR':>6}"
    )
    print("-" * len(hdr))
    for dom, a in agg.items():
        print(
            f"{dom:<16}{a['n']:>4}{a['avg_chars']:>7.0f}{a['avg_words']:>7.0f}"
            f"{a['avg_rules']:>7.1f}{a['complex_count']:>6}{a['min_rules']:>6}{a['max_rules']:>6}"
        )

    # complexity concerns
    print()
    print("COMPLEXITY CONCERNS (not complex / very few rules):")
    for r in rows:
        if not r["complex"] or r["rules"] < 5:
            print(
                f"  {r['persona_id']:<20} rules={r['rules']} words={r['words']} complex={r['complex']}"
            )

    OUT.write_text(json.dumps({"rows": rows, "agg": agg}, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
