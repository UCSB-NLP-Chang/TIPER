#!/usr/bin/env python3
"""Convert high-k rule-generated seed personas into trait cards.

High-k clustering (for example k=100) produces many tiny clusters. Treating each
cluster as a full stable persona overstates what the data supports. This script
keeps the same data-grounded evidence but renders each cluster as a trait card:
support, signature traits, examples, metric profile, and usage guidance.

Example:
    .venv/bin/python experiments/persona_search/scripts/trait_cards_from_seed_personas.py \
        --seed-dir data/persona_opt/seeds_brinks_k100_rule \
        --out-dir data/persona_opt/trait_cards_brinks_k100
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _read_seed(seed_file: Path) -> dict:
    text = seed_file.read_text()
    title = re.search(
        r"^# Persona: (?P<name>.*?)  \(effect: (?P<effect>.*?)\)", text, re.M
    )
    profile = re.search(r"^cluster metric profile: (?P<profile>.*)$", text, re.M)
    provenance = re.search(r"^provenance: (?P<provenance>.*)$", text, re.M)

    bullets: list[str] = []
    in_behaviors = False
    for line in text.splitlines():
        if line.strip() == "## How you behave":
            in_behaviors = True
            continue
        if in_behaviors and line.startswith("## "):
            break
        if in_behaviors and line.startswith("- "):
            b = line[2:].strip()
            if b.startswith("You "):
                b = b[4:]
                b = b[:1].upper() + b[1:]
            bullets.append(b)

    examples: list[str] = []
    for m in re.finditer(r'^\s+- "(?P<ex>.*)"$', text, re.M):
        examples.append(m.group("ex"))

    return {
        "name": title.group("name") if title else seed_file.parent.name,
        "effect": title.group("effect") if title else "UNKNOWN",
        "profile": profile.group("profile") if profile else "",
        "provenance": provenance.group("provenance") if provenance else "",
        "bullets": bullets,
        "examples": examples,
        "source_text": text,
    }


def _trait_tokens(name: str) -> list[str]:
    return [t.strip() for t in name.split("&") if t.strip()]


def _render_card(row: dict, parsed: dict, total: int) -> str:
    cluster = row["cluster"]
    name = row["name"]
    support = row["n"]
    frac = row["frac"]
    effect = row["effect"]
    tokens = _trait_tokens(name)
    token_s = " + ".join(tokens) if tokens else name

    lines = [
        f"# Trait Cluster {cluster:02d}: {token_s}",
        "",
        "## Support",
        f"- Real chats: {support} / {total} ({frac:.1%})",
        f"- Effect heuristic: {effect}",
        "- Source: Brinks real customer chats, k=100 style-subspace KMeans",
        f"- Source seed file: `{row['seed_file']}`",
        "",
        "## Signature Traits",
    ]
    if parsed["bullets"]:
        lines.extend(f"- {b}" for b in parsed["bullets"])
    else:
        lines.append("- No rule-generated trait bullets were available.")

    lines += [
        "",
        "## Behavioral Examples",
    ]
    if parsed["examples"]:
        lines.extend(f'- "{e}"' for e in parsed["examples"])
    else:
        lines.append(
            "- No contentful masked example turns were saved for this cluster."
        )

    lines += [
        "",
        "## Metric Profile",
    ]
    if parsed["profile"]:
        for part in parsed["profile"].split(", "):
            lines.append(f"- {part}")
    else:
        lines.append("- No metric profile was found in the source seed file.")

    lines += [
        "",
        "## How To Use This Trait",
        "- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.",
        "- Combine it with other trait cards when constructing a stable seed persona for optimization.",
        "- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.",
    ]
    if support < 10:
        lines.append(
            "- Low-support cluster: use cautiously; validate against more examples before turning it into a seed."
        )
    elif effect.startswith("HARMS"):
        lines.append(
            "- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support."
        )

    lines += [
        "",
        "<!--",
        parsed["provenance"] or "provenance unavailable",
        "-->",
        "",
    ]
    return "\n".join(lines)


def _render_readme(
    rows: list[dict], trait_stats: dict[str, dict], total: int, out_dir: Path
) -> str:
    effect_clusters = Counter(r["effect"].split()[0] for r in rows)
    effect_chats = Counter()
    buckets = Counter()
    for r in rows:
        effect_chats[r["effect"].split()[0]] += r["n"]
        n = r["n"]
        if n == 1:
            bucket = "1"
        elif n <= 3:
            bucket = "2-3"
        elif n <= 5:
            bucket = "4-5"
        elif n <= 10:
            bucket = "6-10"
        elif n <= 20:
            bucket = "11-20"
        elif n <= 50:
            bucket = "21-50"
        else:
            bucket = ">50"
        buckets[bucket] += 1

    harms = [r for r in rows if r["effect"].startswith("HARMS") and r["n"] >= 10]
    harms = sorted(harms, key=lambda r: (-r["n"], r["cluster"]))
    traits = sorted(trait_stats.items(), key=lambda kv: (-kv[1]["chats"], kv[0]))

    lines = [
        "# Brinks k=100 Trait Cards",
        "",
        "These cards convert high-k rule-generated Brinks clusters into trait-level evidence.",
        "They should be used as a trait bank, not as 100 independent stable personas.",
        "",
        "## Summary",
        f"- Source clusters: {len(rows)}",
        f"- Source chats: {total}",
        f"- Output directory: `{out_dir}`",
        f"- Effect by clusters: HELPS={effect_clusters['HELPS']}, HARMS={effect_clusters['HARMS']}",
        f"- Effect by covered chats: HELPS={effect_chats['HELPS']}, HARMS={effect_chats['HARMS']}",
        "",
        "## Cluster Size Buckets",
        "",
    ]
    for bucket in ["1", "2-3", "4-5", "6-10", "11-20", "21-50", ">50"]:
        lines.append(f"- `{bucket}` chats: {buckets[bucket]} clusters")

    lines += [
        "",
        "## Top Traits By Covered Chats",
        "",
        "| trait | clusters | chats | HELPS chats | HARMS chats |",
        "|---|---:|---:|---:|---:|",
    ]
    for trait, s in traits:
        lines.append(
            f"| {trait} | {s['clusters']} | {s['chats']} | {s['helps']} | {s['harms']} |"
        )

    lines += [
        "",
        "## HARMS Trait Clusters With `n >= 10`",
        "",
        "| cluster | trait card | n | name |",
        "|---:|---|---:|---|",
    ]
    for r in harms:
        rel = Path(r["trait_file"]).relative_to(out_dir)
        lines.append(f"| {r['cluster']} | [`{rel}`]({rel}) | {r['n']} | {r['name']} |")

    lines += [
        "",
        "## Usage Guidance",
        "",
        "- Use `n >= 10` as a rough minimum when selecting traits for optimization seeds.",
        "- Combine 2-4 complementary traits into one coherent persona instead of running every trait card directly.",
        "- Prefer HARMS clusters for adversarial seed construction, but keep an eye on support size and examples.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    index_path = args.seed_dir / "index.json"
    rows = json.loads(index_path.read_text())
    total = sum(r["n"] for r in rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trait_stats: dict[str, dict] = defaultdict(
        lambda: {
            "clusters": 0,
            "chats": 0,
            "helps": 0,
            "harms": 0,
        }
    )
    out_index = []
    for r in rows:
        parsed = _read_seed(Path(r["seed_file"]))
        slug = re.sub(r"[^a-z0-9]+", "_", r["name"].lower()).strip("_")
        card_dir = args.out_dir / f"trait_{r['cluster']:02d}_{slug}"
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / "trait_card.md"
        card_path.write_text(_render_card(r, parsed, total))

        nr = dict(r)
        nr["trait_file"] = str(card_path)
        nr["traits"] = _trait_tokens(r["name"])
        nr["examples"] = parsed["examples"]
        nr["metric_profile"] = parsed["profile"]
        out_index.append(nr)

        for t in nr["traits"]:
            trait_stats[t]["clusters"] += 1
            trait_stats[t]["chats"] += r["n"]
            if r["effect"].startswith("HELPS"):
                trait_stats[t]["helps"] += r["n"]
            else:
                trait_stats[t]["harms"] += r["n"]

    (args.out_dir / "index.json").write_text(
        json.dumps(out_index, indent=2, ensure_ascii=False)
    )
    (args.out_dir / "README.md").write_text(
        _render_readme(out_index, trait_stats, total, args.out_dir)
    )

    print(f"wrote {len(out_index)} trait cards to {args.out_dir}")
    print(args.out_dir / "README.md")


if __name__ == "__main__":
    main()
