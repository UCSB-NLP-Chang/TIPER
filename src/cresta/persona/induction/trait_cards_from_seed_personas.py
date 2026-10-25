#!/usr/bin/env python3
"""Convert high-k induced seed personas into trait cards.

High-k clustering can produce many tiny clusters. Treating each cluster as a
complete stable persona overstates what the data supports. This script keeps the
same data-grounded evidence but renders each cluster as a trait card: support,
signature traits, examples, metric profile, and usage guidance.

Example:
  .venv/bin/python -m cresta.persona.induction.trait_cards_from_seed_personas \
    --seed-dir experiments/persona_search/seeds/generated_brinks_k100_rule
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cresta.paths import PERSONA_SEEDS_DIR, REPO_ROOT

REPO = REPO_ROOT
DEFAULT_OUT_DIR = PERSONA_SEEDS_DIR / "trait_cards" / "generated"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return REPO / path


def _read_seed(seed_file: Path) -> dict[str, Any]:
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
            bullet = line[2:].strip()
            if bullet.startswith("You "):
                bullet = bullet[4:]
                bullet = bullet[:1].upper() + bullet[1:]
            bullets.append(bullet)

    examples = [
        match.group("example")
        for match in re.finditer(r'^\s+- "(?P<example>.*)"$', text, re.M)
    ]
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
    return [token.strip() for token in name.split("&") if token.strip()]


def _render_card(row: dict[str, Any], parsed: dict[str, Any], total: int) -> str:
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
        "- Source: real customer-support chats, style-subspace KMeans",
        f"- Source seed file: `{row['seed_file']}`",
        "",
        "## Signature Traits",
    ]
    if parsed["bullets"]:
        lines.extend(f"- {bullet}" for bullet in parsed["bullets"])
    else:
        lines.append("- No generated trait bullets were available.")

    lines += ["", "## Behavioral Examples"]
    if parsed["examples"]:
        lines.extend(f'- "{example}"' for example in parsed["examples"])
    else:
        lines.append(
            "- No contentful masked example turns were saved for this cluster."
        )

    lines += ["", "## Metric Profile"]
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
        "- Keep examples as style evidence only; do not copy specific customer facts into simulator scenarios.",
    ]
    if support < 10:
        lines.append(
            "- Low-support cluster: use cautiously; validate against more examples before using it as a seed."
        )
    elif effect.startswith("HARMS"):
        lines.append(
            "- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support."
        )

    lines += ["", "<!--", parsed["provenance"] or "provenance unavailable", "-->", ""]
    return "\n".join(lines)


def _render_readme(
    rows: list[dict[str, Any]],
    trait_stats: dict[str, dict[str, int]],
    total: int,
    out_dir: Path,
) -> str:
    effect_clusters = Counter(row["effect"].split()[0] for row in rows)
    effect_chats: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    for row in rows:
        effect_chats[row["effect"].split()[0]] += row["n"]
        n = row["n"]
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

    harms = [
        row for row in rows if row["effect"].startswith("HARMS") and row["n"] >= 10
    ]
    harms = sorted(harms, key=lambda row: (-row["n"], row["cluster"]))
    traits = sorted(trait_stats.items(), key=lambda kv: (-kv[1]["chats"], kv[0]))

    lines = [
        "# Trait Cards",
        "",
        "These cards convert high-k induced clusters into trait-level evidence.",
        "Use them as a trait bank, not as many independent stable personas.",
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
    for trait, stats in traits:
        lines.append(
            f"| {trait} | {stats['clusters']} | {stats['chats']} | {stats['helps']} | {stats['harms']} |"
        )

    lines += [
        "",
        "## HARMS Trait Clusters With n >= 10",
        "",
        "| cluster | trait card | n | name |",
        "|---:|---|---:|---|",
    ]
    for row in harms:
        rel = Path(row["trait_file"]).relative_to(out_dir)
        lines.append(
            f"| {row['cluster']} | [`{rel}`]({rel}) | {row['n']} | {row['name']} |"
        )

    lines += [
        "",
        "## Usage Guidance",
        "",
        "- Use `n >= 10` as a rough minimum when selecting traits for optimization seeds.",
        "- Combine 2-4 complementary traits into one coherent persona instead of running every trait card directly.",
        "- Prefer HARMS clusters for adversarial seed construction, while checking support size and examples.",
        "",
    ]
    return "\n".join(lines)


def build_trait_cards(seed_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    index_path = seed_dir / "index.json"
    rows = json.loads(index_path.read_text())
    total = sum(row["n"] for row in rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    trait_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"clusters": 0, "chats": 0, "helps": 0, "harms": 0}
    )
    out_index = []
    for row in rows:
        parsed = _read_seed(_resolve_path(row["seed_file"]))
        slug = re.sub(r"[^a-z0-9]+", "_", row["name"].lower()).strip("_")
        card_dir = out_dir / f"trait_{row['cluster']:02d}_{slug}"
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / "trait_card.md"
        card_path.write_text(_render_card(row, parsed, total))

        output_row = dict(row)
        output_row["trait_file"] = str(card_path)
        output_row["traits"] = _trait_tokens(row["name"])
        output_row["examples"] = parsed["examples"]
        output_row["metric_profile"] = parsed["profile"]
        out_index.append(output_row)

        for trait in output_row["traits"]:
            trait_stats[trait]["clusters"] += 1
            trait_stats[trait]["chats"] += row["n"]
            if row["effect"].startswith("HELPS"):
                trait_stats[trait]["helps"] += row["n"]
            else:
                trait_stats[trait]["harms"] += row["n"]

    (out_dir / "index.json").write_text(
        json.dumps(out_index, indent=2, ensure_ascii=False)
    )
    (out_dir / "README.md").write_text(
        _render_readme(out_index, trait_stats, total, out_dir)
    )
    return out_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_trait_cards(args.seed_dir, args.out_dir)
    print(f"wrote {len(rows)} trait cards to {args.out_dir}")
    print(args.out_dir / "README.md")


if __name__ == "__main__":
    main()
