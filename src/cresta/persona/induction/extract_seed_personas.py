#!/usr/bin/env python3
"""Extract data-grounded seed personas from a real customer-chat corpus.

This is the persona induction entrypoint for the standalone persona toolkit.
It turns a customer-support chat corpus into drop-in seed persona markdown files
that can be passed to `code/search/parallel_gepa.py --seed-persona`.

Input corpus format:
  JSON list (or dict of records) with each conversation shaped like:
    {"messages": [{"speaker": "VISITOR", "text": "..."}, ...]}

Two induction methods are supported:

  llm
    Cluster real customers on a rule-based style subspace, pick representative
    and hard exemplar chats per cluster, then ask an LLM to infer a reusable
    customer disposition persona. Requires `OPENAI_API_KEY`.

  rule
    No-API fallback. Renders names and behavior bullets from the cluster metric
    profile using fixed rules. Cheaper and deterministic, but more generic.

Examples:
  # Deterministic, no-API seed extraction.
  .venv/bin/python -m cresta.persona.induction.extract_seed_personas \
    --data /path/to/user_simulator_dataset_v2.json \
    --method rule --k 5 \
    --out-dir experiments/persona_search/seeds/generated_brinks_rule

  # LLM disposition induction dry-run: print prompts without calling the API.
  .venv/bin/python -m cresta.persona.induction.extract_seed_personas \
    --data /path/to/user_simulator_dataset_v2.json \
    --method llm --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.vq import kmeans2

from cresta.evaluation.features import feature_matrix
from cresta.evaluation.usi import sim2real_metrics as M
from cresta.paths import PERSONA_SEEDS_DIR, REPO_ROOT

REPO = REPO_ROOT

DEFAULT_OUT_DIR = PERSONA_SEEDS_DIR / "generated_brinks"

_RE_DIGITS = re.compile(r"\d[\d\-\s]{2,}\d")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_BANNED_META = re.compile(
    r"\b(dataset|cluster|scenario|benchmark|simulat|persona|AI|model)\b", re.I
)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def mask(text: str) -> str:
    """Lightweight redaction for examples sent to the LLM and saved as comments."""
    text = _RE_EMAIL.sub("<email>", text)
    text = _RE_DIGITS.sub("<redacted>", text)
    return " ".join(text.split())


def load_corpus(path: str | Path, speaker: str, min_turns: int) -> list[dict[str, Any]]:
    """Load conversations and attach customer turns plus hardness metadata."""
    data = json.loads(Path(path).read_text())
    convs = list(data.values()) if isinstance(data, dict) else data
    records = []
    for conv in convs:
        messages = conv.get("messages", [])
        customer_turns = [
            msg.get("text", "")
            for msg in messages
            if msg.get("speaker") == speaker and msg.get("text", "").strip()
        ]
        if len(customer_turns) < min_turns:
            continue

        speakers = [msg.get("speaker") for msg in messages]
        escalated = (
            "BOT" in speakers
            and "AGENT" in speakers
            and speakers.index("AGENT")
            > (speakers.index("BOT") if "BOT" in speakers else 10**9)
        )
        records.append(
            {
                "conv": conv,
                "vis": customer_turns,
                "n_vis": len(customer_turns),
                "n_msg": len(messages),
                "escalated": bool(escalated),
            }
        )
    return records


# Rule-trusted style metrics. This gives diversity without letting raw length dominate.
STYLE_SUBSPACE = [
    "Polite%",
    "Formal%",
    "VerbCV",
    "Repeat%",
    "FrntLoad%",
    "InfoQ%",
    "Certn%",
    "Wds/trn",
]
ADVERSARIAL = {
    "Uncert%",
    "Pushbk%",
    "Accuse%",
    "Emot%",
    "Pivot%",
    "ClarfyQ%",
    "IDConf%",
}


def cluster(records: list[dict[str, Any]], k: int, seed: int):
    """Cluster conversations in style space and return features plus hardness proxy."""
    M.set_mode("strict")
    features = feature_matrix([record["vis"] for record in records])
    order = M.METRIC_ORDER
    style_idx = [order.index(metric) for metric in STYLE_SUBSPACE]
    z_style = (features[:, style_idx] - features[:, style_idx].mean(0)) / (
        features[:, style_idx].std(0) + 1e-9
    )
    z_white = z_style
    centroids, labels = kmeans2(z_white, k, seed=seed, minit="++", missing="warn")

    hardness_raw = np.array(
        [[r["escalated"], r["n_msg"], r["n_vis"]] for r in records], float
    )
    hardness = (
        (hardness_raw - hardness_raw.mean(0)) / (hardness_raw.std(0) + 1e-9)
    ).sum(1)
    return features, order, labels, centroids, z_white, hardness


PHRASE = {
    "Wds/trn": (
        "write long, detailed messages",
        "answer in very short, terse messages",
    ),
    "Short%": (
        "often reply with just a few words",
        "rarely send one-word replies; you always elaborate",
    ),
    "Polite%": (
        "stay very polite - frequent please/thanks/sorry",
        "skip pleasantries and stay blunt",
    ),
    "Formal%": ("write in a formal register", None),
    "Ack%": ("often reply with bare acknowledgements (ok, yes, thanks)", None),
    "VerbCV": (
        "vary message length a lot (mix one-liners with long messages)",
        "keep message length very uniform",
    ),
    "Repeat%": ("repeat the same phrasing across turns", None),
    "FrntLoad%": (
        "front-load most details in the first couple of messages",
        "reveal information gradually, a little at a time",
    ),
    "Open.Wds": (
        "open with a long, detailed first message",
        "open with a minimal first message",
    ),
    "Uncert%": (
        'sound unsure/hesitant ("not sure", "I think", "maybe")',
        "speak plainly without hedging",
    ),
    "Certn%": ('speak with strong certainty ("definitely", "for sure")', None),
    "Pushbk%": (
        'push back and contradict the agent ("that\'s not right", "I already told you")',
        None,
    ),
    "ClarfyQ%": ('ask clarifying questions when confused ("what do you mean?")', None),
    "InfoQ%": (
        "ask many information-seeking questions (status, price, how-to)",
        "rarely ask questions; you mostly state things",
    ),
    "Emot%": ('express frustration/emotion ("ugh", "annoyed", "frustrated")', None),
    "Accuse%": ("blame or accuse the agent/company", None),
    "Pivot%": (
        'change your mind or switch topic mid-conversation ("actually...", "never mind")',
        None,
    ),
    "IDConf%": ("sometimes muddle roles / identifiers", None),
    "IDs/trn": ("pack many account ids/numbers into messages", "give few identifiers"),
}
NAME = {
    ("Short%", "+"): "Terse",
    ("Wds/trn", "+"): "Verbose",
    ("Wds/trn", "-"): "Terse",
    ("Polite%", "+"): "Polite",
    ("Polite%", "-"): "Blunt",
    ("Uncert%", "+"): "Hesitant",
    ("Certn%", "+"): "Confident",
    ("Pushbk%", "+"): "Pushy",
    ("Accuse%", "+"): "Combative",
    ("Emot%", "+"): "Frustrated",
    ("InfoQ%", "+"): "Inquisitive",
    ("ClarfyQ%", "+"): "Confused",
    ("InfoQ%", "-"): "Declarative",
    ("Pivot%", "+"): "Wavering",
    ("FrntLoad%", "+"): "Front-loading",
    ("FrntLoad%", "-"): "Drip-feeding",
    ("VerbCV", "+"): "Erratic",
    ("Formal%", "+"): "Formal",
    ("Open.Wds", "-"): "Abrupt",
}


def rule_persona(cz: np.ndarray, craw: np.ndarray, order: list[str]):
    """Render a deterministic persona from cluster z-scores."""
    notable = sorted(range(len(order)), key=lambda i: -abs(cz[i]))
    bullets, name_bits = [], []
    for i in notable:
        if abs(cz[i]) < 0.4 and len(bullets) >= 3:
            break
        direction = "+" if cz[i] > 0 else "-"
        phrase = PHRASE.get(order[i], (None, None))[0 if direction == "+" else 1]
        if phrase and phrase not in bullets:
            bullets.append(phrase)
        name = NAME.get((order[i], direction))
        if name and name not in name_bits:
            name_bits.append(name)

    bullets = [f"You {bullet}." for bullet in bullets[:6]] or [
        "You behave close to the corpus-average customer."
    ]
    name = " & ".join(name_bits[:2]) or "Archetype"
    adv_score = float(np.mean([cz[order.index(metric)] for metric in ADVERSARIAL]))
    effect = "HARMS the task" if adv_score > 0 else "HELPS the task"
    intro = f"You are a real customer contacting support. Your messaging style is {name.lower()}."
    tone = (
        "Brief and to the point"
        if (craw[order.index("Short%")] > 40 or craw[order.index("Wds/trn")] < 9)
        else "Conversational"
    )
    return name, effect, intro, bullets, tone


LLM_SYSTEM_PROMPT = (
    "You are a UX researcher building USER-SIMULATOR personas for an adversarial "
    "benchmark. You are given several lightly redacted real customer-support chat excerpts "
    "that all belong to ONE behavioral cluster. Infer the customer's underlying "
    "DISPOSITION - their goal stance, cooperativeness, emotional state, knowledge "
    "level, trust in the agent, and durable communication habits - NOT the specific "
    "topic or wording. Then write a reusable persona a simulator can act out on "
    "DIFFERENT tasks.\n\n"
    "Return STRICT JSON with keys:\n"
    '  "name": 2-3 word Title Case archetype name (e.g. "Concise & Trusting")\n'
    '  "effect": "HELPS the task" or "HARMS the task" - does this disposition make a '
    "support agent's job easier or harder?\n"
    '  "intro": ONE in-character sentence describing the customer. NEVER mention '
    'datasets, clusters, "the scenario", benchmarks, or AI.\n'
    '  "behaviors": 4-6 strings, each a full sentence starting with "You ", describing '
    "disposition-driven behavior.\n"
    '  "tone": a short phrase (e.g. "Brief, friendly, low-friction").\n\n'
    "Match the spirit of this reference persona:\n"
    "# Persona: Concise & Trusting  (effect: HELPS the task)\n"
    "You are an efficient customer who trusts the agent's expertise and keeps things moving.\n"
    "## How you behave\n- Answer questions in one or two short sentences - but always answer fully.\n"
    "- You assume the agent knows their job; you do what they suggest.\n"
    "## Tone\n- Brief, friendly, low-friction."
)


def build_user_prompt(
    craw: np.ndarray, order: list[str], exemplars: list[dict[str, Any]]
) -> str:
    profile = ", ".join(
        f"{metric}={craw[order.index(metric)]:.1f}"
        for metric in (
            "Wds/trn",
            "Short%",
            "Polite%",
            "Uncert%",
            "InfoQ%",
            "Pivot%",
            "Emot%",
        )
    )
    parts = [f"Cluster behavioral profile (strict-mode metrics): {profile}", ""]
    for j, exemplar in enumerate(exemplars, 1):
        body = "\n".join(
            f"  CUSTOMER: {mask(turn)[:200]}" for turn in exemplar["vis"][:25]
        )
        tag = "escalated-to-human, " if exemplar["escalated"] else ""
        parts.append(
            f"--- Chat {j} ({tag}{exemplar['n_vis']} customer turns) ---\n{body}"
        )
    parts.append("\nReturn ONLY the JSON object.")
    return "\n".join(parts)


def _as_you(text: str) -> str:
    text = _BANNED_META.sub("", str(text).strip())
    text = " ".join(text.split())
    if not text:
        return ""
    if not text.startswith("You "):
        text = "You " + text[0].lower() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def llm_persona(
    craw: np.ndarray,
    order: list[str],
    exemplars: list[dict[str, Any]],
    model: str,
    effort: str,
):
    from openai import OpenAI

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    client = OpenAI()
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(craw, order, exemplars)},
        ],
        "response_format": {"type": "json_object"},
    }
    if model.startswith("gpt-5") or model.startswith("o"):
        body["reasoning_effort"] = effort
    else:
        body["temperature"] = 1.0

    resp = client.chat.completions.create(**body)
    obj = json.loads(resp.choices[0].message.content)
    name = str(obj.get("name", "Archetype")).strip()
    effect_raw = str(obj.get("effect", "HARMS the task"))
    effect = "HELPS the task" if "HELP" in effect_raw.upper() else "HARMS the task"
    intro = str(obj.get("intro", "")).strip()
    bullets = [_as_you(item) for item in obj.get("behaviors", []) if str(item).strip()]
    bullets = [bullet for bullet in bullets if bullet][:6]
    tone = str(obj.get("tone", "Conversational")).strip().rstrip(".")
    return name, effect, intro, bullets, tone


def render_persona(
    name: str,
    effect: str,
    intro: str,
    bullets: list[str],
    tone: str,
    examples: list[str],
    craw: np.ndarray,
    order: list[str],
    n: int,
    total: int,
    method: str,
) -> str:
    profile = ", ".join(
        f"{metric}={craw[order.index(metric)]:.1f}"
        for metric in ("Wds/trn", "Short%", "Polite%", "Uncert%", "InfoQ%", "Pivot%")
    )
    lines = [
        f"# Persona: {name}  (effect: {effect})",
        "",
        intro,
        "",
        "## How you behave",
    ]
    lines += [
        bullet if bullet.startswith("- ") else f"- {bullet}" for bullet in bullets
    ]
    lines += [
        "",
        "## Tone",
        f"- {tone}; write the way a real customer texts support - lowercase, fragments, and typos are fine.",
        "",
        "## Invariants (never break these)",
        '- Stay in character as a real customer; never mention "the scenario" or that you are an AI.',
        "- Only state facts grounded in your scenario or in the actual results of tools you ran. Never invent tool results.",
        "",
    ]
    example_block = (
        "\n".join(f'     - "{example}"' for example in examples)
        if examples
        else "     (no long-form example turns in the representative chat)"
    )
    lines += [
        "<!--",
        f"provenance: {method}-method seed from a {n}/{total} ({n / total:.0%}) behavioral "
        "cluster of real customer-support chats (style-subspace KMeans).",
        f"cluster metric profile: {profile}",
        "example turns (lightly redacted, STYLE REFERENCE ONLY - do not reuse content):",
        example_block,
        "-->",
        "",
    ]
    return "\n".join(lines)


def induce_seed_personas(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_corpus(args.data, args.speaker, args.min_turns)
    if not records:
        raise ValueError(
            "No conversations survived filtering; lower --min-turns or check --speaker."
        )

    features, order, labels, centroids, z_white, hardness = cluster(
        records, args.k, args.seed
    )
    print(f"corpus: {len(records)} conversations | method={args.method} | k={args.k}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for cluster_id in range(args.k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        cz = ((features[members] - features.mean(0)) / (features.std(0) + 1e-9)).mean(0)
        craw = features[members].mean(0)

        dist = ((z_white[members] - centroids[cluster_id]) ** 2).sum(1)
        nearest = members[int(dist.argmin())]
        hardest = members[np.argsort(-hardness[members])][: args.exemplars]
        chosen, seen = [], set()
        for idx in [nearest, *hardest]:
            if idx not in seen:
                seen.add(idx)
                chosen.append(records[idx])
            if len(chosen) >= args.exemplars:
                break

        examples = [mask(turn) for turn in records[nearest]["vis"]]
        examples = [example for example in examples if len(example.split()) >= 4][
            : args.examples
        ]

        if args.method == "rule":
            name, effect, intro, bullets, tone = rule_persona(cz, craw, order)
        elif args.dry_run:
            print(f"\n===== cluster {cluster_id} (n={len(members)}) PROMPT =====")
            print(build_user_prompt(craw, order, chosen)[:2500])
            continue
        else:
            name, effect, intro, bullets, tone = llm_persona(
                craw, order, chosen, args.model, args.effort
            )

        markdown = render_persona(
            name,
            effect,
            intro,
            bullets,
            tone,
            examples,
            craw,
            order,
            len(members),
            len(records),
            args.method,
        )
        slug = (
            re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            or f"archetype_{cluster_id}"
        )
        persona_dir = out_dir / f"cluster_{cluster_id:02d}_{slug}"
        persona_dir.mkdir(parents=True, exist_ok=True)
        persona_path = persona_dir / "gen_00.md"
        persona_path.write_text(markdown)

        index.append(
            {
                "cluster": cluster_id,
                "name": name,
                "effect": effect,
                "n": int(len(members)),
                "frac": round(len(members) / len(records), 3),
                "seed_file": _repo_relative(persona_path),
            }
        )
        print(
            f"  [{cluster_id}] {name:30} n={len(members):4d} ({len(members) / len(records):.0%})  {effect}"
        )

    if not args.dry_run:
        (out_dir / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False)
        )
        print(f"\nwrote {len(index)} seed personas + index.json to {out_dir}")
    return index


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="real conversation corpus JSON")
    ap.add_argument("--speaker", default="VISITOR")
    ap.add_argument("--k", type=int, default=5, help="number of induced archetypes")
    ap.add_argument("--method", choices=["llm", "rule"], default="llm")
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument(
        "--exemplars", type=int, default=4, help="chats shown to the LLM per cluster"
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=4,
        help="example turns saved in the persona comment",
    )
    ap.add_argument("--model", default=os.environ.get("SIM2REAL_LLM_MODEL", "gpt-5.2"))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM method: print prompts and do not call the API",
    )
    return ap.parse_args()


def main() -> None:
    induce_seed_personas(parse_args())


if __name__ == "__main__":
    main()
