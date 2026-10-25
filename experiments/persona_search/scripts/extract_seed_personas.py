#!/usr/bin/env python3
"""Extract data-grounded SEED personas from a real customer-chat corpus.

Two methods (``--method``):

  llm  (default) -- DISPOSITION personas. Cluster real customers on a *style
        subspace* (for diversity), pick the most representative + HARDEST
        exemplar chats per cluster (hardness proxy = escalated-to-human +
        long + many customer turns), then ask an LLM to infer the latent
        customer DISPOSITION from the (PII-masked) transcripts and write a
        persona in the exact format of data/persona_opt/lineages/p000x/gen_00.md.
        Needs OPENAI_API_KEY (and OPENAI_BASE_URL for a proxy). Reuses the
        OpenAI() client pattern from sim2real_llm_metrics.py.

  rule  -- no-API fallback. Names + behavior bullets are templated from the
        cluster's metric profile via a fixed lookup. Cheap but generic; the
        text captures surface style, not disposition.

Both emit drop-in ``--seed-persona`` inputs for parallel_gepa.py plus index.json.
Input corpus: JSON list of {messages:[{speaker,text}]} with customer speaker
"VISITOR" (e.g. virtual_agent/llm_clean_3k.json -- Brinks Home).

    .venv/bin/python experiments/persona_search/scripts/extract_seed_personas.py \
        --data .../llm_clean_3k.json --k 5 --method llm \
        --out-dir data/persona_opt/seeds_brinks
    # verify selection/prompt without calling the API:
    ... --method llm --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2, whiten

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import feature_matrix  # noqa: E402

# --- PII masking -------------------------------------------------------------
_RE_DIGITS = re.compile(r"\d[\d\-\s]{2,}\d")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def mask(t: str) -> str:
    t = _RE_EMAIL.sub("<email>", t)
    t = _RE_DIGITS.sub("<redacted>", t)
    return " ".join(t.split())


# --- corpus -> conversations with features + hardness metadata ---------------
def load_corpus(path: str, speaker: str, min_turns: int):
    d = json.load(open(path))
    convs = list(d.values()) if isinstance(d, dict) else d
    out = []
    for c in convs:
        msgs = c.get("messages", [])
        vis = [
            m.get("text", "")
            for m in msgs
            if m.get("speaker") == speaker and m.get("text", "").strip()
        ]
        if len(vis) < min_turns:
            continue
        spk = [m.get("speaker") for m in msgs]
        # escalated-to-human proxy: a BOT handled it, then a human AGENT took over
        escalated = (
            "BOT" in spk
            and "AGENT" in spk
            and spk.index("AGENT") > (spk.index("BOT") if "BOT" in spk else 1e9)
        )
        out.append(
            {
                "conv": c,
                "vis": vis,
                "n_vis": len(vis),
                "n_msg": len(msgs),
                "escalated": bool(escalated),
            }
        )
    return out


# clustering subspace: rule-TRUSTED style metrics (drop the noisy LLM-owned
# adversarial metrics, which are unreliable in strict/rule mode, and the
# redundant pure-length columns). Gives diversity without length-domination.
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


def cluster(records, k, seed):
    M.set_mode("strict")
    X = feature_matrix([r["vis"] for r in records])  # (n,19) raw metrics
    order = M.METRIC_ORDER
    idx = [order.index(m) for m in STYLE_SUBSPACE]
    Z = (X[:, idx] - X[:, idx].mean(0)) / (X[:, idx].std(0) + 1e-9)
    Zw = whiten(Z) if len(Z) > k else Z
    centroids, labels = kmeans2(Zw, k, seed=seed, minit="++", missing="warn")
    # hardness proxy (standardized): escalation + length + #customer turns
    H = np.array([[r["escalated"], r["n_msg"], r["n_vis"]] for r in records], float)
    Hz = ((H - H.mean(0)) / (H.std(0) + 1e-9)).sum(1)
    return X, order, labels, centroids, Zw, Hz


# =========================== RULE method (no API) ============================
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
        "stay very polite — frequent please/thanks/sorry",
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


def rule_persona(cz, craw, order):
    notable = sorted(range(len(order)), key=lambda i: -abs(cz[i]))
    bullets, name_bits = [], []
    for i in notable:
        if abs(cz[i]) < 0.4 and len(bullets) >= 3:
            break
        d = "+" if cz[i] > 0 else "-"
        ph = PHRASE.get(order[i], (None, None))[0 if d == "+" else 1]
        if ph and ph not in bullets:
            bullets.append(ph)
        nm = NAME.get((order[i], d))
        if nm and nm not in name_bits:
            name_bits.append(nm)
    bullets = [f"You {b}." for b in bullets[:6]] or [
        "You behave close to the corpus-average customer."
    ]
    name = " & ".join(name_bits[:2]) or "Archetype"
    adv = float(np.mean([cz[order.index(m)] for m in ADVERSARIAL]))
    effect = "HARMS the task" if adv > 0 else "HELPS the task"
    intro = f"You are a real customer contacting support. Your messaging style is {name.lower()}."
    tone = (
        "Brief and to the point"
        if (craw[order.index("Short%")] > 40 or craw[order.index("Wds/trn")] < 9)
        else "Conversational"
    )
    return name, effect, intro, bullets, tone


# =========================== LLM method (disposition) ========================
SYS = (
    "You are a UX researcher building USER-SIMULATOR personas for an adversarial "
    "benchmark. You are given several anonymized real customer-support chat excerpts "
    "that all belong to ONE behavioral cluster. Infer the customer's underlying "
    "DISPOSITION — their goal stance, cooperativeness, emotional state, knowledge "
    "level, trust in the agent, and durable communication habits — NOT the specific "
    "topic or wording. Then write a reusable persona a simulator can act out on "
    "DIFFERENT tasks.\n\n"
    "Return STRICT JSON with keys:\n"
    '  "name": 2-3 word Title Case archetype name (e.g. "Concise & Trusting")\n'
    '  "effect": "HELPS the task" or "HARMS the task" — does this disposition make a '
    "support agent's job easier or harder?\n"
    '  "intro": ONE in-character sentence describing the customer. NEVER mention '
    'datasets, clusters, "the scenario", benchmarks, or AI.\n'
    '  "behaviors": 4-6 strings, each a full sentence starting with "You ", describing '
    "disposition-driven behavior (how they answer, question, react to friction, reveal "
    "info, handle uncertainty). Concrete and durable, not topic-specific.\n"
    '  "tone": a short phrase (e.g. "Brief, friendly, low-friction").\n\n'
    "Match the spirit of this reference persona:\n"
    "# Persona: Concise & Trusting  (effect: HELPS the task)\n"
    "You are an efficient customer who trusts the agent's expertise and keeps things moving.\n"
    "## How you behave\n- Answer questions in one or two short sentences — but always answer fully.\n"
    "- You assume the agent knows their job; you do what they suggest.\n"
    "## Tone\n- Brief, friendly, low-friction."
)


def build_user_prompt(craw, order, exemplars):
    prof = ", ".join(
        f"{order[order.index(k)]}={craw[order.index(k)]:.1f}"
        for k in (
            "Wds/trn",
            "Short%",
            "Polite%",
            "Uncert%",
            "InfoQ%",
            "Pivot%",
            "Emot%",
        )
    )
    parts = [f"Cluster behavioral profile (strict-mode metrics): {prof}", ""]
    for j, ex in enumerate(exemplars, 1):
        turns = ex["vis"][:25]
        body = "\n".join(f"  CUSTOMER: {mask(t)[:200]}" for t in turns)
        tag = "escalated-to-human, " if ex["escalated"] else ""
        parts.append(f"--- Chat {j} ({tag}{ex['n_vis']} customer turns) ---\n{body}")
    parts.append("\nReturn ONLY the JSON object.")
    return "\n".join(parts)


def llm_persona(craw, order, exemplars, model, effort):
    from openai import OpenAI

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    client = OpenAI()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
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
    effect = obj.get("effect", "HARMS the task")
    effect = "HELPS the task" if "HELP" in effect.upper() else "HARMS the task"
    intro = str(obj.get("intro", "")).strip()
    bullets = [_as_you(b) for b in obj.get("behaviors", []) if str(b).strip()][:6]
    tone = str(obj.get("tone", "Conversational")).strip().rstrip(".")
    return name, effect, intro, bullets, tone


_BANNED = re.compile(
    r"\b(dataset|cluster|scenario|benchmark|simulat|persona|AI|model)\b", re.I
)


def _as_you(b: str) -> str:
    b = str(b).strip()
    b = _BANNED.sub("", b)  # scrub any leaked meta words
    b = " ".join(b.split())
    if not b.startswith("You "):
        b = "You " + b[0].lower() + b[1:]
    if not b.endswith("."):
        b += "."
    return b


# =============================== render + main ===============================
def render(
    name,
    effect,
    intro,
    bullets,
    tone,
    examples,
    craw,
    order,
    n,
    total,
    method,
    corpus_name="Brinks Home",
) -> str:
    prof = ", ".join(
        f"{order[order.index(k)]}={craw[order.index(k)]:.1f}"
        for k in ("Wds/trn", "Short%", "Polite%", "Uncert%", "InfoQ%", "Pivot%")
    )
    L = [f"# Persona: {name}  (effect: {effect})", "", intro, "", "## How you behave"]
    L += [b if b.startswith("- ") else f"- {b}" for b in bullets]
    L += [
        "",
        "## Tone",
        f"- {tone}; write the way a real customer texts support — lowercase, fragments, and typos are fine.",
        "",
        "## Invariants (never break these)",
        '- Stay in character as a real customer; never mention "the scenario" or that you are an AI.',
        "- Only state facts grounded in your scenario or in the actual results of tools you ran. "
        "Never invent tool results.",
        "",
    ]
    ex_block = (
        "\n".join(f'     - "{e}"' for e in examples)
        if examples
        else "     (no long-form example turns in the representative chat)"
    )
    L += [
        "<!--",
        f"provenance: {method}-method seed from a {n}/{total} ({n / total:.0%}) behavioral "
        f"cluster of real {corpus_name} support chats (style-subspace KMeans).",
        f"cluster metric profile: {prof}",
        "example turns (PII-masked, STYLE REFERENCE ONLY — do not reuse content):",
        ex_block,
        "-->",
        "",
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--speaker", default="VISITOR")
    ap.add_argument("--k", type=int, default=5, help="number of archetypes")
    ap.add_argument("--method", choices=["llm", "rule"], default="llm")
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument(
        "--exemplars", type=int, default=4, help="chats shown to the LLM per cluster"
    )
    ap.add_argument(
        "--examples", type=int, default=4, help="example turns saved in the comment"
    )
    ap.add_argument("--model", default=os.environ.get("SIM2REAL_LLM_MODEL", "gpt-5.2"))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "data/persona_opt/seeds_brinks"))
    ap.add_argument(
        "--corpus-name", default="Brinks Home", help="name used in provenance comment"
    )
    ap.add_argument(
        "--prefix", default="brinks", help="directory prefix for persona folders"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM method: print prompts, don't call API",
    )
    a = ap.parse_args()

    recs = load_corpus(a.data, a.speaker, a.min_turns)
    X, order, labels, centroids, Zw, Hz = cluster(recs, a.k, a.seed)
    print(f"corpus: {len(recs)} conversations | method={a.method} | k={a.k}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for c in range(a.k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        cz = ((X[members] - X.mean(0)) / (X.std(0) + 1e-9)).mean(0)
        craw = X[members].mean(0)
        # exemplars = nearest-to-centroid + hardest, for the LLM
        dist = ((Zw[members] - centroids[c]) ** 2).sum(1)
        nearest = members[int(dist.argmin())]
        hardest = members[np.argsort(-Hz[members])][: a.exemplars]
        chosen, seen = [], set()
        for i in [nearest, *hardest]:
            if i not in seen:
                seen.add(i)
                chosen.append(recs[i])
            if len(chosen) >= a.exemplars:
                break
        # comment examples (contentful, masked) from the nearest-centroid chat
        ex = [mask(t) for t in recs[nearest]["vis"]]
        ex = [e for e in ex if len(e.split()) >= 4][: a.examples]

        if a.method == "rule":
            name, effect, intro, bullets, tone = rule_persona(cz, craw, order)
        elif a.dry_run:
            print(f"\n===== cluster {c} (n={len(members)}) PROMPT =====")
            print(build_user_prompt(craw, order, chosen)[:2500])
            continue
        else:
            name, effect, intro, bullets, tone = llm_persona(
                craw, order, chosen, a.model, a.effort
            )

        md = render(
            name,
            effect,
            intro,
            bullets,
            tone,
            ex,
            craw,
            order,
            len(members),
            len(recs),
            a.method,
            corpus_name=a.corpus_name,
        )
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or f"archetype_{c}"
        pdir = out_dir / f"{a.prefix}_{c:02d}_{slug}"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "gen_00.md").write_text(md)
        index.append(
            {
                "cluster": c,
                "name": name,
                "effect": effect,
                "n": int(len(members)),
                "frac": round(len(members) / len(recs), 3),
                "seed_file": str(pdir / "gen_00.md"),
            }
        )
        print(
            f"  [{c}] {name:30} n={len(members):4d} ({len(members) / len(recs):.0%})  {effect}"
        )

    if not a.dry_run:
        (out_dir / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False)
        )
        print(f"\nwrote {len(index)} seed personas + index.json to {out_dir}")


if __name__ == "__main__":
    main()
