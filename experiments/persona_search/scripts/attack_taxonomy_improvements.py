#!/usr/bin/env python3
"""Three improvements to the attack taxonomy:
1. Per-code κ (from existing annotations, zero cost)
2. Causal link annotation (attack → mechanism: direct/enabling/coincidental)
3. Step-level consistency (dual-judge decisive_msg_index comparison)

Usage:
  .venv/bin/python experiments/persona_search/scripts/attack_taxonomy_improvements.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)

ANNOT_DIR = REPO / "data/persona_opt/rootcause_flips/attack_annotations"
FLIP_DIR = REPO / "data/persona_opt/rootcause_flips"
REPORT_PATH = FLIP_DIR / "ATTACK_TAXONOMY_REPORT.md"


def strip_persona(tid: str) -> str:
    return re.sub(r"\[PERSONA:[^\]]*\]", "", tid).strip()


# ============================================================
# 1. Per-code κ
# ============================================================


def load_paired_annotations() -> list[dict]:
    by_key = defaultdict(dict)
    for jf in ANNOT_DIR.glob("*__gpt54.json"):
        d = json.loads(jf.read_text())
        k = jf.name.rsplit("__gpt54", 1)[0]
        by_key[k]["gpt54"] = d
    for jf in ANNOT_DIR.glob("*__glm52.json"):
        d = json.loads(jf.read_text())
        k = jf.name.rsplit("__glm52", 1)[0]
        by_key[k]["glm52"] = d

    items = []
    for k, judges in by_key.items():
        if "gpt54" in judges and "glm52" in judges:
            items.append(
                {
                    "key": k,
                    "domain": judges["gpt54"].get("_domain", ""),
                    "island": judges["gpt54"].get("_island", ""),
                    "task_id": judges["gpt54"].get("_task_id", ""),
                    "attribution": judges["gpt54"].get("_attribution", ""),
                    "gpt54_primary": judges["gpt54"].get("primary", "?"),
                    "glm52_primary": judges["glm52"].get("primary", "?"),
                    "agree": judges["gpt54"].get("primary", "")
                    == judges["glm52"].get("primary", ""),
                }
            )
    return items


def per_code_kappa(items: list[dict]) -> dict[str, dict]:
    """Compute per-code binary κ (code-vs-rest for each code)."""
    all_codes = sorted(
        set(
            [it["gpt54_primary"] for it in items]
            + [it["glm52_primary"] for it in items]
        )
    )
    results = {}
    for code in all_codes:
        a = [1 if it["gpt54_primary"] == code else 0 for it in items]
        b = [1 if it["glm52_primary"] == code else 0 for it in items]
        n = len(items)
        agree = sum(1 for x, y in zip(a, b) if x == y)
        po = agree / n

        pa1 = sum(a) / n
        pa0 = 1 - pa1
        pb1 = sum(b) / n
        pb0 = 1 - pb1
        pe = pa1 * pb1 + pa0 * pb0

        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

        tp = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        fp = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        fn = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        results[code] = {
            "kappa": round(kappa, 3),
            "agree_rate": round(po, 3),
            "n_gpt54": sum(a),
            "n_glm52": sum(b),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }
    return results


# ============================================================
# 2+3. Causal link + decisive step (dual judge)
# ============================================================


def load_v4_for_key(key: str) -> str:
    """Get v4 agent-side code for a given annotation key."""
    # Try existing v4 annotation
    v4_path = ANNOT_DIR / f"{key}__v4_agent.json"
    if v4_path.exists():
        return json.loads(v4_path.read_text()).get("primary", "UNKNOWN")

    # Try loading from existing gt_full
    gt_path = REPO / "data/persona_opt/gt_full_gpt54.jsonl"
    if not gt_path.exists():
        return "UNKNOWN"

    # Parse key to extract domain/island/task
    parts = key.split("__")
    if len(parts) < 3:
        return "UNKNOWN"
    domain = parts[0]
    island = parts[1]

    for line in open(gt_path):
        d = json.loads(line)
        if (
            d.get("domain") == domain
            and d.get("persona") == island
            and d.get("cond") == "opt"
        ):
            task_str = strip_persona(str(d.get("task", "")))
            if task_str in key:
                return d.get("primary", "UNKNOWN")
    return "UNKNOWN"


CAUSAL_SYS = """\
You are analyzing the causal relationship between a user's attack strategy and an agent's failure mechanism in a customer service conversation.

Given:
- ATTACK STRATEGY: what the adversarial user did differently from a baseline user
- AGENT MECHANISM: how the agent failed
- ANALYSIS: full context of what happened

Determine:

1. CAUSAL_LINK: the relationship between attack and mechanism
   - "direct": the attack DIRECTLY triggered the agent mechanism (e.g., user demanded transfer → agent transferred prematurely)
   - "enabling": the attack created conditions that made the mechanism more likely, but the agent could have avoided it (e.g., user withheld context → agent COULD have asked for it but instead guessed)
   - "coincidental": the attack and mechanism are largely independent (the agent would likely have failed the same way regardless)

2. DECISIVE_USER_MSG: the index (0-based) of the single most decisive USER message that caused or enabled the failure. Must be a user turn, not an assistant turn.

3. CAUSAL_CHAIN: one sentence describing the causal path: "User did X → this caused/enabled agent to Y → resulting in Z"

Output JSON:
{
  "causal_link": "<direct|enabling|coincidental>",
  "decisive_user_msg": <int>,
  "causal_chain": "<one sentence>"
}

Output ONLY JSON."""


def build_causal_prompt(item: dict, attack_code: str, v4_code: str) -> str:
    # Load original flip annotation for analysis text
    flip_data = None
    for jf in FLIP_DIR.rglob("*.json"):
        if jf.name in ("cluster_summary.json", "attack_codebook_v1.json"):
            continue
        if jf.parent.name == "attack_annotations":
            continue
        d = json.loads(jf.read_text())
        if d.get("domain") == item["domain"] and d.get("island") == item["island"]:
            tid = strip_persona(str(d.get("task_id", "")))
            if tid == item["task_id"]:
                flip_data = d
                break

    analysis = flip_data.get("analysis", "") if flip_data else ""
    opt_strategy = flip_data.get("opt_strategy", "") if flip_data else ""
    seed_strategy = flip_data.get("seed_strategy", "") if flip_data else ""
    decisive = flip_data.get("decisive_step", {}) if flip_data else {}

    return (
        f"DOMAIN: {item['domain']}\n"
        f"TASK: {item['task_id']}\n"
        f"ATTRIBUTION: {item['attribution']}\n\n"
        f"ATTACK STRATEGY: {attack_code}\n"
        f"AGENT MECHANISM: {v4_code}\n\n"
        f"OPT user strategy: {opt_strategy}\n"
        f"SEED user strategy: {seed_strategy}\n\n"
        f"Original decisive step: msg[{decisive.get('msg_index', '?')}] ({decisive.get('role', '?')}): "
        f"{decisive.get('quote', '')[:200]}\n"
        f"Explanation: {decisive.get('explanation', '')[:200]}\n\n"
        f"Full analysis: {analysis}\n\n"
        f"Based on the above, determine the causal link, decisive user message index, and causal chain."
    )


def call_llm(model: str, sys_prompt: str, user_msg: str, max_tokens: int = 800) -> dict:
    import litellm

    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"parse_error": raw[:300]}


def run_causal_annotation(items: list[dict], concurrency: int = 5):
    """Dual-judge causal link + decisive step annotation for agreed items."""
    agreed = [it for it in items if it["agree"] and it["gpt54_primary"] != "NONE"]
    models = [
        ("gpt-5.4", "gpt54"),
        ("anthropic/accounts/fireworks/models/glm-5p2", "glm52"),
    ]

    # Build v4 map
    v4_map = {}
    gt_path = REPO / "data/persona_opt/gt_full_gpt54.jsonl"
    if gt_path.exists():
        existing_v4 = {}
        for line in open(gt_path):
            d = json.loads(line)
            stripped = strip_persona(str(d.get("task", "")))
            key = (d["domain"], d.get("persona", ""), "opt", stripped)
            existing_v4[key] = d.get("primary", "UNKNOWN")

    for it in agreed:
        # Try existing v4
        key = (it["domain"], it["island"], "opt", it["task_id"])
        if key in existing_v4:
            v4_map[it["key"]] = existing_v4[key]
        else:
            v4_path = ANNOT_DIR / f"{it['key']}__v4_agent.json"
            if v4_path.exists():
                v4_map[it["key"]] = json.loads(v4_path.read_text()).get(
                    "primary", "UNKNOWN"
                )

    todo = []
    for it in agreed:
        v4 = v4_map.get(it["key"], "UNKNOWN")
        if v4 == "UNKNOWN":
            continue
        for model, label in models:
            out_path = ANNOT_DIR / f"{it['key']}__causal_{label}.json"
            if not out_path.exists():
                todo.append((it, model, label, out_path, it["gpt54_primary"], v4))

    print(
        f"Causal annotation: {len(todo)} calls ({len(agreed)} agreed non-NONE items × 2 judges)"
    )
    if not todo:
        print("  All done.")
        return

    done = 0

    def _do(it, model, label, out_path, attack, v4):
        user_msg = build_causal_prompt(it, attack, v4)
        result = call_llm(model, CAUSAL_SYS, user_msg)
        result["_key"] = it["key"]
        result["_domain"] = it["domain"]
        result["_attack"] = attack
        result["_v4"] = v4
        result["_judge"] = label
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_do, *args): args for args in todo}
        for fut in as_completed(futs):
            done += 1
            it, model, label, out_path, attack, v4 = futs[fut]
            try:
                r = fut.result()
                cl = r.get("causal_link", "?")
                dm = r.get("decisive_user_msg", "?")
                print(
                    f"  [{done}/{len(todo)}] {label}: {it['domain']}/{it['task_id'][:25]} → {cl} msg={dm}"
                )
            except Exception as e:
                print(f"  [{done}/{len(todo)}] ERROR: {e}")


# ============================================================
# Analysis & report update
# ============================================================


def analyze_causal(items: list[dict]) -> dict:
    """Analyze causal annotations."""
    agreed = [it for it in items if it["agree"] and it["gpt54_primary"] != "NONE"]

    causal_pairs = []
    step_diffs = []

    for it in agreed:
        gpt_path = ANNOT_DIR / f"{it['key']}__causal_gpt54.json"
        glm_path = ANNOT_DIR / f"{it['key']}__causal_glm52.json"
        if not gpt_path.exists() or not glm_path.exists():
            continue
        gpt = json.loads(gpt_path.read_text())
        glm = json.loads(glm_path.read_text())

        causal_pairs.append(
            {
                "key": it["key"],
                "domain": it["domain"],
                "attack": it["gpt54_primary"],
                "gpt54_link": gpt.get("causal_link", "?"),
                "glm52_link": glm.get("causal_link", "?"),
                "agree_link": gpt.get("causal_link") == glm.get("causal_link"),
                "gpt54_msg": gpt.get("decisive_user_msg"),
                "glm52_msg": glm.get("decisive_user_msg"),
                "gpt54_chain": gpt.get("causal_chain", ""),
                "glm52_chain": glm.get("causal_chain", ""),
            }
        )

        gm = gpt.get("decisive_user_msg")
        lm = glm.get("decisive_user_msg")
        if isinstance(gm, (int, float)) and isinstance(lm, (int, float)):
            step_diffs.append(abs(gm - lm))

    # Causal link IAA
    link_labels = [(p["gpt54_link"], p["glm52_link"]) for p in causal_pairs]
    link_agree = sum(1 for a, b in link_labels if a == b)
    link_n = len(link_labels)
    link_agree_rate = link_agree / link_n if link_n else 0

    # Cohen's κ for causal link
    all_link_labels = sorted(
        set([a for a, _ in link_labels] + [b for _, b in link_labels])
    )
    if link_n > 0 and len(all_link_labels) > 1:
        la = [a for a, _ in link_labels]
        lb = [b for _, b in link_labels]
        k = len(all_link_labels)
        l2i = {l: i for i, l in enumerate(all_link_labels)}
        ca = np.zeros(k)
        cb = np.zeros(k)
        for a, b in zip(la, lb):
            ca[l2i[a]] += 1
            cb[l2i[b]] += 1
        po = link_agree / link_n
        pe = float(np.sum((ca / link_n) * (cb / link_n)))
        link_kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    else:
        link_kappa = 0

    # Causal link distribution (agreed only)
    agreed_links = [p for p in causal_pairs if p["agree_link"]]
    link_dist = Counter(p["gpt54_link"] for p in agreed_links)

    # Per-attack causal link
    attack_causal = defaultdict(lambda: Counter())
    for p in agreed_links:
        attack_causal[p["attack"]][p["gpt54_link"]] += 1

    # Step-level consistency
    step_stats = {
        "n": len(step_diffs),
        "exact_match": sum(1 for d in step_diffs if d == 0),
        "within_2": sum(1 for d in step_diffs if d <= 2),
        "within_5": sum(1 for d in step_diffs if d <= 5),
        "mean_diff": round(np.mean(step_diffs), 1) if step_diffs else 0,
        "median_diff": round(float(np.median(step_diffs)), 1) if step_diffs else 0,
        "max_diff": max(step_diffs) if step_diffs else 0,
    }

    return {
        "causal_pairs": causal_pairs,
        "link_kappa": round(link_kappa, 3),
        "link_agree_rate": round(link_agree_rate, 3),
        "link_n": link_n,
        "link_agree_count": link_agree,
        "link_dist": dict(link_dist),
        "attack_causal": {k: dict(v) for k, v in attack_causal.items()},
        "step_stats": step_stats,
        "step_diffs": step_diffs,
    }


def append_to_report(per_code: dict[str, dict], causal: dict):
    """Append improvement sections to the report."""
    report = REPORT_PATH.read_text()

    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Appendix: Taxonomy Improvements")
    lines.append("")

    # A. Per-code κ
    lines.append("## A. Per-Code Agreement")
    lines.append("")
    lines.append("Binary κ (code-vs-rest) for each attack code:")
    lines.append("")
    lines.append("| Code | n(GPT) | n(GLM) | TP | κ | F1 | Agree |")
    lines.append("|------|--------|--------|----|----|----|----|")
    for code in sorted(per_code, key=lambda c: -per_code[c]["f1"]):
        s = per_code[code]
        lines.append(
            f"| `{code[:35]}` | {s['n_gpt54']} | {s['n_glm52']} | {s['tp']} "
            f"| {s['kappa']:.3f} | {s['f1']:.3f} | {s['agree_rate']:.3f} |"
        )
    lines.append("")

    # Interpretation
    high_k = [c for c, s in per_code.items() if s["kappa"] >= 0.8 and s["n_gpt54"] >= 2]
    low_k = [c for c, s in per_code.items() if s["kappa"] < 0.6 and s["n_gpt54"] >= 2]
    lines.append(
        f"**Highly reliable codes** (κ ≥ 0.8): {', '.join(f'`{c[:20]}`' for c in high_k) or 'none'}"
    )
    lines.append("")
    lines.append(
        f"**Less reliable codes** (κ < 0.6): {', '.join(f'`{c[:20]}`' for c in low_k) or 'none'}"
    )
    lines.append("")

    # B. Causal link
    lines.append("## B. Causal Link Analysis")
    lines.append("")
    lines.append(
        f"- **Causal link κ**: {causal['link_kappa']} (3-way: direct/enabling/coincidental)"
    )
    lines.append(
        f"- **Agree rate**: {causal['link_agree_rate']:.1%} ({causal['link_agree_count']}/{causal['link_n']})"
    )
    lines.append("")

    lines.append("### Causal link distribution (agree-filtered)")
    lines.append("")
    total_links = sum(causal["link_dist"].values())
    lines.append("| Link type | Count | % |")
    lines.append("|-----------|-------|---|")
    for link, cnt in sorted(causal["link_dist"].items(), key=lambda x: -x[1]):
        lines.append(f"| **{link}** | {cnt} | {cnt / total_links * 100:.0f}% |")
    lines.append("")

    # Per-attack causal breakdown
    lines.append("### Attack strategy → causal link type")
    lines.append("")
    lines.append("| Attack | direct | enabling | coincidental |")
    lines.append("|--------|--------|----------|--------------|")
    for atk in sorted(
        causal["attack_causal"], key=lambda a: -sum(causal["attack_causal"][a].values())
    ):
        ac = causal["attack_causal"][atk]
        d = ac.get("direct", 0)
        e = ac.get("enabling", 0)
        c = ac.get("coincidental", 0)
        total = d + e + c
        lines.append(
            f"| `{atk[:30]}` | {d} ({d / total * 100:.0f}%) | {e} ({e / total * 100:.0f}%) | {c} ({c / total * 100:.0f}%) |"
        )
    lines.append("")

    lines.append("**Interpretation:**")
    lines.append("")
    # Find most direct and most enabling attacks
    most_direct = max(
        causal["attack_causal"].items(),
        key=lambda x: x[1].get("direct", 0) / max(sum(x[1].values()), 1),
        default=("?", {}),
    )
    most_enabling = max(
        causal["attack_causal"].items(),
        key=lambda x: x[1].get("enabling", 0) / max(sum(x[1].values()), 1),
        default=("?", {}),
    )
    lines.append(
        f"- Most **direct** attack: `{most_direct[0][:30]}` — the attack itself triggers the failure mechanism."
    )
    lines.append(
        f"- Most **enabling** attack: `{most_enabling[0][:30]}` — creates conditions for failure, but agent could have avoided it."
    )
    lines.append("")

    # C. Step-level consistency
    lines.append("## C. Step-Level Consistency")
    lines.append("")
    ss = causal["step_stats"]
    lines.append(f"Dual-judge decisive user message index comparison (n={ss['n']}):")
    lines.append("")
    lines.append(
        f"- **Exact match** (same msg index): {ss['exact_match']}/{ss['n']} ({ss['exact_match'] / ss['n'] * 100:.0f}%)"
        if ss["n"]
        else ""
    )
    lines.append(
        f"- **Within ±2 messages**: {ss['within_2']}/{ss['n']} ({ss['within_2'] / ss['n'] * 100:.0f}%)"
        if ss["n"]
        else ""
    )
    lines.append(
        f"- **Within ±5 messages**: {ss['within_5']}/{ss['n']} ({ss['within_5'] / ss['n'] * 100:.0f}%)"
        if ss["n"]
        else ""
    )
    lines.append(f"- **Mean |diff|**: {ss['mean_diff']} messages")
    lines.append(f"- **Median |diff|**: {ss['median_diff']} messages")
    lines.append(f"- **Max |diff|**: {ss['max_diff']} messages")
    lines.append("")

    lines.append("**Interpretation:**")
    lines.append("")
    if ss["n"] > 0:
        exact_pct = ss["exact_match"] / ss["n"] * 100
        w2_pct = ss["within_2"] / ss["n"] * 100
        if exact_pct >= 50:
            lines.append(
                f"Strong step-level agreement: {exact_pct:.0f}% exact match means both judges identify the same decisive user turn."
            )
        elif w2_pct >= 70:
            lines.append(
                f"Moderate step-level agreement: {w2_pct:.0f}% within ±2 messages. Judges agree on the critical region but may differ on the exact trigger message."
            )
        else:
            lines.append(
                f"Weak step-level agreement: only {exact_pct:.0f}% exact match. Case-level code agreement (κ=0.853) does not imply step-level causal agreement — judges may assign the same code for different reasons."
            )
    lines.append("")

    # Distribution of diffs
    if causal["step_diffs"]:
        diff_bins = Counter()
        for d in causal["step_diffs"]:
            if d == 0:
                diff_bins["0 (exact)"] += 1
            elif d <= 2:
                diff_bins["1-2"] += 1
            elif d <= 5:
                diff_bins["3-5"] += 1
            elif d <= 10:
                diff_bins["6-10"] += 1
            else:
                diff_bins["11+"] += 1

        lines.append("| |diff| range | Count | % |")
        lines.append("|-------------|-------|---|")
        for bin_label in ["0 (exact)", "1-2", "3-5", "6-10", "11+"]:
            cnt = diff_bins.get(bin_label, 0)
            lines.append(f"| {bin_label} | {cnt} | {cnt / ss['n'] * 100:.0f}% |")
        lines.append("")

    report += "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(f"Appended improvements to {REPORT_PATH}")


def main():
    items = load_paired_annotations()
    print(f"Loaded {len(items)} paired annotations")

    # 1. Per-code κ
    print("\n=== 1. Per-Code κ ===")
    pc = per_code_kappa(items)
    for code in sorted(pc, key=lambda c: -pc[c]["f1"]):
        s = pc[code]
        print(
            f"  {code[:35]:35s}  κ={s['kappa']:.3f}  F1={s['f1']:.3f}  n={s['n_gpt54']}/{s['n_glm52']}"
        )

    # 2+3. Causal link + decisive step
    print("\n=== 2+3. Causal Link + Step-Level ===")
    run_causal_annotation(items, concurrency=5)

    # Analyze
    print("\n=== Analysis ===")
    causal = analyze_causal(items)
    print(
        f"Causal link: κ={causal['link_kappa']}, agree={causal['link_agree_rate']:.1%}"
    )
    print(f"Link distribution: {causal['link_dist']}")
    ss = causal["step_stats"]
    print(
        f"Step consistency: exact={ss['exact_match']}/{ss['n']}, "
        f"within±2={ss['within_2']}/{ss['n']}, mean|diff|={ss['mean_diff']}"
    )

    # Update report
    append_to_report(pc, causal)


if __name__ == "__main__":
    main()
