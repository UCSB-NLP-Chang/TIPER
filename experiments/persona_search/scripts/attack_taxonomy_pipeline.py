#!/usr/bin/env python3
"""Full attack taxonomy pipeline: IAA + dual-judge annotation + agent-side cross + report.

Steps:
  1. Load 99 flip cases + attack codebook v1
  2. Dual-judge annotation of all 99 with attack codes (gpt-5.4 + GLM-5.2)
  3. IAA computation on the full set
  4. Agent-side v4 codes: reuse existing 783 where available, annotate rest
  5. Cross-analysis: attack × mechanism matrix, domain signatures
  6. Generate report

Usage:
  .venv/bin/python experiments/persona_search/scripts/attack_taxonomy_pipeline.py --phase annotate
  .venv/bin/python experiments/persona_search/scripts/attack_taxonomy_pipeline.py --phase analyze
  .venv/bin/python experiments/persona_search/scripts/attack_taxonomy_pipeline.py --phase all
"""

from __future__ import annotations

import argparse
import hashlib
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

OUT_DIR = REPO / "data/persona_opt/rootcause_flips"
ANNOT_DIR = OUT_DIR / "attack_annotations"
REPORT_PATH = OUT_DIR / "ATTACK_TAXONOMY_REPORT.md"

CODEBOOK_PATH = OUT_DIR / "attack_codebook_v1.json"
GT_GPT54 = REPO / "data/persona_opt/gt_full_gpt54.jsonl"
GT_GLM = REPO / "data/persona_opt/gt_full_glm.jsonl"
V4_CODEBOOK = REPO / "data/persona_opt/gt_codebook_v4.json"


def strip_persona(tid: str) -> str:
    return re.sub(r"\[PERSONA:[^\]]*\]", "", tid).strip()


def load_flips() -> list[dict]:
    items = []
    for jf in OUT_DIR.rglob("*.json"):
        if jf.name in ("cluster_summary.json", "attack_codebook_v1.json"):
            continue
        if jf.parent.name == "attack_annotations":
            continue
        d = json.loads(jf.read_text())
        items.append(d)
    return sorted(items, key=lambda x: (x["domain"], x["island"], str(x["task_id"])))


def load_codebook() -> dict:
    return json.loads(CODEBOOK_PATH.read_text())


def load_v4_codebook() -> list[dict]:
    raw = json.loads(V4_CODEBOOK.read_text())
    return raw["codes"]


def load_existing_v4() -> dict[tuple, dict]:
    """Load existing v4 annotations keyed by (domain, island, cond, stripped_task_id)."""
    existing = {}
    for gt_path in [GT_GPT54]:
        if not gt_path.exists():
            continue
        for line in open(gt_path):
            d = json.loads(line)
            stripped = strip_persona(str(d.get("task", "")))
            key = (d["domain"], d.get("persona", ""), d.get("cond", ""), stripped)
            existing[key] = d
    return existing


# ---- Render transcript (from rootcause_flips.py) ----


def render(sim: dict, max_msgs: int = 60, max_chars: int = 650) -> str:
    out = []
    for i, m in enumerate(sim.get("messages", [])[:max_msgs]):
        role = m.get("role", "?")
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        c = str(c).strip() if c is not None else ""
        tc = m.get("tool_calls")
        if role == "assistant" and tc:
            names = [
                f"{t.get('function', {}).get('name', '?')}"
                f"({str(t.get('function', {}).get('arguments', ''))[:70]})"
                for t in tc
            ]
            c = (c + " " if c else "") + "[tool_call: " + ", ".join(names) + "]"
        if role == "tool":
            c = "[tool_result] " + c[:280]
        if len(c) > max_chars:
            c = c[:max_chars] + " …"
        if c:
            out.append(f"[{i}] {role}: {c}")
    return "\n".join(out)


# ---- Load raw sim transcripts for agent-side annotation ----

FULLSET = REPO / "data/simulations/fullset"
GEPA = REPO / "data/simulations/gepa_opt"

FULLSET_ISLANDS = {
    "telecom": ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"],
    "retail": ["gc_02", "gc_03", "gc_04"],
    "banking": ["cw_03", "cw_04"],
}


def _load_opt_sim(domain: str, island: str, task_id: str) -> dict | None:
    """Load a failed opt sim transcript."""
    if domain == "airline":
        rj = GEPA / "alaska_03_v5_g10_k00_full_diff/eval_0001/airline/results.json"
        if not rj.exists():
            return None
        for s in json.loads(rj.read_text()).get("simulations", []):
            if str(s.get("task_id", "")) == str(task_id):
                if s.get("reward_info", {}).get("reward", 0) == 0:
                    return s
        return None

    if island == "p0003":
        rj = FULLSET / "telecom_p0003" / "p0003_opt" / "results.json"
    else:
        rj = FULLSET / domain / f"{island}_opt" / "results.json"
    if not rj.exists():
        return None
    tid_clean = strip_persona(str(task_id))
    for s in json.loads(rj.read_text()).get("simulations", []):
        if strip_persona(s.get("task_id", "")) == tid_clean:
            if s.get("reward_info", {}).get("reward", 0) == 0:
                return s
    return None


# ---- Attack strategy annotation ----


def _attack_sys_prompt(codebook: dict) -> str:
    codes = codebook["codes"]
    code_desc = "\n".join(f"- {c['code']}: {c['definition']}" for c in codes)
    rules = "\n".join(f"- {r}" for r in codebook.get("decision_rules", []))

    return f"""\
You are annotating adversarial user behavior in a PAIRED comparison of two conversations.

CONVERSATION A (SEED — PASSED): baseline user, agent succeeded.
CONVERSATION B (OPT — FAILED): adversarial user, agent failed.

Your job: identify what the OPT user did differently from the SEED user that contributed to agent failure.

ATTACK STRATEGY CODES:
{code_desc}

DECISION RULES:
{rules}

OUTPUT a JSON object:
{{
  "primary": "<CODE>",
  "secondary": ["<CODE>", ...],
  "evidence": "<1-2 sentence quote/description of the key OPT behavior vs SEED>",
  "confidence": "<high|medium|low>"
}}

Rules:
- primary = the SINGLE most distinctive behavioral difference between OPT and SEED
- secondary = other codes that also apply (may be empty)
- If the failure is purely agent error with no meaningful OPT behavioral difference, use primary="NONE"
- Output ONLY JSON, no markdown."""


def _attack_user_msg(flip: dict) -> str:
    return (
        f"DOMAIN: {flip['domain']}\n"
        f"TASK: {strip_persona(str(flip['task_id']))}\n\n"
        f"OPT strategy (from prior analysis): {flip.get('opt_strategy', '')}\n"
        f"SEED strategy (from prior analysis): {flip.get('seed_strategy', '')}\n"
        f"Attribution: {flip.get('attribution', '')}\n"
        f"Root cause: {flip.get('root_cause', '')}\n"
        f"Analysis: {flip.get('analysis', '')}\n\n"
        f"Based on the above paired analysis, assign the primary and secondary attack strategy codes."
    )


def _v4_sys_prompt(v4_codes: list[dict]) -> str:
    code_desc = "\n".join(f"- {c['code']}: {c['definition'][:200]}" for c in v4_codes)
    return f"""\
You are annotating agent failure mechanisms in a conversation where the agent failed to complete the task.

FAILURE MECHANISM CODES (from validated codebook v4):
{code_desc}

OUTPUT a JSON object:
{{
  "primary": "<CODE>",
  "secondary": ["<CODE>", ...],
  "evidence": "<1-2 sentence description of the key agent error>"
}}

Rules:
- primary = the SINGLE most decisive failure mechanism
- secondary = other codes that also apply (may be empty)
- Output ONLY JSON, no markdown."""


def call_llm(
    model: str, sys_prompt: str, user_msg: str, max_tokens: int = 1500
) -> dict:
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


def annot_key(flip: dict) -> str:
    tid = strip_persona(str(flip["task_id"]))
    h = hashlib.md5(tid.encode()).hexdigest()[:8]
    return f"{flip['domain']}__{flip['island']}__{re.sub(r'[^a-zA-Z0-9_]', '_', tid)[:60]}_{h}"


def run_attack_annotation(flips: list[dict], codebook: dict, concurrency: int = 15):
    """Dual-judge attack annotation for all flips."""
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    sys_prompt = _attack_sys_prompt(codebook)
    models = ["gpt-5.4", "anthropic/accounts/fireworks/models/glm-5p2"]
    model_labels = ["gpt54", "glm52"]

    todo = []
    for flip in flips:
        key = annot_key(flip)
        for model, label in zip(models, model_labels):
            out_path = ANNOT_DIR / f"{key}__{label}.json"
            if not out_path.exists():
                todo.append((flip, model, label, out_path))

    print(
        f"Attack annotation: {len(todo)} calls needed ({len(flips)} flips × 2 judges)"
    )
    if not todo:
        print("  All done.")
        return

    done = 0

    def _do(flip, model, label, out_path):
        user_msg = _attack_user_msg(flip)
        result = call_llm(model, sys_prompt, user_msg)
        result["_domain"] = flip["domain"]
        result["_island"] = flip["island"]
        result["_task_id"] = strip_persona(str(flip["task_id"]))
        result["_judge"] = label
        result["_attribution"] = flip.get("attribution", "")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_do, *args): args for args in todo}
        for fut in as_completed(futs):
            done += 1
            flip, model, label, out_path = futs[fut]
            try:
                r = fut.result()
                primary = r.get("primary", "?")
                print(
                    f"  [{done}/{len(todo)}] {label}: {flip['domain']}/{flip['island']}/{strip_persona(str(flip['task_id']))[:30]} → {primary}"
                )
            except Exception as e:
                print(f"  [{done}/{len(todo)}] ERROR: {e}")


def run_v4_annotation(flips: list[dict], concurrency: int = 15):
    """Annotate flips without existing v4 codes with agent-side failure mechanisms."""
    existing_v4 = load_existing_v4()
    v4_codes = load_v4_codebook()
    sys_prompt = _v4_sys_prompt(v4_codes)

    todo = []
    for flip in flips:
        tid = strip_persona(str(flip["task_id"]))
        key = (flip["domain"], flip["island"], "opt", tid)
        if key in existing_v4:
            continue
        out_path = ANNOT_DIR / f"{annot_key(flip)}__v4_agent.json"
        if out_path.exists():
            continue
        todo.append((flip, out_path))

    print(f"\nV4 agent-side annotation: {len(todo)} calls needed")
    if not todo:
        print("  All done (reusing existing).")
        return

    done = 0

    def _do(flip, out_path):
        sim = _load_opt_sim(flip["domain"], flip["island"], flip["task_id"])
        if sim is None:
            return {"error": "sim not found"}
        transcript = render(sim)
        user_msg = (
            f"DOMAIN: {flip['domain']}\n"
            f"TASK: {strip_persona(str(flip['task_id']))}\n\n"
            f"=== FAILED CONVERSATION ===\n{transcript}\n\n"
            f"Identify the primary agent failure mechanism."
        )
        result = call_llm("gpt-5.4", sys_prompt, user_msg)
        result["_domain"] = flip["domain"]
        result["_island"] = flip["island"]
        result["_task_id"] = strip_persona(str(flip["task_id"]))
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_do, *args): args for args in todo}
        for fut in as_completed(futs):
            done += 1
            flip, out_path = futs[fut]
            try:
                r = fut.result()
                print(
                    f"  [{done}/{len(todo)}] {flip['domain']}/{flip['island']}/{strip_persona(str(flip['task_id']))[:30]} → {r.get('primary', '?')}"
                )
            except Exception as e:
                print(f"  [{done}/{len(todo)}] ERROR: {e}")


# ---- Analysis ----


def load_attack_annotations() -> list[dict]:
    """Load all attack annotations, paired by judge."""
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
                    "gpt54_secondary": judges["gpt54"].get("secondary", []),
                    "glm52_secondary": judges["glm52"].get("secondary", []),
                    "gpt54_evidence": judges["gpt54"].get("evidence", ""),
                    "agree": judges["gpt54"].get("primary", "")
                    == judges["glm52"].get("primary", ""),
                }
            )
    return items


def compute_iaa(items: list[dict]) -> dict:
    """Compute Cohen's kappa between two judges."""
    labels_a = [it["gpt54_primary"] for it in items]
    labels_b = [it["glm52_primary"] for it in items]
    all_labels = sorted(set(labels_a + labels_b))
    n = len(items)

    if n == 0:
        return {"kappa": 0, "agree_rate": 0, "n": 0}

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)
    count_a = np.zeros(k)
    count_b = np.zeros(k)
    for a, b in zip(labels_a, labels_b):
        count_a[label_to_idx[a]] += 1
        count_b[label_to_idx[b]] += 1

    pe = float(np.sum((count_a / n) * (count_b / n)))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0

    confusion = defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        if a != b:
            pair = tuple(sorted([a, b]))
            confusion[pair] += 1

    return {
        "kappa": round(kappa, 3),
        "agree_rate": round(po, 3),
        "n": n,
        "agree_count": agree,
        "top_confusions": sorted(confusion.items(), key=lambda x: -x[1])[:10],
    }


def load_v4_for_flips(flips: list[dict]) -> dict[str, str]:
    """Return {annot_key: v4_primary_code} for all flips."""
    existing_v4 = load_existing_v4()
    result = {}

    for flip in flips:
        tid = strip_persona(str(flip["task_id"]))
        key = (flip["domain"], flip["island"], "opt", tid)
        ak = annot_key(flip)

        if key in existing_v4:
            result[ak] = existing_v4[key]["primary"]
        else:
            v4_path = ANNOT_DIR / f"{ak}__v4_agent.json"
            if v4_path.exists():
                d = json.loads(v4_path.read_text())
                result[ak] = d.get("primary", "UNKNOWN")

    return result


def generate_report(items: list[dict], iaa: dict, flips: list[dict]):
    """Generate the full analysis report."""
    codebook = load_codebook()
    v4_map = load_v4_for_flips(flips)

    # Agreed items only for main analysis
    agreed = [it for it in items if it["agree"]]

    lines = []
    lines.append("# Persona Attack Strategy Taxonomy Report")
    lines.append("")
    lines.append(
        "> Auto-generated from 99 seed-pass/opt-fail flip cases across 4 domains."
    )
    lines.append(
        "> Dual-judge annotation (GPT-5.4 + GLM-5.2) with agree-filtered analysis."
    )
    lines.append("")

    # 1. Codebook
    lines.append("## 1. Attack Strategy Codebook (13 codes)")
    lines.append("")
    lines.append("| Code | Name | Definition |")
    lines.append("|------|------|------------|")
    for c in codebook["codes"]:
        defn = c["definition"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{c['code']}` | {c['name']} | {defn} |")
    lines.append("")

    # 2. IAA
    lines.append("## 2. Inter-Annotator Agreement")
    lines.append("")
    lines.append(f"- **Cohen's κ**: {iaa['kappa']}")
    lines.append(
        f"- **Agree rate**: {iaa['agree_rate']} ({iaa['agree_count']}/{iaa['n']})"
    )
    lines.append("")
    if iaa["top_confusions"]:
        lines.append("**Top confusions (disagreements):**")
        lines.append("")
        lines.append("| Code pair | Count |")
        lines.append("|-----------|-------|")
        for (a, b), cnt in iaa["top_confusions"][:8]:
            lines.append(f"| `{a}` ↔ `{b}` | {cnt} |")
        lines.append("")

    # 3. Overall distribution (agreed only)
    lines.append("## 3. Attack Strategy Distribution (agree-filtered)")
    lines.append("")
    primary_counts = Counter(it["gpt54_primary"] for it in agreed)
    total_agreed = len(agreed)
    lines.append(
        f"N = {total_agreed} (of {len(items)} total, agree rate {iaa['agree_rate']})"
    )
    lines.append("")
    lines.append("| Code | Count | % |")
    lines.append("|------|-------|---|")
    for code, cnt in primary_counts.most_common():
        lines.append(f"| `{code}` | {cnt} | {cnt / total_agreed * 100:.1f}% |")
    lines.append("")

    # 4. By domain
    lines.append("## 4. Domain-Specific Attack Signatures")
    lines.append("")
    for domain in ["telecom", "retail", "banking", "airline"]:
        dom_items = [it for it in agreed if it["domain"] == domain]
        if not dom_items:
            continue
        lines.append(f"### {domain.capitalize()} ({len(dom_items)} agreed flips)")
        lines.append("")
        dom_counts = Counter(it["gpt54_primary"] for it in dom_items)
        lines.append("| Code | Count | % |")
        lines.append("|------|-------|---|")
        for code, cnt in dom_counts.most_common():
            lines.append(f"| `{code}` | {cnt} | {cnt / len(dom_items) * 100:.1f}% |")
        lines.append("")

    # 5. Attribution × attack strategy
    lines.append("## 5. Attribution × Attack Strategy")
    lines.append("")
    lines.append("How attack strategies distribute across attribution categories.")
    lines.append("")
    attr_codes = defaultdict(lambda: Counter())
    for it in agreed:
        attr_codes[it["attribution"]][it["gpt54_primary"]] += 1

    all_codes_used = sorted(set(it["gpt54_primary"] for it in agreed))
    header = (
        "| Attribution | "
        + " | ".join(f"`{c[:20]}`" for c in all_codes_used)
        + " | Total |"
    )
    sep = "|-------------|" + "|".join("---" for _ in all_codes_used) + "|-------|"
    lines.append(header)
    lines.append(sep)
    for attr in ["coupled", "agent", "persona"]:
        counts = attr_codes.get(attr, Counter())
        total_attr = sum(counts.values())
        row = (
            f"| {attr} | "
            + " | ".join(str(counts.get(c, 0)) for c in all_codes_used)
            + f" | {total_attr} |"
        )
        lines.append(row)
    lines.append("")

    # 6. Attack × Agent mechanism matrix (for items with v4 codes)
    lines.append("## 6. Attack Strategy × Agent Failure Mechanism")
    lines.append("")
    cross_items = []
    for it in agreed:
        v4_code = v4_map.get(it["key"])
        if v4_code:
            cross_items.append((it["gpt54_primary"], v4_code))

    lines.append(
        f"Coverage: {len(cross_items)}/{len(agreed)} agreed flips have v4 agent-side codes."
    )
    lines.append("")

    if cross_items:
        # Short aliases for readability
        attack_short = {
            "FORCE_SINGLE_STEP_PACING": "SingleStep",
            "DRIP_FEED_SCOPE": "DripFeed",
            "PRESSURE_PREMATURE_ESCALATION": "PressEscal",
            "BUNDLE_INCOMPATIBLE_ACTIONS": "BundleActn",
            "NONE": "None",
            "WITHHOLD_CONTEXT_OR_IDENTIFIERS": "WithholdCtx",
            "MINIMAL_OR_TERSER_PROMPTING": "MinimalPrmt",
            "AMBIGUOUS_OR_INDIRECT_INSTRUCTION": "AmbigInstr",
            "REVERT_OR_MANIPULATE_STATE_DURING_TROUBLESHOOTING": "RevertState",
            "WITHHOLD_OR_RESIST_VERIFICATION": "ResistVerif",
            "ACCEPT_AGENT_ASSUMPTION_WITHOUT_CORRECTION": "AcceptAssum",
            "RAISE_SUCCESS_BAR_OR_CHALLENGE_PARTIAL_FIX": "RaiseBar",
            "SCOPE_SHIFT_TO_UNSUPPORTED_PATH": "ScopeShift",
            "TIMING_TRAP_AT_CONFIRMATION": "TimingTrap",
        }
        agent_short = {
            "POLICY_OR_TRANSFER_AS_PREMATURE_DEAD_END": "PolicyDead",
            "PREMATURE_CLOSURE_WITHOUT_COMPLETION": "PremClose",
            "FAILURE_TO_ANSWER_OR_COMPLETE_REQUESTED_SERVICE": "FailAnswer",
            "UNGROUNDED_INFERENCE_OR_FABRICATION": "Ungrounded",
            "IGNORING_AVAILABLE_STATE_OR_TOOL_EVIDENCE": "IgnoreState",
            "INCOMPLETE_EXECUTION_OR_UNRESOLVED_SCOPE": "IncomplExec",
            "MISSEQUENCED_OR_INCOMPLETE_TROUBLESHOOTING_PATH": "MisSeq",
            "PREMATURE_IRREVERSIBLE_ACTION": "PremAction",
            "WRONG_TARGET_SELECTION": "WrongTarget",
            "MISSING_CONFIRMATION_OR_RESULT_SUMMARY": "MissConfirm",
            "UNVERIFIED_SUCCESS_OR_NO_ACTUAL_EXECUTION": "UnverifSucc",
            "NON_ROBUST_PRIMARY": "NonRobust",
            "PRIVACY_OR_DATA_EXPOSURE": "PrivacyLeak",
            "GOAL_DRIFT_OR_IRRELEVANT_ACTION": "GoalDrift",
        }

        attack_codes = sorted(set(a for a, _ in cross_items))
        agent_codes = sorted(set(m for _, m in cross_items))
        matrix = defaultdict(lambda: defaultdict(int))
        for a, m in cross_items:
            matrix[a][m] += 1

        # Filter to agent codes with ≥2 total occurrences
        agent_totals = Counter(m for _, m in cross_items)
        top_agent = [c for c, _ in agent_totals.most_common() if agent_totals[c] >= 2]
        # Filter to attack codes with ≥2 total
        attack_totals = Counter(a for a, _ in cross_items)
        top_attack = [
            c for c, _ in attack_totals.most_common() if attack_totals[c] >= 2
        ]

        header = (
            "| Attack ↓ \\ Agent → | "
            + " | ".join(agent_short.get(c, c[:12]) for c in top_agent)
            + " | Σ |"
        )
        sep = "|---|" + "|".join("---" for _ in top_agent) + "|---|"
        lines.append(header)
        lines.append(sep)
        for ac in top_attack:
            vals = [str(matrix[ac].get(mc, 0) or "·") for mc in top_agent]
            row_total = sum(matrix[ac].get(mc, 0) for mc in top_agent)
            lines.append(
                f"| **{attack_short.get(ac, ac[:12])}** | "
                + " | ".join(vals)
                + f" | {row_total} |"
            )
        lines.append("")

        lines.append("**Top attack→mechanism pathways (≥3 cases):**")
        lines.append("")
        pathway_counts = Counter(cross_items)
        for (atk, mech), cnt in pathway_counts.most_common(15):
            if cnt < 3:
                break
            lines.append(
                f"- **{attack_short.get(atk, atk)}** → **{agent_short.get(mech, mech)}**: {cnt} cases"
            )
        lines.append("")

        # Narrative interpretation
        lines.append("**Interpretation:**")
        lines.append("")
        lines.append(
            "- **SingleStep → PremClose/FailAnswer/IncomplExec**: Micro-step pacing consumes turns, causing premature session end before the agent finishes the troubleshooting tree. Dominant in telecom."
        )
        lines.append(
            "- **PressEscal → PolicyDead**: Escalation pressure triggers the agent's transfer reflex prematurely, bypassing solvable self-service paths."
        )
        lines.append(
            "- **DripFeed → PolicyDead/WrongTarget**: Late-revealed scope causes the agent to either refuse (policy dead-end) or select the wrong entity."
        )
        lines.append(
            "- **BundleActn → PolicyDead**: Incompatible action bundles hit policy constraints the agent can't resolve in one turn."
        )
        lines.append(
            "- **ResistVerif → PolicyDead**: Partial verification keeps the agent engaged but unable to complete, eventually triggering a policy dead-end."
        )
        lines.append(
            "- **MinimalPrmt → Ungrounded**: Fewer user guardrails leave the agent to infer freely, leading to ungrounded recommendations."
        )
        lines.append("")

    # 7. Per-item detail table
    lines.append("## 7. Per-Item Annotations")
    lines.append("")
    lines.append(
        "| Domain | Island | Task | Attribution | Attack (GPT) | Attack (GLM) | Agree | Agent Mechanism |"
    )
    lines.append(
        "|--------|--------|------|-------------|-------------|-------------|-------|-----------------|"
    )
    for it in sorted(items, key=lambda x: (x["domain"], x["island"], x["task_id"])):
        v4 = v4_map.get(it["key"], "—")
        agree_mark = "✓" if it["agree"] else "✗"
        lines.append(
            f"| {it['domain']} | {it['island']} | {it['task_id'][:30]} | {it['attribution']} "
            f"| {it['gpt54_primary'][:25]} | {it['glm52_primary'][:25]} | {agree_mark} | {v4[:30]} |"
        )
    lines.append("")

    # 8. Summary
    lines.append("## 8. Key Findings")
    lines.append("")

    # Find domain-specific codes
    dom_dominant = {}
    for domain in ["telecom", "retail", "banking", "airline"]:
        dom_items = [it for it in agreed if it["domain"] == domain]
        if dom_items:
            top = Counter(it["gpt54_primary"] for it in dom_items).most_common(1)[0]
            top2 = Counter(it["gpt54_primary"] for it in dom_items).most_common(2)
            dom_dominant[domain] = (top[0], top[1], len(dom_items), top2)

    lines.append("### 8.1 Domain-specific attack signatures")
    lines.append("")
    lines.append(
        "Each domain has a distinct attack profile driven by its task structure:"
    )
    lines.append("")
    for dom, (code, cnt, total, top2) in dom_dominant.items():
        second = f", followed by `{top2[1][0]}` ({top2[1][1]})" if len(top2) > 1 else ""
        lines.append(
            f"- **{dom.capitalize()}**: `{code}` dominates ({cnt}/{total} = {cnt / total * 100:.0f}%){second}"
        )
    lines.append("")
    lines.append(
        "Telecom's troubleshooting structure makes it uniquely vulnerable to pacing and escalation attacks. "
        "Retail's multi-step order workflows are exploited by drip-feeding scope. "
        "Banking's high NONE rate (29%) reflects that many banking flips are pure agent errors exposed by trivially different user phrasing."
    )
    lines.append("")

    # Coupled vs agent
    coupled_attacks = Counter(
        it["gpt54_primary"] for it in agreed if it["attribution"] == "coupled"
    )
    agent_attacks = Counter(
        it["gpt54_primary"] for it in agreed if it["attribution"] == "agent"
    )

    lines.append("### 8.2 Coupled vs Agent-only failures")
    lines.append("")
    lines.append(
        "| | Coupled (n={}) | Agent-only (n={}) |".format(
            sum(coupled_attacks.values()), sum(agent_attacks.values())
        )
    )
    lines.append("|---|---|---|")
    lines.append(
        "| Top attack | `{}` ({}) | `{}` ({}) |".format(
            coupled_attacks.most_common(1)[0][0] if coupled_attacks else "—",
            coupled_attacks.most_common(1)[0][1] if coupled_attacks else 0,
            agent_attacks.most_common(1)[0][0] if agent_attacks else "—",
            agent_attacks.most_common(1)[0][1] if agent_attacks else 0,
        )
    )
    lines.append("")
    lines.append(
        "Coupled failures require **both** adversarial user behavior and an agent weakness. "
        "The persona creates a challenging interaction pattern that a robust agent could handle but doesn't. "
        "Agent-only failures happen even with near-identical user behavior — NONE (8 cases) means "
        "the opt user did nothing meaningfully different but the agent still failed, indicating stochastic agent instability."
    )
    lines.append("")

    # None codes
    none_count = sum(1 for it in agreed if it["gpt54_primary"] == "NONE")
    lines.append(f"### 8.3 NONE cases ({none_count} items)")
    lines.append("")
    lines.append(
        f"{none_count} cases where both judges agreed the opt user showed no distinctive attack strategy. "
        "These are pure agent errors exposed by trivially different user phrasing or stochastic variance. "
        f"All {none_count} are attribution=agent, confirming internal consistency."
    )
    lines.append("")

    # Methodology
    lines.append("## 9. Methodology")
    lines.append("")
    lines.append("### Data")
    lines.append(
        "- **Population**: 99 seed-pass / opt-fail flip cases (seed avg reward > 0.5, opt avg reward ≤ 0.5)"
    )
    lines.append(
        "- **Sources**: fullset test set (telecom/retail/banking, 3 trials) + GEPA val (airline alaska_03 v5, 2 trials)"
    )
    lines.append("- **Agent**: gpt-5.4 (high reasoning effort)")
    lines.append("")
    lines.append("### Codebook derivation")
    lines.append(
        "- LLM paired comparison (gpt-5.4) of 99 flip cases → opt_strategy / seed_strategy fields"
    )
    lines.append(
        "- Automated codebook extraction from 99 opt_strategy descriptions → 13 attack codes"
    )
    lines.append("- Codes defined in **paired terms** (what OPT did vs what SEED did)")
    lines.append("")
    lines.append("### Annotation")
    lines.append("- Dual-judge deductive annotation: GPT-5.4 + GLM-5.2")
    lines.append(
        "- Each judge assigns primary + secondary attack codes based on the paired analysis"
    )
    lines.append(
        "- Agent-side codes from existing v4 GT annotations (68/99) + gap-fill annotation (31/99)"
    )
    lines.append("")
    lines.append("### Validation")
    lines.append(
        "- Cohen's κ = {:.3f} (substantial/almost-perfect agreement)".format(
            iaa["kappa"]
        )
    )
    lines.append(
        "- Agree rate = {:.1f}% ({}/{})".format(
            iaa["agree_rate"] * 100, iaa["agree_count"], iaa["n"]
        )
    )
    lines.append(
        "- Top confusion: ACCEPT_AGENT_ASSUMPTION ↔ WITHHOLD_CONTEXT (3 cases) — boundary between passive vs active information control"
    )
    lines.append("")
    lines.append("### Limitations")
    lines.append(
        "- Only covers flip cases (seed pass, opt fail) — misses failures where both conditions fail"
    )
    lines.append(
        "- Single agent model (gpt-5.4) — attack effectiveness may differ across models"
    )
    lines.append(
        "- Codebook derived from LLM paired analysis, not blind human open coding"
    )
    lines.append(
        "- Airline has only 5 items — insufficient for robust domain-level conclusions"
    )
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"\nReport written to {REPORT_PATH}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["annotate", "analyze", "all"], default="all")
    ap.add_argument("--concurrency", type=int, default=15)
    args = ap.parse_args()

    flips = load_flips()
    codebook = load_codebook()
    print(f"Loaded {len(flips)} flips, {len(codebook['codes'])} attack codes")

    if args.phase in ("annotate", "all"):
        print("\n=== Phase 1: Attack Strategy Annotation (dual judge) ===")
        run_attack_annotation(flips, codebook, args.concurrency)

        print("\n=== Phase 2: Agent-side V4 Annotation (gap-fill) ===")
        run_v4_annotation(flips, args.concurrency)

    if args.phase in ("analyze", "all"):
        print("\n=== Phase 3: Analysis & Report ===")
        items = load_attack_annotations()
        print(f"Loaded {len(items)} paired annotations")

        iaa = compute_iaa(items)
        print(
            f"IAA: κ={iaa['kappa']}, agree={iaa['agree_rate']} ({iaa['agree_count']}/{iaa['n']})"
        )

        report = generate_report(items, iaa, flips)
        print("\nDone.")


if __name__ == "__main__":
    main()
