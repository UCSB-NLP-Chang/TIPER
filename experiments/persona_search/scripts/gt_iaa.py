#!/usr/bin/env python3
"""GT IAA: two judges independently assign the 16 emergent codes (primary + up to
2 secondary) to the same traces. Compute Cohen's kappa on primary + confusion
matrix to find codes that are confused (= non-mutually-exclusive definitions to
refine). This is the legitimacy source for the GT taxonomy.

Usage:
  .venv/bin/python experiments/persona_search/scripts/gt_iaa.py code --judge gpt54
  .venv/bin/python experiments/persona_search/scripts/gt_iaa.py code --judge glm
  .venv/bin/python experiments/persona_search/scripts/gt_iaa.py kappa
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)
ROOT = REPO / "data/simulations/fullset"
OUT_DIR = REPO / "data/persona_opt"
CATS_PATH = OUT_DIR / "gt_axial_categories.json"
SAMPLES_PATH = OUT_DIR / "gt_opencode_samples.jsonl"

DOMAINS = {
    "airline": ["alaska_00", "alaska_01"],
    "retail": ["gc_02", "gc_03", "gc_04"],
    "telecom": ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"],
    "banking": ["cw_03", "cw_04"],
}
DOMAIN_DIR = {
    "airline": ROOT / "airline",
    "retail": ROOT / "retail",
    "telecom": ROOT / "telecom",
    "telecom_p0003": ROOT / "telecom_p0003",
    "banking": ROOT / "banking",
}

JUDGES = {
    "gpt54": {"model": "gpt-5.4", "via": "litellm"},
    "glm": {
        "model": "anthropic/accounts/fireworks/models/glm-5p2",
        "via": "shim",
        "base": "http://127.0.0.1:4114",
        "key_env": "CRESTA_VERTEX_TOKEN",
    },
}


def cell_path(domain, persona, cond):
    if persona == "p0003":
        return DOMAIN_DIR["telecom_p0003"] / f"p0003_{cond}"
    return DOMAIN_DIR[domain] / f"{persona}_{cond}"


def find_sim(domain, persona, cond, task):
    rp = cell_path(domain, persona, cond) / "results.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text())
    for s in r.get("simulations", []):
        if s.get("reward_info", {}).get("reward", 1) <= 0 and s.get("task_id") == task:
            return s
    return None


def render(sim, max_msgs=60, max_chars=650):
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
            out.append(f"{i:03d} [{role}] {c}")
    return "\n".join(out)


def codebook_block(cats):
    lines = [
        "Assign PRIMARY = single most decisive failure mechanism. "
        "SECONDARY = up to 2 contributing mechanisms (or empty). "
        "Use ONLY these codes:"
    ]
    # v2 codebook has per-code decision_rule/inclusion/exclusion
    for c in cats.get("codes", cats.get("categories", [])):
        lines.append(f"  {c['code']}: {c['definition']}")
        if c.get("inclusion"):
            lines.append(f"     INCL: {c['inclusion']}")
        if c.get("exclusion"):
            lines.append(f"     EXCL: {c['exclusion']}")
        if c.get("decision_rule"):
            lines.append(f"     RULE: {c['decision_rule']}")
    # global decision rules
    gr = cats.get("decision_rules_global", [])
    if gr:
        lines.append("GLOBAL RULES:")
        for r in gr:
            lines.append(f"  - {r}")
    return "\n".join(lines)


def call(judge, sys_msg, user_msg, max_tokens=400):
    import litellm

    j = JUDGES[judge]
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    if j["via"] == "shim":
        resp = litellm.completion(
            model=j["model"],
            messages=msgs,
            temperature=0,
            max_tokens=max_tokens,
            api_base=j["base"],
            api_key=os.environ.get(j["key_env"], ""),
        )
    else:
        resp = litellm.completion(
            model=j["model"], messages=msgs, temperature=0, max_tokens=max_tokens
        )
    return resp.choices[0].message.content.strip()


def code(judge, codebook_path=None, tag=""):
    cb_path = Path(codebook_path) if codebook_path else CATS_PATH
    cats = json.loads(cb_path.read_text())
    valid = {c["code"] for c in cats.get("codes", cats.get("categories", []))}
    cb = codebook_block(cats)
    out_path = (
        OUT_DIR / f"gt_iaa_{tag}{judge}.jsonl"
        if tag
        else OUT_DIR / f"gt_iaa_{judge}.jsonl"
    )
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                done.add((a["domain"], a["persona"], a["cond"], a["task"]))
    samples = [
        json.loads(l) for l in SAMPLES_PATH.read_text().splitlines() if l.strip()
    ]
    samples = [
        s
        for s in samples
        if (s["domain"], s["persona"], s["cond"], s["task"]) not in done
    ]
    print(
        f"judge={judge} codebook={cb_path.name}: {len(samples)} to code ({len(done)} done), {len(valid)} codes"
    )

    SYS = (
        "You are a failure-mode annotator. You read a FAILED customer-service "
        "agent transcript and assign failure-mechanism codes from a fixed codebook. "
        "Pick the PRIMARY (single most decisive) and up to 2 SECONDARY (contributing) "
        'codes. OUTPUT ONLY JSON: {"primary": "CODE", "secondary": ["CODE",...], '
        '"evidence": "<short quote with msg idx>"}. No prose, no markdown.'
    )

    def one(s):
        sim = find_sim(s["domain"], s["persona"], s["cond"], s["task"])
        if sim is None:
            return {**s, "judge": judge, "primary": "SIM_NOT_FOUND", "secondary": []}
        tr = render(sim)
        term = sim.get("termination_reason", "?")
        nmsg = len(sim.get("messages", []))
        tid = re.sub(r"\[PERSONA:.*?\]", "", sim.get("task_id", ""))
        user = (
            f"DOMAIN: {s['domain']}\nTERMINATION: {term} (n={nmsg})\nTASK: {tid}\n"
            f"TRANSCRIPT:\n{tr}\n\n{cb}\n\nOutput only the JSON."
        )
        raw = call(judge, SYS, user)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            a = json.loads(raw)
        except Exception:
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            try:
                a = (
                    json.loads(m.group(0))
                    if m
                    else {"primary": "PARSE_ERROR", "secondary": []}
                )
            except Exception:
                a = {"primary": "PARSE_ERROR", "secondary": []}
        if a.get("primary") not in valid:
            a["primary"] = (
                "PARSE_ERROR" if a.get("primary") not in valid else a["primary"]
            )
        a["domain"] = s["domain"]
        a["persona"] = s["persona"]
        a["cond"] = s["cond"]
        a["task"] = s["task"]
        a["judge"] = judge
        a["termination"] = term
        a["n_messages"] = nmsg
        return a

    n = 0
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(one, s) for s in samples]
        for fut in as_completed(futs):
            a = fut.result()
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  [{n}/{len(samples)}]")


def kappa(tag=""):
    """Cohen's kappa on primary + confusion matrix + per-code agreement."""
    from collections import Counter

    def load(j):
        p = OUT_DIR / f"gt_iaa_{tag}{j}.jsonl" if tag else OUT_DIR / f"gt_iaa_{j}.jsonl"
        if not p.exists():
            return {}
        out = {}
        for l in p.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                out[(a["domain"], a["persona"], a["cond"], a["task"])] = a["primary"]
        return out

    g = load("gpt54")
    l = load("glm")
    keys = sorted(set(g) & set(l))
    # drop parse errors / sim_not_found
    bad = {"PARSE_ERROR", "SIM_NOT_FOUND", "None", None}
    pairs = [(g[k], l[k]) for k in keys if g[k] not in bad and l[k] not in bad]
    n = len(pairs)
    print(f"paired traces: {n} (excluded {len(keys) - n} parse/missing)")

    codes = sorted(set(a for a, b in pairs) | set(b for a, b in pairs))
    # observed agreement
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    # expected agreement
    ca = Counter(a for a, b in pairs)
    cb = Counter(b for a, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in codes)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"\n=== Cohen's kappa (primary) ===")
    print(f"  observed agreement: {agree}/{n} = {100 * po:.1f}%")
    print(f"  expected (chance):  {100 * pe:.1f}%")
    print(f"  kappa = {k:.3f}")
    if k >= 0.81:
        print("  strength: almost perfect")
    elif k >= 0.61:
        print("  strength: substantial")
    elif k >= 0.41:
        print("  strength: moderate")
    elif k >= 0.21:
        print("  strength: fair")
    else:
        print("  strength: poor")

    # confusion matrix: which codes get confused
    print(f"\n=== Confusion (gpt54 -> glm), top off-diagonal pairs ===")
    conf = Counter((a, b) for a, b in pairs if a != b)
    for (a, b), cnt in conf.most_common(15):
        print(f"  {a:55s} -> {b:55s}  {cnt}")

    # per-code: agreement when either judge used it
    print(f"\n=== Per-code reliability ===")
    print(f"  {'code':55s} {'either':>6s} {'agree':>6s} {'agree%':>7s}")
    for c in codes:
        either = sum(1 for a, b in pairs if a == c or b == c)
        ag = sum(1 for a, b in pairs if a == c and b == c)
        if either:
            print(f"  {c:55s} {either:6d} {ag:6d} {100 * ag / either:6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("code")
    sc.add_argument("--judge", choices=list(JUDGES), required=True)
    sc.add_argument("--codebook", default=None)
    sc.add_argument("--tag", default="")
    sk = sub.add_parser("kappa")
    sk.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "code":
        code(a.judge, a.codebook, a.tag)
    elif a.cmd == "kappa":
        kappa(a.tag)


if __name__ == "__main__":
    main()
