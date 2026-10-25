#!/usr/bin/env python3
"""Full annotation of all 783 failures with the v4 robust codebook, by two judges.
PRIMARY must be one of the 10 robust codes; SECONDARY can be any of 16.
Resumable per-judge JSONL. Used to produce the agree-filtered robust distribution.

Usage:
  .venv/bin/python experiments/persona_search/scripts/gt_annotate_full.py code --judge gpt54
  .venv/bin/python experiments/persona_search/scripts/gt_annotate_full.py code --judge glm
  .venv/bin/python experiments/persona_search/scripts/gt_annotate_full.py agree   # agree-filtered dist
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
CB_PATH = OUT_DIR / "gt_codebook_v4.json"

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
MAX_PER_CELL = 40

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


def failed_trials(cell):
    rp = cell / "results.json"
    if not rp.exists():
        return []
    r = json.loads(rp.read_text())
    seen, out = set(), []
    for s in r.get("simulations", []):
        if s.get("reward_info", {}).get("reward", 1) > 0:
            continue
        tid = s.get("task_id")
        if tid in seen:
            continue
        seen.add(tid)
        out.append(s)
    return out


def render(sim, max_msgs=200, max_chars=700):
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


def codebook_block(cb):
    lines = [
        "Assign PRIMARY = single most decisive failure mechanism. "
        "PRIMARY MUST be one of the 10 ROBUST codes (primary_eligible=true). "
        "SECONDARY = up to 2 contributing mechanisms (any code)."
    ]
    for c in cb["codes"]:
        tag = (
            "[PRIMARY-ELIGIBLE]"
            if c.get("primary_eligible")
            else "[SECONDARY-ONLY/merged]"
        )
        lines.append(f"  {tag} {c['code']}: {c['definition']}")
        if c.get("decision_rule"):
            lines.append(f"     RULE: {c['decision_rule']}")
        if c.get("merged_to"):
            lines.append(
                f"     (if this fits, use {c['merged_to']} as PRIMARY and this as SECONDARY)"
            )
    lines.append("GLOBAL RULES:")
    for r in cb.get("decision_rules_global", []):
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


def code(judge):
    cb = json.loads(CB_PATH.read_text())
    robust = {c["code"] for c in cb["codes"] if c.get("primary_eligible")}
    all_codes = {c["code"] for c in cb["codes"]}
    cb_txt = codebook_block(cb)
    out_path = OUT_DIR / f"gt_full_{judge}.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                done.add((a["domain"], a["persona"], a["cond"], a["task"]))

    # enumerate all failed tasks
    pairs = []
    for dom, personas in DOMAINS.items():
        for p in personas:
            for cond in ("opt", "seed"):
                trials = failed_trials(cell_path(dom, p, cond))
                if len(trials) > MAX_PER_CELL:
                    trials = random.Random(300).sample(trials, MAX_PER_CELL)
                for s in trials:
                    pairs.append((dom, p, cond, s))
    todo = [
        (d, p, c, s)
        for (d, p, c, s) in pairs
        if (d, p, c, s.get("task_id")) not in done
    ]
    print(
        f"judge={judge}: {len(todo)} to code ({len(done)} done), {len(robust)} robust primary codes"
    )

    SYS = (
        "You are a failure-mode annotator. Read a FAILED customer-service agent transcript "
        "and assign failure-mechanism codes from a fixed codebook. PRIMARY must be a ROBUST "
        "(primary_eligible) code. SECONDARY (up to 2, any code) are contributing. "
        'OUTPUT ONLY JSON: {"primary": "CODE", "secondary": ["CODE",...], '
        '"evidence": "<short quote with msg idx>"}. No prose, no markdown.'
    )

    def one(args):
        i, (dom, p, cond, sim) = args
        tr = render(sim)
        term = sim.get("termination_reason", "?")
        nmsg = len(sim.get("messages", []))
        tid = re.sub(r"\[PERSONA:.*?\]", "", sim.get("task_id", ""))
        user = (
            f"DOMAIN: {dom}\nTERMINATION: {term} (n={nmsg})\nTASK: {tid}\n"
            f"TRANSCRIPT:\n{tr}\n\n{cb_txt}\n\nOutput only the JSON."
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
        # if primary not robust, try to map merged code to its robust parent
        if a.get("primary") not in robust:
            for c in cb["codes"]:
                if c["code"] == a.get("primary") and c.get("merged_to"):
                    a["secondary"] = list(
                        set((a.get("secondary") or []) + [a["primary"]])
                    )
                    a["primary"] = c["merged_to"]
                    break
        if a.get("primary") not in robust and a.get("primary") in all_codes:
            # secondary-only used as primary: demote to secondary, leave primary as PARSE_ERROR for review
            a["secondary"] = list(set((a.get("secondary") or []) + [a["primary"]]))
            a["primary"] = "NON_ROBUST_PRIMARY"
        a["domain"] = dom
        a["persona"] = p
        a["cond"] = cond
        a["task"] = sim.get("task_id")
        a["termination"] = term
        a["n_messages"] = nmsg
        return i, a

    n = 0
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(one, (i, x)) for i, x in enumerate(todo)]
        for fut in as_completed(futs):
            i, a = fut.result()
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 40 == 0:
                print(f"  [{n}/{len(todo)}]")


def agree():
    """Agree-filtered robust distribution across full 783."""
    from collections import Counter, defaultdict

    def load(j):
        p = OUT_DIR / f"gt_full_{j}.jsonl"
        out = {}
        for l in p.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                out[(a["domain"], a["persona"], a["cond"], a["task"])] = a
        return out

    g = load("gpt54")
    l = load("glm")
    bad = {"PARSE_ERROR", "SIM_NOT_FOUND", "NON_ROBUST_PRIMARY", "None", None}
    keys = [
        k
        for k in g
        if k in l and g[k]["primary"] not in bad and l[k]["primary"] not in bad
    ]
    agree_keys = [k for k in keys if g[k]["primary"] == l[k]["primary"]]
    print(
        f"full set: paired {len(keys)}, agreed {len(agree_keys)} ({100 * len(agree_keys) / len(keys):.1f}%)"
    )

    # agree-filtered primary distribution
    c = Counter(g[k]["primary"] for k in agree_keys)
    print(f"\n=== AGREE-FILTERED robust primary taxonomy (n={len(agree_keys)}) ===")
    print(f"  {'code':55s} {'n':>4s} {'%':>6s}")
    for code, n in c.most_common():
        print(f"  {code:55s} {n:4d} {100 * n / len(agree_keys):5.1f}%")

    # per-judge distribution (for interval)
    print(f"\n=== per-judge primary distribution (n={len(keys)}) ===")
    print(f"  {'code':55s} {'gpt54':>6s} {'glm':>6s}")
    gc = Counter(g[k]["primary"] for k in keys)
    lc = Counter(l[k]["primary"] for k in keys)
    for code in sorted(set(gc) | set(lc), key=lambda c: -(gc.get(c, 0) + lc.get(c, 0))):
        print(
            f"  {code:55s} {100 * gc.get(code, 0) / len(keys):5.1f}% {100 * lc.get(code, 0) / len(keys):5.1f}%"
        )

    # by domain (agreed)
    print(f"\n=== agreed by domain ===")
    bydom = defaultdict(list)
    for k in agree_keys:
        bydom[k[0]].append(g[k]["primary"])
    for dom in ["airline", "retail", "telecom", "banking"]:
        lst = bydom.get(dom, [])
        if not lst:
            continue
        cc = Counter(lst)
        print(
            f"  {dom} (n={len(lst)}): "
            + ", ".join(
                f"{k.split('_')[0]}={100 * v / len(lst):.0f}%"
                for k, v in cc.most_common(4)
            )
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("code")
    sc.add_argument("--judge", choices=list(JUDGES), required=True)
    sub.add_parser("agree")
    a = ap.parse_args()
    if a.cmd == "code":
        code(a.judge)
    elif a.cmd == "agree":
        agree()


if __name__ == "__main__":
    main()
