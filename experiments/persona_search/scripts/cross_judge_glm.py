#!/usr/bin/env python3
"""Cross-judge validation: re-annotate telecom failures with GLM-5.2 and compare
to the gpt-5.4 labels. Tests whether gpt-5.4 over-attributes STATE_MISINFERENCE.

Reads gpt-5.4 labels from failure_taxonomy_annotations.jsonl (telecom only),
re-runs the same transcripts through GLM-5.2 (via the fireworks shim :4114,
blind — no persona/cond), writes GLM labels to a separate JSONL, and prints
agreement + per-judge mechanism distribution.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)

ANN = REPO / "data/persona_opt/failure_taxonomy_annotations.jsonl"
GLM_OUT = REPO / "data/persona_opt/failure_taxonomy_glm_annotations.jsonl"
TAXO = json.load(open(REPO / "data/persona_opt/failure_taxonomy.json"))
TAXO_BLOCK = "\n".join(f"  {c['code']}: {c['definition']}" for c in TAXO["codes"])
CODES = {c["code"] for c in TAXO["codes"]}
USER_CODES = {c for c in CODES if c.startswith("USER_")}

SYS = (
    "You are a JSON-only failure annotator. Output ONLY a single JSON object. "
    "No prose, no reasoning, no markdown fences. Exact format: "
    '{"primary": "CODE", "secondary": "CODE or null", '
    '"evidence": "short quote", "reasoning": "<=40 words"}.'
)


def find_sim(persona, task, cond, domain):
    if persona == "p0003":
        base = REPO / "data/simulations/fullset/telecom_p0003"
        cell = base / f"p0003_{cond}"
    else:
        base = f"{DATA_ROOT}/data/simulations/fullset/{domain}"
        cell = base / f"{persona}_{cond}"
    r = json.load(open(cell / "results.json"))
    for s in r.get("simulations", []):
        if (
            s.get("task_id") == task
            and (s.get("reward_info") or {}).get("reward", 1) <= 0
        ):
            return s
    return None


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


def call_glm(sim, ann):
    import litellm

    tr = render(sim)
    msg = (
        f"TASK: {ann.get('domain', 'telecom')}. termination={ann['termination']} nmsg={ann['n_messages']}\n"
        f"TRANSCRIPT:\n{tr}\n\nTAXONOMY:\n{TAXO_BLOCK}\n\n"
        f"Pick the PRIMARY failure mechanism (single most decisive cause) "
        f"and an optional SECONDARY. OUTPUT ONLY JSON."
    )
    resp = litellm.completion(
        model="anthropic/accounts/fireworks/models/glm-5p2",
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
        temperature=0,
        max_tokens=500,
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
    if not m:
        return {"primary": "PARSE_ERROR", "secondary": None, "evidence": txt[:200]}
    try:
        j = json.loads(m.group(0))
        j.setdefault("secondary", None)
        return j
    except Exception:
        return {
            "primary": "PARSE_ERROR",
            "secondary": None,
            "evidence": m.group(0)[:200],
        }


def main():
    anns = [json.loads(l) for l in ANN.read_text().splitlines() if l.strip()]
    # all domains now (telecom was first; resume logic skips already-done)
    todo_anns = anns
    from collections import Counter as _C

    print(
        f"all annotations (gpt-5.4): {len(todo_anns)} by domain:",
        dict(_C(a["domain"] for a in todo_anns)),
    )

    done = {}
    if GLM_OUT.exists():
        for l in GLM_OUT.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                done[(a["persona"], a["cond"], a["task"])] = a
    todo = [a for a in todo_anns if (a["persona"], a["cond"], a["task"]) not in done]
    print(f"already done: {len(done)}, to annotate: {len(todo)}")

    def one(a):
        sim = find_sim(a["persona"], a["task"], a["cond"], a["domain"])
        if sim is None:
            return a["persona"], a["cond"], a["task"], None
        j = call_glm(sim, a)
        j["persona"] = a["persona"]
        j["cond"] = a["cond"]
        j["task"] = a["task"]
        j["gpt54_primary"] = a["primary"]
        j["gpt54_secondary"] = a.get("secondary")
        j["domain"] = a["domain"]
        return a["persona"], a["cond"], a["task"], j

    with GLM_OUT.open("a") as f, ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(one, a) for a in todo]
        for i, fut in enumerate(as_completed(futs)):
            _, _, _, j = fut.result()
            if j is None:
                continue
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(todo)}]")

    # reload all GLM labels
    glm = {}
    for l in GLM_OUT.read_text().splitlines():
        if l.strip():
            a = json.loads(l)
            glm[(a["persona"], a["cond"], a["task"])] = a

    # agreement — per domain + overall
    domains = sorted({a["domain"] for a in todo_anns})
    all_pairs = []
    for dom in domains:
        pairs = [
            (a, glm[(a["persona"], a["cond"], a["task"])])
            for a in todo_anns
            if a["domain"] == dom and (a["persona"], a["cond"], a["task"]) in glm
        ]
        if not pairs:
            continue
        all_pairs.extend(pairs)
        prim_agree = sum(1 for g, l in pairs if g["primary"] == l["primary"])
        print(f"\n=== GLM vs gpt-5.4 — {dom} (n={len(pairs)}) ===")
        print(
            f"  primary exact agreement: {prim_agree}/{len(pairs)} = {100 * prim_agree / len(pairs):.1f}%"
        )

        # distribution comparison
        print(f"  --- primary distribution by judge ---")
        gptc = Counter(g["primary"] for g, _ in pairs)
        glmc = Counter(l["primary"] for _, l in pairs)
        allc = sorted(
            set(gptc) | set(glmc), key=lambda c: -(gptc.get(c, 0) + glmc.get(c, 0))
        )
        print(f"    {'code':40s} {'gpt-5.4':>8s} {'GLM-5.2':>8s}")
        for c in allc:
            print(f"    {c:40s} {gptc.get(c, 0):8d} {glmc.get(c, 0):8d}")

        # persona-driven share
        gpt_pd = sum(1 for g, _ in pairs if g["primary"] in USER_CODES)
        glm_pd = sum(1 for _, l in pairs if l["primary"] in USER_CODES)
        print(
            f"  persona-driven (primary): gpt-5.4 {100 * gpt_pd / len(pairs):.1f}% -> GLM {100 * glm_pd / len(pairs):.1f}%"
        )
        # STATE_MISINFERENCE specifically
        gpt_sm = sum(
            1 for g, _ in pairs if g["primary"] == "AGENT_FACTUAL_OR_STATE_MISINFERENCE"
        )
        glm_sm = sum(
            1 for _, l in pairs if l["primary"] == "AGENT_FACTUAL_OR_STATE_MISINFERENCE"
        )
        print(
            f"  STATE_MISINFERENCE (primary): gpt-5.4 {100 * gpt_sm / len(pairs):.1f}% -> GLM {100 * glm_sm / len(pairs):.1f}%"
        )

    # overall
    if all_pairs:
        pa = sum(1 for g, l in all_pairs if g["primary"] == l["primary"])
        print(f"\n=== OVERALL (n={len(all_pairs)}) ===")
        print(
            f"  primary exact agreement: {pa}/{len(all_pairs)} = {100 * pa / len(all_pairs):.1f}%"
        )
        gptc = Counter(g["primary"] for g, _ in all_pairs)
        glmc = Counter(l["primary"] for _, l in all_pairs)
        allc = sorted(
            set(gptc) | set(glmc), key=lambda c: -(gptc.get(c, 0) + glmc.get(c, 0))
        )
        print(f"  {'code':40s} {'gpt-5.4':>8s} {'GLM-5.2':>8s}")
        for c in allc:
            print(f"  {c:40s} {gptc.get(c, 0):8d} {glmc.get(c, 0):8d}")
        gpt_pd = sum(1 for g, _ in all_pairs if g["primary"] in USER_CODES)
        glm_pd = sum(1 for _, l in all_pairs if l["primary"] in USER_CODES)
        print(
            f"  persona-driven (primary): gpt-5.4 {100 * gpt_pd / len(all_pairs):.1f}% -> GLM {100 * glm_pd / len(all_pairs):.1f}%"
        )
        gpt_sm = sum(
            1
            for g, _ in all_pairs
            if g["primary"] == "AGENT_FACTUAL_OR_STATE_MISINFERENCE"
        )
        glm_sm = sum(
            1
            for _, l in all_pairs
            if l["primary"] == "AGENT_FACTUAL_OR_STATE_MISINFERENCE"
        )
        print(
            f"  STATE_MISINFERENCE (primary): gpt-5.4 {100 * gpt_sm / len(all_pairs):.1f}% -> GLM {100 * glm_sm / len(all_pairs):.1f}%"
        )


if __name__ == "__main__":
    main()
