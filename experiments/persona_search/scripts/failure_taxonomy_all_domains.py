#!/usr/bin/env python3
"""Comprehensive failure-mode taxonomy + annotation across ALL domains on the
gpt-5.4 fullset.

Two-phase:
  Phase 1 (discover): sample failed transcripts stratified across
    domain x persona x {seed,opt}, ask gpt-5.4 to induce a COMPREHENSIVE
    failure-mode taxonomy (codes + definitions) grounded in what it sees.
  Phase 2 (annotate): for every (domain, persona, condition) cell, take each
    unique failed task's first failed trial, annotate with the discovered
    taxonomy -> {primary, secondary, evidence, reasoning}.
  Phase 3 (report): per-domain x per-persona mode-proportion table +
    seed->opt shift.

Resumable: annotations are appended to a JSONL; already-done (domain,persona,
cond,task) keys are skipped on rerun.

Usage
-----
    .venv/bin/python experiments/persona_search/scripts/failure_taxonomy_all_domains.py discover
    .venv/bin/python experiments/persona_search/scripts/failure_taxonomy_all_domains.py annotate
    .venv/bin/python experiments/persona_search/scripts/failure_taxonomy_all_domains.py report
    # or all phases:
    .venv/bin/python experiments/persona_search/scripts/failure_taxonomy_all_domains.py all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)

ROOT = REPO / "data/simulations/fullset"
JUDGE = "gpt-5.4"

# domain -> persona keys (without _seed/_opt suffix). telecom_p0003 folded into telecom.
DOMAINS = {
    "airline": ["alaska_00", "alaska_01"],
    "retail": ["gc_02", "gc_03", "gc_04"],
    "telecom": ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"],
    "banking": ["cw_03", "cw_04"],
}
# p0003 lives in a separate dir
DOMAIN_DIR = {
    "airline": ROOT / "airline",
    "retail": ROOT / "retail",
    "telecom": ROOT / "telecom",
    "telecom_p0003": ROOT / "telecom_p0003",
    "banking": ROOT / "banking",
}

TAXO_PATH = REPO / "data/persona_opt/failure_taxonomy.json"
ANN_PATH = REPO / "data/persona_opt/failure_taxonomy_annotations.jsonl"
DISC_SAMPLE = 60  # transcripts for discovery
MAX_PER_CELL = 40  # cap failed tasks annotated per (domain,persona,cond)
DISC_MAX_MSGS = 60  # msgs per transcript in discovery (keep prompt small)
ANN_MAX_MSGS = 200  # msgs per transcript in annotation


def cell_path(domain: str, persona: str, cond: str) -> Path:
    if persona == "p0003":
        return DOMAIN_DIR["telecom_p0003"] / f"p0003_{cond}"
    return DOMAIN_DIR[domain] / f"{persona}_{cond}"


def load_results(p: Path) -> dict:
    rp = p / "results.json"
    return json.loads(rp.read_text()) if rp.exists() else {}


def failed_trials(cell: Path) -> list[dict]:
    """First failed (reward<=0) trial per task_id."""
    r = load_results(cell)
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


def render(sim: dict, max_msgs: int, max_chars: int = 700) -> str:
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


def task_ctx(task_id: str, domain: str) -> str:
    t = re.sub(r"\[PERSONA:.*?\]", "", task_id)
    head = t.replace("[", "").replace("]", "")
    parts = head.split("|")
    return (
        f"domain={domain}; issue_or_task={parts[0].strip()}; "
        f"faults_or_details={', '.join(parts[1:]) if len(parts) > 1 else '(none)'}"
    )


# --------------------------------------------------------------------------- #
def call_judge(messages: list[dict], max_tokens: int = 800) -> str:
    import litellm

    resp = litellm.completion(
        model=JUDGE, messages=messages, temperature=0, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def discover():
    """Phase 1: induce a comprehensive taxonomy from sampled failed transcripts."""
    random.seed(300)
    pool = []  # (domain, persona, cond, sim)
    for dom, personas in DOMAINS.items():
        for p in personas:
            for cond in ("opt", "seed"):
                cell = cell_path(dom, p, cond)
                for s in failed_trials(cell):
                    pool.append((dom, p, cond, s))
    sample = random.sample(pool, min(DISC_SAMPLE, len(pool)))
    print(f"discovery sample: {len(sample)} transcripts from {len(pool)} pool")

    blocks = []
    for dom, p, cond, sim in sample:
        tr = render(sim, DISC_MAX_MSGS, max_chars=500)
        blocks.append(
            f"### domain={dom} persona={p} cond={cond} "
            f"term={sim.get('termination_reason')} nmsg={len(sim.get('messages', []))}\n"
            f"task: {task_ctx(sim.get('task_id', ''), dom)}\n"
            f"transcript (truncated):\n{tr}"
        )
    prompt = (
        "You are a failure-mode taxonomist for LLM customer-service agents. "
        "Below are FAILED conversation transcripts (reward=0) across FOUR domains "
        "(airline / retail / telecom / banking) on the tau2bench framework, with a "
        "gpt-5.4 agent and persona-optimized adversarial users. The agent guides a "
        "user through multi-step tasks (telecom device troubleshooting, airline "
        "booking changes, retail order/return, banking disputes/transactions) and is "
        "scored by environment assertions on final state.\n\n"
        "Induce a COMPREHENSIVE failure-mode taxonomy that spans ALL domains. Cover "
        "both persona-driven mechanisms (the user's adversarial behaviour caused the "
        "breakdown) and agent-driven mechanisms (the agent erred regardless). Aim for "
        "8-14 distinct, mutually-exclusive-at-primary-level codes. Each code: a short "
        "UPPER_SNAKE label, a 1-line definition, and 1 tell-tale signal.\n\n"
        "TRANSCRIPTS:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n"
        'Output ONLY a JSON object: {"codes": [{"code": "...", '
        '"name": "...", "definition": "...", "signal": "..."}], '
        '"notes": "<= 60 words on coverage"}.'
    )
    raw = call_judge([{"role": "user", "content": prompt}], max_tokens=1500)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        taxo = json.loads(raw)
    except Exception:
        taxo = {"codes": [], "notes": "PARSE_ERROR", "raw": raw[:1000]}
    TAXO_PATH.write_text(json.dumps(taxo, indent=2, ensure_ascii=False))
    print(f"\nwrote taxonomy to {TAXO_PATH} ({len(taxo.get('codes', []))} codes)")
    for c in taxo.get("codes", []):
        print(f"  {c['code']:18s} {c.get('name', '')}: {c.get('definition', '')}")


def load_taxonomy() -> dict:
    return json.loads(TAXO_PATH.read_text())


def taxonomy_block(taxo: dict) -> str:
    lines = [
        "TAXONOMY (pick PRIMARY = single most decisive cause; "
        "SECONDARY = one contributing cause or null):"
    ]
    for c in taxo.get("codes", []):
        lines.append(
            f"  {c['code']:18s} {c.get('name', '')}. "
            f"{c.get('definition', '')} SIGNAL: {c.get('signal', '')}"
        )
    return "\n".join(lines)


def ann_key(a: dict) -> tuple:
    return (a["domain"], a["persona"], a["cond"], a["task"])


def load_done() -> set:
    if not ANN_PATH.exists():
        return set()
    return {
        ann_key(json.loads(l)) for l in ANN_PATH.read_text().splitlines() if l.strip()
    }


def annotate():
    """Phase 2: annotate all failed (persona,task) pairs with discovered taxonomy."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    taxo = load_taxonomy()
    taxo_txt = taxonomy_block(taxo)
    valid_codes = {c["code"] for c in taxo.get("codes", [])}
    done = load_done()
    print(f"already annotated: {len(done)}; valid codes: {sorted(valid_codes)}")

    pairs = []
    for dom, personas in DOMAINS.items():
        for p in personas:
            for cond in ("opt", "seed"):
                cell = cell_path(dom, p, cond)
                trials = failed_trials(cell)
                if len(trials) > MAX_PER_CELL:
                    trials = random.Random(300).sample(trials, MAX_PER_CELL)
                for s in trials:
                    pairs.append((dom, p, cond, s))
    todo = [
        (d, p, c, s)
        for (d, p, c, s) in pairs
        if (d, p, c, s.get("task_id")) not in done
    ]
    print(f"to annotate: {len(todo)} / {len(pairs)} total")

    lock = threading.Lock()

    def one(args):
        i, (dom, p, cond, sim) = args
        tr = render(sim, ANN_MAX_MSGS)
        ctx = task_ctx(sim.get("task_id", ""), dom)
        term = sim.get("termination_reason", "?")
        nmsg = len(sim.get("messages", []))
        msg = (
            f"TASK: {ctx}\nTERMINATION: {term} (n_messages={nmsg})\n\n"
            f"TRANSCRIPT:\n{tr}\n\n{taxo_txt}\n\n"
            f"OUTPUT strictly JSON: "
            f'{{"primary": "<CODE>", "secondary": "<CODE or null>", '
            f'"evidence": "<short quote/paraphrase>", '
            f'"reasoning": "<=40 words"}}'
        )
        raw = call_judge([{"role": "user", "content": msg}], max_tokens=300)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            a = json.loads(raw)
        except Exception:
            a = {
                "primary": "PARSE_ERROR",
                "secondary": None,
                "evidence": raw[:200],
                "reasoning": "",
            }
        a["domain"] = dom
        a["persona"] = p
        a["cond"] = cond
        a["task"] = sim.get("task_id")
        a["termination"] = term
        a["n_messages"] = nmsg
        return i, a

    CONCURRENCY = 20
    n_done = 0
    with ANN_PATH.open("a") as f, ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one, (i, x)) for i, x in enumerate(todo)]
        for fut in as_completed(futs):
            i, a = fut.result()
            with lock:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                print(
                    f"[{n_done}/{len(todo)}] {a['domain']}/{a['persona']}/{a['cond']}: "
                    f"{a.get('primary')} (+{a.get('secondary')})  term={a.get('termination')}"
                )


def report():
    """Phase 3: per-domain x per-persona mode proportions + seed->opt shift."""
    if not ANN_PATH.exists():
        print("no annotations; run annotate first")
        return
    anns = [json.loads(l) for l in ANN_PATH.read_text().splitlines() if l.strip()]
    print(f"total annotations: {len(anns)}\n")

    # overall distribution
    print("=== OVERALL primary-mode distribution ===")
    cnt = Counter(a["primary"] for a in anns)
    for code, n in cnt.most_common():
        print(f"  {code:18s} {n:4d}  {100 * n / len(anns):5.1f}%")
    print()

    # per domain x persona x cond
    print("=== per (domain, persona, cond) primary-mode proportion ===")
    cells = defaultdict(list)
    for a in anns:
        cells[(a["domain"], a["persona"], a["cond"])].append(a["primary"])
    for dom in DOMAINS:
        print(f"\n--- {dom} ---")
        for p in DOMAINS[dom]:
            for cond in ("seed", "opt"):
                lst = cells.get((dom, p, cond), [])
                if not lst:
                    continue
                c = Counter(lst)
                top = ", ".join(
                    f"{k}={100 * v / len(lst):.0f}%" for k, v in c.most_common(3)
                )
                print(f"  {p:14s} {cond:4s} n={len(lst):3d}  {top}")

    # seed->opt shift per persona
    print("\n=== seed->opt mode shift (primary, top modes) ===")
    for dom in DOMAINS:
        for p in DOMAINS[dom]:
            s = Counter(
                a["primary"]
                for a in anns
                if a["domain"] == dom and a["persona"] == p and a["cond"] == "seed"
            )
            o = Counter(
                a["primary"]
                for a in anns
                if a["domain"] == dom and a["persona"] == p and a["cond"] == "opt"
            )
            if not s and not o:
                continue
            modes = sorted(set(s) | set(o), key=lambda m: -(o.get(m, 0) + s.get(m, 0)))
            line = " ".join(f"{m}:{s.get(m, 0)}→{o.get(m, 0)}" for m in modes[:5])
            print(f"  {dom:8s} {p:14s} {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["discover", "annotate", "report", "all"])
    args = ap.parse_args()
    if args.phase in ("discover", "all"):
        discover()
    if args.phase in ("annotate", "all"):
        annotate()
    if args.phase in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
