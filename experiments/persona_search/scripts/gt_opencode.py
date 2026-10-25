#!/usr/bin/env python3
"""Grounded-Theory open coding of agent failures (full GT, taxonomy-as-contribution).

Stage 0 — theoretical sampling: stratified across domain x persona x {seed,opt},
          comprehensive coverage of every cell, in small batches.
Stage 1 — open coding (BLIND): LLM judge reads each FAILED transcript and names
          the decisive error as an observed PHENOMENON (short noun phrase), NOT
          fitted to any preset category. No taxonomy, no code list, no
          persona-driven/agent-driven prior is shown — categories must emerge.
          Up to 3 phenomena per transcript. Two judges (gpt-5.4, GLM-5.2) code
          independently (anti-bias + feeds IAA later).

Anti-假GT discipline: the system prompt never mentions the old 12 codes, never
asks to "classify", and forces phenomenon-level description with the decisive
moment quoted. If the emerging structure differs from the old taxonomy, that is
a finding, not a failure.

Outputs:
  data/persona_opt/gt_opencode_samples.jsonl   — sample manifest (per trace)
  data/persona_opt/gt_opencode_<judge>.jsonl    — open-code results (resumable)

Usage:
  .venv/bin/python experiments/persona_search/scripts/gt_opencode.py sample --per-cell 2 --batch 1
  .venv/bin/python experiments/persona_search/scripts/gt_opencode.py code --batch 1 --judge gpt54
  .venv/bin/python experiments/persona_search/scripts/gt_opencode.py code --batch 1 --judge glm
  .venv/bin/python experiments/persona_search/scripts/gt_opencode.py status
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

SAMPLES_PATH = REPO / "data/persona_opt/gt_opencode_samples.jsonl"
OUT_DIR = REPO / "data/persona_opt"

JUDGES = {
    "gpt54": {"model": "gpt-5.4", "via": "litellm"},
    "glm": {
        "model": "anthropic/accounts/fireworks/models/glm-5p2",
        "via": "shim",
        "base": "http://127.0.0.1:4114",
        "key_env": "CRESTA_VERTEX_TOKEN",
    },
}


def cell_path(domain: str, persona: str, cond: str) -> Path:
    if persona == "p0003":
        return DOMAIN_DIR["telecom_p0003"] / f"p0003_{cond}"
    return DOMAIN_DIR[domain] / f"{persona}_{cond}"


def load_results(p: Path) -> dict:
    rp = p / "results.json"
    return json.loads(rp.read_text()) if rp.exists() else {}


def failed_trials(cell: Path) -> list[dict]:
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
            out.append(f"{i:03d} [{role}] {c}")
    return "\n".join(out)


def find_sim(domain: str, persona: str, cond: str, task: str) -> dict | None:
    for s in failed_trials(cell_path(domain, persona, cond)):
        if s.get("task_id") == task:
            return s
    return None


# --------------------------------------------------------------------------- #
# Stage 0: stratified sampling


def sample(per_cell: int, batch: int, seed: int = 300):
    """Sample per_cell failed tasks from EVERY (domain,persona,cond) cell."""
    rng = random.Random(seed + batch)  # different batches draw different traces
    existing = set()
    if SAMPLES_PATH.exists():
        for l in SAMPLES_PATH.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                existing.add((a["domain"], a["persona"], a["cond"], a["task"]))
    drawn = []
    for dom, personas in DOMAINS.items():
        for p in personas:
            for cond in ("seed", "opt"):
                trials = failed_trials(cell_path(dom, p, cond))
                # exclude already-sampled traces so batches don't overlap
                fresh = [
                    s
                    for s in trials
                    if (dom, p, cond, s.get("task_id")) not in existing
                ]
                rng.shuffle(fresh)
                pick = fresh[:per_cell]
                for s in pick:
                    drawn.append(
                        {
                            "batch": batch,
                            "domain": dom,
                            "persona": p,
                            "cond": cond,
                            "task": s.get("task_id"),
                            "termination": s.get("termination_reason"),
                            "n_messages": len(s.get("messages", [])),
                        }
                    )
    with SAMPLES_PATH.open("a") as f:
        for d in drawn:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # coverage report
    cells = {}
    for d in drawn:
        cells.setdefault((d["domain"], d["persona"], d["cond"]), 0)
        cells[(d["domain"], d["persona"], d["cond"])] += 1
    print(
        f"batch {batch}: sampled {len(drawn)} traces across {len(cells)} cells "
        f"(target per_cell={per_cell})"
    )
    missing = [
        (d, p, c)
        for d in DOMAINS
        for p in DOMAINS[d]
        for c in ("seed", "opt")
        if (d, p, c) not in cells
    ]
    if missing:
        print(f"  WARNING: {len(missing)} cells with no failures: {missing}")
    for d in DOMAINS:
        for p in DOMAINS[d]:
            for c in ("seed", "opt"):
                if (d, p, c) in cells:
                    print(f"    {d:8s} {p:12s} {c:4s}  +{cells[(d, p, c)]}")


def load_samples(batch: int | None = None) -> list[dict]:
    if not SAMPLES_PATH.exists():
        return []
    out = []
    for l in SAMPLES_PATH.read_text().splitlines():
        if l.strip():
            a = json.loads(l)
            if batch is None or a["batch"] == batch:
                out.append(a)
    return out


# --------------------------------------------------------------------------- #
# Stage 1: blind open coding

# THE anti-假GT prompt: no taxonomy, no code list, no persona/agent prior,
# force phenomenon-level description with the decisive moment quoted.
SYS = (
    "You are analyzing a FAILED conversation between an LLM customer-service "
    "agent and a user. The conversation did not reach the required final state. "
    "Your task is OPEN CODING: describe what went wrong as concretely as possible, "
    "in terms of what you actually observe in the transcript.\n\n"
    "RULES:\n"
    "1. Do NOT use any predefined category, checklist, or taxonomy. There is no "
    "list to choose from. Do NOT 'classify' the failure into a type.\n"
    "2. Name each decisive error as a PHENOMENON — a short concrete noun phrase "
    "describing the specific thing that happened (e.g. 'agent ran a tool that "
    "contradicted the state it had just read', 'user withheld the required ID and "
    "agent ended the call without resolving', 'agent cancelled the old booking "
    "before confirming the new one was available'). Describe the behaviour, not a "
    "category label.\n"
    "3. A failure may have up to 3 contributing phenomena; list each separately. "
    "Order them by how decisive they were (most decisive first).\n"
    "4. For each phenomenon, give: a short label (<=12 words), the single most "
    "decisive moment (quote or paraphrase WITH the message index, e.g. msg 014 "
    "agent says it will cancel despite user only asking to check), and one line "
    "on why it was decisive for the failure.\n"
    "5. Do not mention 'persona-driven', 'agent-driven', or any prior scheme. "
    "Just describe what you see.\n\n"
    'OUTPUT ONLY a JSON object: {"phenomena": [{"label": "...", '
    '"moment": "...", "why_decisive": "..."}]}. No prose, no markdown.'
)


def call(judge: str, user_msg: str, max_tokens: int = 700) -> str:
    import litellm

    j = JUDGES[judge]
    if j["via"] == "shim":
        token = os.environ.get(j["key_env"], "")
        resp = litellm.completion(
            model=j["model"],
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=max_tokens,
            api_base=j["base"],
            api_key=token,
        )
    else:
        resp = litellm.completion(
            model=j["model"],
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
    return resp.choices[0].message.content.strip()


def open_code_one(judge: str, sample: dict) -> dict:
    sim = find_sim(sample["domain"], sample["persona"], sample["cond"], sample["task"])
    if sim is None:
        return {**sample, "judge": judge, "phenomena": [], "error": "sim_not_found"}
    tr = render(sim)
    term = sim.get("termination_reason", "?")
    nmsg = len(sim.get("messages", []))
    # task context: strip persona tag, keep issue description only (no persona leak)
    tid = re.sub(r"\[PERSONA:.*?\]", "", sim.get("task_id", ""))
    user_msg = (
        f"DOMAIN: {sample['domain']}\nTERMINATION: {term} (n_messages={nmsg})\n"
        f"TASK: {tid}\n\nTRANSCRIPT:\n{tr}\n\n"
        f"Open-code the decisive failure phenomenon/tha. Output only the JSON."
    )
    raw = call(judge, user_msg)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        j = json.loads(raw)
        if "phenomena" not in j:
            j = (
                {"phenomena": j}
                if isinstance(j, list)
                else {"phenomena": [], "raw": raw[:300]}
            )
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            j = json.loads(m.group(0)) if m else {"phenomena": []}
        except Exception:
            j = {"phenomena": [], "parse_error": raw[:300]}
    return {
        "batch": sample["batch"],
        "domain": sample["domain"],
        "persona": sample["persona"],
        "cond": sample["cond"],
        "task": sample["task"],
        "termination": term,
        "n_messages": nmsg,
        "judge": judge,
        "phenomena": j.get("phenomena", []),
    }


def code(batch: int, judge: str, concurrency: int = 20):
    out_path = OUT_DIR / f"gt_opencode_{judge}.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                a = json.loads(l)
                if a["batch"] == batch:
                    done.add((a["domain"], a["persona"], a["cond"], a["task"]))
    samples = [
        s
        for s in load_samples(batch)
        if (s["domain"], s["persona"], s["cond"], s["task"]) not in done
    ]
    print(
        f"judge={judge} batch={batch}: {len(samples)} to code "
        f"({len(done)} already done)"
    )

    def one(s):
        return open_code_one(judge, s)

    n = 0
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one, s) for s in samples]
        for fut in as_completed(futs):
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            labels = [p.get("label", "?")[:40] for p in r.get("phenomena", [])[:2]]
            print(
                f"[{n}/{len(samples)}] {r['domain']}/{r['persona']}/{r['cond']} "
                f"term={r.get('termination')}: {labels}"
            )


def status():
    samples = load_samples()
    by_batch = {}
    for s in samples:
        by_batch.setdefault(s["batch"], 0)
        by_batch[s["batch"]] += 1
    print(f"samples: {len(samples)} total, by batch: {dict(sorted(by_batch.items()))}")
    cells = set((s["domain"], s["persona"], s["cond"]) for s in samples)
    total_cells = sum(len(ps) * 2 for ps in DOMAINS.values())
    print(f"cells covered: {len(cells)}/{total_cells}")
    for judge in JUDGES:
        p = OUT_DIR / f"gt_opencode_{judge}.jsonl"
        n = sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0
        print(f"  {judge}: {n} open-coded")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_sample = sub.add_parser("sample")
    s_sample.add_argument("--per-cell", type=int, default=2)
    s_sample.add_argument("--batch", type=int, required=True)
    s_sample.add_argument("--seed", type=int, default=300)
    s_code = sub.add_parser("code")
    s_code.add_argument("--batch", type=int, required=True)
    s_code.add_argument("--judge", choices=list(JUDGES), required=True)
    s_code.add_argument("--concurrency", type=int, default=20)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.per_cell, args.batch, args.seed)
    elif args.cmd == "code":
        code(args.batch, args.judge, args.concurrency)
    elif args.cmd == "status":
        status()


if __name__ == "__main__":
    main()
