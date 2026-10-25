#!/usr/bin/env python3
"""LLM root-cause annotation for seed-pass / opt-fail flip cases.

For each flip (seed avg reward > 0.5, opt avg reward <= 0.5), picks ONE failed
opt trial and ONE passed seed trial for the same task, sends both transcripts to
an LLM judge, and asks for:
  - decisive_step: the single most decisive moment (msg index + quote)
  - root_cause: short label (<=15 words)
  - analysis: full explanation of why opt failed but seed passed
  - attribution: "persona" | "agent" | "task" | "noise"

Data sources:
  - telecom/retail/banking: fullset test set, 3 trials
  - airline: alaska_03 v5 GEPA val set, 2 trials (seed eval_0003 + opt gen10 full_diff)

Outputs per-conversation JSON to data/persona_opt/rootcause_flips/<domain>/<island>__<task_clean>.json

Usage:
  .venv/bin/python experiments/persona_search/scripts/rootcause_flips.py --dry-run
  .venv/bin/python experiments/persona_search/scripts/rootcause_flips.py
  .venv/bin/python experiments/persona_search/scripts/rootcause_flips.py --domain telecom
  .venv/bin/python experiments/persona_search/scripts/rootcause_flips.py --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)

FULLSET = REPO / "data/simulations/fullset"
GEPA = REPO / "data/simulations/gepa_opt"
OUT_DIR = REPO / "data/persona_opt/rootcause_flips"

SPLITS = {
    "telecom": "experiments/persona_search/configs/telecom_island_split.json",
    "retail": "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
    "banking": "experiments/persona_search/configs/banking_knowledge_split_v2.json",
}

FULLSET_ISLANDS = {
    "telecom": ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"],
    "retail": ["gc_02", "gc_03", "gc_04"],
    "banking": ["cw_03", "cw_04"],
}


def strip_persona(tid: str) -> str:
    return re.sub(r"\[PERSONA:[^\]]*\]", "", tid).strip()


def load_test_ids(domain: str) -> set[str]:
    sp = json.loads(Path(SPLITS[domain]).read_text())
    raw = sp.get("test", {})
    key = domain if domain in raw else next(iter(raw))
    return {strip_persona(t) for t in raw[key]}


def _results_dir(domain: str, island: str, cond: str) -> Path:
    if island == "p0003":
        return FULLSET / "telecom_p0003" / f"p0003_{cond}"
    return FULLSET / domain / f"{island}_{cond}"


def _load_sims(path: Path) -> list[dict]:
    rj = path / "results.json"
    if not rj.exists():
        return []
    return json.loads(rj.read_text()).get("simulations", [])


def _task_trials(sims: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for s in sims:
        clean = strip_persona(s.get("task_id", ""))
        out[clean].append(s)
    return dict(out)


# ---- Airline: GEPA val data ----


def _airline_sims() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Return (seed_by_task, opt_by_task) for alaska_03 v5 val set."""
    v5 = json.loads(
        Path(
            "experiments/persona_search/configs/airline_alaska_split_v5.json"
        ).read_text()
    )
    val_ids = set(v5["val"]["airline"])

    seed_rj = GEPA / "alaska_03_v5_g00_seed/eval_0003/airline/results.json"
    opt_rj = GEPA / "alaska_03_v5_g10_k00_full_diff/eval_0001/airline/results.json"

    seed_by_task: dict[str, list[dict]] = defaultdict(list)
    for s in json.loads(seed_rj.read_text()).get("simulations", []):
        if s["task_id"] in val_ids:
            seed_by_task[s["task_id"]].append(s)

    opt_by_task: dict[str, list[dict]] = defaultdict(list)
    for s in json.loads(opt_rj.read_text()).get("simulations", []):
        if s["task_id"] in val_ids:
            opt_by_task[s["task_id"]].append(s)

    return dict(seed_by_task), dict(opt_by_task)


# ---- Collect all flip cases ----


def collect_flips(domains: list[str] | None = None) -> list[dict]:
    """Return list of flip case dicts with seed_sim and opt_sim attached."""
    targets = domains or ["telecom", "retail", "banking", "airline"]
    flips = []

    for dom in targets:
        if dom == "airline":
            seed_bt, opt_bt = _airline_sims()
            for tid in sorted(seed_bt):
                sr = [
                    s.get("reward_info", {}).get("reward", 0)
                    for s in seed_bt[tid]
                    if s.get("reward_info")
                ]
                opr = [
                    s.get("reward_info", {}).get("reward", 0)
                    for s in opt_bt.get(tid, [])
                    if s.get("reward_info")
                ]
                if not opr:
                    continue
                sa, oa = sum(sr) / len(sr), sum(opr) / len(opr)
                if sa > 0.5 and oa <= 0.5:
                    passed = [
                        s
                        for s in seed_bt[tid]
                        if s.get("reward_info", {}).get("reward", 0) > 0
                    ]
                    failed = [
                        s
                        for s in opt_bt[tid]
                        if s.get("reward_info", {}).get("reward", 0) == 0
                    ]
                    if not passed or not failed:
                        continue
                    flips.append(
                        {
                            "domain": "airline",
                            "island": "alaska_03",
                            "task_id": tid,
                            "seed_avg": sa,
                            "opt_avg": oa,
                            "seed_sim": passed[0],
                            "opt_sim": failed[0],
                            "source": "gepa_v5_val",
                        }
                    )
            continue

        test_ids = load_test_ids(dom)
        for isl in FULLSET_ISLANDS[dom]:
            seed_sims = _task_trials(_load_sims(_results_dir(dom, isl, "seed")))
            opt_sims = _task_trials(_load_sims(_results_dir(dom, isl, "opt")))

            for tid in seed_sims:
                if tid not in test_ids:
                    continue
                sr = [
                    s.get("reward_info", {}).get("reward", 0)
                    for s in seed_sims[tid]
                    if s.get("reward_info")
                ]
                opr = [
                    s.get("reward_info", {}).get("reward", 0)
                    for s in opt_sims.get(tid, [])
                    if s.get("reward_info")
                ]
                if not opr:
                    continue
                sa, oa = sum(sr) / len(sr), sum(opr) / len(opr)
                if sa > 0.5 and oa <= 0.5:
                    passed = [
                        s
                        for s in seed_sims[tid]
                        if s.get("reward_info", {}).get("reward", 0) > 0
                    ]
                    failed = [
                        s
                        for s in opt_sims[tid]
                        if s.get("reward_info", {}).get("reward", 0) == 0
                    ]
                    if not passed or not failed:
                        continue
                    flips.append(
                        {
                            "domain": dom,
                            "island": isl,
                            "task_id": tid,
                            "seed_avg": sa,
                            "opt_avg": oa,
                            "seed_sim": passed[0],
                            "opt_sim": failed[0],
                            "source": "fullset_test",
                        }
                    )

    return flips


# ---- Render transcript ----


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


# ---- LLM judge ----

SYS_PROMPT = """\
You are analyzing a PAIRED comparison of two conversations for the SAME task.

CONVERSATION A (SEED — PASSED): The user had a "seed" persona (baseline). The agent succeeded.
CONVERSATION B (OPT — FAILED): The user had an "optimized" adversarial persona. The agent failed.

Your job: determine WHY the optimized persona caused the agent to fail, when the seed persona did not.

INSTRUCTIONS:
1. Read both transcripts carefully. Focus on what the OPT user did differently from the SEED user, and how the agent responded differently.
2. Identify the DECISIVE STEP: the single moment in Conversation B where the trajectory diverged from success. Quote the message index and content.
3. Identify the ROOT CAUSE: a short label (<=15 words) for what fundamentally went wrong.
4. Write a full ANALYSIS (3-8 sentences): explain the causal chain from persona behavior → agent failure. What specific user behavior (withholding info, vague references, drip-feeding, etc.) triggered what specific agent mistake?
5. ATTRIBUTION: who is primarily responsible?
   - "persona": the persona's adversarial behavior directly caused the failure (the agent would have succeeded with normal user behavior)
   - "agent": the agent made an independent error that the persona merely exposed (a competent agent would have handled the persona's behavior)
   - "coupled": the failure required BOTH the persona's adversarial behavior AND an agent weakness (neither alone would have caused failure)
   - "noise": the failure appears random / not reproducibly caused by the persona

OUTPUT a JSON object:
{
  "decisive_step": {"msg_index": <int>, "role": "<user|assistant>", "quote": "<key excerpt>", "explanation": "<why this was the point of no return>"},
  "root_cause": "<short label, <=15 words>",
  "analysis": "<3-8 sentence explanation>",
  "attribution": "<persona|agent|coupled|noise>",
  "seed_strategy": "<what the seed user did that let the agent succeed, 1-2 sentences>",
  "opt_strategy": "<what the opt user did differently that caused failure, 1-2 sentences>"
}

Output ONLY the JSON. No markdown, no commentary."""


def call_judge(user_msg: str, max_tokens: int = 2000) -> str:
    import litellm

    resp = litellm.completion(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def annotate_one(flip: dict) -> dict:
    seed_tr = render(flip["seed_sim"])
    opt_tr = render(flip["opt_sim"])
    tid = strip_persona(flip["task_id"])

    user_msg = (
        f"DOMAIN: {flip['domain']}\n"
        f"TASK: {tid}\n"
        f"SEED avg reward: {flip['seed_avg']:.2f}  |  OPT avg reward: {flip['opt_avg']:.2f}\n\n"
        f"=== CONVERSATION A (SEED — PASSED) ===\n{seed_tr}\n\n"
        f"=== CONVERSATION B (OPT — FAILED) ===\n{opt_tr}\n\n"
        f"Analyze why Conversation B failed. Output only the JSON."
    )

    raw = call_judge(user_msg)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            result = json.loads(m.group(0)) if m else {"parse_error": raw[:500]}
        except Exception:
            result = {"parse_error": raw[:500]}

    return {
        "domain": flip["domain"],
        "island": flip["island"],
        "task_id": tid,
        "seed_avg": flip["seed_avg"],
        "opt_avg": flip["opt_avg"],
        "source": flip["source"],
        "seed_n_messages": len(flip["seed_sim"].get("messages", [])),
        "opt_n_messages": len(flip["opt_sim"].get("messages", [])),
        "opt_termination": flip["opt_sim"].get("termination_reason"),
        **result,
    }


def out_path(flip: dict) -> Path:
    import hashlib

    tid_clean = strip_persona(flip["task_id"])
    tid_safe = re.sub(r"[^\w\-]", "_", tid_clean)[:100]
    h = hashlib.md5(tid_clean.encode()).hexdigest()[:8]
    return OUT_DIR / flip["domain"] / f"{flip['island']}__{tid_safe}_{h}.json"


def run(domains: list[str] | None, concurrency: int, dry: bool):
    flips = collect_flips(domains)
    todo = [f for f in flips if not out_path(f).exists()]
    print(
        f"Total flips: {len(flips)}, already done: {len(flips) - len(todo)}, "
        f"to annotate: {len(todo)}"
    )

    if dry:
        for f in flips:
            done = "✓" if out_path(f).exists() else " "
            print(
                f"  [{done}] {f['domain']}/{f['island']}/{strip_persona(f['task_id'])[:60]} "
                f"seed={f['seed_avg']:.2f} opt={f['opt_avg']:.2f}"
            )
        return

    for dom in set(f["domain"] for f in todo):
        (OUT_DIR / dom).mkdir(parents=True, exist_ok=True)

    done_count = 0

    def _do(f):
        return annotate_one(f), f

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, f) for f in todo]
        for fut in as_completed(futs):
            result, flip = fut.result()
            op = out_path(flip)
            op.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            done_count += 1
            rc = result.get("root_cause", "?")[:50]
            attr = result.get("attribution", "?")
            print(
                f"[{done_count}/{len(todo)}] {flip['domain']}/{flip['island']}/"
                f"{strip_persona(flip['task_id'])[:40]} → {attr}: {rc}"
            )


def summary():
    """Print summary of completed annotations."""
    by_domain = defaultdict(list)
    for jf in OUT_DIR.rglob("*.json"):
        data = json.loads(jf.read_text())
        by_domain[data.get("domain", "?")].append(data)

    print(f"\n{'=' * 60}")
    print(f"Root Cause Annotation Summary")
    print(f"{'=' * 60}")

    attr_total = defaultdict(int)
    for dom in sorted(by_domain):
        items = by_domain[dom]
        attrs = defaultdict(int)
        for d in items:
            a = d.get("attribution", "unknown")
            attrs[a] += 1
            attr_total[a] += 1
        attr_str = " ".join(f"{k}={v}" for k, v in sorted(attrs.items()))
        print(f"\n{dom} ({len(items)} flips): {attr_str}")
        for d in sorted(items, key=lambda x: x.get("task_id", "")):
            rc = d.get("root_cause", "?")[:60]
            a = d.get("attribution", "?")
            print(f"  {d['island']}/{d['task_id'][:50]}: [{a}] {rc}")

    print(f"\n{'─' * 40}")
    total = sum(attr_total.values())
    print(f"Total: {total}")
    for k, v in sorted(attr_total.items()):
        print(f"  {k}: {v} ({v / total * 100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="*", help="restrict to specific domains")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--summary", action="store_true", help="print summary of done annotations"
    )
    args = ap.parse_args()

    if args.summary:
        summary()
        return

    run(args.domain, args.concurrency, args.dry_run)


if __name__ == "__main__":
    main()
