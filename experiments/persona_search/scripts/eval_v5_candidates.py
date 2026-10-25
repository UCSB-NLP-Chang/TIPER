#!/usr/bin/env python3
"""Evaluate v5 GEPA seeds + top mutated personas on the test split.

Reads persona texts directly from the v5 pgepa_state.json, writes them to
temp skill files, then runs tau2 across all 3 domains.

Usage:
    # full test split (134 tasks)
    .venv/bin/python experiments/persona_search/scripts/eval_v5_candidates.py

    # quick smoke (5 tasks per domain)
    .venv/bin/python experiments/persona_search/scripts/eval_v5_candidates.py --max-tasks 5

    # only specific personas
    .venv/bin/python experiments/persona_search/scripts/eval_v5_candidates.py --only-idx 0 6 12

    # custom split
    .venv/bin/python experiments/persona_search/scripts/eval_v5_candidates.py --split val
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA_ROOT,
    REPO_ROOT,
    SIMULATIONS_DIR,
    is_infra_broken,
    read_json,
    write_json,
)

import tau2.utils.cresta_vertex  # noqa: E402, F401

PERSONA_OPT_DIR = DATA_ROOT / "data" / "persona_opt"
STATE_FILE = PERSONA_OPT_DIR / "pgepa_runs" / "xdom_v5_3domain" / "pgepa_state.json"
SPLIT_FILE = PERSONA_OPT_DIR / "crossdomain_variance_split_v3.json"

ALL_DOMAINS = {
    "telecom": "telecom",
    "airline": "airline",
    "retail": "retail",
}


# ---------------------------------------------------------------------------
# Persona selection
# ---------------------------------------------------------------------------


def load_personas(only_idx: list[int] | None) -> list[dict]:
    """Return list of {label, idx, gen, parent, text} sorted by label."""
    state = json.loads(STATE_FILE.read_text())
    candidates = state["candidates"]

    dom_ids = defaultdict(list)
    for iid in state["instance_ids"]:
        dom_ids[iid.split("::")[0]].append(iid)

    personas = []
    for c in candidates:
        idx = c["idx"]
        if only_idx and idx not in only_idx:
            continue
        gen = c.get("gen", 0)
        label = f"seed{idx}" if gen == 0 else f"idx{idx}"
        personas.append(
            {
                "label": label,
                "idx": idx,
                "gen": gen,
                "parent": c.get("parent"),
                "text": c["persona_text"],
                "train_agg": c.get("agg"),
            }
        )

    personas.sort(key=lambda p: (p["gen"], p["idx"]))
    return personas


def select_default(personas: list[dict]) -> list[dict]:
    """Seeds (gen=0) + top 5 mutated by train_agg."""
    seeds = [p for p in personas if p["gen"] == 0]
    mutated = sorted(
        [p for p in personas if p["gen"] > 0 and p["train_agg"] is not None],
        key=lambda p: -(p["train_agg"] or 0),
    )[:5]
    seen = {p["idx"] for p in seeds + mutated}
    return [p for p in personas if p["idx"] in seen]


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(split: str, max_tasks: int) -> dict[str, list[str]]:
    data = json.loads(SPLIT_FILE.read_text())
    split_data = data[split]
    out = {}
    for domain_key, task_ids in split_data.items():
        ids = list(task_ids) if not isinstance(task_ids, list) else task_ids
        if max_tasks:
            ids = ids[:max_tasks]
        out[domain_key] = ids
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _build_cmd(
    args, domain_key: str, task_ids: list[str], run_name: str, persona_text: str
) -> list[str]:
    domain = ALL_DOMAINS[domain_key]
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "tau2"),
        "run",
        "--domain",
        domain,
        "--user",
        args.user_type,
        "--user-llm",
        args.user_llm,
        "--user-llm-args",
        "{}",
        "--user-persona",
        json.dumps({"custom_guidelines": persona_text}),
        "--agent-llm",
        args.agent_llm,
        "--agent-llm-args",
        args.agent_llm_args,
        "--num-trials",
        str(args.num_trials),
        "--max-steps",
        str(args.max_steps),
        "--timeout",
        str(args.sim_timeout),
        "--task-ids",
        *task_ids,
        "--save-to",
        run_name,
        "--max-concurrency",
        str(args.max_concurrency),
        "--auto-resume",
    ]
    if domain == "banking_knowledge":
        cmd += ["--retrieval-config", args.retrieval_config]
    return cmd


def _run_domain(
    args, domain_key: str, task_ids: list[str], run_name: str, persona_text: str
) -> Path:
    rp = SIMULATIONS_DIR / run_name / "results.json"
    cmd = _build_cmd(args, domain_key, task_ids, run_name, persona_text)
    print(f"  [{domain_key}] {len(task_ids)} tasks -> {run_name}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    for attempt in range(1, args.max_infra_retries + 1):
        if not rp.exists():
            break
        data = read_json(rp)
        broken = [s for s in data.get("simulations", []) if is_infra_broken(s)]
        if not broken:
            break
        broken_ids = sorted({s["task_id"] for s in broken})
        print(
            f"  [{domain_key}] infra-retry {attempt}: {len(broken)} sims over {broken_ids}"
        )
        data["simulations"] = [s for s in data["simulations"] if not is_infra_broken(s)]
        write_json(rp, data)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return rp


def _aggregate(rp: Path) -> tuple[float | None, int, int]:
    """Returns (mean_reward, n_scored, n_infra)."""
    if not rp.exists():
        return None, 0, 0
    data = read_json(rp)
    rewards, infra = [], 0
    for s in data.get("simulations", []):
        if is_infra_broken(s):
            infra += 1
            continue
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        tr = s.get("termination_reason", "")
        if tr in ("timeout", "max_steps"):
            continue
        rewards.append(float(r))
    mean = sum(rewards) / len(rewards) if rewards else None
    return mean, len(rewards), infra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    all_personas = load_personas(args.only_idx)
    if args.only_idx:
        personas = all_personas
    else:
        personas = select_default(all_personas)

    tasks = load_tasks(args.split, args.max_tasks)
    total_tasks = sum(len(v) for v in tasks.values())

    print(
        f"eval_v5_candidates: {len(personas)} personas x {total_tasks} tasks "
        f"({args.split} split, {'+'.join(f'{len(v)}{k[0].upper()}' for k, v in sorted(tasks.items()))})"
    )
    print(
        f"  agent={args.agent_llm}  user={args.user_type}/{args.user_llm}"
        f"  trials={args.num_trials}  concurrency={args.max_concurrency}\n"
    )

    results: dict[
        str, dict[str, float | None]
    ] = {}  # label -> domain_key -> mean_reward

    for p in personas:
        label = p["label"]
        train_str = f"{p['train_agg']:.3f}" if p["train_agg"] is not None else "—"
        print(
            f"[{label}]  train_agg={train_str}  gen={p['gen']}  parent={p['parent']}",
            flush=True,
        )

        dom_results: dict[str, float | None] = {}
        for domain_key, task_ids in sorted(tasks.items()):
            run_name = f"gepa_opt/v5_test/{label}_{domain_key}"
            try:
                rp = _run_domain(args, domain_key, task_ids, run_name, p["text"])
                mean, n_scored, n_infra = _aggregate(rp)
                dom_results[domain_key] = mean
                mean_str = f"{mean:.3f}" if mean is not None else "—"
                print(
                    f"  [{domain_key}] reward={mean_str}  n={n_scored}  infra={n_infra}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{domain_key}] ERROR: {e}", flush=True)
                dom_results[domain_key] = None

        valid = [v for v in dom_results.values() if v is not None]
        agg = sum(valid) / len(valid) if valid else None
        dom_results["_agg"] = agg
        results[label] = dom_results
        agg_str = f"{agg:.3f}" if agg is not None else "—"
        print(f"  => agg={agg_str}\n", flush=True)

    # Summary table

    domains = sorted(tasks.keys())
    header = (
        f"{'label':<10} {'train':>6}  "
        + "  ".join(f"{d[:7]:>7}" for d in domains)
        + f"  {'agg':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for p in personas:
        label = p["label"]
        r = results.get(label, {})
        train_str = (
            f"{p['train_agg']:>6.3f}" if p["train_agg"] is not None else f"{'—':>6}"
        )
        row = f"{label:<10} {train_str}  "
        for d in domains:
            v = r.get(d)
            row += f"  {v:>7.3f}" if v is not None else f"  {'—':>7}"
        agg = r.get("_agg")
        row += f"  {agg:>6.3f}" if agg is not None else f"  {'—':>6}"
        print(row)
    print("=" * len(header))

    # Save
    out_path = (
        PERSONA_OPT_DIR
        / "pgepa_runs"
        / "xdom_v5_3domain"
        / f"test_results_{args.split}.json"
    )
    write_json(
        out_path,
        {
            "split": args.split,
            "agent_llm": args.agent_llm,
            "user_llm": args.user_llm,
            "results": results,
            "personas": [{k: v for k, v in p.items() if k != "text"} for p in personas],
        },
    )
    print(f"\nsaved -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="cap tasks per domain (0=all)"
    )
    ap.add_argument(
        "--only-idx",
        type=int,
        nargs="+",
        default=None,
        help="evaluate only these candidate indices (else seeds + top-5 mutated)",
    )

    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="gpt-5.2")
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")

    return ap.parse_args()


if __name__ == "__main__":
    main()
