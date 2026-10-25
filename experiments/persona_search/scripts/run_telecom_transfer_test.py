#!/usr/bin/env python3
"""Cross-domain transfer test of telecom-optimized personas.

Takes the top-2 telecom seeds (by GEPA difficulty delta) plus two *domain-
transferable* optimized personas, and evaluates all of them on the airline,
retail, and banking test sets — to measure how much a telecom-tuned adversarial
persona still hinders an agent in a different domain.

Personas (all from telecom ev2_3islands_v4):
  - seed  'Terse & Blunt'                  (brinks_00 idx0) — pure style baseline
  - seed  'Front-loading & Terse'          (brinks_02 idx0) — pure style baseline
  - best  'Script-Weary Specialist-Seeker' (brinks_02 best) — generic escalation/misroute
  - best  'Meaning-First Translator'       (brinks_04 best) — generic status-collapse

brinks_00's own optimized persona ('Temporary-Test Snapback') is intentionally
EXCLUDED: its attack is a telecom device-settings mechanism with no analogue in
airline/retail/banking, so it cannot transfer by construction.

Settings mirror the top-delta test / telecom certification口径:
  agent = user = gpt-5.4 @ reasoning_effort=high, num_trials=3, verbatim persona.
Banking uses --retrieval-config bm25 (offline, no API key) and the SAME gpt-5.4
agent as the other domains so the persona effect is comparable across domains.

Baselines: airline/retail reuse the no_persona runs from run_topdelta_test.py;
banking runs its own no_persona.

Usage:
  python experiments/persona_search/scripts/run_telecom_transfer_test.py --dry-run
  python experiments/persona_search/scripts/run_telecom_transfer_test.py --max-concurrency 25
  python experiments/persona_search/scripts/run_telecom_transfer_test.py --summarize-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
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
TAU2 = os.path.join(REPO, ".venv/bin/tau2")
TELECOM_RUNS = str(DATA_ROOT / "data/persona_opt/pgepa_runs/ev2_3islands_v4")

AGENT_LLM = "gpt-5.4"
USER_LLM = "gpt-5.4"
EFFORT = '{"reasoning_effort":"high"}'
SAVE_BASE = "gepa_opt/telecom_transfer_test"

# telecom source personas: (run subdir, kind, short label)
PERSONA_SPECS = [
    {"run": "ev2v4_brinks_00", "kind": "seed", "label": "seed_b00_terse_blunt"},
    {"run": "ev2v4_brinks_02", "kind": "seed", "label": "seed_b02_frontload"},
    {"run": "ev2v4_brinks_02", "kind": "best", "label": "best_b02_script_weary"},
    {"run": "ev2v4_brinks_04", "kind": "best", "label": "best_b04_meaning_xlator"},
]

# target domain -> config. `domain` is the tau2 --domain; `key` is the split key.
TARGET_DOMAINS = {
    "airline": {
        "domain": "airline",
        "key": "airline",
        "retrieval": None,
        "split": "experiments/persona_search/configs/airline_alaska_split_v4.json",
        "reuse_baseline": "gepa_opt/airline_topdelta_test/no_persona/airline",
    },
    "retail": {
        "domain": "retail",
        "key": "retail",
        "retrieval": None,
        "split": "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
        "reuse_baseline": "gepa_opt/retail_topdelta_test/no_persona/retail",
    },
    "banking": {
        "domain": "banking_knowledge",
        "key": "banking",
        "retrieval": "bm25",
        "split": "experiments/persona_search/configs/banking_knowledge_split_v2.json",
        "reuse_baseline": None,
    },
}


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def read_test_ids(split_file: str, key: str) -> list:
    d = json.load(open(_abs(split_file)))
    t = d["test"]
    if isinstance(t, dict):
        return list(t[key]) if key in t else [x for v in t.values() for x in v]
    return list(t)


def load_persona(spec: dict) -> tuple[str, str, float]:
    """Return (persona_name, persona_text, difficulty) for a spec."""
    st = json.load(open(_abs(f"{TELECOM_RUNS}/{spec['run']}/pgepa_state.json")))
    cands = st["candidates"]
    by_idx = {int(c.get("idx", i)): c for i, c in enumerate(cands)}
    if spec["kind"] == "seed":
        c = by_idx.get(0, cands[0])
    else:
        bidx = st.get("best_idx")
        c = (
            by_idx.get(int(bidx))
            if bidx is not None
            else max(cands, key=lambda x: x.get("difficulty_agg", 0))
        )
    text = c.get("persona_text") or ""
    name = "?"
    for ln in text.splitlines():
        if ln.strip().lower().startswith("# persona:"):
            name = ln.split(":", 1)[1].strip().split("  ")[0].split(" (")[0]
            break
    diff = float(c.get("difficulty_agg", c.get("agg", 0.0)))
    return name, text, diff


def persona_arg(text: str) -> str:
    return json.dumps({"custom_guidelines": text}, ensure_ascii=False)


def baseline_exists(cfg: dict) -> bool:
    rb = cfg.get("reuse_baseline")
    if not rb:
        return False
    return os.path.isfile(
        _abs(os.path.join(str(DATA_ROOT / "data/simulations"), rb, "results.json"))
    )


def run_condition(
    dom_cfg: dict,
    name: str,
    persona_text: str | None,
    task_ids: list,
    num_trials: int,
    conc: int,
    dry_run: bool,
) -> None:
    domain = dom_cfg["domain"]
    save_to = f"{SAVE_BASE}/{name}/{dom_cfg['key']}"
    cmd = [
        TAU2,
        "run",
        "--domain",
        domain,
        "--user",
        "user_simulator",
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        EFFORT,
        "--agent-llm",
        AGENT_LLM,
        "--agent-llm-args",
        EFFORT,
        "--num-trials",
        str(num_trials),
        "--max-concurrency",
        str(conc),
        "--save-to",
        save_to,
        "--auto-resume",
        "--max-steps",
        "120",
    ]
    if dom_cfg.get("retrieval"):
        cmd += ["--retrieval-config", dom_cfg["retrieval"]]
    if persona_text:
        cmd += ["--user-persona", persona_arg(persona_text)]
    cmd += ["--task-ids", *[str(t) for t in task_ids]]

    print(
        f"\n[RUN] {dom_cfg['key']}/{name}  "
        f"({len(task_ids)} tasks x {num_trials} trials)"
    )
    if dry_run:
        show = " ".join(c if len(c) < 80 else c[:60] + f"...<{len(c)}c>" for c in cmd)
        print("   " + show)
        return
    subprocess.run(cmd, cwd=REPO, check=False)


def load_avg_reward(save_rel: str) -> tuple[int, float]:
    sim_dir = _abs(os.path.join(str(DATA_ROOT / "data/simulations"), save_rel))
    rewards = []
    for f in glob.glob(os.path.join(sim_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for s in d.get("simulations") or []:
            r = (s.get("reward_info") or {}).get("reward")
            if r is not None:
                rewards.append(r)
    n = len(rewards)
    return n, (sum(rewards) / n if n else 0.0)


def baseline_rel(dom_key: str, cfg: dict) -> str:
    if baseline_exists(cfg):
        return cfg["reuse_baseline"]
    return f"{SAVE_BASE}/no_persona/{dom_key}"


def summarize(personas: list[dict]) -> None:
    for dom_key, cfg in TARGET_DOMAINS.items():
        print(
            f"\n{'=' * 74}\nTRANSFER SUMMARY — {dom_key} "
            f"(agent {AGENT_LLM})\n{'=' * 74}"
        )
        print(f"{'condition':<26}{'n':>6}{'ASR':>9}{'diff':>9}{'Δvs base':>10}")
        print("-" * 60)
        bn, bavg = load_avg_reward(baseline_rel(dom_key, cfg))
        base_diff = 1 - bavg
        print(f"{'no_persona':<26}{bn:>6}{bavg:>9.3f}{base_diff:>9.3f}{'—':>10}")
        for p in personas:
            n, avg = load_avg_reward(f"{SAVE_BASE}/{p['label']}/{dom_key}")
            diff = 1 - avg
            d = diff - base_diff
            print(f"{p['label']:<26}{n:>6}{avg:>9.3f}{diff:>9.3f}{d:>+10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains",
        nargs="+",
        default=list(TARGET_DOMAINS),
        choices=list(TARGET_DOMAINS),
    )
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--max-concurrency", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    # resolve personas up-front (so we fail fast if a source is missing)
    personas = []
    for spec in PERSONA_SPECS:
        name, text, diff = load_persona(spec)
        personas.append({**spec, "name": name, "text": text, "src_diff": diff})

    print(f"{'#' * 74}\n# Telecom→cross-domain persona transfer test\n{'#' * 74}")
    print("Personas (telecom source difficulty in parens):")
    for p in personas:
        print(
            f"  {p['label']:<26} {p['kind']:<5} '{p['name']}' "
            f"(telecom diff={p['src_diff']:.3f}, {len(p['text'])} chars)"
        )

    for dom_key in args.domains:
        cfg = TARGET_DOMAINS[dom_key]
        test_ids = read_test_ids(cfg["split"], cfg["key"])
        reuse = baseline_exists(cfg)
        print(
            f"\n### {dom_key}: {len(test_ids)} test tasks | "
            f"baseline={'REUSE ' + cfg['reuse_baseline'] if reuse else 'run fresh'}"
        )

        if args.summarize_only:
            continue

        if not reuse:
            run_condition(
                cfg,
                "no_persona",
                None,
                test_ids,
                args.num_trials,
                args.max_concurrency,
                args.dry_run,
            )
        for p in personas:
            run_condition(
                cfg,
                p["label"],
                p["text"],
                test_ids,
                args.num_trials,
                args.max_concurrency,
                args.dry_run,
            )

    if not args.dry_run:
        summarize(personas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
