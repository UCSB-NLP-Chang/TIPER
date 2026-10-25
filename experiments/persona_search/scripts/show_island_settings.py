"""Print the EFFECTIVE configuration of the telecom island-model persona search.

Single source of truth = run_islands_telecom.sh: we parse its `VAR="${VAR:-default}"`
lines, then let any matching environment variable override (so
`ITERATIONS=30 python show_island_settings.py` shows what an actual run would use).
Also reports the split sizes/distribution, the island seeds, the params that come
from parallel_gepa.py's built-in defaults (not set by the launcher), and a rollout
cost estimate.

Usage:
  .venv/bin/python experiments/persona_search/scripts/show_island_settings.py
  ITERATIONS=30 PER_ISLAND_CONCURRENCY=2 .venv/bin/python experiments/persona_search/scripts/show_island_settings.py
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
LAUNCHER = REPO / "experiments/persona_search/scripts/run_islands_telecom.sh"


def parse_launcher_defaults(path: Path) -> dict[str, str]:
    """Extract NAME -> default from `NAME="${NAME:-DEFAULT}"` lines (one nested
    `${ALT:-x}` fallback is unwrapped to x). Env vars of the same name override,
    then `${OTHER}` references are expanded against the resolved table."""
    out: dict[str, str] = {}
    pat = re.compile(r'^(\w+)="\$\{\1:-(.*)\}"\s*$')
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        name, default = m.group(1), m.group(2)
        # unwrap a single nested ${ALT:-x} -> x (best effort)
        nested = re.match(r"\$\{\w+:-(.*)\}$", default)
        if nested:
            default = nested.group(1)
        out[name] = os.environ.get(name, default)
    # expand ${OTHER} references (e.g. RUN_ROOT -> uses RUN_TAG) to fixpoint
    ref = re.compile(r"\$\{(\w+)\}")
    for _ in range(5):
        changed = False
        for k, v in out.items():
            nv = ref.sub(
                lambda mm: out.get(mm.group(1), os.environ.get(mm.group(1), "")), v
            )
            if nv != v:
                out[k] = nv
                changed = True
        if not changed:
            break
    return out


def parse_launcher_seeds(path: Path) -> list[str]:
    """Parse the SEEDS=( "name:path" ... ) array → list of island names."""
    txt = path.read_text()
    m = re.search(r"SEEDS=\((.*?)\)", txt, re.DOTALL)
    if not m:
        return []
    return [q.split(":", 1)[0] for q in re.findall(r'"([^"]+)"', m.group(1))]


def fmt(label: str, value, note: str = "") -> str:
    n = f"  ({note})" if note else ""
    return f"  {label:<24} {value}{n}"


def issue(t: str) -> str:
    return t[1 : t.index("]")]


def ptag(t: str) -> str:
    m = re.search(r"PERSONA:(\w+)", t)
    return m.group(1) if m else "?"


def main():
    d = parse_launcher_defaults(LAUNCHER)

    def g(k, default=""):
        return d.get(k, os.environ.get(k, default))

    line = "=" * 64
    print(line)
    print("  TELECOM ISLAND-MODEL PERSONA SEARCH — effective settings")
    print(f"  (launcher: {LAUNCHER.relative_to(REPO)}; env overrides applied)")
    print(line)

    # --- split ---
    split_file = g(
        "SPLIT_FILE", "experiments/persona_search/configs/telecom_island_split.json"
    )
    sp = REPO / split_file
    print("\n[SPLIT]  " + split_file)
    if sp.exists():
        js = json.load(open(sp))
        for k in ("train", "val", "test"):
            ids = js[k]["telecom"]
            it = dict(Counter(issue(t) for t in ids))
            pt = dict(Counter(ptag(t) for t in ids))
            role = {
                "train": "minibatch source",
                "val": "selection / Pareto",
                "test": "HELD-OUT (untouched in search)",
            }[k]
            print(f"  {k:<5} n={len(ids):<3} {role}")
            print(f"        issue={it}  ptag={pt}")
        meta = js.get("meta", {})
        print(
            f"  universe={meta.get('universe')}  seed={meta.get('seed')}  "
            f"strata={meta.get('strata')}"
        )
    else:
        print("  !! split file not found")

    # --- islands / seeds (parsed from the launcher's SEEDS array) ---
    seeds = parse_launcher_seeds(LAUNCHER)
    print(f"\n[ISLANDS]  {len(seeds)} islands (one optimization each)")
    for name in seeds:
        print(f"  {name}")

    # --- algorithm ---
    print("\n[ALGORITHM]")
    print(fmt("iterations", g("ITERATIONS"), "parent->child rounds per island"))
    print(fmt("minibatch", g("MB"), "tasks sampled from TRAIN for the pre-screen"))
    print(fmt("min_dissimilarity", g("MIN_DISSIM"), "lexical 3gram-Jaccard floor"))
    print(fmt("max_semantic_similarity", g("MAX_SEM_SIM"), "embedding cosine ceiling"))
    print(fmt("max_mutation_attempts", g("MAX_MUT_ATTEMPTS")))
    print(fmt("severe_gate_rate", g("SEVERE_GATE_RATE")))
    print(
        fmt(
            "summary_inject_limit",
            g("SUMMARY_INJECT_LIMIT"),
            "cross-island summaries in prompt",
        )
    )

    # --- models ---
    print("\n[MODELS]")
    print(fmt("agent (SUT)", g("AGENT_LLM"), g("AGENT_LLM_ARGS").replace('\\"', '"')))
    print(
        fmt(
            "user simulator",
            f"{g('USER_TYPE')} / {g('USER_LLM')}",
            f"effort={g('EFFORT')}",
        )
    )
    print(
        fmt(
            "mutation (proposer)",
            g("MUTATION_MODEL"),
            f"effort={g('MUTATION_EFFORT', 'xhigh')}",
        )
    )
    print(fmt("judge (validity)", g("JUDGE_MODEL")))
    print(fmt("embedding (dedup)", g("EMBEDDING_MODEL")))

    # --- concurrency ---
    per = int(g("PER_ISLAND_CONCURRENCY", "3"))
    n_isl = len(seeds) or 5
    print("\n[CONCURRENCY]")
    print(fmt("per-island", per, "within-eval sim fan-out"))
    print(fmt("total in-flight cap", per * n_isl, f"{per} x {n_isl} islands"))

    # --- routing / auth ---
    print("\n[ROUTING]")
    use_bedrock = g("USE_BEDROCK", "0") == "1"
    if use_bedrock:
        print(fmt("backbone", "AWS Bedrock"))
    else:
        port = g("VERTEX_SHIM_PORT", "4112")
        print(fmt("backbone", f"Anthropic->Vertex shim :{port}"))
        print(
            fmt(
                "vertex proj/loc",
                f"{g('VERTEXAI_PROJECT', 'cresta-ml')}/{g('VERTEXAI_LOCATION', 'us-east5')}",
            )
        )

    # --- paths ---
    run_root = g(
        "RUN_ROOT", f"data/persona_opt/pgepa_runs/{g('RUN_TAG', 'islands_telecom_v1')}"
    )
    print("\n[PATHS]")
    print(fmt("run root", run_root))
    print(
        fmt("shared seen pool", g("SHARED_POOL", f"{run_root}/shared_seen_pool.jsonl"))
    )
    print(fmt("dry run", g("DRY_RUN", "0")))

    # --- params from parallel_gepa.py built-in defaults (NOT set by launcher) ---
    print("\n[PARALLEL_GEPA DEFAULTS]  (not set by launcher; argparse defaults)")
    print(fmt("num_trials", 1, "sims per (persona,task) eval"))
    print(fmt("val_num_trials", "= num_trials", "held-out val confirm"))
    print(fmt("max_steps", 120))
    print(fmt("sim_timeout", "600s"))
    print(fmt("tie_tol", 0.0, "Pareto tie tolerance"))
    print(fmt("lint_persona", "on (maj1, gpt-5.4-mini)"))

    # --- cost estimate ---
    iters = int(g("ITERATIONS", "20"))
    mb = int(g("MB", "10"))
    n_tr = len(json.load(open(sp))["train"]["telecom"]) if sp.exists() else 30
    n_val = len(json.load(open(sp))["val"]["telecom"]) if sp.exists() else 30
    gen0 = n_val + n_tr  # seed: val + train baseline
    per_iter_max = mb + n_val + (n_tr - mb)  # minibatch + val confirm + train top-up
    per_iter_min = mb  # screened out immediately
    isl_max = gen0 + iters * per_iter_max
    isl_min = gen0 + iters * per_iter_min
    print("\n[COST ESTIMATE]  (rollouts = sims; num_trials=1)")
    print(
        f"  per island : {isl_min} (all screened out)  ..  {isl_max} (all pass+accepted)"
    )
    print(f"  all {n_isl} islands: {isl_min * n_isl}  ..  {isl_max * n_isl}")
    print(
        f"    gen0={gen0} (val {n_val}+train {n_tr}); per-iter {per_iter_min}..{per_iter_max}; "
        f"x {iters} iters"
    )
    print(
        f"  NOTE: TEST split ({len(json.load(open(sp))['test']['telecom']) if sp.exists() else 54} "
        "tasks) is NOT evaluated during search — separate post-hoc cost."
    )
    print(line)


if __name__ == "__main__":
    main()
