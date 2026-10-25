#!/usr/bin/env python3
"""Test-set eval of the top-delta optimized personas per domain.

For each domain (airline, retail) this script:
  1. Scans every GEPA run-dir under the domain's pgepa_runs folder.
  2. For each seed, reads pgepa_state.json -> seed candidate (idx0) and best
     candidate (highest difficulty), computes delta = best.diff - seed.diff.
  3. Ranks seeds by delta (desc), keeps the top-K (default 3).
  4. Runs the full test split (task ids read from the *latest* split file) for
     both the seed persona and the best optimized persona, at num_trials trials.
  5. Optionally runs a no_persona baseline once per domain.
  6. Summarizes avg_reward (and difficulty = 1 - avg_reward), seed vs best.

The persona_text stored in pgepa_state.json already contains the full
"method actor" wrapper, so it is passed verbatim as custom_guidelines.

Usage:
  # dry-run: print the selection + planned commands, run nothing
  python experiments/persona_search/scripts/run_topdelta_test.py --dry-run

  # real run
  python experiments/persona_search/scripts/run_topdelta_test.py --domains airline retail \
      --top-k 3 --num-trials 3 --max-concurrency 5

  # summarize already-completed sims without re-running
  python experiments/persona_search/scripts/run_topdelta_test.py --summarize-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
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

# domain -> (pgepa_runs subdir, latest split file, save-to base under data/simulations)
DOMAIN_CONFIG = {
    "airline": {
        "run_glob": str(DATA_ROOT / "data/persona_opt/pgepa_runs/airline_v4/*"),
        "split_file": "experiments/persona_search/configs/airline_alaska_split_v4.json",
        "save_base": "gepa_opt/airline_topdelta_test",
    },
    "retail": {
        "run_glob": str(DATA_ROOT / "data/persona_opt/pgepa_runs/retail_v2/*"),
        "split_file": "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
        "save_base": "gepa_opt/retail_topdelta_test",
    },
}

AGENT_LLM = "gpt-5.4"
USER_LLM = "gpt-5.4"
EFFORT = '{"reasoning_effort":"high"}'


@dataclass
class SeedResult:
    seed_id: str  # e.g. alaska_00 / gc_03
    run_dir: str
    seed_text: str
    seed_diff: float
    best_idx: int
    best_name: str
    best_text: str
    best_diff: float

    @property
    def delta(self) -> float:
        return self.best_diff - self.seed_diff


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def read_test_ids(split_file: str, domain: str) -> list:
    """Read test task ids from a split json.

    The split files store `test` as a domain-keyed dict, e.g.
    ``{"test": {"airline": ["1", "2", ...]}}``. Fall back to a flat list or
    a few alternate key spellings for older splits.
    """
    with open(_abs(split_file)) as fh:
        d = json.load(fh)

    def _extract(v):
        # domain-keyed dict -> pick this domain, else flatten all values
        if isinstance(v, dict):
            if domain in v:
                return list(v[domain])
            flat = []
            for vv in v.values():
                flat.extend(vv)
            return flat
        return list(v)

    for key in ("test", "test_ids", "test_task_ids", "test_instance_ids"):
        if key in d:
            return _extract(d[key])
    for k in ("splits", "split"):
        if k in d and isinstance(d[k], dict):
            for key in ("test", "test_ids"):
                if key in d[k]:
                    return _extract(d[k][key])
    raise KeyError(f"no test ids found in {split_file}; top keys={list(d.keys())}")


def load_seed(run_dir: str) -> SeedResult | None:
    """Parse one GEPA run-dir's pgepa_state.json into a SeedResult."""
    state_path = os.path.join(run_dir, "pgepa_state.json")
    if not os.path.isfile(state_path):
        return None
    with open(state_path) as fh:
        state = json.load(fh)  # duplicate keys collapse to last; that's fine
    cands = state.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    seed_id = state.get("persona_id_base") or os.path.basename(run_dir.rstrip("/"))

    def diff_of(c):
        # difficulty is stored as difficulty_agg (== agg); keep older spellings
        for k in (
            "difficulty_agg",
            "agg",
            "difficulty",
            "exploration_score",
            "raw_difficulty",
        ):
            if isinstance(c.get(k), (int, float)):
                return float(c[k])
        return 0.0

    def text_of(c):
        return c.get("persona_text") or c.get("custom_guidelines") or ""

    def name_of(c):
        # no explicit name field; parse the "# Persona: NAME" header
        if c.get("name") and c["name"] != "?":
            return c["name"]
        for line in text_of(c).splitlines():
            line = line.strip()
            if line.lower().startswith("# persona:"):
                nm = line.split(":", 1)[1].strip()
                return nm.split("  ")[0].split(" (")[0].strip() or "?"
        return "?"

    by_idx = {int(c.get("idx", i)): c for i, c in enumerate(cands)}
    seed_c = by_idx.get(0, cands[0])
    # prefer GEPA's own selection; else the max-difficulty candidate
    best_top = state.get("best_idx")
    best_c = by_idx.get(int(best_top)) if best_top is not None else None
    if best_c is None:
        best_c = max(cands, key=diff_of)

    return SeedResult(
        seed_id=seed_id,
        run_dir=run_dir,
        seed_text=text_of(seed_c),
        seed_diff=diff_of(seed_c),
        best_idx=int(best_c.get("idx", -1)),
        best_name=name_of(best_c),
        best_text=text_of(best_c),
        best_diff=diff_of(best_c),
    )


def select_top_delta(domain: str, top_k: int) -> list[SeedResult]:
    cfg = DOMAIN_CONFIG[domain]
    seeds = []
    for run_dir in sorted(glob.glob(_abs(cfg["run_glob"]))):
        if not os.path.isdir(run_dir):
            continue
        sr = load_seed(run_dir)
        if sr is not None:
            seeds.append(sr)
    seeds.sort(key=lambda s: s.delta, reverse=True)
    return seeds[:top_k]


def persona_arg(text: str) -> str:
    return json.dumps({"custom_guidelines": text}, ensure_ascii=False)


def run_condition(
    domain: str,
    name: str,
    persona_text: str | None,
    task_ids: list,
    save_base: str,
    num_trials: int,
    conc: int,
    dry_run: bool,
) -> None:
    save_to = f"{save_base}/{name}/{domain}"
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
        "--task-ids",
        *[str(t) for t in task_ids],
    ]
    if persona_text:
        cmd[cmd.index("--task-ids") : cmd.index("--task-ids")] = [
            "--user-persona",
            persona_arg(persona_text),
        ]
    print(f"\n[RUN] {domain}/{name}  ({len(task_ids)} tasks x {num_trials} trials)")
    if dry_run:
        printable = " ".join(
            (c if len(c) < 80 else c[:60] + f"...<{len(c)}chars>") for c in cmd
        )
        print("   " + printable)
        return
    subprocess.run(cmd, cwd=REPO, check=False)


def load_avg_reward(save_base: str, name: str, domain: str) -> tuple[int, float]:
    sim_dir = _abs(
        os.path.join(str(DATA_ROOT / "data/simulations"), save_base, name, domain)
    )
    rewards = []
    for f in glob.glob(os.path.join(sim_dir, "eval_*.json")) + glob.glob(
        os.path.join(sim_dir, "*.json")
    ):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # tau2 results.json holds a list of simulations; eval_*.json a single one
        sims = d.get("simulations") if isinstance(d, dict) else None
        if sims:
            for s in sims:
                ri = s.get("reward_info") or {}
                rewards.append(ri.get("reward", 0) or 0)
        else:
            ri = (d.get("reward_info") or {}) if isinstance(d, dict) else {}
            if ri:
                rewards.append(ri.get("reward", 0) or 0)
    n = len(rewards)
    return n, (sum(rewards) / n if n else 0.0)


def summarize(domain: str, selected: list[SeedResult]) -> None:
    cfg = DOMAIN_CONFIG[domain]
    sb = cfg["save_base"]
    print(f"\n{'=' * 72}\nSUMMARY — {domain}\n{'=' * 72}")
    print(f"{'condition':<28}{'n':>6}{'avg_reward':>12}{'difficulty':>12}")
    print("-" * 58)
    n, avg = load_avg_reward(sb, "no_persona", domain)
    if n:
        print(f"{'no_persona':<28}{n:>6}{avg:>12.3f}{1 - avg:>12.3f}")
    for sr in selected:
        for kind in ("seed", "best"):
            name = f"{kind}_{sr.seed_id}"
            n, avg = load_avg_reward(sb, name, domain)
            tag = f"{name}" + (f" (idx{sr.best_idx})" if kind == "best" else "")
            print(f"{tag:<28}{n:>6}{avg:>12.3f}{1 - avg:>12.3f}")
    print("\n(difficulty = 1 - avg_reward; GEPA val delta shown below)")
    for sr in selected:
        print(
            f"  {sr.seed_id:<12} val: seed={sr.seed_diff:.3f} "
            f"best(idx{sr.best_idx} {sr.best_name!r})={sr.best_diff:.3f} "
            f"Δ={sr.delta:+.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["airline", "retail"],
        choices=list(DOMAIN_CONFIG),
    )
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--airline-top-k",
        type=int,
        default=None,
        help="override --top-k for airline (e.g. 2)",
    )
    ap.add_argument(
        "--retail-top-k", type=int, default=None, help="override --top-k for retail"
    )
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--max-concurrency", type=int, default=5)
    ap.add_argument(
        "--no-baseline", action="store_true", help="skip the no_persona baseline"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print selection + commands, run nothing"
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="only read existing sims and print summary",
    )
    args = ap.parse_args()

    per_domain_k = {"airline": args.airline_top_k, "retail": args.retail_top_k}
    for domain in args.domains:
        cfg = DOMAIN_CONFIG[domain]
        k = per_domain_k.get(domain) or args.top_k
        selected = select_top_delta(domain, k)
        test_ids = read_test_ids(cfg["split_file"], domain)

        print(
            f"\n{'#' * 72}\n# {domain}: top-{k} by delta "
            f"({len(test_ids)} test tasks)\n{'#' * 72}"
        )
        for rank, sr in enumerate(selected, 1):
            print(
                f"  {rank}. {sr.seed_id:<12} Δ={sr.delta:+.3f} "
                f"(seed={sr.seed_diff:.3f} -> best idx{sr.best_idx} "
                f"{sr.best_name!r}={sr.best_diff:.3f})"
            )

        if args.summarize_only:
            summarize(domain, selected)
            continue

        if not args.no_baseline:
            run_condition(
                domain,
                "no_persona",
                None,
                test_ids,
                cfg["save_base"],
                args.num_trials,
                args.max_concurrency,
                args.dry_run,
            )
        for sr in selected:
            run_condition(
                domain,
                f"seed_{sr.seed_id}",
                sr.seed_text,
                test_ids,
                cfg["save_base"],
                args.num_trials,
                args.max_concurrency,
                args.dry_run,
            )
            run_condition(
                domain,
                f"best_{sr.seed_id}",
                sr.best_text,
                test_ids,
                cfg["save_base"],
                args.num_trials,
                args.max_concurrency,
                args.dry_run,
            )

        if not args.dry_run:
            summarize(domain, selected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
