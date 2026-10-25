#!/usr/bin/env python3
"""Test-set three-way comparison for the banking CardWorks island run.

Evaluates, on the held-out banking_knowledge split_v2 TEST set (57 tasks):
  - no_persona                         (naked user_simulator, absolute baseline)
  - <island>_seed                      (the real CardWorks cluster seed, idx0)
  - <island>_opt                       (the GEPA-optimized hardest candidate)

for all 4 islands (cw_00/01/03/04). Reuses eval_persona._run_domain_with_retry
so the agent/user/retrieval config AND the method-actor persona wrapper are
IDENTICAL to the search — difficulty (=1-ASR) is directly comparable.

Config mirrors the island search: agent gpt-5.5 (reasoning_effort=high),
user_simulator gpt-5.4 (effort high), bm25 retrieval, no bedrock (vertex),
num_trials=3. no_persona passes persona_text=None so eval_persona skips the
--user-persona injection entirely (a true no-persona baseline).

Usage:
  python experiments/persona_search/scripts/run_test_matrix_banking_cardworks.py            # run all + summary
  python experiments/persona_search/scripts/run_test_matrix_banking_cardworks.py --summarize-only
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
# Runtime data workspace: TAU2BENCH_DATA_ROOT (default ../tau2bench-data).
REPO = Path(
    __import__("os").environ.get(
        "TAU2BENCH_DATA_ROOT", str(REPO_ROOT.parent / "tau2bench-data")
    )
)
sys.path.insert(0, str(REPO_ROOT / "experiments/persona_search/scripts"))
from eval_persona import _run_domain_with_retry  # noqa: E402

RUN_ROOT = REPO / "data/persona_opt/pgepa_runs/islands_banking_cardworks_v2"
SPLIT = REPO_ROOT / "experiments/persona_search/configs/banking_knowledge_split_v2.json"
ISLANDS = ["cw_00", "cw_01", "cw_03", "cw_04"]
SAVE_BASE = "gepa_opt/testmatrix_cardworks"


def _best_cand(state: dict) -> dict:
    """Hardest candidate by difficulty_agg (matches cert口径, not stored best_idx)."""
    return max(state["candidates"], key=lambda c: c.get("difficulty_agg") or 0.0)


def load_conditions() -> list[tuple[str, str | None, float | None]]:
    """(name, persona_text_or_None, search_difficulty_or_None)."""
    conds: list[tuple[str, str | None, float | None]] = [("no_persona", None, None)]
    for isl in ISLANDS:
        st = json.load(open(RUN_ROOT / f"isl_{isl}" / "pgepa_state.json"))
        seed = st["candidates"][0]
        opt = _best_cand(st)
        conds.append((f"{isl}_seed", seed["persona_text"], seed.get("difficulty_agg")))
        conds.append((f"{isl}_opt", opt["persona_text"], opt.get("difficulty_agg")))
    return conds


def test_ids() -> list[str]:
    d = json.load(open(SPLIT))
    t = d["test"]
    if isinstance(t, dict):
        return (
            list(t["banking"]) if "banking" in t else [x for v in t.values() for x in v]
        )
    return list(t)


def make_args(persona_text: str | None, conc: int, trials: int) -> SimpleNamespace:
    return SimpleNamespace(
        skill_name="testmatrix_dummy",  # unused by user_simulator path
        persona_text=persona_text,
        effort="high",
        use_bedrock=False,
        bedrock_profile=None,
        user_type="user_simulator",
        user_llm="gpt-5.4",
        agent_llm="gpt-5.5",
        agent_llm_args='{"reasoning_effort":"high"}',
        num_trials=trials,
        max_concurrency=conc,
        retrieval_config="bm25",
        max_steps=120,
        sim_timeout=600,
        max_infra_retries=2,
    )


def load_reward(save_rel: str) -> tuple[int, float]:
    sim_dir = REPO / "data/simulations" / save_rel
    rs: list[float] = []
    for f in glob.glob(str(sim_dir / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for s in d.get("simulations") or []:
            r = (s.get("reward_info") or {}).get("reward")
            if r is not None:
                rs.append(r)
    return len(rs), (sum(rs) / len(rs) if rs else 0.0)


def summarize() -> None:
    print("\n" + "=" * 78)
    print(
        "TEST-SET THREE-WAY — banking split_v2 test (agent gpt-5.5, difficulty = 1-ASR)"
    )
    print("=" * 78)
    n0, r0 = load_reward(f"{SAVE_BASE}/no_persona/banking")
    d0 = 1 - r0
    print(
        f"{'no_persona':<14}{n0:>5}   ASR={r0:.3f}   diff={d0:.3f}   (absolute baseline)"
    )
    print("-" * 78)
    print(
        f"{'island':<8}{'seed diff':>11}{'Δvs no_p':>10}   {'opt diff':>10}{'Δvs seed':>10}   test-vs-search"
    )
    for isl in ISLANDS:
        ns, rs = load_reward(f"{SAVE_BASE}/{isl}_seed/banking")
        no, ro = load_reward(f"{SAVE_BASE}/{isl}_opt/banking")
        ds, do = 1 - rs, 1 - ro
        st = json.load(open(RUN_ROOT / f"isl_{isl}" / "pgepa_state.json"))
        search_opt = _best_cand(st).get("difficulty_agg") or 0.0
        print(
            f"{isl:<8}{ds:>11.3f}{ds - d0:>+10.3f}   {do:>10.3f}{do - ds:>+10.3f}   "
            f"(search {search_opt:.3f} → test {do:.3f}, n_seed={ns} n_opt={no})"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrency", type=int, default=12)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    conds = load_conditions()
    ids = test_ids()
    print(
        f"### banking CardWorks TEST matrix: {len(ids)} tasks x {a.num_trials} "
        f"trials x {len(conds)} conditions"
    )
    for name, text, sd in conds:
        tag = "no-persona" if text is None else f"{len(text)}c, search_diff={sd}"
        print(f"  {name:<14} {tag}")

    if not a.summarize_only:
        for name, text, _ in conds:
            run_name = f"{SAVE_BASE}/{name}/banking"
            eargs = make_args(text, a.max_concurrency, a.num_trials)
            print(f"\n=== [{name}] ===")
            try:
                _run_domain_with_retry(eargs, "banking", ids, run_name)
            except Exception as e:
                print(f"  WARN {name}: {e}")

    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
