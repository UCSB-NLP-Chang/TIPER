#!/usr/bin/env python3
"""Aggregate top-10 adversarial personas per domain (cross-run).

Pools candidates.json across all optimize runs in a domain, dedups by
persona_text, ranks by difficulty score, takes top 10. Score field is
inconsistent across benchmarks/runs, so use a fallback chain:
  score -> exploration_score -> difficulty_score -> val_difficulty
and backfill from sibling pgepa_state.json when candidates.json lacks scores.

Output: experiments/baseline/scripts/roleplay_probe/personas_selected.json
  {domain: [{persona_id, persona_text, source_run, difficulty}, ...x10]}

Caveat: cross-run difficulty is NOT comparable (different splits / user models
searched) — top10 is within-domain relative picking, accepted by design.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "personas_selected.json"

DOMAIN_GLOBS = {
    "telecom": str(
        DATA_ROOT / "data/persona_opt/pgepa_runs/**/telecom/**/candidates.json"
    ),
    "lic_math": str(
        DATA_ROOT / "data/persona_opt/pgepa_runs/**/lic_math/**/candidates.json"
    ),
    "colbench_code": str(
        REPO_ROOT / "vendor/sweet_rl/data/persona_opt/gepa_runs/*_code/candidates.json"
    ),
    "delivery": str(
        REPO_ROOT
        / "vendor/vitabench/data/persona_opt/vita_pgepa_runs/run_delivery*/candidates.json"
    ),
}

SCORE_KEYS = ("score", "exploration_score", "difficulty_score", "val_difficulty")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _cand_score(c: dict) -> float | None:
    for k in SCORE_KEYS:
        v = c.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _state_backfill(state_path: Path) -> dict[str, float]:
    """Map persona_text -> score from pgepa_state candidates/generations."""
    try:
        s = json.loads(state_path.read_text())
    except Exception:
        return {}
    out: dict[str, float] = {}
    for c in s.get("candidates", []):
        pt = _norm(c.get("persona_text", ""))
        if not pt:
            continue
        sc = _cand_score(c)
        if sc is None:
            sc = (
                float(s.get("best_agg", 0.0))
                if c.get("idx") == s.get("best_idx")
                else None
            )
        if sc is not None:
            out[pt] = sc
    # generations fallback (difficulty_agg per idx)
    gens = s.get("generations", [])
    if isinstance(gens, list):
        for g in gens:
            pt = _norm(g.get("persona_text", ""))
            da = g.get("difficulty_agg")
            if pt and isinstance(da, (int, float)) and pt not in out:
                out[pt] = float(da)
    return out


def aggregate_domain(domain: str, pattern: str) -> list[dict]:
    files = sorted(map(Path, glob.glob(pattern)))
    # dedup by normalized persona_text, keep best score
    best: dict[str, dict] = {}
    n_runs = 0
    for f in files:
        try:
            cands = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(cands, list) or not cands:
            continue
        # detect scoreless run -> backfill from sibling state
        has_scores = any(
            _cand_score(c) is not None
            for c in cands
            if _norm(c.get("persona_text", ""))
        )
        backfill = {}
        if not has_scores:
            state_path = f.parent / "pgepa_state.json"
            if state_path.exists():
                backfill = _state_backfill(state_path)
        n_runs += 1
        for c in cands:
            pt = _norm(c.get("persona_text", ""))
            if len(pt) < 50:  # skip empty / seed / trivial
                continue
            sc = _cand_score(c)
            if sc is None:
                sc = backfill.get(pt)
            if sc is None:
                sc = 0.0  # unscored — ranked last but kept
            rec = {
                "persona_text": c.get("persona_text", "").strip(),
                "source_run": (
                    os.path.relpath(str(f), str(DATA_ROOT))
                    if os.path.isabs(str(f))
                    and str(f).startswith(str(DATA_ROOT) + os.sep)
                    else str(f)
                ),
                "difficulty": round(float(sc), 4),
            }
            cur = best.get(pt)
            if cur is None or sc > cur["difficulty"]:
                best[pt] = rec
    ranked = sorted(best.values(), key=lambda r: r["difficulty"], reverse=True)[:10]
    # assign persona_id
    for i, r in enumerate(ranked):
        r["persona_id"] = f"{domain}_p{i:02d}"
    print(
        f"  {domain}: runs={n_runs} unique_nonempty={len(best)} -> top{len(ranked)} "
        f"diff_range={ranked[0]['difficulty']:.3f}..{ranked[-1]['difficulty']:.3f}"
        if ranked
        else f"  {domain}: NONE"
    )
    return ranked


def main():
    out: dict[str, list] = {}
    print("Aggregating top-10 adversarial personas per domain:")
    for dom, pat in DOMAIN_GLOBS.items():
        out[dom] = aggregate_domain(dom, pat)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT}")
    total = sum(len(v) for v in out.values())
    print(f"Total personas: {total} (expect 40 if every domain has >=10)")
    # sanity: any domain short?
    for dom, v in out.items():
        if len(v) < 10:
            print(f"  WARN: {dom} only has {len(v)} personas")


if __name__ == "__main__":
    main()
