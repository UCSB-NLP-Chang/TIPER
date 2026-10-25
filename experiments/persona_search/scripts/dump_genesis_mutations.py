#!/usr/bin/env python3
"""Dump mutation-proposer responses from each benchmark's optimization process.

For 4 benchmarks (tau2 telecom, VitaBench delivery, LiC math, ColBench code) pick
5 distinct mutation contexts from the genesis/optimization run, call the proposer
LLM once per context, and persist {ctx, prompt, raw, parsed}.

Context sources:
  telecom    : genesis_4domain_enriched pgepa_state candidates as parents +
               that candidate's generation eval trajectory (real traces)
  delivery   : run_delivery_v2 candidates as parents + conversations test set
  lic_math   : published candidates as parents + (no trajectory preserved → blind)
  colbench   : sweetrl_full_001 candidates as parents + (blind)

Usage:
  .venv/bin/python experiments/persona_search/scripts/dump_genesis_mutations.py \
      --model gpt-5.2 --per-bench 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments" / "persona_search" / "scripts"))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except Exception:
    pass

from cresta.persona.mutation.prompt import _build_mutation_prompt  # noqa: E402
from cresta.persona.mutation.proposer import call_mutation_llm  # noqa: E402

DATA = Path(os.environ.get("TAU2BENCH_DATA_ROOT", str(_REPO.parent / "tau2bench-data")))
GEPAP_OPT = DATA / "artifacts" / "simulations" / "gepa_opt"
VITA = DATA / "artifacts" / "benchmarks" / "vita" / "persona_search" / "run_delivery_v2"
COLBENCH_CANDS = Path("/tmp/colbench_cands/candidates.json")


def _snip(messages: list[dict], head: int = 4, tail: int = 8) -> str:
    msgs = [
        {"role": m.get("role", "?"), "content": (m.get("content") or "")[:400]}
        for m in messages
        if m.get("content")
    ]
    parts = [f"[{m['role']}] {m['content']}" for m in msgs[:head]]
    if len(msgs) > head + tail:
        parts.append("   ... (middle omitted) ...")
    parts += [f"[{m['role']}] {m['content']}" for m in msgs[-tail:]]
    return "\n".join(parts)


def _traces_from_tau2_results(rf: Path, max_s: int = 12, max_f: int = 8):
    """Extract success/failure conversation traces from a tau2 results.json."""
    try:
        sims = json.load(open(rf))
    except Exception:
        return [], []
    if isinstance(sims, dict):
        sims = sims.get("simulations", [])
    success, failure = [], []
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        msgs = s.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        snip = _snip(msgs)
        if r == 1.0 and len(success) < max_s:
            success.append(
                {
                    "task_id": s.get("task_id", "?"),
                    "n_turns": len(msgs),
                    "conversation": snip,
                }
            )
        elif r == 0.0 and len(failure) < max_f:
            failure.append({"task_id": s.get("task_id", "?"), "conversation": snip})
    return success, failure


# ---- telecom (genesis_4domain_enriched) -----------------------------------


def telecom_ctxs(n: int):
    state = json.load(
        open(
            _REPO
            / "results"
            / "published"
            / "pgepa_runs"
            / "genesis_4domain_enriched"
            / "genesis_telecom"
            / "pgepa_state.json"
        )
    )
    cands = [c for c in state["candidates"] if c.get("persona_text")]
    ctxs = []
    for c in cands[:n]:
        gen = c.get("gen", 0)
        gdir = GEPAP_OPT / f"genesis_telecom_g{gen:02d}_k00"
        success, failure = [], []
        if gdir.is_dir():
            for rf in sorted(gdir.glob("eval_*/telecom/results.json")):
                s, f = _traces_from_tau2_results(rf)
                success += s
                failure += f
                if len(success) >= 12 and len(failure) >= 8:
                    break
        scored = len(success) + len(failure)
        score = len(success) / scored if scored else 0.0
        ctxs.append(
            {
                "ctx_id": f"gen{gen}_idx{c.get('idx')}",
                "parent_persona": c["persona_text"],
                "score": score,
                "success": success[:12],
                "failure": failure[:8],
                "traces_source": str(gdir.relative_to(DATA))
                if gdir.is_dir()
                else "none",
                "parent_meta": {
                    "idx": c.get("idx"),
                    "gen": gen,
                    "parent": c.get("parent"),
                    "agg": c.get("agg"),
                    "direction": c.get("mutation_direction", "")[:120],
                },
            }
        )
    return ctxs


# ---- delivery (run_delivery_v2) -------------------------------------------


def delivery_ctxs(n: int):
    cands = json.load(open(VITA / "candidates.json"))
    cands = [c for c in cands if c.get("persona_text")]
    # shared traces from the test conversations. Delivery conversations are long
    # (Chinese, multi-turn), so cap counts and snippet length to keep the prompt
    # under ~40K chars (gpt-5.2 reliably returns JSON there).
    N_S, N_F = 6, 4
    success, failure = [], []
    for cond, sink in (("test_no_persona", "s"), ("test_best_persona", "f")):
        d = VITA / "conversations" / cond
        if not d.is_dir():
            continue
        for tf in sorted(d.glob("*.json")):
            try:
                rec = json.load(open(tf))
            except Exception:
                continue
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue
            snip = _snip(msgs, head=3, tail=5)[:1800]
            if rec.get("reward") == 1.0 and len(success) < N_S:
                success.append(
                    {
                        "task_id": str(rec.get("task_id", tf.stem)),
                        "n_turns": len(msgs),
                        "conversation": snip,
                    }
                )
            elif rec.get("reward") == 0.0 and len(failure) < N_F:
                failure.append(
                    {"task_id": str(rec.get("task_id", tf.stem)), "conversation": snip}
                )
    scored = len(success) + len(failure)
    score = len(success) / scored if scored else 0.0
    ctxs = []
    for c in cands[:n]:
        ctxs.append(
            {
                "ctx_id": f"idx{c.get('idx')}_par{c.get('parent')}",
                "parent_persona": c["persona_text"],
                "score": score,
                "success": success[:N_S],
                "failure": failure[:N_F],
                "traces_source": "run_delivery_v2/conversations (shared test set, capped 6+4)",
                "parent_meta": {
                    "idx": c.get("idx"),
                    "parent": c.get("parent"),
                    "direction": c.get("mutation_direction", c.get("direction", ""))[
                        :120
                    ],
                },
            }
        )
    return ctxs


# ---- lic_math (published, blind) ------------------------------------------


def lic_ctxs(n: int):
    f = (
        _REPO
        / "results"
        / "published"
        / "pgepa_runs"
        / "gpt56luna_3domain"
        / "lic_math"
        / "isl_gpt56luna_lic_math_g0"
        / "candidates.json"
    )
    cands = [c for c in json.load(open(f)) if c.get("persona_text")]
    ctxs = []
    for c in cands[:n]:
        diff = c.get("difficulty_score")
        score = 1.0 - diff if isinstance(diff, (int, float)) else 0.0
        ctxs.append(
            {
                "ctx_id": f"idx{c.get('idx')}_par{c.get('parent')}",
                "parent_persona": c["persona_text"],
                "score": score,
                "success": [],
                "failure": [],
                "traces_source": "none (LiC search trajectory not preserved)",
                "parent_meta": {
                    "idx": c.get("idx"),
                    "parent": c.get("parent"),
                    "difficulty": diff,
                    "direction": c.get("mutation_direction", "")[:120],
                },
            }
        )
    return ctxs


# ---- colbench_code (sweetrl_full_001, blind) ------------------------------


def colbench_ctxs(n: int):
    cands = [c for c in json.load(open(COLBENCH_CANDS)) if c.get("persona_text")]
    ctxs = []
    for c in cands[:n]:
        ctxs.append(
            {
                "ctx_id": f"idx{c.get('idx')}_par{c.get('parent')}",
                "parent_persona": c["persona_text"],
                "score": 0.0,
                "success": [],
                "failure": [],
                "traces_source": "none (ColBench search trajectory not preserved)",
                "parent_meta": {
                    "idx": c.get("idx"),
                    "parent": c.get("parent"),
                    "direction": c.get("mutation_direction", c.get("direction", ""))[
                        :120
                    ],
                },
            }
        )
    return ctxs


BENCHES = {
    "telecom": telecom_ctxs,
    "delivery": delivery_ctxs,
    "lic_math": lic_ctxs,
    "colbench": colbench_ctxs,
}


def run_bench(name: str, ctxs: list, model: str, out_root: Path, api_key: str | None):
    bench_out = out_root / name
    bench_out.mkdir(parents=True, exist_ok=True)
    os.environ["MUTATION_DUMP_DIR"] = str(bench_out)
    print(f"\n===== {name} =====  {len(ctxs)} contexts")
    for i, ctx in enumerate(ctxs):
        ft = ctx["failure"][:4]
        prompt = _build_mutation_prompt(
            persona_text=ctx["parent_persona"],
            score=ctx["score"],
            traces=ctx["success"],
            history=[],
            pid=f"{name}_{ctx['ctx_id']}",
            severe_failures=None,
            timeout_rate=0.0,
            failure_traces=ft or None,
            mutation_context="enriched",
        )
        # persist ctx meta (separate from the dump call writes prompt+raw+parsed)
        meta = {
            "ctx_id": ctx["ctx_id"],
            "bench": name,
            "traces_source": ctx["traces_source"],
            "score": ctx["score"],
            "n_success": len(ctx["success"]),
            "n_failure": len(ctx["failure"]),
            "parent_meta": ctx["parent_meta"],
            "parent_persona": ctx["parent_persona"],
            "prompt_len": len(prompt),
        }
        (bench_out / f"ctx_{i:02d}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        print(
            f"  [{i}] {ctx['ctx_id']}  score={ctx['score']:.1%}  "
            f"traces(s={len(ctx['success'])},f={len(ctx['failure'])})  "
            f"prompt={len(prompt)}ch ... ",
            end="",
            flush=True,
        )
        try:
            res = call_mutation_llm(
                prompt, model, reasoning_effort="high", api_key=api_key
            )
            print(f"OK  dir={res.get('mutation_direction', '?')}")
        except Exception as e:
            print(f"FAILED: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--per-bench", type=int, default=5)
    ap.add_argument(
        "--out", default=str(_REPO / "results" / "mutation_artifacts_genesis")
    )
    ap.add_argument("--benches", nargs="+", default=list(BENCHES))
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("<"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    print(f"out -> {out}\nmodel={args.model} per_bench={args.per_bench}")
    for b in args.benches:
        ctxs = BENCHES[b](args.per_bench)
        run_bench(b, ctxs, args.model, out, api_key)

    print("\nArtifacts:")
    for f in sorted(out.glob("*/*.json")):
        print(f"  {f.relative_to(_REPO)}  ({f.stat().st_size}B)")


if __name__ == "__main__":
    main()
