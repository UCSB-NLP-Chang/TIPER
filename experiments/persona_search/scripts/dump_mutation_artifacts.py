#!/usr/bin/env python3
"""Dump real mutation-proposer artifacts (input prompt + raw LLM output + parsed child)
for tau2 telecom and VitaBench delivery, using existing trajectory data to build
reflective context. Does NOT run any agent simulation — only the proposer LLM call.

For each domain:
  1. load a parent persona (best searched persona)
  2. extract success / failure conversation traces from existing results
  3. build the real mutation prompt via ``_build_mutation_prompt`` (enriched)
  4. call the mutation LLM (``call_mutation_llm``)
  5. persist {prompt, raw, parsed} to results/mutation_artifacts/<domain>/

Set ``MUTATION_DUMP_DIR`` to capture every proposer call (see proposer.py).
Usage:
  .venv/bin/python experiments/persona_search/scripts/dump_mutation_artifacts.py \
      --mutations 3 --model gpt-5.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # tau2bench_clean root
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments" / "persona_search" / "scripts"))

try:  # load .env if present (never fatal)
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except Exception:
    pass

from cresta.persona.mutation.prompt import _build_mutation_prompt  # noqa: E402
from cresta.persona.mutation.proposer import call_mutation_llm  # noqa: E402

DATA_ROOT = Path(
    os.environ.get("TAU2BENCH_DATA_ROOT", str(_REPO.parent / "tau2bench-data"))
)
VITA_RUN = (
    DATA_ROOT
    / "artifacts"
    / "benchmarks"
    / "vita"
    / "persona_search"
    / "run_delivery_v2"
)


def _conv_snippet(messages: list[dict], head: int = 4, tail: int = 8) -> str:
    """Render a conversation as text: first `head` + last `tail` messages."""
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


# ---- tau2 telecom ----------------------------------------------------------


def telecom_parent_and_traces():
    """Parent = hardest persona from published telecom candidates; traces from
    existing telecom simulation results.json files in the data workspace."""
    # parent: hardest candidate from any published telecom run
    parent_text = None
    best_diff = -1.0
    for cf in (_REPO / "results" / "published").rglob("candidates.json"):
        if "telecom" not in str(cf).lower():
            continue
        try:
            cands = json.load(open(cf))
        except Exception:
            continue
        if not isinstance(cands, list):
            continue
        for c in cands:
            d = c.get("difficulty_score")
            if isinstance(d, (int, float)) and d > best_diff:
                best_diff = d
                parent_text = c.get("persona_text")
    if parent_text is None:
        # fallback: a known telecom seed persona
        seed = _REPO / "experiments" / "persona_search" / "seeds" / "telecom_seed.md"
        parent_text = seed.read_text() if seed.exists() else "# telecom customer (seed)"

    # traces: scan telecom results.json in the workspace
    success, failure = [], []
    sim_root = DATA_ROOT / "artifacts" / "simulations"
    if sim_root.is_dir():
        for rf in sim_root.glob("*telecom*/results.json"):
            try:
                sims = json.load(open(rf))
            except Exception:
                continue
            if isinstance(sims, dict):
                sims = sims.get("simulations", [])
            for s in sims:
                ri = s.get("reward_info") or {}
                r = ri.get("reward")
                msgs = s.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                snippet = _conv_snippet(msgs)
                if r == 1.0 and len(success) < 12:
                    success.append(
                        {
                            "task_id": s.get("task_id", "?"),
                            "n_turns": len(msgs),
                            "conversation": snippet,
                        }
                    )
                elif r == 0.0 and len(failure) < 8:
                    failure.append(
                        {"task_id": s.get("task_id", "?"), "conversation": snippet}
                    )
    return parent_text, success, failure


# ---- VitaBench delivery ----------------------------------------------------


def delivery_parent_and_traces():
    """Parent = run_delivery_v2/best_persona.md; traces from conversations/."""
    parent_text = (VITA_RUN / "best_persona.md").read_text()
    success, failure = [], []
    for cond, sink in (
        ("test_no_persona", "success"),
        ("test_best_persona", "failure"),
    ):
        d = VITA_RUN / "conversations" / cond
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
            snippet = _conv_snippet(msgs)
            entry = {
                "task_id": str(rec.get("task_id", tf.stem)),
                "conversation": snippet,
            }
            if rec.get("reward") == 1.0 and len(success) < 12:
                success.append({**entry, "n_turns": len(msgs)})
            elif rec.get("reward") == 0.0 and len(failure) < 8:
                failure.append(entry)
    return parent_text, success, failure


def run_domain(
    name: str,
    parent: str,
    success: list,
    failure: list,
    model: str,
    n: int,
    out_dir: Path,
    api_key: str | None,
) -> None:
    n_s, n_f = len(success), len(failure)
    scored = n_s + n_f
    score = n_s / scored if scored else 0.0
    print(
        f"\n=== {name} ===  parent={len(parent)}ch  success={n_s}  failure={n_f}  "
        f"agent_success_rate={score:.1%}"
    )
    if not success and not failure:
        print(f"  [warn] no traces for {name}; prompt will be blind")

    domain_out = out_dir / name
    domain_out.mkdir(parents=True, exist_ok=True)
    os.environ["MUTATION_DUMP_DIR"] = str(domain_out)

    ft = failure[:4]  # enriched uses up to 4 failure traces
    for i in range(n):
        # vary the trace subset across mutations for diversity
        rng = random.Random(1000 + i)
        tr = list(success)
        rng.shuffle(tr)
        tr = tr[:6]
        ft_i = list(ft)
        rng.shuffle(ft_i)

        prompt = _build_mutation_prompt(
            persona_text=parent,
            score=score,
            traces=tr,
            history=[],
            pid=f"{name}_dump",
            severe_failures=None,
            timeout_rate=0.0,
            failure_traces=ft_i or None,
            mutation_context="enriched",
        )
        meta = {
            "domain": name,
            "mutation_idx": i,
            "parent_persona_text": parent,
            "agent_success_rate": score,
            "n_success_traces": len(tr),
            "n_failure_traces": len(ft_i),
        }
        meta_path = domain_out / f"meta_{i:02d}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        print(f"  [{i}] calling {model} (prompt={len(prompt)}ch) ...")
        try:
            result = call_mutation_llm(
                prompt,
                model,
                reasoning_effort="high",
                api_key=api_key,
            )
            print(f"  [{i}] OK  direction={result.get('mutation_direction', '?')}")
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mutations", type=int, default=3, help="mutations per domain (default 3)"
    )
    ap.add_argument("--model", default="gpt-5.2", help="mutation LLM model")
    ap.add_argument(
        "--out",
        default=str(_REPO / "results" / "mutation_artifacts"),
        help="output dir",
    )
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["telecom", "delivery"],
        help="subset of domains",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("<"):
        print("ERROR: OPENAI_API_KEY not set in env", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    print(f"output -> {out_dir}\nmodel={args.model}  mutations/domain={args.mutations}")

    if "telecom" in args.domains:
        p, s, f = telecom_parent_and_traces()
        run_domain("telecom", p, s, f, args.model, args.mutations, out_dir, api_key)
    if "delivery" in args.domains:
        p, s, f = delivery_parent_and_traces()
        run_domain("delivery", p, s, f, args.model, args.mutations, out_dir, api_key)

    print("\nDone. Artifacts:")
    for d in out_dir.glob("*/*.json"):
        print(f"  {d.relative_to(_REPO)}  ({d.stat().st_size}B)")


if __name__ == "__main__":
    main()
