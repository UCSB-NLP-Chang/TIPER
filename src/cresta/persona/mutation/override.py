"""Mutation override, single-mutation, and diversity-gated mutation loop.

Extracted from parallel_gepa.py — handles the GEPA_MODE_OVERRIDE string,
MutationContextBuilder, and the mutate_valid_diverse retry loop.
"""

from __future__ import annotations

import hashlib
from typing import Any

from cresta.persona.diversity.seen_pool import summarize_persona
from cresta.persona.eval import adapter as tpa
from cresta.persona.eval.adapter import build_custom_proposer
from cresta.persona.mutation.lint import (
    _MUTATION_EXAMPLES,
    _render_invalid_cases,
    _struct_check,
    lint_persona,
    register_invalid,
)

# ---------------------------------------------------------------------------
# Mutation-mode override
# ---------------------------------------------------------------------------

_BASE_OVERRIDE = """

## GEPA MODE OVERRIDE (serial Pareto search)
The numeric score above is a noisy diagnostic — do NOT overfit to it. Use the
concrete agent-success traces only to diagnose WHICH KIND of agent behavior to
counter.

Mutation constraints for this mode:
- Substantially rewrite 30% of the persona: restructure freely, REPLACE weak
  rules, merge or drop redundant ones. A near-verbatim copy is a failed mutation.
- Commit to ONE clear adversarial direction this time (a specific loophole or
  misdirection style), not a little of everything.
- The persona MUST stay a GENERAL, reusable behavioral profile for any customer service task. Do NOT insert any
  task-specific or task-conditioned content: no concrete account numbers, customer
  names, dates, dollar amounts, ticket/order/case IDs, task IDs, or any detail
  copied from the example conversations or tied to one particular scenario. The
  traces tell you what kind of agent move to defeat, NEVER content to embed. A
  persona that names, encodes, or branches on a specific task is invalid and will be rejected here.
- Preserve ALL validity-critical constraints: no
  fabrication of facts the user cannot know, no illegal withholding. Staying valid
  is mandatory — a gated persona scores ZERO regardless of difficulty.
"""


# ---------------------------------------------------------------------------
# MutationContextBuilder — pluggable mutation context strategies
# ---------------------------------------------------------------------------


class MutationContextBuilder:
    """Pluggable mutation context strategies.

    Add a new strategy:
    1. Add a build_xxx() method
    2. Add 'xxx' to --mutation-context choices in argparse
    3. Route in build_explored() / build_failure_traces()
    """

    def __init__(
        self,
        mode: str,
        gen_logs: list[dict],
        candidates: list[Any],
        seen_personas: list[str],
    ):
        self.mode = mode
        self.gen_logs = gen_logs
        self.candidates = candidates
        self.seen_personas = seen_personas

    def build_explored(self) -> list[str] | None:
        if self.mode in ("enriched", "twophase"):
            return self._explored_with_outcomes()
        return self._explored_basic()

    def _explored_basic(self) -> list[str] | None:
        if not self.seen_personas:
            return None
        return [summarize_persona(p) for p in self.seen_personas]

    @staticmethod
    def _summarize_persona_brief(text: str, max_sections: int = 4) -> str:
        """Structured summary: opening sentence + key ## sections with first
        bullet (capped). No LLM call."""
        lines_raw = text.splitlines()
        opening = ""
        for ln in lines_raw:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if len(s) < 20:
                continue
            opening = s.lstrip("- ").rstrip()
            break
        sections: list[str] = []
        cur_header: str | None = None
        skip = {"invariants", "invariants (never break these)"}
        for ln in lines_raw:
            s = ln.strip()
            if s.startswith("## "):
                h = s[3:].strip()
                cur_header = None if h.lower() in skip else h
            elif cur_header and s.startswith("- ") and len(s) > 15:
                bullet = s[2:].strip()
                if len(bullet) > 60:
                    bullet = bullet[:57] + "..."
                sections.append(f"[{cur_header}] {bullet}")
                cur_header = None
                if len(sections) >= max_sections:
                    break
        parts: list[str] = []
        if opening:
            if len(opening) > 100:
                opening = opening[:97] + "..."
            parts.append(opening)
        if sections:
            parts.append(" | ".join(sections))
        return "\n  ".join(parts) if parts else ""

    def _explored_with_outcomes(self) -> list[str]:
        lines: list[str] = []
        for entry in self.gen_logs:
            g = entry.get("gen", "?")
            direction = entry.get("mutation_direction", "?")
            accepted = entry.get("accepted", False)
            if accepted:
                idx = entry.get("idx", "?")
                agg = entry.get("difficulty_agg")
                agg_str = f"{agg:.3f}" if agg is not None else "?"
                header = (
                    f"gen{g} dir={direction} val_diff={agg_str} (idx{idx}, ACCEPTED)"
                )
            else:
                reason = entry.get("accept_reason", "rejected")
                mb = entry.get("minibatch_mean", 0)
                pmb = entry.get("parent_minibatch_mean", 0)
                header = (
                    f"gen{g} dir={direction} mb={mb:.3f}(parent={pmb:.3f}) "
                    f"REJECTED: {reason}"
                )
            persona_text = entry.get("persona_text", "")
            brief = self._summarize_persona_brief(persona_text) if persona_text else ""
            if brief:
                lines.append(f"{header}\n  Summary: {brief}")
            else:
                lines.append(header)
        if not lines:
            return self._explored_basic() or []
        return lines


# ---------------------------------------------------------------------------
# Mutation with diversity + lint gates
# ---------------------------------------------------------------------------


def _build_override(
    args,
    invalid_cases: list[dict],
    hint: str,
    explored_summaries: list[str] | None = None,
) -> str:
    """tpa.GEPA_MODE_OVERRIDE for one mutation attempt: base rules + GOOD/BAD
    examples + the run's observed invalid cases + (optionally) short summaries of
    personas ALREADY explored by the other parallel islands + a corrective hint."""
    ov = (
        _BASE_OVERRIDE
        + _MUTATION_EXAMPLES
        + "\n## Invalid cases observed in THIS run (do NOT reproduce)\n"
        + _render_invalid_cases(invalid_cases)
    )
    if explored_summaries:
        ov += (
            "\n## ALREADY EXPLORED personas (do NOT rehash these "
            "directions — your child must be meaningfully DIFFERENT from all of them)\n"
            + "\n".join(f"- {s}" for s in explored_summaries)
            + "\n"
        )
    return ov + (hint or "")


_SENTINEL = object()


def _mutate_once(
    args,
    parent,
    refl_adapter,
    override: str,
    cached_proposals: list[dict] | None = None,
    trace_eval_batch=_SENTINEL,
) -> dict:
    """One reflective mutation of the parent (own proposer instance). `override`
    is injected via tpa.GEPA_MODE_OVERRIDE so each attempt carries fresh hints.
    Returns {"persona_text": ..., "mutation_direction": ...}.

    When *trace_eval_batch* is given it overrides parent.eval_batch for trace
    extraction (Plan C: minibatch traces instead of val traces)."""
    if args.dry_run:
        tag = hashlib.md5((parent.persona_text + override).encode()).hexdigest()[:6]
        return {
            "persona_text": parent.persona_text
            + f"\n\n<!-- dry-run mutation {tag} -->\n",
            "mutation_direction": f"dry_run_{tag}",
        }
    tpa.GEPA_MODE_OVERRIDE = override
    proposer = build_custom_proposer(
        args.mutation_model,
        args.mutation_edit_pct,
        args.mutation_effort,
        mutation_context=getattr(args, "mutation_context", "basic"),
        reflect_model=getattr(args, "reflect_model", None),
        reflect_effort=getattr(args, "reflect_effort", "medium"),
    )
    cand = {"persona_text": parent.persona_text}
    eb = trace_eval_batch if trace_eval_batch is not _SENTINEL else parent.eval_batch
    if eb is None:
        reflective = {"persona_text": []}
    else:
        reflective = refl_adapter.make_reflective_dataset(cand, eb, ["persona_text"])
    result = proposer(
        cand, reflective, ["persona_text"], _cached_proposals=cached_proposals
    )
    _mutate_once._last_proposer = proposer
    return result


_mutate_once._last_proposer = None


def mutate_valid_diverse(
    args,
    parent,
    refl_adapter,
    seen: list[str],
    invalid_cases: list[dict],
    explored_summaries: list[str] | None = None,
    trace_eval_batch=_SENTINEL,
) -> dict:
    """Mutate the parent into ONE child that (a) is novel vs ALL seen personas
    (layered lexical + semantic gate) and (b) passes the static linter. Retries up
    to --max-mutation-attempts, feeding the failure reason + observed invalid cases
    back into the prompt; keeps the best attempt if none fully clears (never
    stalls). Returns a meta dict."""
    # Import here to avoid circular imports at module level; these are defined
    # in parallel_gepa.py (the caller) which wires them up at runtime.
    from cresta.persona.diversity.metrics import min_lexical_dissim as _lexical_dissim
    from cresta.persona.search.parallel_gepa import diversity_check

    if args.dry_run:
        r = _mutate_once(
            args,
            parent,
            refl_adapter,
            _build_override(args, invalid_cases, "", explored_summaries),
            trace_eval_batch=trace_eval_batch,
        )
        text = r["persona_text"]
        return {
            "text": text,
            "mutation_direction": r.get("mutation_direction"),
            "lexical_dissim": _lexical_dissim(text, seen),
            "semantic_sim": None,
            "diversity_ok": True,
            "lint": None,
            "lint_ok": True,
            "attempts": 1,
            "gate_ok": True,
        }

    best: dict | None = None
    hint = ""
    cached_proposals = None
    is_twophase = getattr(args, "mutation_context", "basic") == "twophase"

    for attempt in range(1, args.max_mutation_attempts + 1):
        r = _mutate_once(
            args,
            parent,
            refl_adapter,
            _build_override(args, invalid_cases, hint, explored_summaries),
            cached_proposals=cached_proposals,
            trace_eval_batch=trace_eval_batch,
        )
        text = r["persona_text"]
        direction = r.get("mutation_direction")

        # After first twophase attempt, cache Phase 1 proposals for reuse
        if cached_proposals is None and is_twophase and _mutate_once._last_proposer:
            cached_proposals = _mutate_once._last_proposer.get_cached_proposals()

        div_ok, lex, sem, div_reason, nearest_snippet = diversity_check(
            args, text, seen
        )
        struct = _struct_check(text)
        lint = struct or lint_persona(args, text, invalid_cases)
        lint_ok = lint is None or lint.get("verdict") == "ok"
        meta = {
            "text": text,
            "mutation_direction": direction,
            "lexical_dissim": lex,
            "semantic_sim": sem,
            "diversity_ok": div_ok,
            "lint": lint,
            "lint_ok": lint_ok,
            "attempts": attempt,
            "gate_ok": div_ok and lint_ok,
        }
        if meta["gate_ok"]:
            return meta
        if best is None or (lint_ok, lex) > (best["lint_ok"], best["lexical_dissim"]):
            best = meta
        reasons = []
        if not div_ok:
            near_info = ""
            if nearest_snippet:
                near_info = (
                    f"\nMost similar existing persona (DIFFERENTIATE from this):\n"
                    f"```\n{nearest_snippet}\n```"
                )
            reasons.append(
                f"too similar to an existing persona "
                f"({div_reason}, lexical_dissim={lex:.0%}){near_info}"
            )
        if not lint_ok:
            lr = (lint or {}).get("reasons", "failed lint")
            reasons.append(f"failed persona lint: {lr}")
            register_invalid(invalid_cases, "lint", text, lr)
        hint = (
            f"\n- RETRY {attempt + 1}: the previous attempt was rejected because "
            f"{'; '.join(reasons)}. Produce a DISTINCT, valid, GENERAL persona "
            f"that fixes this."
        )
        print(
            f"    [gate] attempt {attempt}/{args.max_mutation_attempts} rejected: "
            f"{'; '.join(reasons)}; retrying"
        )
    print(
        f"    [gate] no attempt fully cleared; keeping best "
        f"(lexical_dissim={best['lexical_dissim']:.0%}, lint_ok={best['lint_ok']})"
    )
    return best
