"""Mutation LLM callers and parallel propose/aggregate.

Extracted from lineage_loop.py — handles calling the mutation LLM,
distributing traces to groups, and two-phase parallel propose+aggregate.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path


def _extract_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # reasoning models sometimes prepend prose or trailing commentary;
        # fall back to the outermost {...} object.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_mutation_llm(
    prompt: str,
    model: str,
    max_retries: int = 3,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    import litellm

    is_claude = "claude" in model.lower() or "anthropic" in model.lower()
    # "xhigh" is an Anthropic extended-thinking level; OpenAI/o-series historically
    # only accept minimal|low|medium|high. Default per-provider, but allow an
    # explicit override (e.g. gpt-5.4 + xhigh) to pass straight through.
    effort = reasoning_effort or ("xhigh" if is_claude else "high")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": effort,
        "max_tokens": 16384,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if is_claude:
        kwargs["thinking"] = (
            {"type": "adaptive"}
            if "bedrock" in model.lower() or "vertex" in model.lower()
            else {"type": "enabled", "budget_tokens": 16384}
        )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = litellm.completion(**kwargs)
            raw = resp.choices[0].message.content

            if isinstance(raw, list):
                text = ""
                for block in raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            else:
                text = raw or ""

            result = _extract_json(text)
            if "persona_text" not in result:
                raise ValueError("missing 'persona_text' key in LLM response")
            _dump_mutation_call(prompt, model, effort, text, result)
            return result
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            last_err = e
            # dump the raw response on parse failure so failures are diagnosable
            try:
                _dump_mutation_call(
                    prompt, model, effort, text, {"_parse_error": str(e)}
                )
            except Exception:
                pass
            print(f"  [mutation] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
        except Exception as e:
            last_err = e
            print(f"  [mutation] attempt {attempt}/{max_retries} API error: {e}")
            if attempt < max_retries:
                time.sleep(5)
    raise RuntimeError(f"mutation LLM failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Two-phase parallel propose -> aggregate
# ---------------------------------------------------------------------------


def _distribute_traces_to_groups(
    success_traces: list[dict],
    failure_traces: list[dict],
    n_groups: int = 4,
    n_success_per_group: int = 2,
    n_failure_per_group: int = 1,
) -> list[dict]:
    """Distribute traces into n_groups via round-robin, no duplicate originals per group.

    Cycle-extends each pool to fill all slots, then assigns with offset so
    adjacent groups share one overlapping trace but never two identical ones.
    Returns [{"success": [...], "failure": [...]}, ...].
    """
    needed_success = n_groups * n_success_per_group
    needed_failure = n_groups * n_failure_per_group

    s_pool = list(success_traces)
    f_pool = list(failure_traces)
    while s_pool and len(s_pool) < needed_success:
        s_pool.extend(success_traces)
    while f_pool and len(f_pool) < needed_failure:
        f_pool.extend(failure_traces)
    s_pool = s_pool[:needed_success]
    f_pool = f_pool[:needed_failure]

    groups = []
    for g in range(n_groups):
        gs = (
            [s_pool[(g + j) % len(s_pool)] for j in range(n_success_per_group)]
            if s_pool
            else []
        )
        gf = (
            [
                f_pool[(g * n_failure_per_group + j) % len(f_pool)]
                for j in range(n_failure_per_group)
            ]
            if f_pool
            else []
        )
        groups.append({"success": gs, "failure": gf})
    return groups


def build_aggregate_prompt(
    parent_persona: str,
    proposals: list[dict],
    pid: str = "gepa",
) -> str:
    """Build a prompt that synthesizes multiple proposed personas into one."""
    proposals_section = ""
    for i, p in enumerate(proposals):
        direction = p.get("mutation_direction", "unknown")
        reasoning = p.get("reasoning", "")
        text = p.get("persona_text", "")
        proposals_section += f"\n### Proposal {i + 1}: {direction}\n"
        proposals_section += f"**Reasoning**: {reasoning}\n```\n{text}\n```\n"

    return f"""You are an expert adversarial persona designer for a customer-support simulation benchmark.

I provided an assistant with the following current persona instruction ({pid}):
```
{parent_persona}
```

{len(proposals)} parallel processes each independently proposed an updated persona based on
a different subset of evaluation traces. Here are the proposed updates:

{proposals_section}

Your task is to synthesize these proposed persona updates into a single, comprehensive persona.
Incorporate the key improvements and specific insights from all proposals. When proposals offer
complementary guidance, include all relevant details. When they conflict, prefer more specific
and task-relevant guidance.

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- When DIRECTLY asked for known info (name, DOB, account number), provide it accurately
- Never abandon the call while the problem is unresolved
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts
- Never fabricate facts not grounded in the scenario or tool results
- Never switch to a completely different goal mid-conversation
- Never accept outcomes contradicting explicit criteria
- Never make the conversation MEANINGLESS

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown wrapping, no explanation):
{{
  "mutation_direction": "<snake_case label describing the synthesized direction>",
  "reasoning": "<2-3 sentences: which proposal elements you combined and why>",
  "persona_text": "<the FULL synthesized persona text in markdown format>"
}}"""


def _dump_mutation_call(
    prompt: str, model: str, effort: str, raw: str, parsed: dict
) -> None:
    """If MUTATION_DUMP_DIR is set, persist the full proposer I/O for that call.

    Writes one JSON per call: {ts, model, effort, prompt, raw, parsed}. This is the
    only place the proposer's raw LLM output is captured — without it, only the
    parsed ``persona_text`` survives in candidates.json.
    """
    dump_dir = os.environ.get("MUTATION_DUMP_DIR")
    if not dump_dir:
        return
    try:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(prompt.encode()).hexdigest()[:6]
        rec = {
            "ts": ts,
            "model": model,
            "effort": effort,
            "prompt": prompt,
            "raw": raw,
            "parsed": parsed,
        }
        with open(Path(dump_dir) / f"{ts}_{h}.json", "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
    except Exception as e:  # dumping must never break a mutation
        print(f"  [mutation-dump] failed: {e}")


def call_propose_parallel(
    prompts: list[str],
    model: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> list[dict]:
    """Run multiple enriched mutation prompts in parallel via call_mutation_llm."""

    def _call_one(idx: int, prompt: str) -> dict | None:
        try:
            return call_mutation_llm(
                prompt,
                model,
                max_retries=max_retries,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"  [propose] group {idx} failed: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {pool.submit(_call_one, i, p): i for i, p in enumerate(prompts)}
        results = [None] * len(prompts)
        for f in concurrent.futures.as_completed(futures):
            results[futures[f]] = f.result()
    return [r for r in results if r is not None]
