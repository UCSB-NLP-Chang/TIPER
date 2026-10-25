#!/usr/bin/env python3
"""Generate N diverse initial personas for multi-start genesis, grounded in
actual agent rollout traces.

Reads a gen-0 no-persona evaluation (results.json), extracts success+failure
traces, then asks the LLM to propose N INDEPENDENT attack hypotheses — each
targeting a different behavioral mechanism observed in the traces. The output
personas are used as seeds for island-model GEPA.

Usage:
    # From an existing genesis run's gen-0 eval:
    python experiments/persona_search/scripts/generate_genesis_seeds.py \
        --domain airline \
        --results-json data/simulations/gepa_opt/.../results.json \
        --num-seeds 4 \
        --output-dir data/persona_opt/pgepa_runs/multistart_genesis_airline/seeds

    # Or from fullset no-persona data:
    python experiments/persona_search/scripts/generate_genesis_seeds.py \
        --domain airline \
        --results-json data/simulations/fullset/airline/shared_no_persona/results.json \
        --num-seeds 4 \
        --output-dir data/persona_opt/pgepa_runs/multistart_genesis_airline/seeds
"""

import argparse
import json
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lineage_loop import _get_scoring


def _format_msg(m: dict) -> str:
    role = m.get("role", "?")
    content = m.get("content", "")
    tool_calls = m.get("tool_calls")
    if tool_calls:
        calls = "; ".join(
            f"{tc.get('function', {}).get('name', '?')}({tc.get('function', {}).get('arguments', '')})"
            for tc in tool_calls
        )
        return (
            f"[{role}] {content}\n  TOOL_CALLS: {calls}"
            if content
            else f"[{role}] TOOL_CALLS: {calls}"
        )
    return f"[{role}] {content}"


def extract_traces(results_path: str, max_success: int = 10, max_failure: int = 6):
    data = json.loads(Path(results_path).read_text())
    success, failure = [], []
    for sim in data.get("simulations", []):
        reward = (sim.get("reward_info") or {}).get("reward")
        msgs = sim.get("messages", [])
        if not msgs:
            continue
        snippet = "\n".join(_format_msg(m) for m in msgs)
        entry = {
            "task_id": sim.get("task_id", "?"),
            "n_turns": len(msgs),
            "conversation": snippet,
        }
        if reward == 1.0:
            success.append(entry)
        elif reward == 0.0:
            failure.append(entry)
    success.sort(key=lambda t: t["n_turns"])
    failure.sort(key=lambda t: t["n_turns"])
    return success[:max_success], failure[:max_failure]


PROMPT_TEMPLATE = """\
You are analyzing real conversation traces between a customer service agent and a
user (with NO persona instructions). Your goal is to design {num_seeds} COMPLETELY
DIFFERENT adversarial user personas that exploit DIFFERENT vulnerabilities observed
in these traces.

{scoring_section}

## AGENT SUCCESS TRACES (agent solved the task — study what the agent relies on)
{success_traces}

## AGENT FAILURE TRACES (agent failed even without a persona — natural weak points)
{failure_traces}

## YOUR TASK
Based on the traces above, identify {num_seeds} DISTINCT behavioral mechanisms that
could cause the agent to fail MORE. Each mechanism must be:
1. **Trace-grounded**: point to specific moments in the traces that reveal the vulnerability
2. **Mechanistically distinct**: target a DIFFERENT part of the agent's reasoning/workflow
3. **Realistic**: something a real customer might plausibly do

For each, write a detailed user persona in markdown that a user-simulator LLM can follow.
The persona should describe specific BEHAVIORAL PATTERNS (how to speak, what to reveal
when, what to assume, how to refer to things) — not just "be difficult".

Return a JSON array of {num_seeds} objects:
```json
[
  {{
    "hypothesis": "1-2 sentence causal chain: if user does X, agent will mistakenly do Y, because Z (cite trace evidence)",
    "persona_text": "full persona in markdown"
  }},
  ...
]
```

Output ONLY the JSON array (inside a ```json fence is OK), no other text."""


def generate_seeds(
    domain: str, results_path: str, num_seeds: int, model: str, effort: str
) -> list[dict]:
    success, failure = extract_traces(results_path)
    print(f"  Extracted {len(success)} success + {len(failure)} failure traces")

    scoring = _get_scoring(domain)
    prompt = PROMPT_TEMPLATE.format(
        num_seeds=num_seeds,
        scoring_section=scoring,
        success_traces=json.dumps(success, indent=2)
        if success
        else "No success traces available.",
        failure_traces=json.dumps(failure, indent=2)
        if failure
        else "No failure traces available.",
    )

    print(f"  Prompt length: {len(prompt)} chars")
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        reasoning_effort=effort,
    )
    text = resp.choices[0].message.content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument(
        "--results-json",
        required=True,
        help="path to a no-persona results.json with agent traces",
    )
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--model", default="openai/gpt-5.5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_seeds} trace-grounded seeds for {args.domain}")
    print(f"  traces from: {args.results_json}")
    seeds = generate_seeds(
        args.domain, args.results_json, args.num_seeds, args.model, args.effort
    )

    for i, s in enumerate(seeds):
        name = f"genesis_seed_{i:02d}"
        persona_path = out / f"{name}.md"
        persona_path.write_text(s["persona_text"])
        hyp = s.get("hypothesis", "?")
        print(f"  [{name}] {hyp[:100]}")
        print(f"    -> {persona_path} ({len(s['persona_text'])} chars)")

    meta_path = out / "seeds_meta.json"
    json.dump(seeds, meta_path.open("w"), indent=2, ensure_ascii=False)
    print(f"\nMetadata -> {meta_path}")


if __name__ == "__main__":
    main()
