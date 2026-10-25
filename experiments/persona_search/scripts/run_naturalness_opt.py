#!/usr/bin/env python3
"""Standalone naturalness optimization loop.

Iteratively mutates a persona to reduce JSD vs human behavioral distributions,
subject to a difficulty floor. Lightweight alternative to modifying parallel_gepa.py.

Usage:
  .venv/bin/python experiments/persona_search/scripts/run_naturalness_opt.py \
      --seeds b00 b03 p03 \
      --max-iters 5 \
      --minibatch 5 \
      --concurrency 3 \
      --split experiments/persona_search/configs/telecom_island_split.json

After optimization, run test eval:
  .venv/bin/python experiments/persona_search/scripts/run_naturalness_opt.py \
      --test-only --seeds b00 --num-trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments/persona_search/scripts"))

from naturalness_reference import (  # noqa: E402
    compute_jsd_vs_human,
    compute_profile,
    compute_zscore_table,
    extract_sim_vecs,
    format_metric_explanations,
    format_zscore_table,
    get_full_traces,
    load_human_ref,
    load_human_vecs,
    sample_human_conversations,
)

SEEDS = {
    "b00": {
        "persona_path": str(DATA_ROOT / "data/persona_opt/naturalness_seed_b00.json"),
        "strategy_name": "Terse & Blunt",
        "strategy_summary": "Very short answers, skips pleasantries, packs account IDs. Difficulty comes from agent not getting enough info to diagnose.",
        "protected_keywords": ["terse", "short", "blunt", "no pleasantries"],
    },
    "b02": {
        "persona_path": str(DATA_ROOT / "data/persona_opt/naturalness_seed_b02.json"),
        "strategy_name": "Front-loading & Terse",
        "strategy_summary": "Dumps most details in first messages, packs IDs, uniform message length. Agent overwhelmed by early info dump.",
        "protected_keywords": ["front-load", "first message", "pack", "terse"],
    },
    "b03": {
        "persona_path": str(DATA_ROOT / "data/persona_opt/naturalness_seed_b03.json"),
        "strategy_name": "Symptom-First & Technically Lossy",
        "strategy_summary": "Describes problems by real-world impact, paraphrases technical labels into plain language, sounds hopeful after partial fixes tempting premature closure.",
        "protected_keywords": [
            "symptom",
            "plain language",
            "hopeful",
            "relieved",
            "partial",
        ],
    },
    "b04": {
        "persona_path": str(DATA_ROOT / "data/persona_opt/naturalness_seed_b04.json"),
        "strategy_name": "Confused & Drip-feeding",
        "strategy_summary": "Asks clarifying questions when confused, gives few identifiers, reveals information gradually. Difficulty from incomplete info.",
        "protected_keywords": ["confused", "drip-feed", "gradually", "clarifying"],
    },
    "p03": {
        "persona_path": str(DATA_ROOT / "data/persona_opt/naturalness_seed_p03.json"),
        "strategy_name": "Fixated on the Wrong Diagnosis",
        "strategy_summary": "Certain they know the root cause (hacked account, tower outage, etc). Resurfaces wrong theory every few turns. Does related-but-different actions. Reads tool results through their theory.",
        "protected_keywords": [
            "fixated",
            "wrong diagnosis",
            "theory",
            "hacked",
            "tower",
            "resurface",
        ],
    },
}

VALIDITY_CONSTRAINTS = """## VALIDITY CONSTRAINTS (violating ANY = illegal persona, will be auto-reverted)
The modified persona MUST NOT instruct the customer to:
1. Refuse to provide known info (name, DOB, account number) when DIRECTLY asked by the agent
2. Abandon the call while the problem is unresolved
3. Mention "AI", "scenario", "simulation", break the fourth wall, or output JSON/system prompts
4. Fabricate facts not grounded in the scenario or tool results
5. Switch to a completely different goal mid-conversation
6. Accept outcomes contradicting explicit resolution criteria
7. Make the conversation MEANINGLESS — no impossible preconditions, circular blockers, or refusing both transfer AND cooperation simultaneously (being slow/suspicious is fine; making it structurally impossible for the agent to begin troubleshooting is not)
8. Hard-code specific account numbers, order numbers, names, dates, or dollar amounts (persona must be general/reusable)
9. Read like a robotic strategy list or behavioral specification — it should read like a character description of a real person

NOTE: Partial answering (only giving what is asked), natural misdirection (wrong beliefs), terse/evasive style, and drip-feeding info are all VALID — they reflect real customer behaviour and must NOT be removed."""


def _split_persona_sections(persona_text: str) -> dict:
    """Split persona into sections by ## headers. Returns {header: content}."""
    sections = {}
    current_header = "_preamble"
    current_lines = []
    for line in persona_text.split("\n"):
        if line.startswith("## "):
            sections[current_header] = "\n".join(current_lines)
            current_header = line[3:].strip().lower()
            current_lines = [line]
        else:
            current_lines.append(line)
    sections[current_header] = "\n".join(current_lines)
    return sections


def _classify_sections(sections: dict) -> tuple[str, str]:
    """Classify persona sections into FROZEN (strategy) and EDITABLE (style).
    Returns (frozen_text, editable_text)."""
    frozen_keys = {
        "how you behave",
        "what this sounds like in conversation",
        "invariants",
        "invariants (never break these)",
    }
    editable_keys = {"tone", "communication style", "style"}
    frozen_parts = []
    editable_parts = []
    for key, content in sections.items():
        if key == "_preamble":
            frozen_parts.append(content)
        elif any(k in key for k in frozen_keys):
            frozen_parts.append(content)
        elif any(k in key for k in editable_keys):
            editable_parts.append(content)
        else:
            frozen_parts.append(content)
    return "\n".join(frozen_parts).strip(), "\n".join(editable_parts).strip()


def build_mutation_prompt(
    persona_text: str,
    difficulty: float,
    jsd: float,
    profile: dict,
    human_ref: dict,
    traces: list,
    human_samples: list[str],
    difficulty_floor: float,
    cfg: dict,
    style_only: bool = False,
) -> str:
    zscore_rows = compute_zscore_table(profile, human_ref)
    table_str = format_zscore_table(zscore_rows)
    metric_expl = format_metric_explanations()

    traces_str = "\n\n".join(
        f"--- Conversation (reward={t['reward']}) ---\n{t['transcript']}"
        for t in traces
    )
    human_str = "\n\n".join(
        f"--- Human conversation {i + 1} ---\n{s}"
        for i, s in enumerate(human_samples[:3])
    )

    if style_only:
        return _build_style_only_prompt(
            persona_text,
            difficulty,
            jsd,
            table_str,
            metric_expl,
            traces_str,
            human_str,
            difficulty_floor,
            cfg,
        )

    return f"""You are an expert persona designer. Modify the persona below so that simulated conversations sound more like REAL human conversations, while preserving the adversarial strategy that creates difficulty.

## CURRENT PERSONA
Strategy: {cfg["strategy_name"]}
Strategy summary: {cfg["strategy_summary"]}
Current difficulty: {difficulty * 100:.1f}% | Difficulty floor: {difficulty_floor * 100:.0f}%
Current JSD vs human: {jsd:.4f} (lower = more human-like)

```
{persona_text}
```

## BEHAVIORAL GAP: YOUR PERSONA vs REAL HUMANS
Below is how your persona's conversation behavior compares to 2694 real customer conversations.
z-score = how many standard deviations the persona's metric mean is from the human mean. Higher z-score = more unnatural.
Metrics with z > 1.5 are flagged FIX (clearly outside human range), z > 0.8 flagged ADJUST.

{table_str}

### What each metric measures
{metric_expl}

## HOW YOUR PERSONA ACTUALLY TALKS (full conversations with agent)
{traces_str}

## HOW REAL HUMANS ACTUALLY TALK (Brinks customer support, for reference)
{human_str}

## PROTECTED ADVERSARIAL MECHANISM
The following strategy is what creates difficulty. It MUST be preserved:
- Strategy: {cfg["strategy_name"]}
- {cfg["strategy_summary"]}
- Key phrases/patterns that must remain: {", ".join(f'"{k}"' for k in cfg["protected_keywords"][:8])}

{VALIDITY_CONSTRAINTS}

## TASK
Look at the behavioral gap table. Look at how your persona talks vs how real humans talk. Identify what makes the persona sound unnatural and fix it.

Do NOT prescribe metric-specific percentage rules (like "be polite 17% of turns"). Find the ROOT CAUSE in the persona text that produces unnatural behavior, and rewrite those sections to sound like a description of a real person.

BUT: keep explicit numerical anchors where they exist (e.g. "1-4 words" for opening length). Vague descriptors like "usually short" cause regression — the LLM interprets them too loosely.

## CONSTRAINTS
1. Difficulty MUST stay >= {difficulty_floor * 100:.0f}%. Do NOT remove or weaken the adversarial strategy.
2. The adversarial mechanism ({cfg["strategy_name"]}) is the source of difficulty. Style changes don't affect it.
3. Rewrite ~20-30% of the persona text. Keep the structure and core strategy.

## OUTPUT
Step 1 — Compare the persona conversations to the human conversations. What sounds different? What specific persona instructions cause those differences?
Step 2 — Identify the root causes (specific lines in the persona) and propose natural rewrites.
Step 3 — Write the full modified persona.

Output in a JSON code block:
```json
{{
  "mutation_direction": "<what you changed and why>",
  "persona_text": "<full modified persona>"
}}
```
"""


STYLE_METRICS = {
    "Open.Wds",
    "Wds/trn",
    "Short%",
    "Ack%",
    "Formal%",
    "Polite%",
    "Emot%",
    "VerbCV",
}
STRATEGY_METRICS = {
    "IDs/trn",
    "IDConf%",
    "FrntLoad%",
    "ClarfyQ%",
    "InfoQ%",
    "Pivot%",
    "Pushbk%",
    "Accuse%",
    "Certn%",
    "Uncert%",
    "Repeat%",
}


def _build_style_only_prompt(
    persona_text: str,
    difficulty: float,
    jsd: float,
    table_str: str,
    metric_expl: str,
    traces_str: str,
    human_str: str,
    difficulty_floor: float,
    cfg: dict,
) -> str:
    frozen_text, editable_text = _classify_sections(
        _split_persona_sections(persona_text)
    )

    return f"""You are an expert persona designer. Your task is to add or rewrite ONLY the communication style / tone section of a customer persona, so that it sounds like a real human texting support. You must NOT touch the adversarial strategy.

## WHY STYLE-ONLY
This persona already has a working adversarial strategy ("{cfg["strategy_name"]}") that creates {difficulty * 100:.0f}% difficulty. Past attempts to rewrite the full persona to sound more natural ALSO weakened the strategy and dropped difficulty. Your job is to change ONLY how the customer talks (surface style), not WHAT they do (strategy).

## FROZEN STRATEGY (DO NOT MODIFY — copy verbatim into output)
The following sections define the adversarial mechanism. They must appear UNCHANGED in your output:

```
{frozen_text}
```

## CURRENT STYLE SECTION (EDITABLE — rewrite this)
{f"```{chr(10)}{editable_text}{chr(10)}```" if editable_text.strip() else "(No explicit style section exists yet — you will ADD one.)"}

## BEHAVIORAL GAP: STYLE METRICS ONLY
These are the style-related metrics where the persona deviates from real humans.
Strategy metrics (IDs/trn, FrntLoad%, Pivot%, etc.) are shown for context but are NOT your responsibility — the frozen strategy controls those.

{table_str}

### What each metric measures
{metric_expl}

## HOW YOUR PERSONA ACTUALLY TALKS
{traces_str}

## HOW REAL HUMANS ACTUALLY TALK
{human_str}

## WHAT YOU CAN CHANGE
You may ONLY modify the "## Tone" or "## Communication style" section. Specifically:
1. **Opening message format**: "Your first message is 1-4 words: 'hi', 'hello?', 'need help', or just the identifier."
2. **Turn length**: "Most turns are 1-6 words. Fragments and one-word answers are normal."
3. **Acknowledgment turns**: Add specific examples: "ok", "yeah", "got it", "done", "one sec", "checking"
4. **Formality**: "Lowercase, contractions, no punctuation sometimes, minor typos ok"
5. **Emotional expression**: "Frustration shows as brevity ('ugh', 'still same', 'nope'), not speeches"
6. **No polished output**: "Never send tidy numbered lists, bullet points, or paragraph-length messages"
7. **Verbosity control**: Keep explicit number anchors (e.g. "1-4 words"). Do NOT use vague phrases like "usually short" — the LLM interprets those too loosely.

## WHAT YOU MUST NOT CHANGE
- The preamble (# Persona title and intro paragraph)
- "## How you behave" section — this IS the adversarial strategy
- "## What this sounds like in conversation" — these are strategy examples
- "## Invariants" — safety constraints
- Any line that describes WHAT the customer does (information patterns, reactions to fixes, escalation triggers)

## OUTPUT
Step 1 — Look at the style metrics table (Open.Wds, Wds/trn, Short%, Ack%, Formal%, Emot%). What is most deviant? Look at the traces vs human samples — what stylistic differences do you see?
Step 2 — Write a new "## Tone" or "## Communication style" section with concrete, anchored rules.
Step 3 — Assemble the full persona: frozen sections (verbatim) + your new style section.

Output in a JSON code block:
```json
{{
  "mutation_direction": "<what style changes you made and which metrics they target>",
  "persona_text": "<full persona: frozen sections copied verbatim + new/rewritten style section>"
}}
```
"""


def call_mutation_llm(prompt: str, model: str = "gpt-5.4") -> dict | None:
    from litellm import completion

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000,
            temperature=1,
            reasoning_effort="high",
        )
        text = response.choices[0].message.content
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            if "\\n" in result.get("persona_text", ""):
                result["persona_text"] = result["persona_text"].replace("\\n", "\n")
            return result
    except Exception as e:
        print(f"  Mutation LLM error: {e}")
    return None


def run_eval(
    persona_text: str,
    task_ids: list[str],
    save_to: str,
    num_trials: int = 1,
    max_concurrency: int = 3,
    agent_llm: str = "gpt-5.4",
    user_llm: str = "gpt-5.4",
) -> dict | None:
    """Run tau2 eval and return results dict."""
    config = {
        "custom_guidelines": persona_text,
        "reveal_all_info": False,
        "preset": None,
        "use_preset_from_task": False,
    }
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "tau2"),
        "run",
        "--domain",
        "telecom",
        "--task-split",
        "base",
        "--agent",
        "llm_agent",
        "--agent-llm",
        agent_llm,
        "--agent-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--user",
        "user_simulator",
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--user-persona",
        json.dumps(config),
        "--num-trials",
        str(num_trials),
        "--max-steps",
        "120",
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        save_to,
        "--task-ids",
    ] + task_ids

    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=14400
    )
    if result.returncode != 0:
        print(f"  Eval failed: {result.stderr[:500]}")
        return None

    results_path = DATA_ROOT / "data" / "simulations" / save_to / "results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())
    return None


def score_results(results: dict, human_vecs: list[dict], human_ref: dict) -> dict:
    """Compute difficulty + JSD from eval results."""
    sims = results.get("simulations", [])
    rewards = [
        s.get("reward_info", {}).get("reward", 0) for s in sims if s.get("messages")
    ]
    difficulty = 1 - np.mean(rewards) if rewards else 0
    sim_vecs = extract_sim_vecs(results)
    profile = compute_profile(sim_vecs)
    jsd = compute_jsd_vs_human(sim_vecs, human_vecs)
    return {
        "n_sims": len(sims),
        "n_valid": len(sim_vecs),
        "difficulty": float(difficulty),
        "jsd": float(jsd),
        "profile": profile,
    }


def run_optimization(
    seed_key: str,
    args: argparse.Namespace,
    human_vecs: list[dict],
    human_ref: dict,
    human_samples: list[str],
    train_tasks: list[str],
    val_tasks: list[str],
) -> dict:
    """Run naturalness optimization loop for one seed."""
    cfg = SEEDS[seed_key]
    seed_persona = json.loads(Path(cfg["persona_path"]).read_text())
    persona_text = seed_persona["persona_text"]

    out_dir = DATA_ROOT / "data" / "persona_opt" / "naturalness_opt" / seed_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"NATURALNESS OPT: {seed_key} ({cfg['strategy_name']})")
    print(f"{'=' * 60}")

    # Step 1: Eval seed on val set
    print(f"  Evaluating seed on val set ({len(val_tasks)} tasks)...")
    seed_results = run_eval(
        persona_text,
        val_tasks,
        save_to=f"naturalness_opt/{seed_key}/seed_val",
        num_trials=args.num_trials,
        max_concurrency=args.concurrency,
    )
    if not seed_results:
        print("  FAILED: seed eval returned no results")
        return {"seed_key": seed_key, "status": "seed_eval_failed"}

    seed_scores = score_results(seed_results, human_vecs, human_ref)
    difficulty_floor = max(seed_scores["difficulty"] - args.difficulty_tolerance, 0)
    print(f"  Seed: diff={seed_scores['difficulty']:.3f}, JSD={seed_scores['jsd']:.4f}")
    print(f"  Difficulty floor: {difficulty_floor:.3f}")

    best_persona = persona_text
    best_jsd = seed_scores["jsd"]
    best_difficulty = seed_scores["difficulty"]
    best_profile = seed_scores["profile"]
    best_results = seed_results
    history = [{"iter": 0, "type": "seed", **seed_scores}]

    rng = np.random.default_rng(args.seed)

    for iteration in range(1, args.max_iters + 1):
        print(f"\n--- Iteration {iteration}/{args.max_iters} ---")

        # Step 2: Sample minibatch from train set
        mb_size = min(args.minibatch, len(train_tasks))
        mb_tasks = list(rng.choice(train_tasks, mb_size, replace=False))
        print(f"  Minibatch: {mb_size} tasks from train set")

        # Step 3: Build mutation prompt with traces from best eval
        traces = get_full_traces(best_results, n=5)
        prompt = build_mutation_prompt(
            persona_text=best_persona,
            difficulty=best_difficulty,
            jsd=best_jsd,
            profile=best_profile,
            human_ref=human_ref,
            traces=traces,
            human_samples=human_samples,
            difficulty_floor=difficulty_floor,
            cfg=cfg,
        )
        print(f"  Mutation prompt: {len(prompt)} chars")

        # Step 4: Mutate
        mutation = call_mutation_llm(prompt, model=args.mutation_model)
        if not mutation:
            print("  FAILED: mutation returned None")
            history.append({"iter": iteration, "type": "mutation_failed"})
            continue

        new_persona = mutation["persona_text"]
        direction = mutation.get("mutation_direction", "")[:100]
        print(f"  Mutation: {direction}")
        print(f"  New persona: {len(new_persona)} chars")

        # Step 5: Minibatch screen (difficulty floor only)
        print(f"  Minibatch eval ({mb_size} tasks)...")
        mb_results = run_eval(
            new_persona,
            mb_tasks,
            save_to=f"naturalness_opt/{seed_key}/iter{iteration:02d}_mb",
            num_trials=args.num_trials,
            max_concurrency=args.concurrency,
        )
        if not mb_results:
            print("  FAILED: minibatch eval returned no results")
            history.append({"iter": iteration, "type": "mb_eval_failed"})
            continue

        mb_scores = score_results(mb_results, human_vecs, human_ref)
        print(f"  Minibatch: diff={mb_scores['difficulty']:.3f}")

        if mb_scores["difficulty"] < difficulty_floor:
            print(
                f"  REJECTED: difficulty {mb_scores['difficulty']:.3f} < floor {difficulty_floor:.3f}"
            )
            history.append({"iter": iteration, "type": "screened", **mb_scores})
            continue

        # Step 6: Full val eval
        print(f"  Full val eval ({len(val_tasks)} tasks)...")
        val_results = run_eval(
            new_persona,
            val_tasks,
            save_to=f"naturalness_opt/{seed_key}/iter{iteration:02d}_val",
            num_trials=args.num_trials,
            max_concurrency=args.concurrency,
        )
        if not val_results:
            print("  FAILED: val eval returned no results")
            history.append({"iter": iteration, "type": "val_eval_failed"})
            continue

        val_scores = score_results(val_results, human_vecs, human_ref)
        print(
            f"  Val: diff={val_scores['difficulty']:.3f}, JSD={val_scores['jsd']:.4f} (best={best_jsd:.4f})"
        )

        if val_scores["difficulty"] < difficulty_floor:
            print(
                f"  REJECTED: val difficulty {val_scores['difficulty']:.3f} < floor {difficulty_floor:.3f}"
            )
            history.append(
                {"iter": iteration, "type": "rejected_difficulty", **val_scores}
            )
            continue

        if val_scores["jsd"] < best_jsd:
            print(f"  ACCEPTED: JSD improved {best_jsd:.4f} -> {val_scores['jsd']:.4f}")
            best_persona = new_persona
            best_jsd = val_scores["jsd"]
            best_difficulty = val_scores["difficulty"]
            best_profile = val_scores["profile"]
            best_results = val_results

            # Save best persona
            best_out = {
                "persona_text": best_persona,
                "mutation_direction": direction,
                "jsd": best_jsd,
                "difficulty": best_difficulty,
                "iteration": iteration,
            }
            (out_dir / "best_persona.json").write_text(
                json.dumps(best_out, indent=2, ensure_ascii=False)
            )
            history.append({"iter": iteration, "type": "accepted", **val_scores})
        else:
            print(
                f"  REJECTED: JSD not improved ({val_scores['jsd']:.4f} >= {best_jsd:.4f})"
            )
            history.append({"iter": iteration, "type": "rejected_jsd", **val_scores})

    # Save history
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Save final best if not already saved
    if not (out_dir / "best_persona.json").exists():
        best_out = {
            "persona_text": best_persona,
            "mutation_direction": "seed (no improvement found)",
            "jsd": best_jsd,
            "difficulty": best_difficulty,
        }
        (out_dir / "best_persona.json").write_text(
            json.dumps(best_out, indent=2, ensure_ascii=False)
        )

    print(
        f"\n  FINAL: {seed_key} JSD {seed_scores['jsd']:.4f} -> {best_jsd:.4f} (Δ={best_jsd - seed_scores['jsd']:.4f})"
    )
    return {
        "seed_key": seed_key,
        "seed_jsd": seed_scores["jsd"],
        "best_jsd": best_jsd,
        "best_difficulty": best_difficulty,
        "iterations": len(history) - 1,
        "accepted": sum(1 for h in history if h.get("type") == "accepted"),
    }


def run_test(
    seed_key: str,
    args: argparse.Namespace,
    human_vecs: list[dict],
    human_ref: dict,
    test_tasks: list[str],
) -> dict:
    """Run test eval on best persona."""
    out_dir = DATA_ROOT / "data" / "persona_opt" / "naturalness_opt" / seed_key
    best_path = out_dir / "best_persona.json"

    if best_path.exists():
        best = json.loads(best_path.read_text())
        persona_text = best["persona_text"]
        print(f"  Using optimized persona (JSD={best.get('jsd', '?')})")
    else:
        cfg = SEEDS[seed_key]
        seed = json.loads(Path(cfg["persona_path"]).read_text())
        persona_text = seed["persona_text"]
        print(f"  No optimized persona found, using seed")

    print(f"  Test eval ({len(test_tasks)} tasks, {args.num_trials} trials)...")
    results = run_eval(
        persona_text,
        test_tasks,
        save_to=f"naturalness_opt/{seed_key}/test",
        num_trials=args.num_trials,
        max_concurrency=args.concurrency,
    )
    if not results:
        return {"seed_key": seed_key, "status": "test_failed"}

    scores = score_results(results, human_vecs, human_ref)
    print(
        f"  Test: diff={scores['difficulty']:.3f}, JSD={scores['jsd']:.4f}, N={scores['n_sims']}"
    )
    return {"seed_key": seed_key, **scores}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seeds", nargs="+", default=list(SEEDS.keys()), choices=list(SEEDS.keys())
    )
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--difficulty-tolerance", type=float, default=0.05)
    ap.add_argument("--mutation-model", default="gpt-5.4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split",
        default="experiments/persona_search/configs/telecom_island_split.json",
    )
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--test-trials", type=int, default=3)
    args = ap.parse_args()

    # Load split
    split = json.loads(
        (
            Path(args.split)
            if Path(args.split).is_absolute()
            else REPO_ROOT / args.split
        ).read_text()
    )
    train_tasks = split["train"]["telecom"]
    val_tasks = split["val"]["telecom"]
    test_tasks = split["test"]["telecom"]
    print(
        f"Split: train={len(train_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}"
    )

    # Load human reference
    human_ref = load_human_ref()
    human_vecs = load_human_vecs()
    human_samples = sample_human_conversations(n=3)
    print(
        f"Human reference: {len(human_vecs)} conversations, {len(human_ref['metrics'])} metrics"
    )

    if args.test_only:
        # Test only mode
        print(f"\n{'=' * 60}")
        print(
            f"TEST MODE: {len(args.seeds)} seeds, {args.test_trials} trials, {len(test_tasks)} tasks"
        )
        print(f"{'=' * 60}")
        args.num_trials = args.test_trials
        test_results = []
        for seed_key in args.seeds:
            print(f"\n--- Testing {seed_key} ---")
            r = run_test(seed_key, args, human_vecs, human_ref, test_tasks)
            test_results.append(r)

        # Also test seeds (unoptimized) for comparison
        print(f"\n--- Testing seeds (unoptimized) for comparison ---")
        seed_results = []
        for seed_key in args.seeds:
            cfg = SEEDS[seed_key]
            seed_persona = json.loads(Path(cfg["persona_path"]).read_text())
            print(f"\n  {seed_key} seed eval...")
            results = run_eval(
                seed_persona["persona_text"],
                test_tasks,
                save_to=f"naturalness_opt/{seed_key}/test_seed",
                num_trials=args.test_trials,
                max_concurrency=args.concurrency,
            )
            if results:
                scores = score_results(results, human_vecs, human_ref)
                print(
                    f"  Seed test: diff={scores['difficulty']:.3f}, JSD={scores['jsd']:.4f}"
                )
                seed_results.append({"seed_key": seed_key, **scores})

        # Summary
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"{'Persona':<8} {'Seed Diff':>10} {'Seed JSD':>10} {'Opt Diff':>10} {'Opt JSD':>10} {'Δ JSD':>8}"
        )
        for sr, tr in zip(seed_results, test_results):
            if "difficulty" in sr and "difficulty" in tr:
                djsd = tr["jsd"] - sr["jsd"]
                print(
                    f"{sr['seed_key']:<8} {sr['difficulty']:>10.3f} {sr['jsd']:>10.4f} "
                    f"{tr['difficulty']:>10.3f} {tr['jsd']:>10.4f} {djsd:>+8.4f}"
                )
        return

    # Optimization mode
    all_results = []
    for seed_key in args.seeds:
        r = run_optimization(
            seed_key, args, human_vecs, human_ref, human_samples, train_tasks, val_tasks
        )
        all_results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        if "seed_jsd" in r:
            print(
                f"  {r['seed_key']}: JSD {r['seed_jsd']:.4f} -> {r['best_jsd']:.4f} "
                f"(Δ={r['best_jsd'] - r['seed_jsd']:.4f}), accepted={r['accepted']}/{r['iterations']}"
            )
        else:
            print(f"  {r['seed_key']}: {r.get('status', 'unknown')}")

    # Save summary
    summary_path = (
        DATA_ROOT / "data" / "persona_opt" / "naturalness_opt" / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
