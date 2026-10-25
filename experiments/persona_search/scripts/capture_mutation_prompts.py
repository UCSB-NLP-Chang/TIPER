"""Capture 2 mutation prompts + LLM responses and write to markdown."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

from lineage_loop import _build_mutation_prompt
from tau_persona_adapter import GEPA_MODE_OVERRIDE

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

STATE_PATH = str(
    DATA_ROOT
    / "data/persona_opt/pgepa_runs/enriched_brinks_02_gpt54user/enriched_brinks_02/pgepa_state.json"
)
SIM_ROOT = str(DATA_ROOT / "data/simulations/gepa_opt")
MUTATION_MODEL = os.environ.get("MUTATION_MODEL", "openai/gpt-5.4")
MUTATION_EFFORT = os.environ.get("MUTATION_EFFORT", "high")
MUTATION_EDIT_PCT = "40-50%"
OUTPUT_PATH = "experiments/persona_search/scripts/MUTATION_PROMPT_EXAMPLES.md"
N_CAPTURES = 2


def _load_traces(run_tag, domain="telecom"):
    """Load simulation results and extract success/failure traces."""
    run_path = f"{SIM_ROOT}/{run_tag}"
    if not os.path.isdir(run_path):
        return None, None

    # Find the eval dir that has judge results
    eval_dirs = sorted([d for d in os.listdir(run_path) if d.startswith("eval_")])
    chosen_eval = None
    for ed in reversed(eval_dirs):
        judge_path = f"{run_path}/{ed}/judge/{domain}/per_sim.jsonl"
        if os.path.exists(judge_path):
            chosen_eval = ed
            break
    if chosen_eval is None:
        # fallback: use any eval dir with results.json
        for ed in reversed(eval_dirs):
            if os.path.exists(f"{run_path}/{ed}/{domain}/results.json"):
                chosen_eval = ed
                break
    if chosen_eval is None:
        return None, None

    eval_dir = chosen_eval
    path = f"{run_path}/{eval_dir}/{domain}/results.json"
    if not os.path.exists(path):
        return None, None

    data = json.load(open(path))
    sims = data.get("simulations", [])

    # Build reward lookup from judge
    reward_map = {}
    judge_path = f"{run_path}/{eval_dir}/judge/{domain}/per_sim.jsonl"
    if os.path.exists(judge_path):
        for line in open(judge_path):
            j = json.loads(line)
            tid = j.get("task_id", "")
            reward_map[tid] = j.get("reward", 0)

    success_traces = []
    failure_traces = []
    for sim in sims:
        reward = sim.get("reward")
        if reward is None:
            reward = reward_map.get(sim.get("task_id"), 0)

        msgs = sim.get("messages", [])
        trace = {
            "task_id": sim.get("task_id", "?"),
            "reward": reward,
            "n_messages": len(msgs),
            "opening": [_fmt_msg(m) for m in msgs[:4]],
            "resolution": [_fmt_msg(m) for m in msgs[-8:]],
        }
        if reward >= 0.5:
            success_traces.append(trace)
        else:
            failure_traces.append(trace)

    return success_traces, failure_traces


def _fmt_msg(msg):
    role = msg.get("role", "?")
    content = msg.get("content", "") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use: {block.get('name', '?')}]")
                elif block.get("type") == "tool_result":
                    parts.append(
                        f"[tool_result: {str(block.get('content', ''))[:200]}]"
                    )
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    if len(content) > 500:
        content = content[:497] + "..."
    return {"role": role, "content": content}


def main():
    state = json.load(open(STATE_PATH))
    candidates = state.get("candidates", [])
    generations = state.get("generations", state.get("gen_logs", []))

    sections = []

    for cap_idx in range(N_CAPTURES):
        if cap_idx < len(candidates) - 1:
            parent = candidates[cap_idx]
        else:
            parent = candidates[0]

        persona_text = parent.get("persona_text", "")
        gen = parent.get("gen", 0)
        diff = parent.get("difficulty_agg", 0)

        run_tags = [
            f"enriched_brinks_02_g{gen:02d}_k00",
            f"enriched_brinks_02_g{gen:02d}_seed",
            f"enriched_brinks_02_g{gen:02d}_seed_train",
        ]
        success_traces, failure_traces = None, None
        used_tag = None
        for tag in run_tags:
            st, ft = _load_traces(tag)
            if st is not None:
                success_traces, failure_traces = st, ft
                used_tag = tag
                break

        if success_traces is None:
            for tag_suffix in ["_k00", "_parent_fill", "_parent_trace"]:
                for g in range(gen + 1, gen + 4):
                    tag = f"enriched_brinks_02_g{g:02d}{tag_suffix}"
                    st, ft = _load_traces(tag)
                    if st is not None:
                        success_traces, failure_traces = st, ft
                        used_tag = tag
                        break
                if success_traces is not None:
                    break

        if success_traces is None:
            print(
                f"  [capture {cap_idx}] no traces found for parent idx={parent['idx']} gen={gen}, skipping"
            )
            continue

        print(
            f"  [capture {cap_idx}] parent idx={parent['idx']} gen={gen} diff={diff:.3f} traces from {used_tag}"
        )
        print(f"    success={len(success_traces)} failure={len(failure_traces or [])}")

        history = []
        for g in generations[: gen + 1]:
            entry = {
                "kind": "minibatch_diagnostic",
                "batch_size": g.get("minibatch_size", 5),
                "noisy_agent_success_rate": g.get("minibatch_mean", 0),
                "direction": g.get("direction", "unknown"),
            }
            history.append(entry)

        ft = (failure_traces or [])[:4]
        prompt = _build_mutation_prompt(
            persona_text=persona_text,
            score=1 - diff,
            traces=success_traces,
            history=history[-5:],
            pid="gepa",
            severe_failures=None,
            timeout_rate=0.0,
            failure_traces=ft or None,
            mutation_context="enriched",
        )

        anchor = "Rewrite ~30% of the persona"
        if anchor in prompt:
            prompt = prompt.replace(
                anchor,
                f"Rewrite {MUTATION_EDIT_PCT} of the persona",
            )
        mode_override = GEPA_MODE_OVERRIDE.replace("10-15%", MUTATION_EDIT_PCT)
        prompt += mode_override

        prompt_len = len(prompt)
        prompt_tokens_est = prompt_len // 4

        print(f"    prompt length: {prompt_len} chars (~{prompt_tokens_est} tokens)")
        print(f"    calling {MUTATION_MODEL} (effort={MUTATION_EFFORT})...")

        import litellm

        is_claude = "claude" in MUTATION_MODEL.lower()
        kwargs = {
            "model": MUTATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": MUTATION_EFFORT,
            "max_tokens": 16384,
        }
        if is_claude:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 16384}

        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content
        if isinstance(raw, list):
            text = ""
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        else:
            text = raw or ""

        usage = resp.usage
        usage_str = ""
        if usage:
            usage_str = f"input={getattr(usage, 'prompt_tokens', '?')} output={getattr(usage, 'completion_tokens', '?')} reasoning={getattr(usage, 'completion_tokens_details', {})}"

        sections.append(
            {
                "cap_idx": cap_idx,
                "parent_idx": parent["idx"],
                "parent_gen": gen,
                "parent_diff": diff,
                "trace_source": used_tag,
                "n_success": len(success_traces),
                "n_failure": len(ft),
                "prompt": prompt,
                "prompt_chars": prompt_len,
                "prompt_tokens_est": prompt_tokens_est,
                "response_raw": text,
                "model": MUTATION_MODEL,
                "effort": MUTATION_EFFORT,
                "usage": usage_str,
            }
        )
        print(f"    done. response length: {len(text)} chars")

    # Write markdown
    md = ["# Mutation Prompt & LLM Proposer Response Examples\n"]
    md.append(f"Captured from: `enriched_brinks_02` experiment\n")
    md.append(f"Model: `{MUTATION_MODEL}` | Effort: `{MUTATION_EFFORT}`\n")
    md.append(f"Date: {__import__('datetime').date.today()}\n")
    md.append("---\n")

    for s in sections:
        md.append(
            f"## Example {s['cap_idx'] + 1}: Parent idx={s['parent_idx']} gen={s['parent_gen']} (diff={s['parent_diff']:.3f})\n"
        )
        md.append(f"- Trace source: `{s['trace_source']}`")
        md.append(
            f"- Success traces: {s['n_success']}, Failure traces: {s['n_failure']}"
        )
        md.append(
            f"- Prompt: {s['prompt_chars']} chars (~{s['prompt_tokens_est']} tokens)"
        )
        md.append(f"- Usage: {s['usage']}\n")

        md.append("### Mutation Prompt\n")
        md.append("```")
        md.append(s["prompt"])
        md.append("```\n")

        md.append("### LLM Proposer Response\n")
        md.append("```")
        md.append(s["response_raw"])
        md.append("```\n")
        md.append("---\n")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(md))

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
