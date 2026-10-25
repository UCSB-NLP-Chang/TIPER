"""
Generate persona policy text only — no τ² rollouts, no discriminator, no scoring.

Supports two task sources:
  - tau2 domains (e.g. telecom): loads tasks from a τ²-bench checkout and builds
    context ``c`` via ``tau_train_context.format_user_scenario_c``.
  - vitabench domains (delivery / instore / ota / cross_domain): loads tasks from
    a vitabench checkout and builds ``c`` from ``user_scenario.user_profile`` +
    top-level ``instructions`` (vitabench's flat schema, not τ²'s nested one).

Each (task, persona) result is the dict returned by
``generate_personas_detailed(c, D, N)``; ``expanded_instruction`` is the text
that would be injected into a user simulator's system prompt.

Requires only ``litellm`` + an LLM API key — does NOT import or run τ²-bench.
Run from the repo root with the project venv, e.g.:
  .venv/bin/python -m persona_policies.scripts.generate_personas_only \
      --dataset telecom --domain telecom --split test --n-personas 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# τ² is the shared framework at the monorepo root; VitaBench is a sibling
# benchmark under benchmarks/.
_DEFAULT_TAUBENCH_ROOT = _REPO.parents[1]
_DEFAULT_VITABENCH_ROOT = _REPO.parent / "vitabench"


# ---------------------------------------------------------------------------
# Task loaders → list of (task_id, context_c)
# ---------------------------------------------------------------------------


def load_tau2_tasks(
    domain: str,
    split: str,
    taubench_root: Path,
    split_file: str | None = None,
    split_section: str = "test",
) -> List[Tuple[str, str]]:
    """Load τ² tasks for a split and format context ``c`` for each.

    If ``split_file`` is given, read task ids from that JSON file instead of
    τ²'s ``split_tasks.json``. The file is expected to be a dict with
    ``{meta, train, val, test}`` where each of train/val/test is a
    ``{domain: [task_id, ...]}`` mapping (the PGEPA island-split layout, e.g.
    ``experiments/persona_search/configs/telecom_island_split.json``). ``split_section`` selects
    which section to load (default ``test``).
    """
    from persona_policies.tau_train_context import (
        format_user_scenario_c,
        load_domain_tasks,
    )

    all_tasks = load_domain_tasks(domain, taubench_root)
    by_id = {str(t.get("id")): t for t in all_tasks}

    if split_file:
        data = json.load(open(split_file, encoding="utf-8"))
        section = data.get(split_section, {})
        # section is {domain: [ids]} (possibly multiple domains for combined).
        split_ids: List[str] = []
        for _d, ids in section.items():
            split_ids.extend(str(i) for i in ids)
        split_label = f"{split_file}::{split_section}"
    else:
        from persona_policies.tau_train_context import load_split_task_ids
        split_ids = load_split_task_ids(domain, split, taubench_root)
        split_label = f"tau2 {split}"

    tasks = [by_id[i] for i in split_ids if i in by_id]
    if not tasks:
        raise RuntimeError(
            f"No τ² tasks for domain={domain!r} split={split_label!r} "
            f"(split_ids={len(split_ids)}, matched={len(tasks)}); "
            f"check taubench_root={taubench_root}"
        )
    return [(str(t["id"]), format_user_scenario_c(t)) for t in tasks]


def format_vita_c(task: dict) -> str:
    """Build context ``c`` from a vitabench task (flat user_profile + instructions)."""
    import textwrap

    us = task.get("user_scenario") or {}
    profile = us.get("user_profile") or {}
    lines: List[str] = []
    if profile:
        lines.append("Base Persona (user_profile):")
        for k, v in profile.items():
            lines.append(textwrap.indent(f"{k}: {v}", "\t"))
    instructions = task.get("instructions")
    lines.append("Given Instructions:")
    if isinstance(instructions, dict):
        for k, v in sorted(instructions.items()):
            lines.append(textwrap.indent(f"{k}: {v}", "\t"))
    else:
        lines.append(textwrap.indent(str(instructions or "").strip(), "\t"))
    return "\n".join(lines)


def _pgepa_config_path(domain: str, vitabench_root: Path) -> Path:
    """Path to the PGEPA run config for a domain.

    Run dirs strip underscores from the domain name: ``cross_domain`` →
    ``run_crossdomain_v2``. Tries ``_v2`` then the plain ``run_<domain>`` form.
    """
    compact = domain.replace("_", "")
    candidates = [
        vitabench_root / "data/persona_opt/vita_pgepa_runs" / f"run_{compact}_v2" / "pgepa_config.json",
        vitabench_root / "data/persona_opt/vita_pgepa_runs" / f"run_{compact}" / "pgepa_config.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No pgepa_config.json for domain={domain!r}; looked at {[str(c) for c in candidates]}"
    )


def load_vita_tasks(
    domain: str,
    vitabench_root: Path,
    n_tasks: int | None = None,
    use_pgepa_split: bool = False,
) -> List[Tuple[str, str]]:
    """Load vitabench tasks for a domain.

    With ``use_pgepa_split=True``, reproduce the PGEPA test split: read the
    domain's ``pgepa_config.json`` (seed / train_size / val_size / test_size),
    shuffle all task ids with that seed (same logic as
    ``vita_pgepa.split_tasks`` / ``vita_test_eval.split_test_ids``), and take the
    test slice ``ids[train+val : train+val+test]``. Tasks are returned in test-id
    order. Otherwise the first ``n_tasks`` tasks (or all) are returned.
    """
    import random

    p = vitabench_root / "data/vita/domains" / domain / "tasks.json"
    if not p.is_file():
        raise FileNotFoundError(f"vitabench tasks not found: {p}")
    tasks = json.load(open(p, encoding="utf-8"))

    if use_pgepa_split:
        cfg = json.load(open(_pgepa_config_path(domain, vitabench_root), encoding="utf-8"))
        ids = [str(t.get("id")) for t in tasks]
        random.Random(cfg["seed"]).shuffle(ids)
        tr, va, ts = cfg["train_size"], cfg["val_size"], cfg["test_size"]
        test_ids = ids[tr + va : tr + va + ts]
        by_id = {str(t.get("id")): t for t in tasks}
        tasks = [by_id[i] for i in test_ids if i in by_id]
    elif n_tasks:
        tasks = tasks[:n_tasks]

    return [(str(t.get("id")), format_vita_c(t)) for t in tasks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate persona policy text only.")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["tau2", "vitabench"],
        help="tau2 = τ²-bench domain (telecom, retail, airline, ...); "
        "vitabench = vitabench domain (delivery/instore/ota/cross_domain)",
    )
    ap.add_argument("--domain", required=True, help="domain name within the dataset")
    ap.add_argument(
        "--split",
        default="test",
        help="τ² task split (train/test/small/base/full); ignored for vitabench",
    )
    ap.add_argument("--n-tasks", type=int, default=None, help="limit number of tasks")
    ap.add_argument(
        "--n-personas", type=int, default=10, help="personas per task (N)"
    )
    ap.add_argument(
        "--language",
        default=None,
        choices=["en", "zh"],
        help="hint for persona output language; zh prepends a Chinese-language "
        "instruction to the context (vitabench tasks are already Chinese)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="override PersonaPoliciesConfig.llm_model (LiteLLM id, e.g. "
        "openrouter/google/gemini-3-flash-preview or gemini/gemini-3-flash-preview)",
    )
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
        help="reasoning_effort for gpt-5.x models (injected via litellm). "
        "When set, --max-tokens defaults to 16000 unless overridden (gpt-5 "
        "reasoning tokens count against the output budget; the default Stage2 "
        "max_tokens=800 is far too small for high/xhigh thinking).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override max_tokens for LLM calls (mainly for gpt-5 reasoning, "
        "where thinking consumes the output budget)",
    )
    ap.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="parallelism across tasks (1 = serial). Each task still runs "
        "Stage2 personas in parallel internally, so total LLM concurrency ≈ "
        "task_workers × n_personas unless --llm-concurrency caps it.",
    )
    ap.add_argument(
        "--llm-concurrency",
        type=int,
        default=None,
        help="global cap on simultaneous LLM calls via a semaphore (across all "
        "tasks and Stage2 workers). Recommended when raising --task-workers to "
        "stay within provider rate limits, e.g. --task-workers 30 --llm-concurrency 30.",
    )
    ap.add_argument(
        "--taubench-root",
        default=str(_DEFAULT_TAUBENCH_ROOT),
        help="τ²-bench checkout root (defaults to the monorepo root)",
    )
    ap.add_argument(
        "--tau2-split-file",
        default=None,
        help="tau2 only: read task ids from this PGEPA island-split JSON "
        "(format {meta,train,val,test}, each {domain:[ids]}) instead of τ²'s "
        "split_tasks.json. e.g. experiments/persona_search/configs/telecom_island_split.json",
    )
    ap.add_argument(
        "--tau2-split-section",
        default="test",
        choices=["train", "val", "test"],
        help="which section of --tau2-split-file to load (default test)",
    )
    ap.add_argument(
        "--vitabench-root",
        default=str(_DEFAULT_VITABENCH_ROOT),
        help="vitabench checkout root",
    )
    ap.add_argument(
        "--use-pgepa-split",
        action="store_true",
        help="vitabench only: reproduce the PGEPA test split from the domain's "
        "pgepa_config.json (seed/train_size/val_size/test_size). Same logic as "
        "vita_pgepa.split_tasks / vita_test_eval.split_test_ids.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path; default persona_policies/outputs/generated_personas/"
        "<dataset>_<domain>[_<split>]_nN.json",
    )
    ap.add_argument(
        "--env-file",
        default=None,
        help="load KEY=VALUE lines from a .env file into os.environ before LLM "
        "calls (use when keys are not already exported in the shell)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="load tasks and build contexts only; do not call the LLM",
    )
    args = ap.parse_args()

    # Load credentials from a .env file if requested (python parsing avoids
    # shell quoting issues, e.g. values containing <>).
    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    # When using an OpenAI model, drop ANTHROPIC_* env so litellm does not route
    # the openai/ call through an Anthropic-compatible proxy (e.g. a Fireworks
    # dev proxy bound via ANTHROPIC_BASE_URL).
    if args.model and args.model.lower().startswith("openai/"):
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    # Load tasks → (id, c)
    if args.dataset == "tau2":
        task_pairs = load_tau2_tasks(
            args.domain,
            args.split,
            Path(args.taubench_root),
            split_file=args.tau2_split_file,
            split_section=args.tau2_split_section,
        )
    else:
        task_pairs = load_vita_tasks(
            args.domain,
            Path(args.vitabench_root),
            args.n_tasks,
            use_pgepa_split=args.use_pgepa_split,
        )

    # For tau2 the split already selected tasks; --n-tasks only trims further.
    # For vitabench with --use-pgepa-split the test slice is already exact; only
    # trim if explicitly requested.
    if args.n_tasks and not args.use_pgepa_split and not args.tau2_split_file:
        task_pairs = task_pairs[: args.n_tasks]

    if args.dataset == "tau2":
        split_tag = "pgepa_test" if args.tau2_split_file else args.split
    else:
        split_tag = "pgepa_test" if args.use_pgepa_split else "n/a"
    print(
        f"[load] dataset={args.dataset} domain={args.domain} "
        f"split={split_tag} "
        f"tasks={len(task_pairs)} n_personas={args.n_personas}"
    )

    # Optional language hint prepended to each context c.
    lang_hint = ""
    if args.language == "zh":
        lang_hint = (
            "【生成语言要求】请用中文撰写 persona 的 description 和 "
            "expanded_instruction（角色扮演指令）。\n\n"
        )
    elif args.language == "en":
        lang_hint = (
            "[Language requirement] Write the persona description and the "
            "expanded_instruction (roleplay instructions) in English.\n\n"
        )

    if args.dry_run:
        print("\n[dry-run] first task context preview:")
        tid, c = task_pairs[0]
        print(f"task_id={tid}")
        print((lang_hint + c)[:800])
        print(f"\n[dry-run] {len(task_pairs)} tasks ready, "
              f"{len(task_pairs) * (1 + args.n_personas)} LLM calls would be issued.")
        return

    # Configure generator model (override the cached config value before first call).
    from persona_policies import llm_bedrock
    from persona_policies.evolution import _generator_utils
    from persona_policies.evolution.initial_generator import (
        DIVERSITY_AXES,
        generate_personas_detailed,
    )

    if args.model:
        _generator_utils._cached_llm_model = args.model
        # User picked a model explicitly — don't silently fall back to the
        # config default (gemini/...), which would mask the real error.
        llm_bedrock._cached_fallback_models = []

    # reasoning_effort + max_tokens override for gpt-5.x models. completion_text
    # does not pass reasoning_effort for openai models and Stage2's max_tokens=800
    # is too small for high/xhigh thinking, so patch at the litellm.completion
    # layer (used by llm_bedrock.completion_text).
    if args.reasoning_effort:
        import litellm as _litellm
        _orig_completion = _litellm.completion
        _re = args.reasoning_effort
        _mt = args.max_tokens if args.max_tokens else 16000

        def _patched_completion(*a, **kw):
            m = str(kw.get("model", "")).lower()
            if "gpt-5" in m:
                kw["reasoning_effort"] = _re
                kw["max_tokens"] = _mt
                # gpt-5 reasoning models only support temperature=1.
                kw["temperature"] = 1
            return _orig_completion(*a, **kw)

        _litellm.completion = _patched_completion

    # Global cap on simultaneous LLM calls (across tasks and Stage2 workers).
    # Wrapped at the _chat layer (the single LLM-call entrypoint in
    # _generator_utils), so both Stage1 and Stage2 calls are gated.
    if args.llm_concurrency:
        import threading
        _sem = threading.Semaphore(args.llm_concurrency)
        _orig_chat = _generator_utils._chat

        def _sem_chat(*a, **kw):
            with _sem:
                return _orig_chat(*a, **kw)

        _generator_utils._chat = _sem_chat

    def _gen_one(i: int, tid: str, ctx: str):
        try:
            personas = generate_personas_detailed(ctx, DIVERSITY_AXES, args.n_personas)
            ids = [p.get("persona_id") for p in personas]
            print(f"[{i}/{len(task_pairs)}] task {tid}: {len(personas)} personas → {ids}", flush=True)
            return {"task_id": tid, "context_c": ctx, "personas": personas}
        except Exception as e:
            print(f"[{i}/{len(task_pairs)}] task {tid} FAILED: {e}", flush=True)
            return {"task_id": tid, "error": str(e), "personas": []}

    results: List[Dict[str, Any]] = [None] * len(task_pairs)  # type: ignore[list-item]
    t0 = time.time()
    if args.task_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.task_workers) as pool:
            futs = {
                pool.submit(_gen_one, i, tid, lang_hint + c): i
                for i, (tid, c) in enumerate(task_pairs, 1)
            }
            for fut in as_completed(futs):
                results[futs[fut] - 1] = fut.result()
    else:
        for i, (tid, c) in enumerate(task_pairs, 1):
            results[i - 1] = _gen_one(i, tid, lang_hint + c)
    dt = time.time() - t0
    print(f"\n[done] {len(results)} tasks in {dt:.1f}s", flush=True)

    # Write output.
    out_dir = _REPO / "persona_policies/outputs/generated_personas"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        if args.dataset == "tau2":
            name_split = "_pgepa_test" if args.tau2_split_file else f"_{args.split}"
        elif args.use_pgepa_split:
            name_split = "_pgepa_test"
        else:
            name_split = ""
        out_path = out_dir / f"{args.dataset}_{args.domain}{name_split}_n{args.n_personas}.json"
    model_used = args.model or _generator_utils._llm_model_id()
    payload = {
        "dataset": args.dataset,
        "domain": args.domain,
        "split": args.split if args.dataset == "tau2" else None,
        "n_personas": args.n_personas,
        "model": model_used,
        "language_hint": args.language,
        "n_tasks": len(results),
        "elapsed_seconds": round(dt, 1),
        "tasks": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
