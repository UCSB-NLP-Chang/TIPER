#!/usr/bin/env python3
"""Cross-domain GEPA with multi-seed initialization.

Extends parallel_gepa.py to:
1. Load ALL seed personas into the candidate pool at gen 0
2. Use a variance-aware task list (variable tasks only) from a JSON split file
3. Support telecom/airline/retail domains

Usage:
    python experiments/persona_search/scripts/crossdomain_gepa.py \
        --split-file experiments/persona_search/configs/crossdomain_variance_split.json \
        --split train \
        --seed-personas data/persona_opt/seeds_brinks/brinks_0*/gen_00.md \
        --iterations 20 --minibatch 4

    # dry-run smoke test
    python experiments/persona_search/scripts/crossdomain_gepa.py \
        --split-file experiments/persona_search/configs/crossdomain_variance_split.json \
        --seed-personas data/persona_opt/seeds_brinks/brinks_00_terse_blunt/gen_00.md \
        --dry-run --iterations 3 --minibatch 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tau_persona_adapter as tpa  # noqa: E402
from _common import PERSONA_OPT_DIR, PERSONAS_DIR  # noqa: E402
from parallel_gepa import (  # noqa: E402
    _BASE_OVERRIDE,
    Candidate,
    _all_feasible,
    _mean,
    _score_views,
    eval_one,
    load_resume_state,
    make_child_adapter,
    run_iteration,
    write_candidates_json,
    write_state,
)
from tau_persona_adapter import TauTask  # noqa: E402

import tau2.utils.cresta_vertex  # noqa: E402, F401

tpa.GEPA_MODE_OVERRIDE = _BASE_OVERRIDE


def _stratified_sample(
    tasks: list[TauTask], cap: int, seed: int, min_domain_size: int = 5
) -> list[TauTask]:
    """Sample `cap` tasks from `tasks` proportionally by domain.

    小域（任务数 <= min_domain_size）全部保留，剩余名额按比例分配给其他域。
    默认 min_domain_size=5，airline（4 tasks）自动全保留。
    """
    import math
    import random as _rng

    rng = _rng.Random(seed)
    by_domain: dict[str, list[TauTask]] = {}
    for t in tasks:
        by_domain.setdefault(t.domain, []).append(t)

    # 小域全保留
    small_doms = {dom for dom, ts in by_domain.items() if len(ts) <= min_domain_size}
    result: list[TauTask] = []
    for dom in small_doms:
        result += by_domain[dom]
    allocated = len(result)

    # 剩余名额按比例分配给大域
    large_doms = {dom: ts for dom, ts in by_domain.items() if dom not in small_doms}
    large_total = sum(len(ts) for ts in large_doms.values())
    remaining_cap = cap - allocated
    remainder: list[tuple[float, str]] = []
    for dom, dom_tasks in sorted(large_doms.items()):
        exact = remaining_cap * len(dom_tasks) / large_total if large_total else 0
        floored = math.floor(exact)
        result += rng.sample(dom_tasks, min(floored, len(dom_tasks)))
        allocated += floored
        remainder.append((exact - floored, dom))
    remainder.sort(reverse=True)
    for _, dom in remainder:
        if allocated >= cap:
            break
        already = {t.task_id for t in result if t.domain == dom}
        extras = [t for t in large_doms[dom] if t.task_id not in already]
        if extras:
            result.append(rng.choice(extras))
            allocated += 1
    return result


def build_task_list_from_split(split_file: str, split: str) -> list[TauTask]:
    """Load tasks from a variance-aware split JSON file."""
    data = json.loads(Path(split_file).read_text())
    split_data = data[split]
    tasks: list[TauTask] = []
    for domain, task_ids in split_data.items():
        for tid in task_ids:
            tasks.append(TauTask(task_id=tid, domain=domain))
    return tasks


def main():
    args = parse_args()

    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(PERSONA_OPT_DIR / "pgepa_runs" / f"xdom_{ts}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # train split（minibatch 用）
    valset = build_task_list_from_split(args.split_file, args.split)
    if args.valset_cap and args.valset_cap < len(valset):
        valset = _stratified_sample(valset, args.valset_cap, args.seed)
    instance_ids = [f"{t.domain}::{t.task_id}" for t in valset]

    # val split（held-out，Pareto 用）— 小域全保留
    val_tasks: list[TauTask] | None = None
    val_instance_ids: list[str] | None = None
    if args.val_split:
        val_tasks_all = build_task_list_from_split(args.split_file, args.val_split)
        if args.val_cap and args.val_cap < len(val_tasks_all):
            val_tasks = _stratified_sample(
                val_tasks_all,
                args.val_cap,
                args.seed,
                min_domain_size=args.val_min_domain_size,
            )
        else:
            val_tasks = val_tasks_all
        val_instance_ids = [f"{t.domain}::{t.task_id}" for t in val_tasks]
        val_dom_counts = {}
        for t in val_tasks:
            val_dom_counts[t.domain] = val_dom_counts.get(t.domain, 0) + 1
        val_dom_str = "+".join(
            f"{v}{k[0].upper()}" for k, v in sorted(val_dom_counts.items())
        )
    else:
        val_dom_str = "none"

    seed_texts = []
    seed_paths = []
    for p in args.seed_personas:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"seed persona not found: {path}")
        seed_texts.append(path.read_text())
        seed_paths.append(path)

    dom_counts = {}
    for t in valset:
        dom_counts[t.domain] = dom_counts.get(t.domain, 0) + 1
    dom_str = "+".join(f"{v}{k[0].upper()}" for k, v in sorted(dom_counts.items()))

    print(
        f"crossdomain_gepa: {len(valset)} train instances ({dom_str})  "
        f"{len(seed_texts)} seeds  run_dir={run_dir}"
    )
    print(
        f"  val split: {len(val_tasks) if val_tasks else 0} instances ({val_dom_str})"
    )
    print(f"  split_file={args.split_file}  split={args.split}")
    print(
        f"  iterations={args.iterations} minibatch={args.minibatch} "
        f"agent={args.agent_llm} user={args.user_type}/{args.user_llm}"
    )

    refl_adapter = make_child_adapter(args, f"{args.persona_id_base}_refl")

    # resume 时用 val_instance_ids 匹配（state 保存的是 val 分数的 instance_ids）
    resume_check_ids = val_instance_ids if val_instance_ids else instance_ids
    resumed = load_resume_state(run_dir, resume_check_ids) if args.resume else None
    if resumed is not None:
        candidates, seen_personas, invalid_cases, gen_logs, start_iter = resumed
        print(
            f"  [resume] restored {len(candidates)} candidates, "
            f"{len(gen_logs)} completed iters -> continuing at iter {start_iter}"
        )
        # If gen0 was interrupted, evaluate any seeds not yet in the pool.
        if len(candidates) < len(seed_texts):
            print(
                f"  [resume] gen0 incomplete ({len(candidates)}/{len(seed_texts)} seeds) "
                f"— evaluating remaining seeds...",
                flush=True,
            )
            for i, (text, path) in enumerate(zip(seed_texts, seed_paths)):
                if any(c.persona_text == text for c in candidates):
                    continue  # already evaluated
                pid = f"{args.persona_id_base}_g00_seed{i}"
                print(f"    seed {i}: {path.name} ({len(text)} chars)", flush=True)
                eb = eval_one(args, pid, valset, text, True)
                seed = Candidate(
                    idx=len(candidates), persona_text=text, parent=None, gen=0
                )
                obj, diff, val = _score_views(eb)
                seed.subscores = dict(zip(instance_ids, obj))
                seed.difficulty_subscores = dict(zip(instance_ids, diff))
                seed.validity_subscores = dict(zip(instance_ids, val))
                seed.agg = _mean(obj)
                seed.difficulty_agg = _mean(diff)
                seed.validity_agg = _mean(val)
                seed.feasible = _all_feasible(eb)
                seed.eval_batch = eb
                candidates.append(seed)
                seen_personas.append(text)
                print(
                    f"    seed {i}: score={seed.agg:.3f}  difficulty={seed.difficulty_agg:.3f}  "
                    f"validity={seed.validity_agg:.3f}  feasible={seed.feasible}",
                    flush=True,
                )
                best_idx = max(
                    range(len(candidates)),
                    key=lambda j: (
                        candidates[j].agg if candidates[j].agg is not None else -1.0
                    ),
                )
                write_state(
                    run_dir,
                    candidates,
                    instance_ids,
                    gen_logs,
                    best_idx,
                    args,
                    invalid_cases,
                )
    else:
        if args.resume:
            print("  [resume] no resumable state -> starting fresh.")

        candidates: list[Candidate] = []
        seen_personas: list[str] = []
        invalid_cases: list[dict] = []
        gen_logs: list[dict] = []

        # 加载上一次 run 的 seed train + val 分数（避免重复跑 gen0 eval）
        # 按 persona 文本匹配（不按 idx），这样换 seed 顺序/子集也不会错配。
        # 一份 subscores 是 train 还是 val，靠它的 instance key 落在哪个 split
        # 来判定（config 不一定保存 val_split，故不依赖 config）。
        prev_train_by_text: dict[
            str, dict[str, float]
        ] = {}  # persona_text -> train_subscores
        prev_val_by_text: dict[str, dict] = {}  # persona_text -> {subscores, ...} (val)
        if args.resume_seeds_from:
            train_keyset = set(instance_ids)
            val_keyset = set(val_instance_ids or [])

            def _which_split(scores: dict) -> str | None:
                """按 key 归属判定一份 subscores 属于 train 还是 val。"""
                ks = set(scores.keys())
                if not ks:
                    return None
                tr = len(ks & train_keyset)
                va = len(ks & val_keyset)
                if va > tr:
                    return "val"
                if tr > 0:
                    return "train"
                return None

            for path in args.resume_seeds_from:
                try:
                    prev_state = json.loads(Path(path).read_text())
                except Exception as e:
                    print(f"  [resume-seeds] 加载失败 {path}: {e}", flush=True)
                    continue
                for c in prev_state.get("candidates", []):
                    if c.get("gen", -1) != 0:
                        continue
                    text = (c.get("persona_text") or "").strip()
                    if not text:
                        continue
                    # 显式的 train_subscores 一律当 train
                    if c.get("train_subscores") and text not in prev_train_by_text:
                        prev_train_by_text[text] = c["train_subscores"]
                    # subscores 按 key 归属判定
                    sub = c.get("subscores")
                    if sub:
                        kind = _which_split(sub)
                        if kind == "train" and text not in prev_train_by_text:
                            prev_train_by_text[text] = sub
                        elif kind == "val" and text not in prev_val_by_text:
                            prev_val_by_text[text] = c
            print(
                f"  [resume-seeds] 从 {len(args.resume_seeds_from)} 个文件复用了 "
                f"{len(prev_train_by_text)} 个 seed 的 train 分数、"
                f"{len(prev_val_by_text)} 个 seed 的 val 分数（按文本匹配）",
                flush=True,
            )

        # Gen 0: 每个 seed 先跑 train（存 train_subscores，minibatch 比较用）
        #        再跑 val（存 subscores/agg，Pareto 用）
        print(
            f"  [gen 0] evaluating {len(seed_texts)} seed personas "
            f"on train({len(valset)}) + val({len(val_tasks) if val_tasks else 0})...",
            flush=True,
        )
        for i, (text, path) in enumerate(zip(seed_texts, seed_paths)):
            pid_train = f"{args.persona_id_base}_g00_seed{i}"
            print(f"    seed {i}: {path.name} ({len(text)} chars)", flush=True)

            # train eval（可从上一次 run 复用）
            text_key = text.strip()
            if text_key in prev_train_by_text:
                # 复用旧 train 分数，不重新跑
                train_diff_map = prev_train_by_text[text_key]
                # 只保留当前 instance_ids 中存在的 key
                train_diff = [train_diff_map.get(iid, 0.0) for iid in instance_ids]
                print(f"      复用 train subscores (skipping train eval)", flush=True)
                train_eb = None
            else:
                train_eb = eval_one(args, pid_train, valset, text, True)
                _, train_diff, _ = _score_views(train_eb)

            seed = Candidate(idx=i, persona_text=text, parent=None, gen=0)
            seed.train_subscores = dict(zip(instance_ids, train_diff))
            seed.train_agg = _mean(train_diff)
            if train_eb is not None:
                seed.eval_batch = train_eb  # 保留 train traces 用于 mutation reflective

            # val eval（Pareto 权威分数）
            if val_tasks and text_key in prev_val_by_text:
                # 复用上一次 run 的 held-out val 分数（同 split/cap/seed → 键对齐）
                pc = prev_val_by_text[text_key]
                seed.subscores = {
                    k: pc["subscores"].get(k, 0.0) for k in val_instance_ids
                }
                seed.difficulty_subscores = {
                    k: (pc.get("difficulty_subscores") or {}).get(k, 0.0)
                    for k in val_instance_ids
                }
                seed.validity_subscores = {
                    k: (pc.get("validity_subscores") or {}).get(k, 1.0)
                    for k in val_instance_ids
                }
                seed.agg = _mean(seed.subscores.values())
                seed.difficulty_agg = _mean(seed.difficulty_subscores.values())
                seed.validity_agg = _mean(seed.validity_subscores.values())
                seed.feasible = pc.get("feasible", True)
                print(f"      复用 val subscores (skipping val eval)", flush=True)
            elif val_tasks:
                pid_val = f"{args.persona_id_base}_g00_seed{i}_val"
                val_eb = eval_one(
                    args,
                    pid_val,
                    val_tasks,
                    text,
                    False,
                    num_trials_override=getattr(args, "val_num_trials", None),
                )
                obj, diff, val_scores = _score_views(val_eb)
                seed.subscores = dict(zip(val_instance_ids, obj))
                seed.difficulty_subscores = dict(zip(val_instance_ids, diff))
                seed.validity_subscores = dict(zip(val_instance_ids, val_scores))
                seed.agg = _mean(obj)
                seed.difficulty_agg = _mean(diff)
                seed.validity_agg = _mean(val_scores)
                seed.feasible = _all_feasible(val_eb)
            else:
                # 没有 val split 时退回 train（兼容旧行为）
                obj, diff, val_scores = _score_views(train_eb)
                seed.subscores = dict(zip(instance_ids, obj))
                seed.difficulty_subscores = dict(zip(instance_ids, diff))
                seed.validity_subscores = dict(zip(instance_ids, val_scores))
                seed.agg = _mean(obj)
                seed.difficulty_agg = _mean(diff)
                seed.validity_agg = _mean(val_scores)
                seed.feasible = _all_feasible(train_eb)

            candidates.append(seed)
            seen_personas.append(text)
            print(
                f"    seed {i}: train_agg={seed.train_agg:.3f}  val_agg={seed.agg:.3f}  "
                f"feasible={seed.feasible}",
                flush=True,
            )
            best_idx = max(
                range(len(candidates)),
                key=lambda j: (
                    candidates[j].agg if candidates[j].agg is not None else -1.0
                ),
            )
            write_state(
                run_dir,
                candidates,
                val_instance_ids or instance_ids,
                gen_logs,
                best_idx,
                args,
                invalid_cases,
            )

        start_iter = 1
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        print(
            f"  [gen 0] {len(candidates)} seeds loaded. best=idx{best_idx} "
            f"val_agg={candidates[best_idx].agg:.3f}",
            flush=True,
        )

    # Pareto 和 write_state 统一用 val_instance_ids（若有），否则退回 train instance_ids
    pareto_ids = val_instance_ids if val_instance_ids else instance_ids

    for g in range(start_iter, args.iterations + 1):
        gen_logs.append(
            run_iteration(
                g,
                args,
                candidates,
                valset,
                instance_ids,
                refl_adapter,
                seen_personas,
                invalid_cases,
                val_tasks=val_tasks,
                val_instance_ids=val_instance_ids,
                val_num_trials=getattr(args, "val_num_trials", None),
            )
        )
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        write_state(
            run_dir, candidates, pareto_ids, gen_logs, best_idx, args, invalid_cases
        )

    best_idx = max(
        range(len(candidates)),
        key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
    )
    best = candidates[best_idx]
    write_state(
        run_dir, candidates, pareto_ids, gen_logs, best_idx, args, invalid_cases
    )
    write_candidates_json(run_dir, candidates, best_idx)
    (run_dir / "exploration_best_persona.md").write_text(best.persona_text)

    print(f"\n{'=' * 60}\n  CROSS-DOMAIN GEPA COMPLETE\n{'=' * 60}")
    print(
        f"  candidates: {len(candidates)}  best=idx{best_idx}  "
        f"difficulty={best.agg:.3f}  raw_difficulty={best.difficulty_agg:.3f}  "
        f"validity={best.validity_agg:.3f}"
    )
    print(f"  state       -> {run_dir / 'pgepa_state.json'}")
    print(f"  candidates  -> {run_dir / 'candidates.json'}")
    print(f"  best persona-> {run_dir / 'exploration_best_persona.md'}")

    if not args.dry_run:
        from lineage_loop import materialize

        best_pid = f"{args.persona_id_base}_exploration_best"
        materialize(best_pid, best.persona_text)
        print(f"  skill file  -> {PERSONAS_DIR / f'opt_{best_pid}.skill'}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Cross-domain GEPA with multi-seed initialization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- multi-seed & split ---
    ap.add_argument(
        "--seed-personas",
        nargs="+",
        required=True,
        help="paths to seed persona files (all loaded into pool at gen 0)",
    )
    ap.add_argument(
        "--split-file",
        default=str(PERSONA_OPT_DIR / "crossdomain_variance_split.json"),
        help="JSON file with {train/val/test: {domain: [task_ids]}}",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="which split to use as the training valset (minibatch)",
    )
    ap.add_argument(
        "--val-split",
        default="val",
        help="held-out split for full eval + Pareto（空字符串则不用）",
    )
    ap.add_argument(
        "--val-cap",
        type=int,
        default=25,
        help="val split 最大 task 数（0=全部）；小域自动全保留",
    )
    ap.add_argument(
        "--val-min-domain-size",
        type=int,
        default=5,
        help="task 数 <= 此值的域全部保留不按比例砍",
    )
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--persona-id-base", default="xdom")
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument(
        "--resume-seeds-from",
        nargs="+",
        default=None,
        help="一个或多个上一次 run 的 pgepa_state.json 路径，复用其中 gen=0 seed 的 "
        "train/val 分数（按 persona 文本匹配，按 instance key 自动判定 train/val）",
    )
    ap.add_argument(
        "--val-num-trials",
        type=int,
        default=None,
        help="val eval 的 trials 数（不指定则与 --num-trials 相同；minibatch 仍用 --num-trials）",
    )
    ap.add_argument("--seed", type=int, default=42)

    # --- algorithm ---
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--minibatch", type=int, default=4)
    ap.add_argument(
        "--valset-cap",
        type=int,
        default=0,
        help="if >0, stratified-subsample the valset to this size (proportional by domain)",
    )
    ap.add_argument("--tie-tol", type=float, default=0.0)
    ap.add_argument("--min-dissimilarity", type=float, default=0.30)
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--no-embedding", action="store_true", default=False)
    ap.add_argument("--max-semantic-similarity", type=float, default=0.92)
    ap.add_argument("--max-mutation-attempts", type=int, default=5)
    ap.add_argument(
        "--lint-persona", dest="lint_persona", action="store_true", default=True
    )
    ap.add_argument("--no-lint-persona", dest="lint_persona", action="store_false")
    ap.add_argument("--lint-model", default="openai/gpt-5.4-mini")
    ap.add_argument("--lint-maj", type=int, default=1)

    # --- model / infra (defaults for this cross-domain run) ---
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator")
    ap.add_argument("--user-llm", default="gpt-5.2")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=False
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument("--anthropic-base-url", default=None)
    ap.add_argument("--anthropic-api-key", default=None)
    ap.add_argument("--mutation-model", default="gpt-5.4")
    ap.add_argument("--mutation-edit-pct", default="40-50%")
    ap.add_argument("--max-concurrency", type=int, default=10)
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "airline", "retail"])

    # --- judge ---
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--judge-maj", type=int, default=1)
    ap.add_argument("--judge-valset", action="store_true", default=False)
    ap.add_argument("--severe-gate-rate", type=float, default=0.05)
    ap.add_argument("--valset-judge-max-failures", type=int, default=15)

    # --- testing ---
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="cap tasks PER DOMAIN (0=all)"
    )
    ap.add_argument("--print-pareto", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


if __name__ == "__main__":
    main()
