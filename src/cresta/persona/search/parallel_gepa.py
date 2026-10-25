#!/usr/bin/env python3
"""Serial, Pareto-frontier persona search.

A drop-in alternative to gepa_runner.py's GEPAEngine loop. ONE parent, ONE child
per iteration (no parallel fan-out, no length control):

    gen 0   : confirm the seed on the full valset (per-instance subscores + traces)
    iter g  : 1. select ONE parent from GEPA's per-instance Pareto frontier
                 (gepa.gepa_utils.select_program_candidate_from_pareto_front --
                 instance-win-frequency weighted, dominance-pruned; same logic as
                 gepa.strategies.ParetoCandidateSelector, fed a frontier mapping we
                 build from our own confirmed+feasible candidate pool).
              2. reflective-mutate it into ONE child. The child MUST differ by
                 >= --min-dissimilarity from EVERY persona sampled so far (retry
                 the mutation otherwise), and the mutation MUST NOT insert any
                 task-specific / task-conditioned content (general profile only).
              3. minibatch screen vs the parent (using the parent's CACHED
                 full-valset subscores -- no parent re-eval).
              4. full-valset difficulty eval + validity judge (two-stage).
              5. if minibatch passed AND feasible -> add to the pool.

Search score = difficulty (1 - agent reward). Validity/feasibility is a hard gate:
invalid candidates are rejected, not softly downweighted. Reuses
eval.adapter (eval + reflection + mutation), search.gepa_runner.build_task_list
(valset), and gepa.gepa_utils (Pareto sampling).

    # dry-run smoke (no Bedrock, seconds, $0)
    python -m cresta.persona.search.parallel_gepa \
        --seed-persona experiments/persona_search/seeds/lineages/p0001/gen_00.md \
        --dry-run --iterations 3 --minibatch 2 --max-tasks 2 --print-pareto
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import tau2.utils.cresta_vertex  # noqa: E402, F401
from cresta.persona.core._common import (  # noqa: E402
    PERSONA_OPT_DIR,
    PERSONAS_DIR,
    materialize,
)
from cresta.persona.core.candidate import (  # noqa: E402
    Candidate,
    _all_feasible,
    _mean,
    _score_views,
    build_pareto,
    sample_parent,
)
from cresta.persona.core.state import (  # noqa: E402
    _collect_train_traces,
    load_resume_state,
    write_candidates_json,
    write_state,
)
from cresta.persona.diversity.metrics import (  # noqa: E402
    DiversityConfig,
    Embedder,
)
from cresta.persona.diversity.metrics import (
    diversity_check as _diversity_check_core,
)
from cresta.persona.diversity.seen_pool import SeenPool, summarize_persona  # noqa: E402
from cresta.persona.eval import adapter as tpa  # noqa: E402
from cresta.persona.eval.adapter import (  # noqa: E402
    TauPersonaAdapter,
)
from cresta.persona.eval.scoring import (  # noqa: E402
    TauTask,
)
from cresta.persona.eval.sim import (  # noqa: E402
    build_reuse_cache,
    build_tasks_from_split,
    eval_existing_with_validity,
    eval_one,
    eval_one_difficulty_only,
    make_child_adapter,
)
from cresta.persona.mutation.lint import register_invalid  # noqa: E402
from cresta.persona.mutation.override import (  # noqa: E402
    _BASE_OVERRIDE,
    MutationContextBuilder,
    mutate_valid_diverse,
)
from cresta.persona.search.gepa_runner import build_task_list  # noqa: E402

tpa.GEPA_MODE_OVERRIDE = _BASE_OVERRIDE


# ---------------------------------------------------------------------------
# Diversity wrapper
# ---------------------------------------------------------------------------

_RUN_EMBEDDER: Embedder | None = None
_EMBED_WARNED = False


def _diversity_cfg(args) -> DiversityConfig:
    return DiversityConfig(
        embedding_model=None if args.no_embedding else args.embedding_model,
        min_dissimilarity=args.min_dissimilarity,
        max_semantic_similarity=args.max_semantic_similarity,
    )


def diversity_check(
    args, text: str, seen: list[str]
) -> tuple[bool, float, float | None, str, str]:
    global _RUN_EMBEDDER, _EMBED_WARNED
    if _RUN_EMBEDDER is None:
        _RUN_EMBEDDER = Embedder(None if args.no_embedding else args.embedding_model)
    ok, lex, sem, reason, nearest = _diversity_check_core(
        _diversity_cfg(args), text, seen, embedder=_RUN_EMBEDDER
    )
    if _RUN_EMBEDDER.disable_reason and not _EMBED_WARNED:
        print(
            f"    [embed] disabled (semantic layer off -> lexical-only): "
            f"{_RUN_EMBEDDER.disable_reason}"
        )
        _EMBED_WARNED = True
    return ok, lex, sem, reason, nearest


# ---------------------------------------------------------------------------
# One iteration: select parent -> mutate (diverse, no task-specific) -> confirm
# ---------------------------------------------------------------------------


def run_iteration(
    g: int,
    args,
    candidates: list[Candidate],
    valset: list[TauTask],
    instance_ids: list[str],
    refl_adapter: TauPersonaAdapter,
    seen_personas: list[str],
    invalid_cases: list[dict],
    val_tasks: list[TauTask] | None = None,
    val_instance_ids: list[str] | None = None,
    val_num_trials: int | None = None,
    seen_pool: SeenPool | None = None,
    island_name: str = "",
    summary_limit: int = 40,
    gen_logs: list[dict] | None = None,
) -> dict:
    base = args.persona_id_base

    # 1. select ONE parent from the per-instance Pareto frontier (gepa lib).
    #    Candidate.subscores are keyed by the VAL instance_ids (held-out), so the
    #    Pareto frontier MUST be built over val keys — using train instance_ids
    #    here makes every subscores.get(train_key) == 0, collapsing the frontier
    #    to a single "highest-agg" winner (always the seed). Fall back to train
    #    ids only when no val split is configured (legacy single-split callers).
    pareto_ids = val_instance_ids if val_instance_ids is not None else instance_ids
    pareto = build_pareto(candidates, pareto_ids, args.tie_tol)
    parent_idx = sample_parent(candidates, pareto, random.Random(args.seed + 1000 + g))
    parent = candidates[parent_idx]
    if args.print_pareto:
        pool = {
            c.idx: round(c.agg or 0.0, 4)
            for c in candidates
            if c.subscores is not None and c.feasible
        }
        print(
            f"  [iter {g}] pareto_size={len(pareto)} parent=idx{parent_idx} pool={pool}"
        )

    # 2. sample minibatch from train BEFORE mutation, so we can collect parent
    #    traces on these tasks and feed them to the proposer (Plan C: no val
    #    leakage — proposer only sees train/minibatch traces).
    mb_rng = random.Random(args.seed * 7919 + g)
    if args.minibatch > 0:
        import math as _math

        by_dom: dict[str, list[TauTask]] = {}
        for t in valset:
            by_dom.setdefault(t.domain, []).append(t)
        mb_cap = min(args.minibatch, len(valset))
        mb_list: list[TauTask] = []
        leftover: list[tuple[float, str]] = []
        n_alloc = 0
        for dom in sorted(by_dom):
            exact = mb_cap * len(by_dom[dom]) / len(valset)
            fl = _math.floor(exact)
            mb_list += mb_rng.sample(by_dom[dom], min(fl, len(by_dom[dom])))
            n_alloc += fl
            leftover.append((exact - fl, dom))
        leftover.sort(reverse=True)
        for _, dom in leftover:
            if n_alloc >= mb_cap:
                break
            already = {t.task_id for t in mb_list if t.domain == dom}
            extras = [t for t in by_dom[dom] if t.task_id not in already]
            if extras:
                mb_list.append(mb_rng.choice(extras))
                n_alloc += 1
        minibatch = mb_list
    else:
        minibatch = valset
    mb_ids = [f"{t.domain}::{t.task_id}" for t in minibatch]
    mb_bare_ids = {t.task_id for t in minibatch}

    # 2b. collect parent's traces on minibatch tasks from disk for mutation;
    #     if not on disk, run parent on minibatch to capture traces.
    mb_trace_eb = _collect_train_traces(base, parent.gen, mb_bare_ids, args.domains)
    if mb_trace_eb is not None:
        n_traj = len(mb_trace_eb.trajectories)
        print(f"  [gen {g}] collected {n_traj} parent traces on minibatch from disk")
    elif not args.dry_run:
        print(
            f"  [gen {g}] no parent traces on disk → running parent on minibatch for traces"
        )
        mb_trace_eb = eval_one(
            args,
            f"{base}_g{g:02d}_parent_trace",
            minibatch,
            parent.persona_text,
            capture=True,
            scoring_mode_override="mean",
        )
        n_traj = (
            len(mb_trace_eb.trajectories)
            if mb_trace_eb and mb_trace_eb.trajectories
            else 0
        )
        print(f"  [gen {g}] captured {n_traj} parent traces on minibatch")

    # 3. mutate into ONE child that is novel vs every seen persona (layered
    #    lexical+semantic gate), passes the static linter, and carries NO
    #    task-specific content (enforced in the override prompt + examples).
    # Cross-island uniqueness: fold every OTHER island's persona texts into the
    # diversity gate's seen list, and inject short summaries of what they explored
    # into the mutation prompt (dedup, not distance-maximization).
    if seen_pool is not None:
        pool_texts = seen_pool.texts(exclude_island=island_name)
        seen_for_gate = seen_personas + pool_texts
        explored = seen_pool.summaries(limit=summary_limit, exclude_island=island_name)
        if pool_texts:
            print(
                f"    [pool] {len(pool_texts)} personas from other islands fold into "
                f"diversity gate; {len(explored)} summaries injected"
            )
    else:
        seen_for_gate = seen_personas
        ctx_builder = MutationContextBuilder(
            mode=getattr(args, "mutation_context", "basic"),
            gen_logs=gen_logs or [],
            candidates=candidates,
            seen_personas=seen_personas,
        )
        explored = ctx_builder.build_explored()

    m = mutate_valid_diverse(
        args,
        parent,
        refl_adapter,
        seen_for_gate,
        invalid_cases,
        explored_summaries=explored,
        trace_eval_batch=mb_trace_eb,
    )
    child_text = m["text"]
    seen_personas.append(child_text)  # tried, so it counts toward future diversity
    # publish to the shared pool immediately (even before accept/reject) so a
    # concurrent island never regenerates this same direction.
    if seen_pool is not None:
        seen_pool.add(
            island_name, f"g{g:02d}", summarize_persona(child_text), child_text
        )

    # 4. minibatch screen vs the parent's CACHED train difficulty (no re-eval).
    # 用 train_subscores 做 minibatch 比较。parent 缺失的 task 按需补评（而非
    # 跳过或当 0），确保预筛总是在完整 minibatch 上公平比较。
    parent_train = parent.train_subscores or parent.difficulty_subscores or {}
    missing_ids = [i for i in mb_ids if i not in parent_train]
    if missing_ids and not args.dry_run:
        missing_tasks = [
            t for t in minibatch if f"{t.domain}::{t.task_id}" in set(missing_ids)
        ]
        print(
            f"  [gen {g}] parent missing {len(missing_tasks)} train scores → on-demand eval"
        )
        p_eb, _ = eval_one_difficulty_only(
            args,
            f"{base}_g{g:02d}_parent_fill",
            missing_tasks,
            parent.persona_text,
            scoring_mode_override="mean",
        )
        _, p_diff, _ = _score_views(p_eb)
        for tid, d in zip(missing_ids, p_diff):
            parent_train[tid] = d
        if parent.train_subscores is None:
            parent.train_subscores = {}
        parent.train_subscores.update(zip(missing_ids, p_diff))
    parent_mb = _mean(parent_train.get(i, 0.0) for i in mb_ids)

    pid = f"{base}_g{g:02d}_k00"
    child_mb_eb = eval_one(
        args, pid, minibatch, child_text, False, scoring_mode_override="mean"
    )
    _, child_mb_diff, _ = _score_views(child_mb_eb)
    child_mb = _mean(child_mb_diff)
    # Allow ties through to full-set confirm. Minibatch is the admission gate;
    # full valset is for per-instance subscores + validity, not a second
    # difficulty-vs-parent filter.
    minibatch_passed = child_mb >= parent_mb

    log = {
        "gen": g,
        "parent": parent_idx,
        "mutation_direction": m.get("mutation_direction"),
        "lexical_dissim": round(m["lexical_dissim"], 4),
        "semantic_sim": (
            round(m["semantic_sim"], 4) if m["semantic_sim"] is not None else None
        ),
        "diversity_ok": m["diversity_ok"],
        "lint_verdict": (m["lint"] or {}).get("verdict") if m["lint"] else None,
        "lint_ok": m["lint_ok"],
        "gate_ok": m["gate_ok"],
        "mutation_attempts": m["attempts"],
        "minibatch": mb_ids,
        "minibatch_mean": round(child_mb, 4),
        "parent_minibatch_mean": round(parent_mb, 4),
        "minibatch_passed": minibatch_passed,
        "persona_text": child_text,
        "difficulty_agg": None,
        "parent_difficulty_agg": parent.difficulty_agg,
        "difficulty_passed": None,
        "validity_agg": None,
        "feasible": None,
        "accepted": False,
        "accept_reason": None,
        "idx": None,
        "pareto_size": len(pareto),
    }

    if not minibatch_passed:
        log["accept_reason"] = "below_parent_on_minibatch"
        return _finish_iteration_log(g, args, candidates, parent_idx, log)

    # 有 val_tasks 则用 held-out val 做 confirm，否则退回 train valset（兼容旧调用）
    confirm_tasks = val_tasks if val_tasks is not None else valset
    confirm_ids = val_instance_ids if val_instance_ids is not None else instance_ids
    # When a train/val split is active, an accepted child also needs train scores
    # (so it can be a future parent in the minibatch screen). Pass the train set +
    # the minibatch scores we ALREADY computed, so confirm only evals the rest.
    train_tasks = valset if val_tasks is not None else None
    mb_scores = dict(zip(mb_ids, child_mb_diff)) if val_tasks is not None else None
    return _run_full_valset_confirm(
        g,
        args,
        parent,
        child_text,
        confirm_tasks,
        confirm_ids,
        candidates,
        invalid_cases,
        log,
        lexical_dissim=m["lexical_dissim"],
        val_num_trials=val_num_trials,
        train_tasks=train_tasks,
        train_ids=instance_ids,
        mb_scores=mb_scores,
    )


def _run_full_valset_confirm(
    g: int,
    args,
    parent: Candidate,
    child_text: str,
    valset: list[TauTask],
    instance_ids: list[str],
    candidates: list[Candidate],
    invalid_cases: list[dict],
    log: dict,
    *,
    lexical_dissim: float,
    val_num_trials: int | None = None,
    train_tasks: list[TauTask] | None = None,
    train_ids: list[str] | None = None,
    mb_scores: dict[str, float] | None = None,
) -> dict:
    """Steps 4a–5: full-valset difficulty eval, validity judge, accept if feasible.

    When ``train_tasks`` is given (train/val split mode), an accepted child is also
    scored on the full TRAIN set to populate ``train_subscores`` for the minibatch
    pre-screen when it later serves as a parent. ``mb_scores`` (the minibatch
    results already computed for this child) are reused so only the un-scored train
    tasks are re-evaluated."""
    base = args.persona_id_base
    parent_idx = log["parent"]

    # 4a. full-valset difficulty eval (no judge, no traces) — populates subscores.
    pid = f"{base}_g{g:02d}_k00_full_diff"
    diff_eb, artifacts = eval_one_difficulty_only(
        args, pid, valset, child_text, num_trials_override=val_num_trials
    )
    _, diff_scores, _ = _score_views(diff_eb)
    child_diff = _mean(diff_scores)
    log["difficulty_agg"] = round(child_diff, 4)
    log["difficulty_passed"] = (
        parent.difficulty_agg is None or child_diff > parent.difficulty_agg
    )

    # 4b. validity judge + trace capture on the SAME sims; feasible is a hard gate.
    pid = f"{base}_g{g:02d}_k00_full"
    eb = eval_existing_with_validity(
        args, pid, valset, child_text, artifacts, num_trials_override=val_num_trials
    )
    obj_scores, diff_scores, val_scores = _score_views(eb)
    child_score = _mean(obj_scores)
    feasible = _all_feasible(eb)
    log["validity_agg"] = round(_mean(val_scores), 4)
    log["feasible"] = feasible
    log["agg"] = round(child_score, 4)

    if not feasible:
        log["accept_reason"] = "failed_feasibility_gate"
        register_invalid(
            invalid_cases,
            "confirmed_infeasible",
            child_text,
            "passed lint but failed sim validity (severe/timeout gate)",
        )
        return _finish_iteration_log(g, args, candidates, parent_idx, log)

    # 5. accept: add to the pool (becomes Pareto-eligible next iteration).
    c = Candidate(
        idx=len(candidates),
        persona_text=child_text,
        parent=parent_idx,
        gen=g,
        dissimilarity=lexical_dissim,
    )
    c.subscores = dict(zip(instance_ids, obj_scores))
    c.difficulty_subscores = dict(zip(instance_ids, diff_scores))
    c.validity_subscores = dict(zip(instance_ids, val_scores))
    c.agg = child_score
    c.difficulty_agg = _mean(diff_scores)
    c.validity_agg = _mean(val_scores)
    c.feasible = feasible
    c.eval_batch = eb
    # train scores for the minibatch screen when this child later parents: keep
    # only the minibatch results already computed (no top-up eval). The pre-screen
    # compares on the intersection of parent/child scored tasks, so partial
    # coverage is fine — saves ~20 rollouts per accepted child.
    if train_tasks is not None and train_ids is not None:
        have = dict(mb_scores or {})
        c.train_subscores = {i: have[i] for i in train_ids if i in have}
        c.train_agg = _mean(c.train_subscores.values()) if c.train_subscores else 0.0
    candidates.append(c)
    log["idx"] = c.idx
    log["accepted"] = True
    log["accept_reason"] = "minibatch_passed_and_feasible"
    return _finish_iteration_log(g, args, candidates, parent_idx, log)


def confirm_generation(
    g: int,
    args,
    candidates: list[Candidate],
    valset: list[TauTask],
    instance_ids: list[str],
    gen_logs: list[dict],
    seen_personas: list[str],
    invalid_cases: list[dict],
    persona_path: Path | None = None,
) -> dict:
    """Resume mid-iteration: skip mutation/minibatch, run full-valset confirm for gen *g*.

    Reads the child persona from ``opt_{base}_g{g:02d}_k00.skill`` (or *persona_path*),
    uses the parent idx recorded in that generation's log entry, then runs steps 4a–5.
    """
    entry = next((x for x in gen_logs if x.get("gen") == g), None)
    if entry is None:
        raise SystemExit(
            f"--confirm-gen {g}: no generation log entry in pgepa_state.json"
        )
    if entry.get("accepted"):
        print(
            f"  [confirm gen {g}] already accepted as idx{entry['idx']} — nothing to do"
        )
        return entry

    base = args.persona_id_base
    if persona_path is not None:
        skill = persona_path
    else:
        skill = PERSONAS_DIR / f"opt_{base}_g{g:02d}_k00.skill"
    if not skill.exists():
        raise SystemExit(f"--confirm-gen {g}: persona file not found: {skill}")
    child_text = skill.read_text()

    parent_idx = entry["parent"]
    if parent_idx < 0 or parent_idx >= len(candidates):
        raise SystemExit(
            f"--confirm-gen {g}: invalid parent idx {parent_idx} "
            f"(pool size {len(candidates)})"
        )
    parent = candidates[parent_idx]

    log = dict(entry)
    log["minibatch_passed"] = True
    log["confirm_rerun"] = True
    log["accepted"] = False
    log["accept_reason"] = None
    log["idx"] = None
    log["difficulty_agg"] = None
    log["difficulty_passed"] = None
    log["validity_agg"] = None
    log["feasible"] = None
    log.pop("agg", None)

    lex = float(entry.get("lexical_dissim") or 0.0)
    print(
        f"  [confirm gen {g}] skip mutation/minibatch -> full valset confirm "
        f"(parent=idx{parent_idx}, persona={skill.name}, {len(child_text)} chars)"
    )
    log = _run_full_valset_confirm(
        g,
        args,
        parent,
        child_text,
        valset,
        instance_ids,
        candidates,
        invalid_cases,
        log,
        lexical_dissim=lex,
    )
    if log.get("accepted") and child_text not in seen_personas:
        seen_personas.append(child_text)
    return log


def _finish_iteration_log(
    g: int, args, candidates: list[Candidate], parent_idx: int, log: dict
) -> dict:
    best_idx = max(
        range(len(candidates)),
        key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
    )
    log["best_idx"] = best_idx
    log["best_agg"] = candidates[best_idx].agg
    acc = f"idx{log['idx']}" if log["accepted"] else f"rejected({log['accept_reason']})"
    sem = log.get("semantic_sim")
    sem_s = f" sem={sem:.2f}" if sem is not None else ""
    print(
        f"  [iter {g}] parent=idx{parent_idx} lex_dissim={log['lexical_dissim']:.0%}"
        f"{'' if log['gate_ok'] else '!'}{sem_s} lint={log.get('lint_verdict')} "
        f"mb={log['minibatch_mean']:.3f}(p={log['parent_minibatch_mean']:.3f}) -> {acc}  "
        f"best=idx{best_idx} diff={candidates[best_idx].agg:.3f}"
    )
    return log


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.confirm_gen is not None and not args.resume:
        raise SystemExit(
            "--confirm-gen requires --resume (loads pgepa_state.json from --run-dir)"
        )

    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(PERSONA_OPT_DIR / "pgepa_runs" / f"run_{ts}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Task universe. With --split-file: valset = TRAIN (minibatch source), and a
    # separate held-out VAL set drives selection/Pareto; TEST stays untouched.
    # Without it: legacy single set from load_rep_tasks (val_tasks stays None).
    val_tasks = None
    val_instance_ids = None
    if args.split_file:
        valset = build_tasks_from_split(args.split_file, "train", args.domains)
        val_tasks = build_tasks_from_split(args.split_file, "val", args.domains)
        val_instance_ids = [f"{t.domain}::{t.task_id}" for t in val_tasks]
    else:
        valset = build_task_list(args.domains, args.max_tasks)
    instance_ids = [f"{t.domain}::{t.task_id}" for t in valset]
    seed_text = Path(args.seed_persona).read_text()

    # shared cross-island seen pool (file-locked); None if not running as an island
    seen_pool = SeenPool(args.shared_seen_pool) if args.shared_seen_pool else None
    if seen_pool is not None:
        print(
            f"  [island {args.island_name or '?'}] shared seen pool "
            f"{args.shared_seen_pool} (currently {seen_pool.count()} personas)"
        )
    if val_tasks is not None:
        print(
            f"  [split] train={len(valset)} (minibatch source)  "
            f"val={len(val_tasks)} (selection)  -> {args.split_file}"
        )

    print(
        f"parallel_gepa (serial Pareto): {len(valset)} instances "
        f"({sum(t.domain == 'telecom' for t in valset)}T+"
        f"{sum(t.domain == 'banking' for t in valset)}B)  run_dir={run_dir}"
    )
    sem_desc = (
        "off"
        if (args.no_embedding or not args.embedding_model)
        else f"{args.embedding_model}<= {args.max_semantic_similarity}"
    )
    print(
        f"  iterations={args.iterations} minibatch={args.minibatch} "
        f"parent=pareto_instance_win  max_concurrency={args.max_concurrency}"
    )
    print(
        f"  diversity: lexical_jaccard>={args.min_dissimilarity:.0%} + semantic={sem_desc}; "
        f"lint={'on' if args.lint_persona else 'off'}"
        f"{'(maj' + str(args.lint_maj) + ')' if args.lint_persona else ''}; "
        f"<={args.max_mutation_attempts} attempts/child"
    )

    refl_adapter = make_child_adapter(args, f"{args.persona_id_base}_refl")

    resumed = (
        load_resume_state(
            run_dir,
            instance_ids,
            persona_id_base=args.persona_id_base,
            domains=args.domains,
        )
        if args.resume
        else None
    )
    if resumed is not None:
        candidates, seen_personas, invalid_cases, gen_logs, start_iter = resumed
        print(
            f"  [resume] restored {len(candidates)} candidates, "
            f"{len(invalid_cases)} invalid cases, {len(gen_logs)} completed iters "
            f"-> continuing at iter {start_iter}/{args.iterations}"
        )
        if start_iter > args.iterations:
            print(
                "  [resume] nothing to do: completed iters >= --iterations "
                "(raise --iterations to continue)."
            )
    else:
        if args.resume:
            print("  [resume] no resumable pgepa_state.json found -> starting fresh.")
        # gen 0: confirm the seed (full valset, capture traces) so it has
        # subscores/agg/trajectories before any sampling.
        # Score the seed on the SELECTION set (held-out VAL if split, else the
        # single valset) so its subscores are keyed by the same ids the Pareto
        # frontier is later built over.
        sel_tasks = val_tasks if val_tasks is not None else valset
        sel_ids = val_instance_ids if val_instance_ids is not None else instance_ids
        # Cross-run warm-start: reuse the SEED's prior sims (gen 0 only; the seed
        # persona is identical across runs). Restricted to the current split ids.
        seed_reuse: dict = {}
        if getattr(args, "seed_reuse_glob", None):
            # Use bare task_ids (no domain:: prefix) — old results.json stores
            # bare ids; sel_ids/instance_ids carry a "telecom::" prefix.
            bare_allowed = {t.task_id for t in sel_tasks} | {t.task_id for t in valset}
            seed_reuse = build_reuse_cache(args.seed_reuse_glob, bare_allowed)
            print(
                f"  [gen 0] seed warm-start cache: {len(seed_reuse)} task(s) "
                f"reusable from prior runs ({len(args.seed_reuse_glob)} glob(s))"
            )
        print(f"  [gen 0] confirming seed on {len(sel_tasks)} selection tasks...")
        seed_eb = eval_one(
            args,
            f"{args.persona_id_base}_g00_seed",
            sel_tasks,
            seed_text,
            True,
            reuse_cache=seed_reuse,
        )
        seed = Candidate(idx=0, persona_text=seed_text, parent=None, gen=0)
        seed_obj, seed_diff, seed_val = _score_views(seed_eb)
        seed.subscores = dict(zip(sel_ids, seed_obj))
        seed.difficulty_subscores = dict(zip(sel_ids, seed_diff))
        seed.validity_subscores = dict(zip(sel_ids, seed_val))
        seed.agg = _mean(seed_obj)
        seed.difficulty_agg = _mean(seed_diff)
        seed.validity_agg = _mean(seed_val)
        seed.feasible = _all_feasible(seed_eb)
        seed.eval_batch = seed_eb
        # With a train/val split, also score the seed on the FULL TRAIN set so the
        # minibatch pre-screen has a cached parent baseline on train tasks (the
        # frontier itself is built on VAL subscores above; train is prescreen-only).
        if val_tasks is not None:
            print(
                f"  [gen 0] scoring seed on {len(valset)} train tasks (minibatch baseline)..."
            )
            seed_tr_eb, _ = eval_one_difficulty_only(
                args,
                f"{args.persona_id_base}_g00_seed_train",
                valset,
                seed_text,
                reuse_cache=seed_reuse,
                scoring_mode_override="mean",
            )
            _, seed_tr_diff, _ = _score_views(seed_tr_eb)
            seed.train_subscores = dict(zip(instance_ids, seed_tr_diff))
            seed.train_agg = _mean(seed_tr_diff)
        candidates = [seed]
        seen_personas = [seed_text]
        invalid_cases = []  # observed bad personas, reused as few-shot
        gen_logs = []
        start_iter = 1
        if seen_pool is not None:
            seen_pool.add(
                args.island_name, "g00_seed", summarize_persona(seed_text), seed_text
            )
        print(
            f"  [gen 0] seed score={seed.agg:.3f}  raw_difficulty={seed.difficulty_agg:.3f}  "
            f"validity={seed.validity_agg:.3f}  feasible={seed.feasible}"
        )
        if not seed.feasible:
            print(
                "  WARNING: seed persona is validity-gated (infeasible); the Pareto "
                "frontier will be empty and search will mutate from the seed text only."
            )

    if args.confirm_gen is not None:
        if resumed is None:
            raise SystemExit(
                f"--confirm-gen {args.confirm_gen}: no resumable "
                f"pgepa_state.json in {run_dir}"
            )
        persona_path = Path(args.confirm_persona) if args.confirm_persona else None
        log = confirm_generation(
            args.confirm_gen,
            args,
            candidates,
            valset,
            instance_ids,
            gen_logs,
            seen_personas,
            invalid_cases,
            persona_path,
        )
        for i, lg in enumerate(gen_logs):
            if lg.get("gen") == args.confirm_gen:
                gen_logs[i] = log
                break
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        write_state(
            run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
        )
        if not args.continue_after_confirm:
            write_candidates_json(run_dir, candidates, best_idx)
            best = candidates[best_idx]
            (run_dir / "exploration_best_persona.md").write_text(best.persona_text)
            print(
                f"\n{'=' * 60}\n  CONFIRM-GEN {args.confirm_gen} COMPLETE\n{'=' * 60}"
            )
            print(
                f"  accepted={log.get('accepted')}  reason={log.get('accept_reason')}  "
                f"difficulty={log.get('difficulty_agg')}  validity={log.get('validity_agg')}"
            )
            print(
                f"  candidates: {len(candidates)}  best=idx{best_idx}  "
                f"difficulty={best.agg:.3f}  raw_difficulty={best.difficulty_agg:.3f}"
            )
            print(f"  state -> {run_dir / 'pgepa_state.json'}")
            if log.get("accepted") and not args.dry_run:
                pid = f"{args.persona_id_base}_g{args.confirm_gen:02d}_k00"
                materialize(pid, candidates[log["idx"]].persona_text)
            return
        start_iter = max(start_iter, args.confirm_gen + 1)
        print(
            f"  [confirm-gen] continuing iterations from {start_iter}/{args.iterations}"
        )

    consec_gated = 0
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
                val_num_trials=args.val_num_trials,
                seen_pool=seen_pool,
                island_name=args.island_name,
                summary_limit=args.summary_inject_limit,
                gen_logs=gen_logs,
            )
        )
        best_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
        )
        write_state(
            run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
        )

        last_log = gen_logs[-1]
        if last_log.get(
            "mutation_attempts", 0
        ) >= args.max_mutation_attempts and not last_log.get("gate_ok", True):
            consec_gated += 1
        else:
            consec_gated = 0
        if (
            args.early_stop_consecutive_gated > 0
            and consec_gated >= args.early_stop_consecutive_gated
        ):
            print(
                f"\n  EARLY STOP: {consec_gated} consecutive iterations exhausted "
                f"all {args.max_mutation_attempts} mutation attempts (diversity-gated)."
            )
            break

    best_idx = max(
        range(len(candidates)),
        key=lambda i: candidates[i].agg if candidates[i].agg is not None else -1.0,
    )
    best = candidates[best_idx]
    write_state(
        run_dir, candidates, instance_ids, gen_logs, best_idx, args, invalid_cases
    )
    write_candidates_json(run_dir, candidates, best_idx)
    (run_dir / "exploration_best_persona.md").write_text(best.persona_text)

    print(f"\n{'=' * 60}\n  SERIAL-PARETO-GEPA COMPLETE\n{'=' * 60}")
    print(
        f"  candidates: {len(candidates)}  best=idx{best_idx}  "
        f"difficulty={best.agg:.3f}  raw_difficulty={best.difficulty_agg:.3f}  "
        f"validity={best.validity_agg:.3f}"
    )
    print(f"  state       -> {run_dir / 'pgepa_state.json'}")
    print(f"  candidates  -> {run_dir / 'candidates.json'}")
    print(f"  best persona-> {run_dir / 'exploration_best_persona.md'}")

    if not args.dry_run:
        best_pid = f"{args.persona_id_base}_exploration_best"
        materialize(best_pid, best.persona_text)
        print(f"  skill file  -> {PERSONAS_DIR / f'opt_{best_pid}.skill'}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Serial, Pareto-frontier persona search (1 parent -> 1 child per iter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--seed-persona", required=True, help="path to initial persona .skill/.md"
    )
    ap.add_argument("--run-dir", default=None, help="output dir (default: auto)")
    ap.add_argument(
        "--persona-id-base",
        default="pgepa",
        help="namespace for skill files + sim dirs (must be unique per run)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="continue from <run-dir>/pgepa_state.json: restore candidates, "
        "invalid cases, and completed iterations (skips gen0 seed "
        "confirm). Use the same --domains/--max-tasks/--persona-id-base.",
    )
    ap.add_argument(
        "--confirm-gen",
        type=int,
        default=None,
        metavar="G",
        help="with --resume: skip mutation/minibatch for generation G and "
        "run the full-valset confirm (4a difficulty + 4b judge) on "
        "the saved child persona (opt_{base}_g{G:02d}_k00.skill). "
        "Updates that generation's log entry in state.",
    )
    ap.add_argument(
        "--confirm-persona",
        default=None,
        help="override persona file for --confirm-gen (default: "
        "<persona-state>/personas/opt_{persona-id-base}_g{G:02d}_k00.skill)",
    )
    ap.add_argument(
        "--continue-after-confirm",
        action="store_true",
        default=False,
        help="after --confirm-gen, keep running iterations from G+1 "
        "(default: exit once confirm finishes)",
    )
    ap.add_argument("--seed", type=int, default=0)

    # algorithm
    ap.add_argument(
        "--iterations", type=int, default=24, help="number of parent->child iterations"
    )
    ap.add_argument(
        "--minibatch",
        type=int,
        default=10,
        help="minibatch size for the cheap pre-screen vs the parent "
        "(<=0 skips straight to the full-valset confirm)",
    )
    ap.add_argument(
        "--tie-tol",
        type=float,
        default=0.0,
        help="Pareto tie tolerance: a candidate within tie_tol of the "
        "per-instance best co-wins that instance (coarse scores)",
    )
    ap.add_argument(
        "--min-dissimilarity",
        type=float,
        default=0.30,
        help="lexical novelty floor: child must have word-3gram Jaccard "
        "dissimilarity >= this vs EVERY persona sampled so far",
    )
    ap.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="litellm embedding model for the SEMANTIC novelty layer "
        "(OpenAI via OPENAI_API_KEY/OPENAI_BASE_URL); "
        "auto-disables (lexical-only) if unavailable",
    )
    ap.add_argument(
        "--no-embedding",
        action="store_true",
        default=False,
        help="disable the semantic (embedding) novelty layer entirely",
    )
    ap.add_argument(
        "--max-semantic-similarity",
        type=float,
        default=0.92,
        help="reject a child whose max embedding cosine vs any prior "
        "persona exceeds this (semantic near-duplicate)",
    )
    ap.add_argument(
        "--max-mutation-attempts",
        type=int,
        default=5,
        help="max reflective re-mutations to satisfy the diversity + "
        "lint gates before keeping the best attempt",
    )
    ap.add_argument(
        "--lint-persona",
        dest="lint_persona",
        action="store_true",
        default=True,
        help="static persona-text gate (natural/valid/no-task-specific) "
        "before any sim; on by default",
    )
    ap.add_argument(
        "--no-lint-persona",
        dest="lint_persona",
        action="store_false",
        help="disable the static persona linter",
    )
    ap.add_argument(
        "--lint-model",
        default="openai/gpt-5.4-mini",
        help="model for the persona linter (OpenAI via "
        "OPENAI_API_KEY/OPENAI_BASE_URL); empty falls back to --mutation-model",
    )
    ap.add_argument(
        "--lint-maj",
        type=int,
        default=1,
        help="majority votes for the persona linter verdict",
    )

    # model / infra (mirror gepa_runner defaults)
    ap.add_argument("--agent-llm", default="gpt-5.4")
    ap.add_argument("--agent-llm-args", default='{"reasoning_effort":"high"}')
    ap.add_argument("--user-type", default="user_simulator", choices=["user_simulator"])
    ap.add_argument("--user-llm", default="claude-opus-4-7")
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--use-bedrock", dest="use_bedrock", action="store_true", default=True
    )
    ap.add_argument("--no-bedrock", dest="use_bedrock", action="store_false")
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument(
        "--anthropic-base-url",
        default=None,
        help="optional Anthropic-compatible base URL for model routing",
    )
    ap.add_argument(
        "--anthropic-api-key",
        default=None,
        help="api key sent with --anthropic-base-url (the vertex shim ignores it)",
    )
    ap.add_argument("--mutation-model", default="vertex_ai/claude-opus-4-7")
    ap.add_argument(
        "--mutation-effort",
        default="xhigh",
        help="reasoning_effort passed to the mutation model; overrides the "
        "per-provider default (e.g. xhigh for gpt-5.4)",
    )
    ap.add_argument(
        "--mutation-edit-pct",
        default="40-50%",
        help="how much of the persona the proposer rewrites per mutation",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="sim fan-out within a single evaluate() (tau2 internal; "
        "this is NOT cross-child parallelism)",
    )
    ap.add_argument("--sim-timeout", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument("--retrieval-config", default="terminal_use")
    ap.add_argument("--domains", nargs="+", default=["telecom", "banking"])

    # judge
    ap.add_argument("--judge-model", default="gpt-5.2")
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument(
        "--judge-maj",
        type=int,
        default=1,
        help="validity-judge votes per task (1 = single judgment)",
    )
    ap.add_argument("--judge-valset", action="store_true", default=False)
    ap.add_argument("--severe-gate-rate", type=float, default=None)
    ap.add_argument("--valset-judge-max-failures", type=int, default=15)

    # island model: train/val/test split + shared cross-island seen pool
    ap.add_argument(
        "--split-file",
        default=None,
        help="train/val/test split JSON ({train/val/test: {domain: [ids]}}). "
        "When set: minibatch is sampled from TRAIN, selection/Pareto "
        "uses VAL (held out), TEST is left untouched for post-hoc eval. "
        "Overrides --max-tasks/load_rep_tasks for the task universe.",
    )
    ap.add_argument(
        "--seed-reuse-glob",
        nargs="*",
        default=None,
        help="glob(s) of PRIOR results.json files whose SEED sims are "
        "reused at gen 0 (cross-run warm-start). Only tasks in the "
        "current split are reused; the seed eval simulates only the "
        "misses. Apply ONLY when the seed persona matches the prior "
        "run's. No effect on mutated children.",
    )
    ap.add_argument(
        "--shared-seen-pool",
        default=None,
        help="path to a file-locked JSONL pool shared across parallel "
        "islands; children from every island fold into this island's "
        "diversity gate so generated personas stay unique across runs.",
    )
    ap.add_argument(
        "--island-name",
        default="",
        help="this island's id (e.g. brinks_00), used to tag its rows in "
        "the shared pool and exclude them from its own dedup view.",
    )
    ap.add_argument(
        "--summary-inject-limit",
        type=int,
        default=40,
        help="max recent cross-island persona summaries injected into the "
        "mutation prompt (caps prompt growth as the pool fills)",
    )
    ap.add_argument(
        "--val-num-trials",
        type=int,
        default=None,
        help="num_trials for the held-out VAL confirm (default: --num-trials)",
    )
    ap.add_argument(
        "--scoring-mode",
        choices=["mean", "fail_squared"],
        default="mean",
        help="task-level aggregation: mean (default) or fail_squared "
        "(all trials must fail to score 1.0, else 0.0)",
    )
    ap.add_argument(
        "--mutation-context",
        choices=["basic", "enriched", "twophase"],
        default="basic",
        help="mutation prompt context mode: basic (title-only explored summaries, "
        "no failure traces), enriched (direction+score explored summaries + "
        "failure traces), or twophase (parallel reflect then propose)",
    )
    ap.add_argument(
        "--reflect-model",
        default=None,
        help="model for Phase-1 parallel propose calls in twophase mode "
        "(default: same as --mutation-model)",
    )
    ap.add_argument(
        "--reflect-effort",
        default="medium",
        help="reasoning effort for Phase-1 propose calls (default: medium)",
    )
    ap.add_argument(
        "--early-stop-consecutive-gated",
        type=int,
        default=0,
        metavar="N",
        help="stop early if N consecutive iterations exhaust all mutation "
        "attempts without passing the diversity gate (0 = disabled)",
    )

    # testing / debug
    ap.add_argument("--max-tasks", type=int, default=35, help="cap tasks PER DOMAIN")
    ap.add_argument(
        "--print-pareto",
        action="store_true",
        help="dump the Pareto pool + selected parent each iteration",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="no sims/LLM; synthetic scores"
    )

    # --- deprecated (accepted as no-ops so old launchers don't crash) ----------
    ap.add_argument(
        "--generations",
        type=int,
        default=None,
        help="[deprecated] alias for --iterations",
    )
    for dep in (
        "--children-per-gen",
        "--parents-per-gen",
        "--top-t",
        "--max-concurrency-children",
        "--child-max-concurrency",
        "--phase-max-concurrency",
        "--length-fix-attempts",
    ):
        ap.add_argument(dep, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--parent-softmax-temperature", type=float, default=None, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--max-length-ratio", type=float, default=None, help=argparse.SUPPRESS
    )
    ap.add_argument("--mutation-mode", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()
    if args.generations is not None:
        args.iterations = args.generations
    _deprecated = [
        d
        for d in (
            "children_per_gen",
            "parents_per_gen",
            "top_t",
            "max_concurrency_children",
            "child_max_concurrency",
            "phase_max_concurrency",
            "length_fix_attempts",
            "parent_softmax_temperature",
            "max_length_ratio",
            "mutation_mode",
        )
        if getattr(args, d, None) is not None
    ]
    if _deprecated:
        print(
            f"  [deprecated] ignoring removed flags: {', '.join(_deprecated)} "
            "(serial single-child search; no length control)"
        )
    return args


if __name__ == "__main__":
    main()
