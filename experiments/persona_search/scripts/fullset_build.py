#!/usr/bin/env python3
"""Build the FULL-SET (train + val + test @ 3 trials) persona-diff dataset.

For every (domain, island, condition) cell it:
  1. ASSEMBLES all config-compatible sims already on disk (search-phase train/val
     @2t, the fullset_brinks telecom dumps, and the cert test @3t), gated by an
     exact persona fingerprint check (info.user_info.persona_config.custom_guidelines),
     keeps only canonical-seed trials (config.seed=300 -> [626729,373753,361454]),
     dedups by (trial, task_id, seed), caps at 3 trials/task, and writes a single
     json-format checkpoint at the cell's save path.
  2. RUNS `tau2 run --num-trials 3 --auto-resume` with the exact locked persona +
     config over the full (train+val+test) id list. auto-resume recognises the
     prepopulated (trial,task_id,seed) tuples and only fills the missing trials
     (opt/no_persona train+val from scratch; seed 3rd-trial gaps; test already
     present -> skipped).
  3. VERIFIES every split task ends at exactly 3 trials.

Persona fingerprints guarantee we never merge a wrong-candidate's sims (the
GEPA gNN_kMM eval dirs hold a DIFFERENT candidate than the selected winner, so
opt train/val is NOT salvaged -- it is re-run).

Usage:
  python experiments/persona_search/scripts/fullset_build.py --dry-run              # assemble+report gaps, run nothing
  python experiments/persona_search/scripts/fullset_build.py --cells banking:cw_03:seed --smoke 1
  python experiments/persona_search/scripts/fullset_build.py --domains banking      # real run, one domain
  python experiments/persona_search/scripts/fullset_build.py                        # everything
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
TAU2 = str(REPO / ".venv/bin/tau2")
SIM = REPO / "data/simulations"

# Output prefix for --save-to (overridable via --save-prefix so GLM/deepseek runs
# land in fullset_max_glm/ etc. without touching the canonical fullset/).
SAVE_PREFIX = "fullset"

# canonical trial seeds — generate enough for up to 10 trials, slice to NUM_TRIALS.
random.seed(300)
_ALL_SEEDS = [random.randint(0, 1000000) for _ in range(10)]
NUM_TRIALS = 3
CANON_SEEDS = _ALL_SEEDS[:NUM_TRIALS]

ROLEPLAY_PREAMBLE = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)


def _h(s: str | None) -> str:
    return hashlib.md5((s or "").strip().encode()).hexdigest()[:10]


def read_json(p: Path) -> dict:
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Per-domain configuration                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class DomainCfg:
    key: str
    split_file: str
    agent_llm: str
    agent_args: dict
    user_llm: str
    user_args: dict
    retrieval: str | None
    timeout: int | None
    max_steps: int = 120
    concurrency: int = 12
    cli_domain: str | None = None  # tau2 --domain name if it differs from key
    split_key: str | None = None  # key inside split_file if it differs from `key`

    @property
    def domain_flag(self) -> str:
        return self.cli_domain or self.key

    @property
    def split_lookup(self) -> str:
        return self.split_key or self.key


DOMAINS = {
    "airline": DomainCfg(
        "airline",
        "experiments/persona_search/configs/airline_alaska_split_v4.json",
        "gpt-5.4",
        {"reasoning_effort": "high"},
        "gpt-5.4",
        {"reasoning_effort": "high"},
        None,
        None,
    ),
    "retail": DomainCfg(
        "retail",
        "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
        "gpt-5.4",
        {"reasoning_effort": "high"},
        "gpt-5.4",
        {"reasoning_effort": "high"},
        None,
        None,
    ),
    "banking": DomainCfg(
        "banking",
        "experiments/persona_search/configs/banking_knowledge_split_v2.json",
        "gpt-5.5",
        {"reasoning_effort": "high"},
        "gpt-5.4",
        {"reasoning_effort": "high"},
        "bm25",
        600,
        cli_domain="banking_knowledge",
    ),
    "telecom": DomainCfg(
        "telecom",
        "experiments/persona_search/configs/telecom_island_split.json",
        "gpt-5.4",
        {"reasoning_effort": "high"},
        "gpt-5.4",
        {"reasoning_effort": "high"},
        None,
        600,
        concurrency=25,
    ),
    # p0003 is a SEPARATE telecom island run at user=xhigh / agent=high (its own
    # setting → its own summary table). cli_domain=telecom, split reuses telecom.
    "telecom_p0003": DomainCfg(
        "telecom_p0003",
        "experiments/persona_search/configs/telecom_island_split.json",
        "gpt-5.4",
        {"reasoning_effort": "high"},
        "gpt-5.4",
        {"reasoning_effort": "xhigh"},
        None,
        600,
        concurrency=25,
        cli_domain="telecom",
        split_key="telecom",
    ),
}


# --------------------------------------------------------------------------- #
# Cell definition + persona/source resolution                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    domain: str
    island: str
    cond: str  # seed | opt | no_persona
    persona: str | None  # custom_guidelines text (None = no_persona)
    salvage: list[str]  # glob patterns for compatible train/val salvage sources
    test_src: list[str]  # glob patterns for the cert test source(s)
    user_effort: str | None = (
        None  # per-cell override of the domain's user reasoning_effort
    )
    # (telecom opt is xhigh while its seed/no_persona stay high)
    agent_llm: str | None = (
        None  # per-cell override of the domain's agent model (GLM/deepseek)
    )
    agent_effort: str | None = (
        None  # per-cell override of the domain's agent reasoning_effort
    )

    @property
    def name(self) -> str:
        return f"{self.island}_{self.cond}"

    @property
    def save_rel(self) -> str:
        # tau2 treats --save-to as a directory: results.json is written inside it.
        return f"{SAVE_PREFIX}/{self.domain}/{self.name}"

    @property
    def effective_agent_llm(self) -> str:
        return self.agent_llm or DOMAINS[self.domain].agent_llm

    @property
    def effective_agent_args(self) -> dict:
        args = dict(DOMAINS[self.domain].agent_args)
        if self.agent_effort is not None:
            if self.agent_effort == "none":
                args.pop("reasoning_effort", None)
            else:
                args["reasoning_effort"] = self.agent_effort
        return args

    @property
    def persona_hash(self) -> str | None:
        return None if self.persona is None else _h(self.persona)


def _state(path: str) -> dict:
    return read_json(REPO / path)


def _airline_retail_personas(sub: str, isl: str) -> tuple[str, str]:
    st = _state(f"{DATA_ROOT}/data/persona_opt/pgepa_runs/{sub}/{isl}/pgepa_state.json")
    by_idx = {int(c.get("idx", i)): c for i, c in enumerate(st["candidates"])}
    seed = by_idx.get(0, st["candidates"][0])["persona_text"]
    bi = st.get("best_idx")
    best = (
        by_idx.get(int(bi))
        if bi is not None
        else max(
            st["candidates"], key=lambda c: c.get("difficulty_agg") or c.get("agg") or 0
        )
    )
    return seed, best["persona_text"]


def _banking_personas(isl: str) -> tuple[str, str]:
    st = _state(
        f"{DATA_ROOT}/data/persona_opt/pgepa_runs/islands_banking_cardworks_v2/isl_{isl}/pgepa_state.json"
    )
    seed = st["candidates"][0]["persona_text"]
    opt = max(st["candidates"], key=lambda c: c.get("difficulty_agg") or 0.0)[
        "persona_text"
    ]
    return seed, opt


def _persona_from_source(patterns: list[str]) -> str | None | bool:
    """Read the exact custom_guidelines recorded in the first readable source.

    Returns the persona string, None if the source explicitly has no persona
    (no_persona), or False if no source is readable (caller falls back).
    """
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            try:
                data = read_json(Path(f))
            except Exception:
                continue
            pc = ((data.get("info") or {}).get("user_info") or {}).get("persona_config")
            if isinstance(pc, dict) and pc.get("custom_guidelines"):
                return pc["custom_guidelines"]
            return None  # source exists but carries no persona (true no_persona)
    return False


def _persona_from_results(rel: str) -> str | None:
    """Read info.user_info.persona_config.custom_guidelines from one results.json."""
    p = SIM / rel
    if not p.exists():
        return None
    pc = ((read_json(p).get("info") or {}).get("user_info") or {}).get(
        "persona_config"
    ) or {}
    return pc.get("custom_guidelines") if isinstance(pc, dict) else None


# airline opt winners — v5 GEPA gen12 (cert Δ=−0.065, stronger than v4 best_idx).
# test_src points at the cert run so disk-persona canonicalisation picks up the
# v5 gen12 persona; under a non-gpt-5.4 agent (GLM/deepseek) _config_ok rejects
# those gpt-5.4 sims so nothing is merged — only the persona is borrowed.
AIRLINE_V5_OPT_SRC = {
    "alaska_01": "cert_alaska01_v5/optim/results.json",
}
AIRLINE_V5_SEED_SRC = {
    "alaska_01": "cert_alaska01_v5/seed/results.json",  # v5 gen0, the paired seed for Δ
}


# telecom opt winners — AUTHORITATIVE ev2v4_cert (user XHIGH / agent high). These
# are the certified winners whose test diff (1-avg@3) matches the reference table
# (UNIFIED_RESULTS.md:54-69): b00 0.438 / b02 0.432 / b03 0.290 / b04 0.278.
# NOTE: the earlier high-config picks (test_gpt54user/test_enriched_3seeds) were a
# double error (wrong effort + wrong candidate) and are retired — see
# FULLSET_DATA_SOURCES.md 🚨. Each cert dir has test@3t only; train/val are
# salvaged (xhigh, canonical-seed) from the ev2v4_<isl>_g* GEPA run and the
# remaining gap (~464 slots total) is re-run at xhigh.
TELECOM_OPT_WINNER = {
    "brinks_00": "ev2v4_cert/b00_top1/results.json",
    "brinks_02": "ev2v4_cert/b02_top1/results.json",
    "brinks_03": "ev2v4_cert/b03_top1/results.json",
    "brinks_04": "ev2v4_cert/b04_top1/results.json",
}


def build_cells() -> list[Cell]:
    cells: list[Cell] = []

    def gp(*p):
        return [f"{SIM}/{x}" for x in p]

    # ---- AIRLINE / RETAIL (seed + opt + no_persona) ----
    for dom, sub, islands, certbase in [
        (
            "airline",
            "airline_v4",
            ["alaska_00", "alaska_01"],
            "gepa_opt/airline_topdelta_test",
        ),
        (
            "retail",
            "retail_v2",
            ["gc_02", "gc_03", "gc_04"],
            "gepa_opt/retail_topdelta_test",
        ),
    ]:
        for isl in islands:
            seed_txt, opt_txt = _airline_retail_personas(sub, isl)
            cells.append(
                Cell(
                    dom,
                    isl,
                    "seed",
                    seed_txt,
                    salvage=gp(
                        f"gepa_opt/{isl}_g00_seed_train/eval_*/{dom}/results.json",
                        f"gepa_opt/{isl}_g00_seed/eval_*/{dom}/results.json",
                    ),
                    test_src=gp(
                        AIRLINE_V5_SEED_SRC.get(
                            isl, f"{certbase}/seed_{isl}/{dom}/results.json"
                        )
                    ),
                )
            )
            cells.append(
                Cell(
                    dom,
                    isl,
                    "opt",
                    opt_txt,
                    salvage=[],  # opt train/val NOT salvageable (kNN dir = wrong candidate)
                    test_src=gp(
                        AIRLINE_V5_OPT_SRC.get(
                            isl, f"{certbase}/best_{isl}/{dom}/results.json"
                        )
                    ),
                )
            )
        # one shared no_persona per domain (island label = "shared")
        cells.append(
            Cell(
                dom,
                "shared",
                "no_persona",
                None,
                salvage=[],
                test_src=gp(f"{certbase}/no_persona/{dom}/results.json"),
            )
        )

    # ---- BANKING bm25 (seed + opt + no_persona), cw_03 + cw_04 only ----
    for isl in ["cw_03", "cw_04"]:
        seed_txt, opt_txt = _banking_personas(isl)
        cells.append(
            Cell(
                "banking",
                isl,
                "seed",
                seed_txt,
                salvage=gp(
                    f"gepa_opt/isl_{isl}_g00_seed_train/eval_*/banking/results.json",
                    f"gepa_opt/isl_{isl}_g00_seed/eval_*/banking/results.json",
                ),
                test_src=gp(
                    f"gepa_opt/testmatrix_cardworks/{isl}_seed/banking/results.json"
                ),
            )
        )
        cells.append(
            Cell(
                "banking",
                isl,
                "opt",
                opt_txt,
                salvage=[],
                test_src=gp(
                    f"gepa_opt/testmatrix_cardworks/{isl}_opt/banking/results.json"
                ),
            )
        )
    cells.append(
        Cell(
            "banking",
            "shared",
            "no_persona",
            None,
            salvage=[],
            test_src=gp(
                "gepa_opt/testmatrix_cardworks/no_persona/banking/results.json"
            ),
        )
    )

    # ---- TELECOM ev2 seed (4 islands, user gpt-5.4 HIGH) ----
    # The complete 3-trial seed test for ALL 4 islands lives under
    # test_enriched_3seeds/ (user high), with train/val 2-trial salvage under
    # enriched_<isl>_g00_seed{,_train}/ (also user high). The xhigh ev2v4_cert
    # only ever ran seed_00 (3t) + seed_04 (1t), so telecom is aligned to HIGH.
    for isl in ["brinks_00", "brinks_02", "brinks_03", "brinks_04"]:
        seed_md = (
            f"{DATA_ROOT}/data/persona_opt/pgepa_runs/ev2_3islands_v4/seeds/{isl}.md"
        ).read_text()
        persona = ROLEPLAY_PREAMBLE + seed_md
        salv = [
            f"{SIM}/gepa_opt/enriched_{isl}_g00_seed_train/eval_*/telecom/results.json",
            f"{SIM}/gepa_opt/enriched_{isl}_g00_seed/eval_*/telecom/results.json",
        ]
        test_src = gp(f"gepa_opt/test_enriched_3seeds/{isl}_seed/results.json")
        cells.append(
            Cell("telecom", isl, "seed", persona, salvage=salv, test_src=test_src)
        )

    # ---- TELECOM opt (4 islands, user XHIGH via per-cell override) ----
    # Authoritative ev2v4_cert winners (xhigh). test@3t comes straight from the
    # cert dir; train/val are salvaged (xhigh + canonical seed) from the
    # ev2v4_<isl>_g* GEPA run (persona-hash + effort gated) and the ~464-slot gap
    # is re-run at xhigh. seed/no_persona stay at the domain default (high) — so
    # telecom's Δ(opt−seed) is cross-effort (accepted, see FULLSET_DATA_SOURCES.md).
    for isl in ["brinks_00", "brinks_02", "brinks_03", "brinks_04"]:
        win = TELECOM_OPT_WINNER[isl]
        opt_txt = _persona_from_results(win)
        cells.append(
            Cell(
                "telecom",
                isl,
                "opt",
                opt_txt,
                salvage=[f"{SIM}/gepa_opt/ev2v4_{isl}_g*/**/telecom/results.json"],
                test_src=gp(win),
                user_effort="xhigh",
            )
        )

    # ---- TELECOM no_persona (shared, user HIGH) ----
    # nopersona_gpt54 holds test (54@1t) + val (30@1t) at high; train is re-run.
    cells.append(
        Cell(
            "telecom",
            "shared",
            "no_persona",
            None,
            salvage=gp("gepa_opt/nopersona_gpt54/val/results.json"),
            test_src=gp("gepa_opt/nopersona_gpt54/test/results.json"),
        )
    )

    # ---- TELECOM p0003 (separate island, user XHIGH / agent HIGH) ----
    # seed + opt(best). Both cert tests are complete @3t (xhigh). The serial
    # ev2v4_p0003 GEPA run serialised personas → train/val partly salvageable via
    # the broad ev2v4_p0003_g* glob (hash-gated to seed 2af9017cec / opt 5dd389d157).
    for cond, win in [
        ("seed", "ev2v4_cert/p0003_seed/results.json"),
        ("opt", "ev2v4_cert/p0003_best/results.json"),
    ]:
        cells.append(
            Cell(
                "telecom_p0003",
                "p0003",
                cond,
                _persona_from_results(win),
                salvage=[f"{SIM}/gepa_opt/ev2v4_p0003_g*/**/telecom/results.json"],
                test_src=gp(win),
            )
        )

    # Canonicalise each cell's persona to EXACTLY what its cert test recorded on
    # disk (that is the "what was tested" ground truth: airline/retail were run
    # WITHOUT the method-actor preamble, banking/telecom WITH it). Fall back to
    # the constructed persona only when the cell has no cert test source
    # (telecom brinks_02/03).
    for c in cells:
        if c.cond == "no_persona":
            c.persona = None
            continue
        disk = _persona_from_source(c.test_src)
        if isinstance(disk, str):
            if disk.strip() != (c.persona or "").strip():
                print(
                    f"[persona] {c.domain}:{c.name} using disk-recorded persona "
                    f"(disk_h={_h(disk)} != constructed_h={_h(c.persona)})"
                )
            c.persona = disk
        elif disk is None:
            print(
                f"[WARN] {c.domain}:{c.name} test source has NO persona but cell is "
                f"'{c.cond}' — keeping constructed persona"
            )
        # disk is False -> no test source, keep constructed fallback

    return cells


# --------------------------------------------------------------------------- #
# Split ids                                                                    #
# --------------------------------------------------------------------------- #
def split_ids(dom: str) -> dict[str, list[str]]:
    cfg = DOMAINS[dom]
    d = read_json(REPO / cfg.split_file)
    out = {}
    for sp in ("train", "val", "test"):
        v = d[sp]
        out[sp] = list(v[cfg.split_lookup]) if isinstance(v, dict) else list(v)
    return out


# --------------------------------------------------------------------------- #
# Assemble: collect compatible sims -> prepopulated checkpoint                 #
# --------------------------------------------------------------------------- #
def _source_persona_hash(data: dict) -> str | None:
    pc = ((data.get("info") or {}).get("user_info") or {}).get("persona_config") or {}
    cg = pc.get("custom_guidelines") if isinstance(pc, dict) else None
    return _h(cg) if cg else None


def _config_ok(data: dict, cell: Cell) -> bool:
    cfg = DOMAINS[cell.domain]
    i = data.get("info") or {}
    u = i.get("user_info") or {}
    a = i.get("agent_info") or {}
    if u.get("llm") != cfg.user_llm or a.get("llm") != cell.effective_agent_llm:
        return False
    # user effort must match (telecom xhigh vs high matters a lot).
    # cell.user_effort (per-cell override) wins over the domain default.
    want_eff = (
        cell.user_effort
        if cell.user_effort is not None
        else cfg.user_args.get("reasoning_effort")
    )
    if (u.get("llm_args") or {}).get("reasoning_effort") != want_eff:
        return False
    return True


def assemble(
    cell: Cell, all_ids: set[str], verbose: bool = True, resume_path: Path | None = None
) -> tuple[dict, dict]:
    """Return (checkpoint_dict, stats). checkpoint_dict is a valid Results json.

    If resume_path (the cell's own previously-written output) exists, its sims are
    ingested first under the same persona/config gate, so re-running the full build
    after a crash preserves already-completed 补跑 trials instead of re-running them.
    """
    want_hash = cell.persona_hash
    cfg = DOMAINS[cell.domain]
    sims: dict[tuple, dict] = {}  # (trial,task_id,seed) -> sim
    tasks: dict[str, dict] = {}
    chosen_info = None
    stats = {
        "sources": 0,
        "persona_ok": 0,
        "persona_bad": 0,
        "config_bad": 0,
        "sims_kept": 0,
    }

    def ingest(patterns: list[str], is_test: bool):
        nonlocal chosen_info
        for pat in patterns:
            for f in sorted(glob.glob(pat, recursive=True)):
                try:
                    data = read_json(Path(f))
                except Exception:
                    continue
                stats["sources"] += 1
                src_hash = _source_persona_hash(data)
                if src_hash != want_hash:
                    stats["persona_bad"] += 1
                    if verbose:
                        print(
                            f"    ✗ persona mismatch, skip {f.split('/simulations/')[1][:55]} "
                            f"(src={src_hash} want={want_hash})"
                        )
                    continue
                if not _config_ok(data, cell):
                    stats["config_bad"] += 1
                    if verbose:
                        print(
                            f"    ✗ config mismatch, skip {f.split('/simulations/')[1][:55]}"
                        )
                    continue
                stats["persona_ok"] += 1
                if chosen_info is None or is_test:
                    chosen_info = data.get("info")
                for t in data.get("tasks", []):
                    if t.get("id") in all_ids:
                        tasks.setdefault(t["id"], t)
                for s in data.get("simulations", []):
                    tid = str(s.get("task_id"))
                    if tid not in all_ids:
                        continue
                    if (s.get("reward_info") or {}).get("reward") is None:
                        continue
                    if s.get("seed") not in CANON_SEEDS:
                        continue
                    key = (s.get("trial"), tid, s.get("seed"))
                    sims[key] = s

    if resume_path is not None and resume_path.exists():
        ingest([str(resume_path)], is_test=False)
    ingest(cell.test_src, is_test=True)
    ingest(cell.salvage, is_test=False)

    # cap at 3 canonical trials/task (already enforced by CANON_SEEDS set of 3)
    stats["sims_kept"] = len(sims)
    ckpt = {
        "info": chosen_info or {},
        "tasks": list(tasks.values()),
        "simulations": list(sims.values()),
    }
    if chosen_info is not None:
        ckpt["info"]["num_trials"] = NUM_TRIALS
    return ckpt, stats


def coverage(sims: list[dict], ids: dict[str, list[str]]) -> dict[str, str]:
    from collections import defaultdict

    trials = defaultdict(set)
    for s in sims:
        trials[str(s.get("task_id"))].add(s.get("seed"))
    out = {}
    for sp, lst in ids.items():
        have3 = sum(
            1 for t in lst if len(trials.get(t, set()) & set(CANON_SEEDS)) >= NUM_TRIALS
        )
        totrials = sum(len(trials.get(t, set()) & set(CANON_SEEDS)) for t in lst)
        out[sp] = (
            f"{have3}/{len(lst)}@{NUM_TRIALS}t ({totrials}/{len(lst) * NUM_TRIALS} trial-slots)"
        )
    return out


# --------------------------------------------------------------------------- #
# Run                                                                          #
# --------------------------------------------------------------------------- #
def _inject_allowed_params(args: dict, model: str) -> dict:
    """For openai/ models, whitelist reasoning_effort so litellm won't drop it."""
    if model.startswith("openai/") and "reasoning_effort" in args:
        allowed = list(args.get("allowed_openai_params", []))
        if "reasoning_effort" not in allowed:
            allowed.append("reasoning_effort")
        args["allowed_openai_params"] = allowed
    return args


def run_cell(cell: Cell, ids_all: list[str], dry: bool) -> None:
    cfg = DOMAINS[cell.domain]
    save_rel = cell.save_rel
    user_args = dict(cfg.user_args)
    if cell.user_effort is not None:  # per-cell effort override (telecom opt = xhigh)
        user_args["reasoning_effort"] = cell.user_effort
    agent_args = _inject_allowed_params(
        cell.effective_agent_args, cell.effective_agent_llm
    )
    user_args = _inject_allowed_params(user_args, cfg.user_llm)
    cmd = [
        TAU2,
        "run",
        "--domain",
        cfg.domain_flag,
        "--user",
        "user_simulator",
        "--user-llm",
        cfg.user_llm,
        "--user-llm-args",
        json.dumps(user_args),
        "--agent-llm",
        cell.effective_agent_llm,
        "--agent-llm-args",
        json.dumps(agent_args),
        "--num-trials",
        str(NUM_TRIALS),
        "--seed",
        "300",
        "--max-concurrency",
        str(cfg.concurrency),
        "--max-steps",
        str(cfg.max_steps),
        "--save-to",
        save_rel,
        "--auto-resume",
    ]
    if cfg.retrieval:
        cmd += ["--retrieval-config", cfg.retrieval]
    if cfg.timeout:
        cmd += ["--timeout", str(cfg.timeout)]
    if cell.persona is not None:
        cmd += [
            "--user-persona",
            json.dumps({"custom_guidelines": cell.persona}, ensure_ascii=False),
        ]
    cmd += ["--task-ids", *ids_all]

    if dry:
        show = " ".join(c if len(c) < 70 else c[:50] + f"...<{len(c)}c>" for c in cmd)
        print(f"    [would run] {show}")
        return
    subprocess.run(cmd, cwd=str(REPO), check=False)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument(
        "--cells", nargs="+", default=None, help="explicit cells domain:island:cond"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="limit --task-ids to first N (per split) for a cheap smoke",
    )
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="write the prepopulated checkpoints, do not run tau2",
    )
    ap.add_argument(
        "--agent-llm",
        default=None,
        help="override every cell's agent model (e.g. "
        "anthropic/accounts/fireworks/models/glm-5p2)",
    )
    ap.add_argument(
        "--agent-effort",
        default=None,
        help="override every cell's agent reasoning_effort (e.g. max)",
    )
    ap.add_argument(
        "--agent-api-base",
        default=None,
        help="per-call api_base for the agent model (e.g. aliyun endpoint)",
    )
    ap.add_argument(
        "--agent-api-key", default=None, help="per-call api_key for the agent model"
    )
    ap.add_argument(
        "--user-llm",
        default=None,
        help="override every domain's user model (e.g. openai/deepseek-v4-pro)",
    )
    ap.add_argument(
        "--user-effort",
        default=None,
        help="override every cell's user reasoning_effort (e.g. xhigh)",
    )
    ap.add_argument(
        "--save-prefix",
        default=None,
        help="output dir prefix (default 'fullset'; use 'fullset_max_glm' "
        "for GLM/deepseek runs so the canonical fullset/ is untouched)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="override every domain's max-concurrency",
    )
    ap.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="override number of trials per task (default 3)",
    )
    args = ap.parse_args()

    if args.num_trials:
        global NUM_TRIALS, CANON_SEEDS
        NUM_TRIALS = args.num_trials
        CANON_SEEDS = _ALL_SEEDS[:NUM_TRIALS]

    if args.save_prefix:
        global SAVE_PREFIX
        SAVE_PREFIX = args.save_prefix

    cells = build_cells()
    # apply global agent/user/concurrency overrides to every selected cell
    for c in cells:
        if args.agent_llm:
            c.agent_llm = args.agent_llm
        if args.agent_effort:
            c.agent_effort = args.agent_effort
        if args.user_llm:
            DOMAINS[c.domain].user_llm = args.user_llm
        if args.user_effort:
            c.user_effort = args.user_effort
        if args.concurrency:
            DOMAINS[c.domain].concurrency = args.concurrency
        if args.agent_api_base:
            DOMAINS[c.domain].agent_args["api_base"] = args.agent_api_base
        if args.agent_api_key:
            DOMAINS[c.domain].agent_args["api_key"] = args.agent_api_key

    if args.cells:
        want = set(args.cells)
        cells = [c for c in cells if f"{c.domain}:{c.island}:{c.cond}" in want]
    else:
        cells = [c for c in cells if c.domain in args.domains]

    grand = 0
    for cell in cells:
        ids = split_ids(cell.domain)
        if args.smoke:
            run_ids = (
                ids["train"][: args.smoke]
                + ids["val"][: args.smoke]
                + ids["test"][: args.smoke]
            )
        else:
            run_ids = sorted(set(ids["train"]) | set(ids["val"]) | set(ids["test"]))
        all_ids = set(run_ids)  # assemble only what --task-ids will cover
        print(f"\n{'=' * 78}\n{cell.domain}:{cell.name}  persona_h={cell.persona_hash}")
        resume_path = SIM / cell.save_rel / "results.json"
        ckpt, stats = assemble(cell, all_ids, resume_path=resume_path)
        cov = coverage(ckpt["simulations"], ids)
        print(
            f"  salvage sources: {stats['persona_ok']} ok / {stats['persona_bad']} persona-mismatch"
        )
        for sp in ("train", "val", "test"):
            print(f"    {sp:5s}: {cov[sp]}")
        # gap = trial-slots still to fill
        gap = 0
        from collections import defaultdict

        trials = defaultdict(set)
        for s in ckpt["simulations"]:
            trials[str(s["task_id"])].add(s.get("seed"))
        for sp in ("train", "val", "test"):
            for t in ids[sp]:
                gap += NUM_TRIALS - len(trials.get(t, set()) & set(CANON_SEEDS))
        print(f"  => 补跑 trial-slots: {gap}")
        grand += gap

        save_path = SIM / cell.save_rel / "results.json"
        if not args.dry_run:
            if not ckpt["simulations"] and not ckpt.get("info"):
                # No salvage + no prior resume → ckpt info is {}. tau2 --auto-resume
                # calls Results.load which requires info.{git_commit,num_trials,
                # max_steps,max_errors,user_info,agent_info,environment_info}, so an
                # empty-info checkpoint makes tau2 crash before running anything.
                # Remove any stale file so tau2 starts fresh and writes its own.
                if save_path.exists():
                    save_path.unlink()
                print(
                    f"  no salvage/resume — removed stale checkpoint, tau2 starts fresh"
                )
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as fh:
                    json.dump(ckpt, fh)
                print(
                    f"  wrote prepopulated checkpoint: {cell.save_rel}/results.json "
                    f"({len(ckpt['simulations'])} sims, {len(ckpt['tasks'])} tasks)"
                )

        if not args.dry_run and not args.assemble_only:
            run_cell(cell, run_ids, dry=False)

    print(f"\n{'#' * 78}\n合计补跑 trial-slots: {grand}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
