#!/usr/bin/env python3
"""Run the GLM-5.2-agent matrix.

Each domain's BEST-persona island × {no_persona, seed, opt}:
  airline alaska_00 / retail gc_02 / banking cw_03 / telecom brinks_02.

  agent = GLM-5.2 with HIGH reasoning, via the fireworks shim
          (anthropic/accounts/fireworks/models/glm-5p2 -> ANTHROPIC_API_BASE)
  user  = each domain's ORIGINAL config (airline/retail/telecom gpt-5.4,
          banking gpt-5.5; telecom opt keeps user xhigh for parity with fullset)
  split = full (train + val + test) @ 3 canonical trials (seed=300)

Personas are read byte-for-byte from the existing gpt-5.4 fullset cells
(data/simulations/fullset/<domain>/<cell>/results.json), so only the agent
changes -> GLM vs gpt-5.4 is directly comparable. Output goes to
data/simulations/fullset_glm/ and never touches the gpt-5.4 fullset/.
No salvage: GLM is a new agent, everything is re-run; tau2 --auto-resume
makes it crash-safe.

Prereq: fireworks shim up ->
    FIREWORKS_PROXY_TOKEN=<uuid:secret> \
      .venv/bin/python anthropic_to_fireworks_proxy.py --port 4113 &

Usage:
  .venv/bin/python experiments/persona_search/scripts/run_glm_agent.py --dry-run
  .venv/bin/python experiments/persona_search/scripts/run_glm_agent.py --smoke 1 --cells airline:alaska_00:seed
  .venv/bin/python experiments/persona_search/scripts/run_glm_agent.py            # full run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullset_build import DOMAINS, REPO, SIM, TAU2, read_json, split_ids  # noqa: E402

GLM_MODEL = os.environ.get("GLM_MODEL", "anthropic/accounts/fireworks/models/glm-5p2")
SHIM_BASE = os.environ.get("GLM_SHIM_BASE", "http://127.0.0.1:4113")
SAVE_PREFIX = "fullset_glm"

# each domain's BEST-persona island (most-negative opt Δ on the gpt-5.4 fullset)
WANT = {
    "airline": "alaska_00",
    "retail": "gc_02",
    "banking": "cw_03",
    "telecom": "brinks_02",
}


@dataclass
class GlmCell:
    domain: str
    name: str  # e.g. alaska_00_seed / shared_no_persona
    cond: str  # seed / opt / no_persona
    persona: str | None  # disk-recorded custom_guidelines (None = no_persona)
    user_effort: str | None = None  # telecom opt keeps user xhigh (parity w/ fullset)

    @property
    def persona_hash(self) -> str | None:
        if self.persona is None:
            return None
        return hashlib.md5(self.persona.strip().encode()).hexdigest()[:10]


def _persona_of(domain: str, cell_name: str) -> str | None:
    """Read the exact persona recorded in the existing gpt-5.4 fullset cell."""
    p = SIM / "fullset" / domain / cell_name / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"missing fullset cell for persona: {p}")
    info = read_json(p).get("info") or {}
    pc = (info.get("user_info") or {}).get("persona_config") or {}
    return pc.get("custom_guidelines") if isinstance(pc, dict) else None


def select_cells() -> list[GlmCell]:
    out: list[GlmCell] = []
    for dom, isl in WANT.items():
        out.append(GlmCell(dom, f"{isl}_seed", "seed", _persona_of(dom, f"{isl}_seed")))
        out.append(
            GlmCell(
                dom,
                f"{isl}_opt",
                "opt",
                _persona_of(dom, f"{isl}_opt"),
                user_effort="xhigh" if dom == "telecom" else None,
            )
        )
        out.append(GlmCell(dom, "shared_no_persona", "no_persona", None))
    return out


def run_cell(cell: GlmCell, run_ids: list[str], dry: bool) -> None:
    cfg = DOMAINS[cell.domain]
    save_rel = f"{SAVE_PREFIX}/{cell.domain}/{cell.name}"
    user_args = dict(cfg.user_args)
    if (
        cell.user_effort is not None
    ):  # keep telecom opt user at xhigh (parity w/ fullset)
        user_args["reasoning_effort"] = cell.user_effort
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
        GLM_MODEL,
        "--agent-llm-args",
        json.dumps({}),
        "--num-trials",
        "3",
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
    cmd += ["--task-ids", *run_ids]

    if dry:
        print(
            f"  [would run] {cell.domain}:{cell.name} -> {save_rel}  "
            f"({len(run_ids)} tasks x3t)  user={cfg.user_llm}/{user_args}"
        )
        return

    # GLM agent goes through litellm's anthropic provider -> point it at the shim.
    # The user model uses its own provider and is unaffected by ANTHROPIC_API_BASE.
    env = dict(os.environ)
    env["ANTHROPIC_API_BASE"] = SHIM_BASE
    env["ANTHROPIC_API_KEY"] = "dummy"  # the shim carries the real Bearer token itself
    subprocess.run(cmd, cwd=str(REPO), check=False, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="limit --task-ids to first N per split for a cheap smoke",
    )
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="filter to explicit cells domain:island:cond",
    )
    ap.add_argument(
        "--agent-model",
        default=None,
        help="override agent model (e.g. anthropic/accounts/fireworks/models/deepseek-v4-pro)",
    )
    ap.add_argument(
        "--save-prefix",
        default=None,
        help="override save dir prefix (e.g. fullset_ds); default fullset_glm",
    )
    ap.add_argument(
        "--user-effort",
        default=None,
        help="force this reasoning_effort on ALL cells' user sim (e.g. xhigh)",
    )
    args = ap.parse_args()

    global GLM_MODEL, SAVE_PREFIX
    if args.agent_model:
        GLM_MODEL = args.agent_model
    if args.save_prefix:
        SAVE_PREFIX = args.save_prefix

    cells = select_cells()
    if args.user_effort:  # force uniform user reasoning_effort across every cell
        for c in cells:
            c.user_effort = args.user_effort
    if args.cells:
        want = set(args.cells)
        cells = [
            c
            for c in cells
            if f"{c.domain}:{c.name.replace('_' + c.cond, '')}:{c.cond}" in want
            or f"{c.domain}:{c.name}:{c.cond}" in want
        ]

    print(f"GLM agent = {GLM_MODEL}  via  {SHIM_BASE}")
    print(f"cells ({len(cells)}):")
    for c in cells:
        print(
            f"  {c.domain}:{c.name:20s} cond={c.cond:11s} "
            f"persona_h={c.persona_hash} user={DOMAINS[c.domain].user_llm}"
            f"{'/' + c.user_effort if c.user_effort else ''}"
        )

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
        print(f"\n{'=' * 70}\n{cell.domain}:{cell.name}  ({len(run_ids)} tasks)")
        run_cell(cell, run_ids, dry=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
