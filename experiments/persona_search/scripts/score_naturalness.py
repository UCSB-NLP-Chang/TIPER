#!/usr/bin/env python3
"""Absolute NATURALNESS score for certified persona sims.

Companion to certify_gepa_candidates.py. Cert gives full-set DIFFICULTY + validity;
this gives the other axis the goal asks for: how human the optimized customer sounds.

For every cert persona (data/simulations/gepa_certify/<persona_id>/telecom/results.json)
each CUSTOMER transcript is shown to a GPT judge that scores it 1-10 on an ABSOLUTE
realism rubric (anchored to scripts/judge_user_realism.py's validated criteria:
natural/conversational good; robotic phrasing, unnatural formatting, essay-like
over-explaining bad). maj@N median per transcript, then mean over transcripts ->
one naturalness number per persona, directly comparable across all candidates +
seed + incumbent (unlike the pairwise realism ELO, which needs a shared opponent).

  # score every persona named in a cert report (maps persona_id -> arm/idx/difficulty)
  .venv/bin/python experiments/persona_search/scripts/score_naturalness.py \
      --cert-report data/persona_opt/pgepa_runs/cert_pair_full/certification_report.json \
      --maj 3 --max-sims-per-persona 60 --workers 16

  # or score explicit persona_ids
  .venv/bin/python experiments/persona_search/scripts/score_naturalness.py \
      --persona-ids gepa_certify_c00 gepa_certify_incumbent --maj 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "experiments" / "baseline" / "scripts")
)
import os

from judge_user_realism import _instr_text, _render  # noqa: E402  (validated renderers)

from tau2.data_model.message import SystemMessage, UserMessage  # noqa: E402
from tau2.utils.llm_utils import generate  # noqa: E402

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
SIMS = REPO / "data" / "simulations" / "gepa_certify"

C = {
    "g": "\033[0;32m",
    "y": "\033[1;33m",
    "d": "\033[2m",
    "b": "\033[1m",
    "x": "\033[0m",
}


def col(s, k):
    return f"{C[k]}{s}{C['x']}"


RUBRIC_SYS = (
    "You are an expert evaluator of role-played customer-service conversations. "
    "You will see the role-play INSTRUCTIONS given to a simulated customer and the "
    "resulting TRANSCRIPT (customer turns, with agent turns for context). Rate ONLY "
    "the CUSTOMER on how much they sound like a REAL human customer on the phone.\n\n"
    "Score 1-10 (10 = indistinguishable from a real person; 1 = obviously a bot):\n"
    "  - HIGH: natural, conversational, varied phrasing; says only what a real, "
    "possibly-impatient customer would; reveals info gradually.\n"
    "  - LOW: robotic/stilted phrasing; unnatural formatting (markdown, bullet lists, "
    "headings); essay-like over-explaining; dumping every status field at once; "
    "narrating its own role or strategy.\n"
    "Judge ONLY naturalness of expression, NOT whether the customer was cooperative, "
    "difficult, or whether the task succeeded. A difficult customer can still be 10/10 "
    "human; a helpful one can be 2/10 robotic.\n\n"
    "Examples: HIGH(9) -> 'Ugh, it's still not working. I restarted it twice like you "
    "said.' (varied, conversational, gradual). LOW(2) -> 'Issue: connectivity. Steps "
    "taken: 1) reboot 2) reseat SIM. Status: unresolved. Account: 7781.' (robotic, "
    "bulleted, dumps fields).\n\n"
    'Respond STRICT JSON only: {"score": <int 1-10>, "reason": "<=20 words"}'
)


def _score_once(model: str, instr: str, transcript: str) -> int | None:
    msgs = [
        SystemMessage(role="system", content=RUBRIC_SYS),
        UserMessage(
            role="user",
            content=f"ROLE-PLAY INSTRUCTIONS:\n{instr}\n\nTRANSCRIPT:\n{transcript}",
        ),
    ]
    try:
        # per-call HTTP timeout + bounded retries so a hung connection can't block
        # a worker forever (litellm forwards `timeout`/`num_retries` to completion()).
        out = generate(
            model=model,
            messages=msgs,
            call_name="naturalness_judge",
            timeout=30,
            num_retries=2,
        )
        txt = out.content if hasattr(out, "content") else str(out)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        obj = json.loads(m.group(0) if m else txt)
        s = int(obj.get("score"))
        return s if 1 <= s <= 10 else None
    except Exception:
        return None


def score_sim(model: str, maj: int, instr: str, transcript: str) -> float | None:
    votes = [
        s for s in (_score_once(model, instr, transcript) for _ in range(maj)) if s
    ]
    return median(votes) if votes else None


def load_persona_sims(persona_id: str, max_sims: int) -> list[tuple[str, str]]:
    """Return [(instr_text, rendered_transcript)] for a cert persona, capped."""
    rp = SIMS / persona_id / "telecom" / "results.json"
    if not rp.exists():
        return []
    data = json.loads(rp.read_text())
    out = []
    for sim in data.get("simulations", []):
        task = sim.get("task") or {}
        instr = _instr_text(task)
        transcript = _render(sim)
        if transcript.strip():
            out.append((instr, transcript))
    return out[:max_sims] if max_sims else out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cert-report", help="certification_report.json -> persona_id map + difficulty"
    )
    ap.add_argument(
        "--persona-ids", nargs="*", default=[], help="explicit cert persona_ids"
    )
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--maj", type=int, default=3)
    ap.add_argument("--max-sims-per-persona", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--out", default=None, help="output json (default: alongside cert-report)"
    )
    args = ap.parse_args()

    # Build persona list: (persona_id, label, difficulty, severe_rate)
    personas: list[dict] = []
    if args.cert_report:
        rep = json.loads(Path(args.cert_report).read_text())
        inc = rep.get("incumbent", {})
        personas.append(
            {
                "pid": "gepa_certify_incumbent",
                "label": "incumbent(seed p0003 gen00)",
                "difficulty": round(1 - inc["metrics"]["agent_success_rate"], 4),
                "severe_rate": round(inc["metrics"]["severe_rate"], 4),
            }
        )
        for i, c in enumerate(rep.get("candidates", [])):
            m = c["metrics"]
            src = (c.get("source_run") or "").split("/")[-1]
            personas.append(
                {
                    "pid": f"gepa_certify_c{i:02d}",
                    "label": f"{src}#expl{c.get('exploration_score')}",
                    "difficulty": round(1 - m["agent_success_rate"], 4),
                    "severe_rate": round(m["severe_rate"], 4),
                }
            )
    for pid in args.persona_ids:
        personas.append(
            {"pid": pid, "label": pid, "difficulty": None, "severe_rate": None}
        )

    if not personas:
        print(col("  no personas (give --cert-report or --persona-ids)", "y"))
        sys.exit(1)

    print(
        col(
            f"\n  NATURALNESS  {len(personas)} personas  model={args.model} maj={args.maj} "
            f"cap={args.max_sims_per_persona}/persona",
            "b",
        )
    )

    results = []
    for p in personas:
        sims = load_persona_sims(p["pid"], args.max_sims_per_persona)
        if not sims:
            print(f"  {p['pid']:<24} {col('no sims yet', 'y')}")
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            scores = list(
                ex.map(lambda s: score_sim(args.model, args.maj, s[0], s[1]), sims)
            )
        scores = [s for s in scores if s is not None]
        nat = round(mean(scores), 2) if scores else None
        row = {**p, "naturalness": nat, "n_scored": len(scores), "n_sims": len(sims)}
        results.append(row)
        diff = f"diff {p['difficulty']}" if p["difficulty"] is not None else ""
        sev = f"sev {p['severe_rate']}" if p["severe_rate"] is not None else ""
        valid = ""
        if p["severe_rate"] is not None:
            valid = col("VALID", "g") if p["severe_rate"] <= 0.05 else col("GATED", "y")
        print(
            f"  {p['pid']:<24} nat {col(str(nat), 'b'):<14} {diff:<12} {sev:<11} {valid} "
            f"{col(p['label'], 'd')}"
        )

    out = args.out or (
        str(Path(args.cert_report).parent / "naturalness.json")
        if args.cert_report
        else "naturalness.json"
    )
    Path(out).write_text(json.dumps(results, indent=2))
    print(col(f"\n  wrote {out}", "d"))

    # rank valid personas by difficulty, show naturalness alongside
    valid_rows = [
        r
        for r in results
        if r.get("severe_rate") is not None
        and r["severe_rate"] <= 0.05
        and r.get("difficulty") is not None
    ]
    if valid_rows:
        valid_rows.sort(key=lambda r: r["difficulty"], reverse=True)
        print(col("\n  VALID personas ranked by full-set difficulty:", "b"))
        for r in valid_rows:
            print(
                f"    diff {r['difficulty']:<8} nat {str(r['naturalness']):<6} "
                f"{r['pid']}  {col(r['label'], 'd')}"
            )
        best = valid_rows[0]
        print(
            col(
                f"\n  BEST valid & hardest: {best['pid']} "
                f"(diff {best['difficulty']}, nat {best['naturalness']}) {best['label']}",
                "g",
            )
        )


if __name__ == "__main__":
    main()
