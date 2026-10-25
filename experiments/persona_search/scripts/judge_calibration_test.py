#!/usr/bin/env python3
"""Regression test for the user-behavior validity JUDGE.

Runs the REAL judge (the exact JUDGE_SYS prompt + _build_body + maj@N voting from
judge_user_intent_preservation.py) against a fixed, hand-labeled calibration set of
BOUNDARY cases, and checks the judge's verdict against the designed ground truth.

Why this exists: the persona search's achievable difficulty is capped by the judge's
boundary. If a judge prompt edit or model swap silently moves that boundary -- e.g.
starts flagging careless/sloppy verification as fabrication -- the headline result
(p0003's sloppy-verification mechanic) quietly evaporates and the reported severe_rate
becomes incomparable across runs. Run this BEFORE trusting any new severe_rate.

Locked anchors (must always pass):
  sloppy_verification_ok      careless != fabrication        (the difficulty mechanic)
  false_confirm_fabrication   false concrete result = severe (its boundary partner)
  withhold_stall_then_give_ok stall-then-give = ok
  withhold_ultimate_refuse..  ultimate refusal = severe
  slow_suspicious_ok          slow/suspicious-but-cooperates = ok
  insist_wrong_solution_ok    wrong-diagnosis-but-cooperates = ok

Also UNIT-tests the VALIDITY GATE semantics (pure, no LLM): a sim counts toward the
gate only when it is severe AND failed (reward<1); summary.json's severe_failure_rate
counts ALL severe verdicts. This is the idx1/idx2 "contradiction" in
docs/PIPELINE_LLM_PROMPTS.md section 3.

    .venv/bin/python experiments/persona_search/scripts/judge_calibration_test.py
    .venv/bin/python experiments/persona_search/scripts/judge_calibration_test.py --maj 3 --model gpt-5.2
    .venv/bin/python experiments/persona_search/scripts/judge_calibration_test.py --gate-only   # no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "experiments" / "baseline" / "scripts")
)  # for the judge module
import os

from judge_user_intent_preservation import _build_body  # noqa: E402

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
CASES_PATH = REPO / "data" / "persona_opt" / "judge_calibration" / "cases.json"

C = {
    "g": "\033[0;32m",
    "r": "\033[0;31m",
    "y": "\033[1;33m",
    "d": "\033[2m",
    "b": "\033[1m",
    "x": "\033[0m",
}


def col(s, k):
    return f"{C[k]}{s}{C['x']}"


# ---------------------------------------------------------------------------
# Validity-gate semantics (pure; mirrors tau_persona_adapter's gate counting)
# ---------------------------------------------------------------------------


def gate_counts(verdict: str, reward) -> bool:
    """True iff this sim contributes to the validity GATE: severe AND failed
    (reward < 1). NOTE: summary.json's severe_failure_rate counts ALL severe
    verdicts (including reward==1 sims); the gate does NOT. They are different
    numbers by design -- the idx1/idx2 'contradiction'."""
    return verdict == "severe_failure" and (reward is not None and float(reward) < 1.0)


def test_gate_semantics() -> bool:
    """Deterministic checks + a worked idx1/idx2 example. No API calls."""
    ok = True
    checks = [
        ("severe_failure", 0.0, True),
        ("severe_failure", 1.0, False),  # severe but task SUCCEEDED -> not gated
        ("ok", 0.0, False),
        ("ok", 1.0, False),
        ("severe_failure", None, False),  # unknown reward -> not counted
    ]
    print(col("\n  GATE SEMANTICS (pure unit test)", "b"))
    for verdict, reward, expected in checks:
        got = gate_counts(verdict, reward)
        mark = col("ok", "g") if got == expected else col("FAIL", "r")
        if got != expected:
            ok = False
        print(
            f"    gate_counts({verdict:<14} reward={str(reward):<4}) = {str(got):<5} {mark}"
        )

    # worked example: idx1 vs idx2 (50 non-infra sims each)
    # idx1: 4 severe, ALL on failed sims        -> gate 4/50 = 8%  > 5% -> GATED
    # idx2: 3 severe, but 1 on a reward=1.0 sim  -> gate 2/50 = 4%  <= 5% -> valid
    #       (summary.json would report idx2 as 3/50 = 6%)
    idx1 = [("severe_failure", 0.0)] * 4 + [("ok", 0.0)] * 20 + [("ok", 1.0)] * 26
    idx2 = (
        [("severe_failure", 0.0)] * 2
        + [("severe_failure", 1.0)] * 1
        + [("ok", 0.0)] * 21
        + [("ok", 1.0)] * 26
    )
    g1 = sum(gate_counts(v, r) for v, r in idx1)
    g2 = sum(gate_counts(v, r) for v, r in idx2)
    s2 = sum(v == "severe_failure" for v, r in idx2)
    print(col("    worked idx1/idx2 example (50 non-infra sims, gate=5%):", "d"))
    print(
        f"      idx1 gate={g1}/50={g1 / 50:.0%} -> {'GATED' if g1 / 50 > 0.05 else 'valid'} "
        f"(expect GATED)"
    )
    print(
        f"      idx2 gate={g2}/50={g2 / 50:.0%} -> {'GATED' if g2 / 50 > 0.05 else 'valid'} "
        f"(expect valid); summary severe_rate={s2}/50={s2 / 50:.0%}"
    )
    if not (g1 == 4 and g2 == 2 and s2 == 3 and g1 / 50 > 0.05 and g2 / 50 <= 0.05):
        ok = False
        print(col("      FAIL: gate worked-example mismatch", "r"))
    return ok


# ---------------------------------------------------------------------------
# Judge regression
# ---------------------------------------------------------------------------


def _join(x) -> str:
    return "\n".join(x) if isinstance(x, list) else str(x)


def run_judge(
    cases: list[dict], model: str, maj: int, concurrency: int
) -> dict[str, dict]:
    """Return {case_id: {verdict, votes, violations}} from maj@N real judge calls."""
    jobs = []  # (case_id, body)
    for c in cases:
        body = _build_body(model, _join(c["instr"]), _join(c["transcript"]))
        for _ in range(maj):
            jobs.append((c["id"], body))

    from openai import OpenAI

    client = OpenAI()

    def _one(job):
        cid, body = job
        try:
            resp = client.chat.completions.create(**body)
            return cid, resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            return cid, None, str(e)

    votes: dict[str, list[str]] = {c["id"]: [] for c in cases}
    viols: dict[str, list] = {c["id"]: [] for c in cases}
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for res in ex.map(_one, jobs):
            cid = res[0]
            if len(res) == 3 or res[1] is None:
                errors += 1
                continue
            try:
                obj = json.loads(res[1])
            except Exception:
                errors += 1
                continue
            votes[cid].append(obj.get("verdict", "ok"))
            viols[cid].extend(obj.get("violations") or [])
    if errors:
        print(
            col(
                f"  [warn] {errors} judge call(s) errored / unparseable (treated as missing)",
                "y",
            )
        )

    out = {}
    for c in cases:
        vs = votes[c["id"]]
        severe = vs.count("severe_failure")
        verdict = (
            "severe_failure" if severe > len(vs) / 2 else ("ok" if vs else "no_vote")
        )
        out[c["id"]] = {
            "verdict": verdict,
            "votes": dict(Counter(vs)),
            "violations": viols[c["id"]],
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--maj", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=0.9,
        help="overall verdict accuracy required to pass",
    )
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="run only the pure gate-semantics unit test (no API calls)",
    )
    args = ap.parse_args()

    gate_ok = test_gate_semantics()
    if args.gate_only:
        print(("\n  " + (col("GATE OK", "g") if gate_ok else col("GATE FAILED", "r"))))
        sys.exit(0 if gate_ok else 1)

    cases = json.loads(Path(args.cases).read_text())
    print(
        col(
            f"\n  JUDGE REGRESSION  {len(cases)} cases x maj={args.maj}  model={args.model}",
            "b",
        )
    )
    results = run_judge(cases, args.model, args.maj, args.concurrency)

    correct = 0
    locked_fail = []
    confusion = Counter()  # (expected, predicted)
    print(f"  {'id':<28}{'cat':<26}{'exp':<15}{'got':<15}{'votes'}")
    print(col("  " + "-" * 92, "d"))
    for c in cases:
        r = results[c["id"]]
        exp, got = c["expected_verdict"], r["verdict"]
        passed = exp == got
        correct += passed
        confusion[(exp, got)] += 1
        if c.get("locked") and not passed:
            locked_fail.append(c["id"])
        lock = col("LOCK", "y") if c.get("locked") else "    "
        mark = col("PASS", "g") if passed else col("FAIL", "r")
        lid = c["id"][:27]
        print(
            f"  {lid:<28}{c['category'][:25]:<26}{exp:<15}{got:<15}"
            f"{r['votes']!s:<22}{lock} {mark}"
        )

    n = len(cases)
    acc = correct / n if n else 0.0
    print(col("  " + "-" * 92, "d"))
    print(
        f"  accuracy {col(f'{correct}/{n} = {acc:.0%}', 'b')}   "
        f"min required {args.min_accuracy:.0%}"
    )
    # confusion summary (only the off-diagonal matters)
    miss = {f"{e}->{p}": v for (e, p), v in confusion.items() if e != p}
    if miss:
        print(f"  mispredictions: {miss}")
    if locked_fail:
        print(col(f"  LOCKED ANCHORS FAILED: {locked_fail}", "r"))

    ok = gate_ok and acc >= args.min_accuracy and not locked_fail
    print(
        "\n  "
        + (col("PASS", "g") if ok else col("FAIL", "r"))
        + (col("  (gate semantics failed)", "r") if not gate_ok else "")
        + (col("  (locked anchor failed)", "r") if locked_fail else "")
        + (col("  (accuracy below threshold)", "r") if acc < args.min_accuracy else "")
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
