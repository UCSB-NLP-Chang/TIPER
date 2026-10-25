"""Run ONE optimization round FULLY end-to-end, hands-off, including mutation.

This is the single per-round driver. It chains ALL components so a round is one
command with zero human confirmation:

    select -> materialize -> eval_persona (xN, concurrent, infra-retry)
           -> judge_persona (xN) -> record_results -> converge
           -> auto_mutate (LLM-authored children + dedup)

If convergence says should_stop, mutation is skipped and the round ends. Otherwise
the LLM (default Bedrock opus-4.7 with extended thinking) authors children, which
are deduped via add_children.py. The new untested personas are available for the
next round.

It is idempotent / resumable: a persona whose eval (or judge) output already exists
is skipped unless --force is given, so re-running after an interruption is cheap and
safe. Evals themselves use tau2 --auto-resume, so even a forced re-run only fills in
missing sims.

    # finish / run a round (env must have OPENAI_API_KEY + AWS_PROFILE):
    OPENAI_API_KEY=... AWS_PROFILE=us-west-2-staging_dev \
      .venv/bin/python experiments/persona_search/scripts/run_round.py --round 2

    # smoke (1 task per domain, offline-ish), or force a clean re-run:
    .venv/bin/python experiments/persona_search/scripts/run_round.py --round 2 --max-tasks 1
    .venv/bin/python experiments/persona_search/scripts/run_round.py --round 2 --force

    # skip mutation (old behavior):
    .venv/bin/python experiments/persona_search/scripts/run_round.py --round 2 --no-mutate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    GOAL_PATH,
    read_json,
    round_dir,
)

HERE = Path(__file__).resolve().parent
PY = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")


def _preflight_bedrock(profile: str | None) -> None:
    """Quick Bedrock API test — fail fast if SSO token is expired.

    Makes a tiny completion call via litellm. If it hangs or errors, the token
    is stale and we abort immediately instead of silently hanging for hours.
    """
    import litellm

    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile

    old_profile = os.environ.get("AWS_PROFILE")
    if profile:
        os.environ["AWS_PROFILE"] = profile
    try:
        resp = litellm.completion(
            model="bedrock/us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say ok"}],
            max_tokens=4,
            timeout=15,
        )
        text = resp.choices[0].message.content.strip()
        print(f"preflight Bedrock check: OK ({text})")
    except Exception as e:
        print(
            f"\nERROR: Bedrock preflight failed — SSO token likely expired.\n"
            f"Run:  aws sso login --profile {profile or 'us-west-2-staging_dev'}\n"
            f"Then re-run this command.\n\nDetail: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        if old_profile is None and "AWS_PROFILE" in os.environ:
            del os.environ["AWS_PROFILE"]
        elif old_profile is not None:
            os.environ["AWS_PROFILE"] = old_profile


def _run(script: str, *args: str) -> None:
    """Run a component CLI as a subprocess, streaming its output; raise on failure."""
    cmd = [PY, str(HERE / script), *args]
    print(f"\n$ {script} {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=True)


def _capture(script: str, *args: str, log: Path) -> int:
    """Run a component, tee output to a log file, return exit code (no raise)."""
    cmd = [PY, str(HERE / script), *args]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def _bedrock_profile(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        goal = read_json(GOAL_PATH)
        return (goal.get("models", {}).get("user", {}) or {}).get("bedrock_profile")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Drive one optimization round hands-off.")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=3, help="personas to select this round")
    ap.add_argument(
        "--eval-concurrency",
        type=int,
        default=3,
        help="how many personas to evaluate at once",
    )
    ap.add_argument(
        "--total-sim-concurrency",
        type=int,
        default=20,
        help="total concurrent sims across all parallel evals",
    )
    ap.add_argument(
        "--max-tasks", type=int, default=0, help="cap tasks per domain (smoke)"
    )
    ap.add_argument("--max-infra-retries", type=int, default=2)
    ap.add_argument(
        "--sim-timeout", type=int, default=600, help="per-sim wall-clock timeout (s)"
    )
    ap.add_argument("--maj", type=int, default=3, help="judge majority votes")
    ap.add_argument("--max-failures", type=int, default=10)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--bedrock-profile", default=None)
    ap.add_argument(
        "--mutate",
        dest="mutate",
        action="store_true",
        default=True,
        help="auto-mutate after converge (default)",
    )
    ap.add_argument(
        "--no-mutate",
        dest="mutate",
        action="store_false",
        help="skip mutation (stop after converge)",
    )
    ap.add_argument(
        "--mutate-model",
        default="bedrock/us.anthropic.claude-opus-4-7",
        help="LLM model for mutation",
    )
    ap.add_argument("--mutate-effort", default="xhigh")
    ap.add_argument("--mutate-n-children", type=int, default=3)
    ap.add_argument("--mutate-max-retries", type=int, default=3)
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore existing eval/judge outputs and recompute",
    )
    ap.add_argument("--dry-run", action="store_true", help="fabricate eval (no API)")
    args = ap.parse_args()

    rd = round_dir(args.round)
    profile = _bedrock_profile(args.bedrock_profile)

    # 0) preflight: verify Bedrock SSO token is valid --------------------------
    if not args.dry_run:
        _preflight_bedrock(profile)

    # 1) select + 2) materialize ------------------------------------------------
    _run(
        "select.py",
        "--round",
        str(args.round),
        "--n",
        str(args.n),
        "--n-explore",
        str(max(1, args.n - 1)),
    )
    _run("materialize.py", "--round", str(args.round))

    selected = read_json(rd / "selected.json")["selected"]
    pids = [s["persona_id"] for s in selected]
    print(f"\n=== round {args.round}: {len(pids)} persona(s): {pids} ===")

    # 3) eval (concurrent across personas) -------------------------------------
    per_eval_conc = max(1, args.total_sim_concurrency // max(1, args.eval_concurrency))

    def _eval(pid: str) -> tuple[str, str]:
        out = rd / "eval" / pid / "persona_eval.json"
        if out.exists() and not args.force:
            return pid, "eval:cached"
        a = [
            "--persona-id",
            pid,
            "--round",
            str(args.round),
            "--max-infra-retries",
            str(args.max_infra_retries),
            "--max-concurrency",
            str(per_eval_conc),
        ]
        if args.max_tasks:
            a += ["--max-tasks", str(args.max_tasks)]
        if args.dry_run:
            a += ["--dry-run"]
        if profile:
            a += ["--bedrock-profile", profile]
        if args.sim_timeout:
            a += ["--sim-timeout", str(args.sim_timeout)]
        rc = _capture("eval_persona.py", *a, log=rd / "logs" / f"eval_{pid}.log")
        return pid, ("eval:ok" if rc == 0 else f"eval:FAIL({rc})")

    print(
        f"\n--- evaluating {len(pids)} persona(s) "
        f"(concurrency={args.eval_concurrency}) ---"
    )
    with ThreadPoolExecutor(max_workers=args.eval_concurrency) as ex:
        for fut in as_completed([ex.submit(_eval, p) for p in pids]):
            pid, status = fut.result()
            print(f"  [{pid}] {status}", flush=True)

    # 4) judge (concurrent across personas) ------------------------------------
    def _judge(pid: str) -> tuple[str, str]:
        out = rd / "judge" / pid / "judge_status.json"
        if out.exists() and not args.force:
            return pid, "judge:cached"
        a = [
            "--persona-id",
            pid,
            "--round",
            str(args.round),
            "--maj",
            str(args.maj),
            "--max-failures",
            str(args.max_failures),
            "--mode",
            "sync",
            "--concurrency",
            str(args.judge_concurrency),
        ]
        if args.dry_run:
            a += ["--dry-run"]
        rc = _capture("judge_persona.py", *a, log=rd / "logs" / f"judge_{pid}.log")
        return pid, ("judge:ok" if rc == 0 else f"judge:FAIL({rc})")

    print(f"\n--- judging {len(pids)} persona(s) ---")
    with ThreadPoolExecutor(max_workers=len(pids) or 1) as ex:
        for fut in as_completed([ex.submit(_judge, p) for p in pids]):
            pid, status = fut.result()
            print(f"  [{pid}] {status}", flush=True)

    # 5) record + 6) converge ---------------------------------------------------
    _run("record_results.py", "--round", str(args.round))
    _run("converge.py", "--round", str(args.round))

    # Summary -------------------------------------------------------------------
    rec = read_json(rd / "round_record.json")
    state = read_json(round_dir(args.round).parent.parent / "convergence_state.json")
    hist = state.get("history") or [{}]

    print(f"\n{'=' * 60}\nROUND {args.round} EVAL+JUDGE COMPLETE\n{'=' * 60}")
    for r in sorted(rec["personas"], key=lambda x: x.get("avg_success_rate") or 1.0):
        print(
            f"  {r['persona_id']}: success={r.get('persona_score')}  "
            f"status={r.get('status')}  severe={r.get('severe_failure_rate')}  "
            f"infra_dropped={r.get('infra_error_count')}"
        )
    latest = hist[-1] if hist else {}
    print(
        f"\nbest={latest.get('best_id')} avg={latest.get('best_avg')}  "
        f"untested={latest.get('n_untested')}  "
        f"should_stop={state.get('should_stop')}"
    )
    if latest.get("stop_reason"):
        print(f"  stop_reason: {latest['stop_reason']}")

    # 7) auto-mutate (if not converged and not --no-mutate) --------------------
    if state.get("should_stop"):
        print("\nConverged — skipping mutation.")
    elif not args.mutate:
        print("\n--no-mutate: skipping mutation.")
    else:
        print(f"\n--- auto-mutating {args.mutate_n_children} children ---")
        mutate_args = [
            "--round",
            str(args.round),
            "--n-children",
            str(args.mutate_n_children),
            "--model",
            args.mutate_model,
            "--effort",
            args.mutate_effort,
            "--max-retries",
            str(args.mutate_max_retries),
        ]
        if args.dry_run:
            mutate_args += ["--dry-run"]
        rc = _capture(
            "auto_mutate.py",
            *mutate_args,
            log=rd / "logs" / "auto_mutate.log",
        )
        if rc == 0:
            log_path = rd / "logs" / "auto_mutate.log"
            with open(log_path) as f:
                print(f.read())
            print("Mutation complete — children added to library.")
        else:
            print(f"WARNING: auto_mutate exited with code {rc}")
            log_path = rd / "logs" / "auto_mutate.log"
            if log_path.exists():
                with open(log_path) as f:
                    print(f.read())

    print(f"\n{'=' * 60}\nROUND {args.round} FULLY COMPLETE\n{'=' * 60}")


if __name__ == "__main__":
    main()
