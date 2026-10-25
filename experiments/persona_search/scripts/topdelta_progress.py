#!/usr/bin/env python3
"""Live progress bars for run_topdelta_test.py.

Counts completed simulations (reward != None) in each condition's results.json
and renders per-condition + overall progress bars with an ETA derived from the
running test process's elapsed time.

Usage:
  python experiments/persona_search/scripts/topdelta_progress.py            # one-shot snapshot
  python experiments/persona_search/scripts/topdelta_progress.py --watch    # refresh every 20s
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
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
SIMS = os.path.join(REPO, str(DATA_ROOT / "data/simulations/gepa_opt"))

# domain -> (test split file, split key, trials, n_conditions)
PLAN = {
    "airline": {
        "split": "experiments/persona_search/configs/airline_alaska_split_v4.json",
        "base": "airline_topdelta_test",
        "top_k": 2,
    },
    "retail": {
        "split": "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
        "base": "retail_topdelta_test",
        "top_k": 3,
    },
}
TRIALS = 3

# --- telecom→cross-domain transfer test (run_telecom_transfer_test.py) --------
# Fixed persona list evaluated on each target domain; airline/retail reuse their
# top-delta no_persona baselines, banking runs its own fresh baseline.
TRANSFER_BASE = "telecom_transfer_test"
TRANSFER_PERSONAS = [
    "seed_b00_terse_blunt",
    "seed_b02_frontload",
    "best_b02_script_weary",
    "best_b04_meaning_xlator",
]
TRANSFER_DOMAINS = {
    "airline": {
        "split": "experiments/persona_search/configs/airline_alaska_split_v4.json",
        "key": "airline",
        "reuse_baseline": "gepa_opt/airline_topdelta_test/no_persona/airline",
    },
    "retail": {
        "split": "experiments/persona_search/configs/retail_guitar_center_split_v2.json",
        "key": "retail",
        "reuse_baseline": "gepa_opt/retail_topdelta_test/no_persona/retail",
    },
    "banking": {
        "split": "experiments/persona_search/configs/banking_knowledge_split_v2.json",
        "key": "banking",
        "reuse_baseline": None,  # runs its own no_persona
    },
}


def test_size(split_file: str, domain: str) -> int:
    d = json.load(open(os.path.join(REPO, split_file)))
    t = d["test"]
    return len(t[domain]) if isinstance(t, dict) else len(t)


PASS = 1.0 - 1e-9  # reward >= PASS counts as a task success


def done_stats(
    results_path: str, trials: int = TRIALS
) -> tuple[int, float | None, dict, dict]:
    """Return (n_completed, avg_reward, per_trial, agg) for a condition.

    per_trial maps trial_idx -> (n, avg_reward).
    agg holds per-task cross-trial stats over *fully-completed* tasks (those
    with all `trials` trials present):
      {"n_tasks", "pass_k", "fail_k", "pass_frac", "fail_frac"}
    pass^k = all `trials` trials succeeded; fail^k = none did. pass_frac /
    fail_frac are None until at least one task is fully complete.
    """
    try:
        d = json.load(open(results_path))
    except Exception:
        return 0, None, {}, {}
    by_trial: dict[int, list] = {}
    by_task: dict[str, list] = {}
    rewards = []
    for s in d.get("simulations") or []:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        rewards.append(r)
        by_trial.setdefault(s.get("trial", -1), []).append(r)
        by_task.setdefault(s.get("task_id", "?"), []).append(r)
    if not rewards:
        return 0, None, {}, {}
    per_trial = {t: (len(rs), sum(rs) / len(rs)) for t, rs in by_trial.items()}

    n_full = pass_k = fail_k = 0
    for rs in by_task.values():
        if len(rs) < trials:
            continue
        n_full += 1
        if all(x >= PASS for x in rs):
            pass_k += 1
        elif all(x < PASS for x in rs):
            fail_k += 1
    agg = {
        "n_tasks": n_full,
        "pass_k": pass_k,
        "fail_k": fail_k,
        "pass_frac": (pass_k / n_full) if n_full else None,
        "fail_frac": (fail_k / n_full) if n_full else None,
    }
    return len(rewards), sum(rewards) / len(rewards), per_trial, agg


def bar(done: int, total: int, width: int = 34) -> str:
    frac = (done / total) if total else 0.0
    filled = int(frac * width)
    return (
        "█" * filled
        + "░" * (width - filled)
        + f" {done:>4}/{total:<4} {frac * 100:5.1f}%"
    )


def _log_elapsed_sec(path: str) -> int | None:
    """Seconds since a log file was created (birth time on macOS, else mtime-ish)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    start = getattr(st, "st_birthtime", None) or st.st_ctime
    return max(0, int(time.time() - start))


def _pf_str(agg: dict) -> str:
    """'  p^3=NN.N% f^3=NN.N%(n)' over fully-completed tasks, else ''."""
    if not agg or agg.get("pass_frac") is None:
        return ""
    return (
        f"  p^{TRIALS}={agg['pass_frac'] * 100:4.1f}%"
        f"  f^{TRIALS}={agg['fail_frac'] * 100:4.1f}%({agg['n_tasks']})"
    )


def run_elapsed_sec(pattern: str = "run_topdelta_test") -> int | None:
    """Elapsed wall-clock (sec) of the newest process matching `pattern`."""
    try:
        pid = subprocess.check_output(["pgrep", "-f", pattern], text=True).split()[0]
    except Exception:
        return None
    try:
        et = subprocess.check_output(
            ["ps", "-o", "etime=", "-p", pid], text=True
        ).strip()
    except Exception:
        return None
    # etime formats: MM:SS | HH:MM:SS | D-HH:MM:SS
    d, _, hms = et.partition("-")
    if hms == "":
        hms, d = d, "0"
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return int(d) * 86400 + h * 3600 + m * 60 + s


def snapshot() -> None:
    grand_done = grand_total = 0
    running = run_elapsed_sec()
    lines = []
    for domain, cfg in PLAN.items():
        n = test_size(cfg["split"], domain) * TRIALS  # per-condition
        n_cond = 1 + 2 * cfg["top_k"]  # baseline + seed/best*k
        dom_total = n * n_cond
        base = os.path.join(SIMS, cfg["base"])
        # discover condition dirs (stable order: no_persona first, then sorted)
        conds = []
        if os.path.isdir(base):
            conds = sorted(os.listdir(base), key=lambda c: (c != "no_persona", c))
        dom_done = 0
        lines.append(f"\n  {domain.upper()}  ({n_cond} conditions × {n} sims)")
        seen = set()
        perf = {}  # cond -> (n_done, avg_reward, agg)
        for c in conds:
            rp = os.path.join(base, c, domain, "results.json")
            dc, avg, per_trial, agg = done_stats(rp)
            perf[c] = (dc, avg, agg)
            dom_done += dc
            seen.add(c)
            mark = "✔" if dc >= n else ("▶" if 0 < dc < n else " ")
            perf_str = (
                f"  ASR={avg * 100:5.1f}%  diff={1 - avg:.3f}"
                if avg is not None
                else ""
            )
            perf_str += _pf_str(agg)
            lines.append(f"    {mark} {c:<20} {bar(dc, n)}{perf_str}")
            if per_trial:
                tri = "  ".join(
                    f"t{t}={a * 100:.1f}%({nn})"
                    for t, (nn, a) in sorted(per_trial.items())
                )
                lines.append(f"        └ trials: {tri}")
        # not-yet-started conditions
        pending = n_cond - len(seen)
        if pending > 0:
            lines.append(f"      ({pending} condition(s) not started yet)")
        lines.append(f"    {'—' * 4} {domain} subtotal {bar(dom_done, dom_total)}")

        # seed vs best comparison (only where both have data)
        cmp_lines = []
        for c in conds:
            if not c.startswith("seed_"):
                continue
            sid = c[len("seed_") :]
            sd, sa, sagg = perf.get(f"seed_{sid}", (0, None, {}))
            bd, ba, bagg = perf.get(f"best_{sid}", (0, None, {}))
            if sa is not None and ba is not None:
                dd = (1 - ba) - (1 - sa)  # difficulty delta = ASR drop
                line = (
                    f"      {sid:<12} seed diff={1 - sa:.3f} → best diff={1 - ba:.3f}"
                    f"  Δ={dd:+.3f}  (n {sd}/{bd})"
                )
                sf, bf = sagg.get("fail_frac"), bagg.get("fail_frac")
                if sf is not None and bf is not None:
                    line += f"  | f^{TRIALS} {sf * 100:.1f}%→{bf * 100:.1f}% ({bf - sf:+.1%})"
                cmp_lines.append(line)
        if cmp_lines:
            lines.append(f"    seed→best (test difficulty):")
            lines.extend(cmp_lines)
        grand_done += dom_done
        grand_total += dom_total

    print("=" * 62)
    print("  τ² top-delta test — progress")
    print("=" * 62)
    for ln in lines:
        print(ln)
    print("\n" + "=" * 62)
    print(f"  OVERALL            {bar(grand_done, grand_total, 40)}")
    if running is not None and grand_done > 0:
        rate = grand_done / running  # sims/sec
        remain = (grand_total - grand_done) / rate if rate else 0
        print(
            f"  elapsed {running // 60}m{running % 60:02d}s | "
            f"{rate * 60:.1f} sim/min | ETA ~{int(remain // 60)}m "
            f"(~{remain / 3600:.1f}h)"
        )
    elif running is None:
        print("  (test process not running — final/stopped snapshot)")
    print("=" * 62)


def _baseline_path(cfg: dict, dom_key: str) -> str:
    """results.json path for a transfer domain's no_persona baseline."""
    rb = cfg.get("reuse_baseline")
    if rb:
        p = os.path.join(REPO, str(DATA_ROOT / "data/simulations"), rb, "results.json")
        if os.path.isfile(p):
            return p
    return os.path.join(SIMS, TRANSFER_BASE, "no_persona", dom_key, "results.json")


def snapshot_transfer() -> bool:
    """Render telecom→cross-domain transfer-test progress.

    Returns True if a section was printed (dir exists or process running),
    False otherwise so the caller can stay quiet before the run starts.
    """
    base = os.path.join(SIMS, TRANSFER_BASE)
    alive = run_elapsed_sec("run_telecom_transfer_test") is not None
    if not os.path.isdir(base) and not alive:
        return False
    # ps etime is unreliable here: the launcher exec'd into this pid so its
    # start time predates the actual run. Prefer the log file's birth time.
    running = _log_elapsed_sec("/tmp/telecom_transfer_test.log") if alive else None

    grand_done = grand_total = 0  # counts NEW work only (reused baselines excluded)
    lines = []
    for dom_key, cfg in TRANSFER_DOMAINS.items():
        n = test_size(cfg["split"], cfg["key"]) * TRIALS
        reused = bool(cfg.get("reuse_baseline")) and os.path.isfile(
            os.path.join(
                REPO,
                str(DATA_ROOT / "data/simulations"),
                cfg["reuse_baseline"],
                "results.json",
            )
        )
        # new conditions this run: personas (+ own baseline when not reused)
        n_new_cond = len(TRANSFER_PERSONAS) + (0 if reused else 1)
        dom_total = n * n_new_cond
        dom_done = 0

        lines.append(
            f"\n  {dom_key.upper()}  ({n_new_cond} new × {n} sims"
            f"{'  | baseline REUSED' if reused else ''})"
        )

        # baseline row (shown for context; not counted in new-work totals)
        bd, bavg, _, bagg = done_stats(_baseline_path(cfg, cfg["key"]))
        base_diff = (1 - bavg) if bavg is not None else None
        base_fail = bagg.get("fail_frac")
        btag = "no_persona" + (" (reuse)" if reused else "")
        bperf = (
            f"  ASR={bavg * 100:5.1f}%  diff={base_diff:.3f}"
            if bavg is not None
            else "  (pending)"
        )
        bperf += _pf_str(bagg)
        bmark = "·" if reused else ("✔" if bd >= n else ("▶" if bd else " "))
        lines.append(f"    {bmark} {btag:<24} {bar(bd, n)}{bperf}")
        if not reused:
            dom_done += bd

        for cond in TRANSFER_PERSONAS:
            rp = os.path.join(base, cond, cfg["key"], "results.json")
            dc, avg, per_trial, agg = done_stats(rp)
            dom_done += dc
            mark = "✔" if dc >= n else ("▶" if 0 < dc < n else " ")
            if avg is not None:
                dd = ((1 - avg) - base_diff) if base_diff is not None else None
                perf = f"  ASR={avg * 100:5.1f}%  diff={1 - avg:.3f}"
                perf += f"  Δ={dd:+.3f}" if dd is not None else ""
                perf += _pf_str(agg)
                fv = agg.get("fail_frac")
                if fv is not None and base_fail is not None:
                    perf += f" (Δf^{TRIALS}{fv - base_fail:+.1%})"
            else:
                perf = ""
            lines.append(f"    {mark} {cond:<24} {bar(dc, n)}{perf}")
            if per_trial:
                tri = "  ".join(
                    f"t{t}={a * 100:.1f}%({nn})"
                    for t, (nn, a) in sorted(per_trial.items())
                )
                lines.append(f"        └ trials: {tri}")

        lines.append(f"    {'—' * 4} {dom_key} subtotal {bar(dom_done, dom_total)}")
        grand_done += dom_done
        grand_total += dom_total

    print("=" * 62)
    print("  τ² telecom→cross-domain transfer test — progress")
    print("=" * 62)
    for ln in lines:
        print(ln)
    print("\n" + "=" * 62)
    print(f"  OVERALL (new work)  {bar(grand_done, grand_total, 40)}")
    if running is not None and grand_done > 0:
        rate = grand_done / running
        remain = (grand_total - grand_done) / rate if rate else 0
        print(
            f"  elapsed {running // 60}m{running % 60:02d}s | {rate * 60:.1f} sim/min"
            f" | ETA ~{int(remain // 60)}m (~{remain / 3600:.1f}h)"
        )
    elif running is None:
        print("  (transfer process not running — final/stopped snapshot)")
    print("=" * 62)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every 20s")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument(
        "--transfer",
        action="store_true",
        help="show only the transfer test (else: top-delta + "
        "transfer when it has started)",
    )
    args = ap.parse_args()

    def render():
        if args.transfer:
            if not snapshot_transfer():
                print("(transfer test has not started — no artifacts yet)")
            return
        snapshot()
        # auto-append transfer section once it has produced anything
        if snapshot_transfer():
            pass

    def any_running():
        pat = (
            "run_telecom_transfer_test"
            if args.transfer
            else "run_topdelta_test|run_telecom_transfer_test"
        )
        return run_elapsed_sec(pat) is not None

    if not args.watch:
        render()
        return 0
    try:
        while True:
            os.system("clear")
            render()
            if not any_running():
                print("\n(done — process gone)")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
