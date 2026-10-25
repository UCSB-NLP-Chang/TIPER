#!/usr/bin/env python3
"""Re-judge EXISTING V2 conversation traces with a swapped evaluator LLM.

Does NOT re-run any agent/user simulation. Loads the saved representative-trial
trajectories from run_<domain>_v2/conversations/test_{no_persona,best_persona}/
and re-scores them with TrajectoryEvaluator.calculate_reward using a new judge.

To make the comparison apples-to-apples (same conversations, only the judge
changes), each trace is scored with BOTH the V2 judge (gpt-4.1) and the new
judge (deepseek-v4-flash-0731). The V2 test_report.json aggregate used 3 trials;
these traces are 1 representative trial per task, so we re-judge gpt-4.1 on the
SAME traces too — the delta is then pure judge effect, not trial-count effect.

Usage:
    python scripts/vita_rejudge_deepseek.py
    python scripts/vita_rejudge_deepseek.py --domains delivery --max-concurrency 4   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vita.data_model.message import (  # noqa: E402
    AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage,
)
from vita.evaluator.evaluator_traj import TrajectoryEvaluator  # noqa: E402
from vita.run import get_tasks  # noqa: E402

RUN_BASE = Path("data/persona_opt/vita_pgepa_runs")
DIR_KEY = {"cross_domain": "crossdomain"}
WINDOW_SIZE = 10
OVERLAP = 2
LANGUAGE = "chinese"


def dict_to_message(m: dict):
    role = m.get("role")
    content = m.get("content")
    if role == "assistant":
        tcs = m.get("tool_calls") or []
        tool_calls = [ToolCall(name=tc.get("name", ""), arguments=tc.get("arguments", {}))
                      for tc in tcs]
        return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)
    if role == "tool":
        return ToolMessage(id=m.get("id", ""), name=m.get("name", ""),
                           role="tool", content=content)
    if role == "system":
        return SystemMessage(role="system", content=content)
    return UserMessage(role="user", content=content)


def rejudge_one(trace_path: Path, task, llm_eval: str, llm_args: dict,
                retries: int = 2):
    rec = json.loads(trace_path.read_text())
    task_id = rec["task_id"]
    if task is None:
        return task_id, None, None
    msgs = [dict_to_message(m) for m in rec["messages"]]
    last_err = None
    for attempt in range(retries + 1):
        try:
            ri = TrajectoryEvaluator.calculate_reward(
                task, msgs, {}, window_size=WINDOW_SIZE, overlap=OVERLAP,
                llm_evaluator=llm_eval, llm_args_evaluator=llm_args,
                language=LANGUAGE,
            )
            return task_id, ri.reward, ri.reward_breakdown
        except Exception as e:
            last_err = e
            # backoff before retry (dev-proxy rate limiting / transient errors)
            import time
            time.sleep(3 * (attempt + 1))
    print(f"  [rejudge FAIL] {task_id} judge={llm_eval}: {last_err}")
    return task_id, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="delivery,instore,ota,cross_domain")
    ap.add_argument("--judges", default="gpt-4.1,deepseek-v4-flash-0731",
                    help="comma list of evaluator models to re-judge with")
    ap.add_argument("--max-concurrency", type=int, default=3,
                    help="low to avoid dev-proxy rate limiting on the shim path")
    ap.add_argument("--suffix", default="dseval",
                    help="output suffix for report files")
    args = ap.parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    # llm_args per judge: deepseek needs {} (non-reasoning); gpt-4.1 {} too.
    judge_args = {j: {} for j in judges}

    summary = {}
    for d in domains:
        run_dir = RUN_BASE / f"run_{DIR_KEY.get(d, d)}_v2"
        task_map = {t.id: t for t in get_tasks(d, None, None, LANGUAGE)}
        print(f"\n{'=' * 60}\n  DOMAIN {d}  n_traces per cond = "
              f"{sum(1 for _ in (run_dir / 'conversations' / 'test_no_persona').glob('*.json'))}"
              f"\n{'=' * 60}")

        per_cond = {}
        for cond_dirname, cond in [("test_no_persona", "no_persona"),
                                   ("test_best_persona", "best_persona")]:
            conv_dir = run_dir / "conversations" / cond_dirname
            files = sorted(conv_dir.glob("*.json"))
            # score every trace with every judge
            results = {j: {} for j in judges}  # judge -> task_id -> reward
            jobs = [(f, task_map.get(f.stem), j) for f in files for j in judges]

            def _job(tup):
                f, task, j = tup
                tid, rw, _bd = rejudge_one(f, task, j, judge_args[j])
                return j, tid, rw

            with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
                for j, tid, rw in ex.map(_job, jobs):
                    results[j][tid] = rw

            per_cond[cond] = {}
            for j in judges:
                rws = [r for r in results[j].values() if r is not None]
                avg = sum(rws) / len(rws) if rws else 0.0
                fully = sum(1 for r in rws if r >= 1.0)
                per_cond[cond][j] = {"avg_reward": avg, "fully_correct": fully,
                                     "n": len(rws), "rewards": rws}
                print(f"  {cond:12s} judge={j:28s} n={len(rws):2d} "
                      f"avg={avg:.3f} fully={fully}/{len(rws)}")

        report = {"domain": d, "judges": judges, "window_size": WINDOW_SIZE,
                  "overlap": OVERLAP, "language": LANGUAGE, "conditions": per_cond}
        out = run_dir / f"rejudge_report_{args.suffix}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"  report -> {out}")
        summary[d] = {cond: {j: per_cond[cond][j]["avg_reward"] for j in judges}
                      for cond in per_cond}

    print(f"\n{'=' * 60}\n  SUMMARY (re-judge, same traces, judge swap)\n{'=' * 60}")
    print(f"  {'domain':14s}" + "".join(f" {j}/{c:>10s}" for c in ["no", "best"] for j in judges))
    for d, s in summary.items():
        cells = []
        for c in ["no_persona", "best_persona"]:
            for j in judges:
                cells.append(f"{s[c][j]:.3f}")
        print(f"  {d:14s} " + " ".join(f"{x:>12s}" for x in cells))
    (RUN_BASE / f"rejudge_summary_{args.suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n  summary -> {RUN_BASE / f'rejudge_summary_{args.suffix}.json'}")


if __name__ == "__main__":
    main()
