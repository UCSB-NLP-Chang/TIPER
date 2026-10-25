"""
Offline validity judge: re-evaluate τ²-bench telecom conversations with an LLM.

Reads a results JSON and trajectories JSON produced by ``run_tau2_eval.py``,
sends every conversation to a judge LLM, and writes a ``_judged.json`` file
with the original results annotated with ``judge_fair``, ``judge_reward``,
``judge_reasoning``, and ``judge_fault``.

Usage:
    .venv/bin/python -u persona_policies/scripts/offline_judge.py \
        --results persona_policies/outputs/tau2_evals/tau2_telecom_pgepa_test_n10_t1.json \
        --trajectories persona_policies/outputs/tau2_evals/tau2_telecom_pgepa_test_n10_t1.json_trajectories.json \
        --judge-llm gpt-4.1 --max-workers 30 \
        --env-file ../.env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

JUDGE_PROMPT = """You are an offline validity judge for a telecom customer service benchmark.

A customer (with a specific behavioral persona) contacted support about an MMS/network issue. The automated DB-check evaluation gave this interaction a reward of {reward} (1.0 = all issues resolved, 0.0 = nothing resolved).

Your job: read the conversation and judge whether the reward is FAIR.

Rules:
- If the agent transferred to a human before any real troubleshooting → unfair if reward is 0.0 (agent gave up)
- If the agent made substantial progress but the conversation ended before all issues were fixed → partial credit is appropriate
- If the persona's behavior actively hindered resolution → may be fair
- If the task had too many simultaneous issues for the agent to handle → may be fair

Return ONLY valid JSON (no markdown, no backticks):
{{"fair": true/false, "actual_reward": 0.0-1.0, "who_is_at_fault": "agent"|"persona"|"both"|"task_too_hard", "reasoning": "one sentence"}}

=== CONVERSATION ===
{conversation}
=== END ==="""


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LLM validity judge for τ²-bench conversations.")
    ap.add_argument("--results", required=True, help="results JSON from run_tau2_eval.py")
    ap.add_argument("--trajectories", required=True, help="trajectories JSON from run_tau2_eval.py")
    ap.add_argument("--judge-llm", default="gpt-4.1", help="LLM for judging (default gpt-4.1)")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    if args.env_file:
        for line in open(args.env_file, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    import litellm

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    traj_data = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))

    # Index trajectories by (task_id, persona_id)
    traj_map = {}
    for t in traj_data:
        key = (t["task_id"], t["persona_id"])
        traj_map[key] = t["messages"]

    episodes = [r for r in results["results"] if r["status"] == "ok"]
    print(f"[load] {len(episodes)} ok episodes, judge={args.judge_llm}", flush=True)

    judged = []
    n_fair = n_unfair = 0

    from threading import Lock
    _lock = Lock()

    def _judge_one(ep: dict) -> dict:
        nonlocal n_fair, n_unfair
        key = (ep["task_id"], ep["persona_id"])
        msgs = traj_map.get(key, [])
        # Format conversation
        conv = ""
        for m in msgs:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:600]
            conv += f"[{role}] {content}\n\n"
        conv = conv[:8000]

        prompt = JUDGE_PROMPT.format(reward=ep["reward"], conversation=conv)
        try:
            resp = litellm.completion(
                model=args.judge_llm,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            judge = json.loads(raw)
        except Exception as e:
            judge = {"fair": None, "actual_reward": None, "who_is_at_fault": "unknown", "reasoning": f"judge error: {e}"}

        is_fair = judge.get("fair")
        if is_fair:
            n_fair += 1
        else:
            n_unfair += 1

        t0 = time.time()
        with _lock:
            print(f"[{ep['task_id'][:50]}...] {ep['persona_id']}: "
                  f"db_reward={ep['reward']} judge_fair={is_fair} "
                  f"judge_reward={judge.get('actual_reward')} "
                  f"fault={judge.get('who_is_at_fault')}",
                  flush=True)

        out = dict(ep)
        out["judge_fair"] = judge.get("fair")
        out["judge_reward"] = judge.get("actual_reward")
        out["judge_fault"] = judge.get("who_is_at_fault")
        out["judge_reasoning"] = judge.get("reasoning", "")
        return out

    t0 = time.time()
    if args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            fut_map = {pool.submit(_judge_one, ep): ep for ep in episodes}
            for fut in as_completed(fut_map):
                judged.append(fut.result())
    else:
        for ep in episodes:
            judged.append(_judge_one(ep))

    dt = time.time() - t0
    n_total = len(judged)
    print(f"\n[done] {n_total} judged in {dt:.0f}s "
          f"(fair={n_fair} unfair={n_unfair})", flush=True)

    # Write
    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".judged.json")
    payload = dict(results)
    payload["judge_llm"] = args.judge_llm
    payload["results"] = judged
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path}", flush=True)

    # Summary
    db_rewards = [j["reward"] for j in judged]
    judge_rewards = [j["judge_reward"] for j in judged if j["judge_reward"] is not None]
    unfair_eps = [j for j in judged if j["judge_fair"] == False]
    print(f"\n[summary]")
    print(f"  DB mean_reward: {sum(db_rewards)/len(db_rewards):.3f}")
    if judge_rewards:
        print(f"  Judge mean_reward: {sum(judge_rewards)/len(judge_rewards):.3f}")
    print(f"  Fair: {n_fair}/{n_total} ({n_fair/n_total:.1%})")
    print(f"  Unfair: {n_unfair}/{n_total} ({n_unfair/n_total:.1%})")
    if unfair_eps:
        print(f"  Unfair episodes ({len(unfair_eps)}):")
        for j in sorted(unfair_eps, key=lambda x: x["judge_reward"] or 0, reverse=True)[:10]:
            print(f"    {j['persona_id']:<30} db={j['reward']} judge={j['judge_reward']} "
                  f"fault={j['judge_fault']}  | {j['judge_reasoning'][:100]}")


if __name__ == "__main__":
    main()
