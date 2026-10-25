"""
Offline validity judge for vitabench delivery conversations.

Same concept as offline_judge.py for telecom, but with delivery-domain-specific
judge prompt. Reads results + trajectories JSON from run_vitabench_eval.py.

Usage:
    vitabench/.venv/bin/python persona_policies/scripts/offline_judge_vitabench.py \
        --results persona_policies/outputs/vitabench_evals/vitabench_delivery_pgepa_test_n10_t3.json \
        --trajectories ..._trajectories.json \
        --judge-llm gpt-4.1 --max-workers 20 --env-file ../.env
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

JUDGE_PROMPT_EN = """You are an offline validity judge for a food delivery customer service benchmark.

A customer contacted delivery support. The automated evaluation gave this interaction a reward of {reward} (1.0 = all issues resolved, 0.0 = nothing resolved).

Your job: read the conversation and judge whether the reward is FAIR.

Rules:
- If the agent made substantial progress but was interrupted or the conversation ended early → partial credit is appropriate
- If the agent transferred or deflected without real help → unfair if reward is 1.0
- If the persona's behavior actively hindered resolution (refusing to cooperate, giving wrong info, being unresponsive) → may be fair
- If the task had conflicting constraints that made full resolution impossible → may be fair

Return ONLY valid JSON (no markdown, no backticks):
{{"fair": true/false, "actual_reward": 0.0-1.0, "who_is_at_fault": "agent"|"persona"|"both"|"task_too_hard"|"neither", "reasoning": "one sentence"}}

=== CONVERSATION ===
{conversation}
=== END ==="""


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LLM validity judge for vitabench conversations.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--judge-llm", default="gpt-4.1")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--language", default="en", choices=["en", "chinese"])
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

    # Index by (task_id, persona_id, trial_idx)
    traj_map = {}
    for t in traj_data:
        key = (str(t["task_id"]), str(t["persona_id"]))
        traj_map[key] = t["messages"]

    episodes = [r for r in results["results"] if r["status"] == "ok"]
    print(f"[load] {len(episodes)} ok episodes, judge={args.judge_llm} lang={args.language}", flush=True)

    judged = []
    n_fair = n_unfair = 0

    from threading import Lock
    _lock = Lock()

    prompt_template = JUDGE_PROMPT_EN

    def _judge_one(ep: dict) -> dict:
        nonlocal n_fair, n_unfair
        key = (str(ep["task_id"]), str(ep["persona_id"]))
        msgs = traj_map.get(key, [])
        conv = ""
        for m in msgs:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:600]
            conv += f"[{role}] {content}\n\n"
        conv = conv[:8000]

        reward = ep.get("reward", 0.0)
        if reward is None:
            reward = 0.0
        prompt = prompt_template.format(reward=reward, conversation=conv)
        try:
            resp = litellm.completion(
                model=args.judge_llm,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
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

        with _lock:
            tid_short = str(ep["task_id"])[:50]
            print(f"[{tid_short}] {ep['persona_id']}: "
                  f"reward={ep.get('reward')} judge_fair={is_fair} "
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
    print(f"\n[done] {n_total} judged in {dt:.0f}s (fair={n_fair} unfair={n_unfair})", flush=True)

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".judged.json")
    payload = dict(results)
    payload["judge_llm"] = args.judge_llm
    payload["results"] = judged
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path}", flush=True)

    # Summary
    db_rewards = [j.get("reward") for j in judged if j.get("reward") is not None]
    judge_rewards = [j["judge_reward"] for j in judged if j.get("judge_reward") is not None]
    unfair_eps = [j for j in judged if j.get("judge_fair") == False]
    print(f"\n[summary]")
    if db_rewards:
        print(f"  Orig mean_reward: {sum(db_rewards)/len(db_rewards):.3f}")
    if judge_rewards:
        print(f"  Judge mean_reward: {sum(judge_rewards)/len(judge_rewards):.3f}")
    print(f"  Fair: {n_fair}/{n_total} ({n_fair/n_total:.1%})")
    print(f"  Unfair: {n_unfair}/{n_total} ({n_unfair/n_total:.1%})")
    if unfair_eps:
        print(f"  Top unfair ({min(10, len(unfair_eps))}):")
        for j in sorted(unfair_eps, key=lambda x: abs((x.get("judge_reward") or 0) - (x.get("reward") or 0)), reverse=True)[:10]:
            print(f"    {j['persona_id']:<30} reward={j.get('reward')} judge={j['judge_reward']} "
                  f"fault={j['judge_fault']}  | {(j.get('judge_reasoning','') or '')[:100]}")


if __name__ == "__main__":
    main()
