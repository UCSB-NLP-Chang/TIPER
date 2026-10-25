#!/usr/bin/env python3
"""Validity judge for pgepa dsv4 tau2 cert (ds4flash_test results.json format).

Reads {domain}_{nop,opt}/results.json (simulations[] with inline messages +
reward_info.reward, tasks[] with user_scenario), judges every FAILED sim
(reward<1) with the severe-categories JUDGE_SYS. opt uses the genesis
exploration_best_persona.md; nop uses empty persona. severe_rate<=5% => VALID.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3]
        / "vendor"
        / "persona_policies"
        / "persona_policies"
        / "scripts"
    ),
)
from offline_validity_judge import JUDGE_SYS, _transcript  # reuse

ROOT = os.environ.get("TAU2BENCH_DATA_ROOT", str(Path(__file__).resolve().parents[3]))
PGEPA = f"{ROOT}/data/persona_opt/pgepa_runs/genesis_ds4flash_4domain"
TEST = f"{ROOT}/data/simulations/ds4flash_test"
BEST = {  # domain -> best persona md
    "telecom": f"{PGEPA}/telecom/isl_telecom_g0/exploration_best_persona.md",
    "retail": f"{PGEPA}/retail/isl_retail_g0/exploration_best_persona.md",
    "airline": f"{PGEPA}/airline/isl_airline_g0/exploration_best_persona.md",
}


def rew(e):
    v = (e.get("reward_info") or {}).get("reward")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def judge_one(messages, persona_text, scenario_text, task_id, trial, model):
    user_prompt = (
        f"## CUSTOMER SCENARIO INFORMATION (the reason for contact + known facts the customer was given — the ONLY source of truth for fabrication checks)\n"
        f"{(scenario_text or '')[:6000] or '(scenario not available)'}\n\n"
        f"## PERSONA the customer was following\n{persona_text[:4000] or '(no persona — neutral customer)'}\n\n"
        f"## CONVERSATION TRANSCRIPT (task {str(task_id)[:60]}, trial {trial})\n{_transcript(messages)}\n\n"
        f"Agent reward (rubric score): {rew({'reward_info': {'reward': 0}}):.2f}. "
        f"Audit the CUSTOMER's behavior per the rules. Return JSON."
    )
    # need actual reward
    return user_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-llm", default="gpt-4.1")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--severe-threshold", type=float, default=0.05)
    ap.add_argument("--env-file", default=f"{ROOT}/.env")
    args = ap.parse_args()
    if args.env_file:
        for line in open(args.env_file):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)
    import litellm

    litellm.drop_params = True

    doms = ["airline", "retail", "telecom"]
    results = {}
    for dom in doms:
        best_ptext = Path(BEST[dom]).read_text() if os.path.isfile(BEST[dom]) else ""
        for cond, ptext in (("nop", ""), ("opt", best_ptext)):
            rp = f"{TEST}/{dom}_{cond}/results.json"
            if not os.path.isfile(rp):
                print(f"[{dom}/{cond}] MISSING {rp}")
                continue
            d = json.load(open(rp))
            tmap = {
                t["id"]: str(t.get("user_scenario", "")) for t in d.get("tasks", [])
            }
            sims = d.get("simulations", [])
            failed = [e for e in sims if rew(e) < 1.0]
            print(
                f"[{dom}/{cond}] {len(sims)} sims, {len(failed)} failed to judge",
                flush=True,
            )
            if not failed:
                results[(dom, cond)] = {
                    "n_failed": 0,
                    "n_severe": 0,
                    "severe_rate": 0.0,
                    "cats": {},
                    "valid": True,
                }
                print(f"  -> severe_rate=0/0=0.0% VALID (no failures)")
                continue
            cats = {}
            n_sev = 0
            lock = Lock()
            judged = []

            def work(e):
                nonlocal n_sev
                msgs = e.get("messages", [])
                scen = tmap.get(e["task_id"], "")
                r = rew(e)
                up = (
                    f"## CUSTOMER SCENARIO INFORMATION\n{scen[:6000] or '(n/a)'}\n\n"
                    f"## PERSONA the customer was following\n{ptext[:4000] or '(no persona)'}\n\n"
                    f"## CONVERSATION TRANSCRIPT (task {str(e['task_id'])[:55]}, trial {e.get('trial')})\n{_transcript(msgs)}\n\n"
                    f"Agent reward (rubric score): {r:.2f}. Audit the CUSTOMER's behavior. Return JSON."
                )
                v = {
                    "verdict": "genuine_failure",
                    "category": "none",
                    "reason": "judge error",
                }
                for _ in range(3):
                    try:
                        resp = litellm.completion(
                            model=args.judge_llm,
                            messages=[
                                {"role": "user", "content": JUDGE_SYS + "\n\n" + up}
                            ],
                            max_tokens=300,
                            temperature=0,
                        )
                        raw = resp.choices[0].message.content.strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[-1]
                            if raw.endswith("```"):
                                raw = raw[:-3]
                        m = re.search(r"\{.*\}", raw, re.DOTALL)
                        if m:
                            v = json.loads(m.group(0))
                            v.setdefault("category", "none")
                            v.setdefault("reason", "")
                            break
                    except Exception:
                        time.sleep(1.0)
                with lock:
                    if v.get("verdict") == "severe_failure":
                        n_sev += 1
                        c = v.get("category", "?")
                        cats[c] = cats.get(c, 0) + 1
                return v

            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {pool.submit(work, e): e for e in failed}
                for f in as_completed(futs):
                    judged.append(f.result())
            sr = n_sev / len(failed)
            valid = sr <= args.severe_threshold
            results[(dom, cond)] = {
                "n_failed": len(failed),
                "n_severe": n_sev,
                "severe_rate": sr,
                "cats": cats,
                "valid": valid,
            }
            print(
                f"  -> severe_rate={n_sev}/{len(failed)}={sr:.1%} cats={cats} {'VALID' if valid else 'SUSPECT'}",
                flush=True,
            )
    print("\n=== pgepa dsv4 tau2 severe rate summary ===")
    print(f"{'domain':8} {'cond':4} {'severe_rate':>11} {'cats':20} valid")
    for dom in doms:
        for cond in ("nop", "opt"):
            r = results.get((dom, cond))
            if not r:
                continue
            cats = ",".join(f"{k}:{v}" for k, v in r["cats"].items()) or "-"
            print(
                f"{dom:8} {cond:4} {r['severe_rate']:>10.1%} {cats:20} {'✅' if r['valid'] else '❌'}"
            )


if __name__ == "__main__":
    main()
