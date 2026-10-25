"""Probe whether a CONDITIONAL hybrid persona can be hard on BOTH telecom and
airline simultaneously, where the two seed specialists are mutually exclusive:
  idx1 Front-loading&Terse: telecom 0.61 / airline 0.25
  idx2 Verbose&Polite     : airline 0.62 / telecom 0.11

Naive blend would average down (cf. idx3 Confused&Drip-feeding 0.28/0.50, a
"master of none"). The hybrid below does NOT pick a point on the terse<->verbose
axis. Instead it stacks two NON-conflicting difficulty mechanisms:
  * drip-feed structured info / withhold ids   -> hurts telecom (multi-step debug)
  * change-mind / hesitate / bury constraints  -> hurts airline (constraint-heavy)

Eval: telecom+airline val subset only, gpt-5.2 user / gpt-5.4 high agent,
2 trials, LOW concurrency (v7 GEPA still running). Compare vs idx1/idx2/idx3.
"""

import json
import os
import subprocess
from pathlib import Path

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
VENV = REPO / ".venv" / "bin"
VAL = json.load(open(REPO / "data/persona_opt/val25_taskids.json"))
DOMAINS = ["telecom", "airline"]
NUM_TRIALS = 2
CONCURRENCY = 3  # low: avoid contending with the live v7 GEPA run

HYBRID = """# Persona: Conditional Hybrid (drip-feed + mind-changing)

You are a real customer contacting support. You combine two habits that make you
hard to help, and you keep BOTH at all times.

## How you behave
- You reveal information gradually, a little at a time; never dump everything at once.
- You give very few identifiers up front; the agent has to ask for account ids / numbers one by one.
- When the agent asks something, you answer partially, leaving out a detail you only mention later.
- You change your mind or switch topic mid-conversation ("actually...", "wait, never mind", "oh and also").
- You sound unsure/hesitant about your own situation ("not sure", "i think", "maybe it was").
- You occasionally restate a constraint differently than before, so the key details are easy to miss.

## Tone
- Brief, slightly scattered; write the way a real customer texts support — lowercase, fragments, and typos are fine.

## Invariants (never break these)
- Stay in character as a real customer; never mention "the scenario" or that you are an AI.
- Only state facts grounded in your scenario or in the actual results of tools you ran. Never invent tool results.
"""

env = dict(os.environ)
env.setdefault("OPENAI_API_KEY", "REDACTED_USE_OPENAI_API_KEY_ENV")

persona_json = json.dumps({"custom_guidelines": HYBRID})
for domain in DOMAINS:
    tids = VAL[domain]
    save_to = f"gepa_opt/hybrid_probe/cond_hybrid/{domain}"
    cmd = [
        str(VENV / "tau2"),
        "run",
        "--domain",
        domain,
        "--user",
        "user_simulator",
        "--user-llm",
        "gpt-5.2",
        "--agent-llm",
        "gpt-5.4",
        "--agent-llm-args",
        json.dumps({"reasoning_effort": "high"}),
        "--num-trials",
        str(NUM_TRIALS),
        "--max-concurrency",
        str(CONCURRENCY),
        "--max-steps",
        "120",
        "--timeout",
        "600",
        "--seed",
        "300",
        "--save-to",
        save_to,
        "--auto-resume",
        "--user-persona",
        persona_json,
        "--task-ids",
        *tids,
    ]
    print(
        f"\n==== hybrid [{domain}] {len(tids)} tasks x{NUM_TRIALS} -> {save_to} ====",
        flush=True,
    )
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"  WARNING: {domain} exited {r.returncode}", flush=True)

print("\nhybrid probe done.", flush=True)
