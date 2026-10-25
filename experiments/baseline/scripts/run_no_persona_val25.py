"""Run the no_persona baseline on the EXACT val-25 task set (telecom/airline/
retail), matching the GEPA seed-val config (agent gpt-5.4 high, user
user_simulator gpt-5.2, no persona, max_steps 120, seed 300). Produces the
empirical noise floor for persona_dispersion.py (Tier 2).
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
NUM_TRIALS = 4  # 25 tasks x 4 = 100 rollouts for a stable bootstrap floor

env = dict(os.environ)
env.setdefault("OPENAI_API_KEY", "REDACTED_USE_OPENAI_API_KEY_ENV")

for domain, tids in VAL.items():
    save_to = f"gepa_opt/no_persona_val25/{domain}"
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
        "8",
        "--max-steps",
        "120",
        "--timeout",
        "600",
        "--seed",
        "300",
        "--save-to",
        save_to,
        "--auto-resume",
        "--task-ids",
        *tids,
    ]
    print(
        f"\n==== no_persona [{domain}] {len(tids)} tasks x{NUM_TRIALS} -> {save_to} ====",
        flush=True,
    )
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"  WARNING: {domain} exited {r.returncode}", flush=True)

print("\nno_persona val25 done.", flush=True)
