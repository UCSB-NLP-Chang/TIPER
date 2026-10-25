"""Does a REASONING user simulator follow long/abstract personas better — and
does that help or HURT induced difficulty?

Two arms, identical except the user simulator's reasoning effort:
  * base   : gpt-5.2 user, no reasoning_effort (current GEPA setting)
  * reason : gpt-5.2 user, reasoning_effort=high

Personas: 2 SHORT seeds (concrete style: i1 terse, i2 verbose) where following is
already saturated, + 2 LONG mutations (abstract semantic rules: i6 scoped-assent,
i12 front-end-labels) where following is suspect. telecom+airline val, 2 trials.

For each (arm, persona, domain) we later compare:
  * difficulty (1 - reward): does reasoning make the persona harder or easier?
  * (separately) LLM-judge adherence on the saved rollouts.

LOW concurrency: the v7 test eval is still running; avoid oversubscription.
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
STATE = json.load(
    open(REPO / "data/persona_opt/pgepa_runs/xdom_v7_3domain/pgepa_state.json")
)
BY = {c["idx"]: c for c in STATE["candidates"]}

PERSONAS = [
    (1, "i1_terse_short"),
    (2, "i2_verbose_short"),
    (6, "i6_scoped_long"),
    (12, "i12_frontend_long"),
]
# small subset: 3 telecom + 2 airline tasks x 2 trials = ~10 sims per (persona, arm)
DOMAIN_TASKS = {
    "telecom": VAL["telecom"][:3],
    "airline": VAL["airline"][:2],
}
DOMAINS = list(DOMAIN_TASKS)
ARMS = [("base", {}), ("reason", {"reasoning_effort": "high"})]
NUM_TRIALS = 2
CONCURRENCY = 3

env = dict(os.environ)
env.setdefault("OPENAI_API_KEY", "REDACTED_USE_OPENAI_API_KEY_ENV")

for idx, plabel in PERSONAS:
    persona_json = json.dumps({"custom_guidelines": BY[idx]["persona_text"]})
    for arm, uargs in ARMS:
        for domain in DOMAINS:
            tids = DOMAIN_TASKS[domain]
            save_to = f"gepa_opt/reasoning_user_probe/{arm}/{plabel}/{domain}"
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
            if uargs:
                cmd += ["--user-llm-args", json.dumps(uargs)]
            print(
                f"\n==== [{arm}] {plabel} [{domain}] {len(tids)}x{NUM_TRIALS} "
                f"user_args={uargs} -> {save_to} ====",
                flush=True,
            )
            r = subprocess.run(cmd, env=env)
            if r.returncode != 0:
                print(
                    f"  WARNING {arm}/{plabel}/{domain} exited {r.returncode}",
                    flush=True,
                )

print("\nreasoning_user_probe done.", flush=True)
