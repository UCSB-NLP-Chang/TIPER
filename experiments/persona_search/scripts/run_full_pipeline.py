#!/usr/bin/env python3
"""
Full pipeline orchestration:
  Phase 1: Resume airline v3 GEPA → 20 iterations
  Phase 2: Airline test eval (seed + optimized + no_persona)
  Phase 3: Retail v1 GEPA → 20 iterations
  Phase 4: Retail test eval (seed + optimized + no_persona)

Global concurrency cap: 25.

Usage:
  nohup .venv/bin/python experiments/persona_search/scripts/run_full_pipeline.py > /tmp/full_pipeline.log 2>&1 &
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

PY = ".venv/bin/python"
TAU2 = ".venv/bin/tau2"
GLOBAL_MAX_CONC = 25

AGENT_LLM = "gpt-5.4"
AGENT_LLM_ARGS = '{"reasoning_effort":"high"}'
USER_TYPE = "user_simulator"
USER_LLM = "gpt-5.4"
EFFORT = "high"
MUTATION_MODEL = "openai/gpt-5.4"
MUTATION_EFFORT = "xhigh"
JUDGE_MODEL = "gpt-5.2"

MB = 5
NUM_TRIALS = 2
VAL_NUM_TRIALS = 2
MIN_DISSIM = 0.30
MAX_SEM_SIM = 0.90
MAX_MUT_ATTEMPTS = 8
EMBEDDING_MODEL = "text-embedding-3-small"
SEVERE_GATE = 0.05
TEST_NUM_TRIALS = 3

ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "REDACTED_ANTHROPIC_VERTEX_TOKEN",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CRESTA_VERTEX_TOKEN = os.environ.get("CRESTA_VERTEX_TOKEN", ANTHROPIC_API_KEY)
VERTEX_SHIM_PORT = int(os.environ.get("VERTEX_SHIM_PORT", "4112"))
ANTHROPIC_BASE_URL = f"http://127.0.0.1:{VERTEX_SHIM_PORT}"

ROLEPLAY = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)

# ── Airline config ───────────────────────────────────────────────────────
AIRLINE_SEEDS = [
    (
        "alaska_00",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_alaska_k4/brinks_00_polite_goal_getter/gen_00.md"
        ),
    ),
    (
        "alaska_01",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_alaska_k4/brinks_01_polite_practical_solver/gen_00.md"
        ),
    ),
    (
        "alaska_02",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_alaska_k4/brinks_02_friendly_detail_seeker/gen_00.md"
        ),
    ),
    (
        "alaska_03",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_alaska_k4/brinks_03_task_focused_impatient/gen_00.md"
        ),
    ),
]
AIRLINE_SPLIT = "experiments/persona_search/configs/airline_alaska_split_v3.json"
AIRLINE_RUN_TAG = "alaska_airline_v3"
AIRLINE_RUN_ROOT = f"{DATA_ROOT}/data/persona_opt/pgepa_runs/{AIRLINE_RUN_TAG}"

# ── Retail config ────────────────────────────────────────────────────────
RETAIL_SEEDS = [
    (
        "gc_00",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_guitar_center_k5/gc_00_cautious_info_seeker/gen_00.md"
        ),
    ),
    (
        "gc_01",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_guitar_center_k5/gc_01_pragmatic_cooperative/gen_00.md"
        ),
    ),
    (
        "gc_02",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_guitar_center_k5/gc_02_polite_but_exacting/gen_00.md"
        ),
    ),
    (
        "gc_03",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_guitar_center_k5/gc_03_practical_deal_seeker/gen_00.md"
        ),
    ),
    (
        "gc_04",
        str(
            DATA_ROOT
            / "data/persona_opt/seeds_guitar_center_k5/gc_04_direct_confirming/gen_00.md"
        ),
    ),
]
RETAIL_SPLIT = "experiments/persona_search/configs/retail_guitar_center_split.json"
RETAIL_RUN_TAG = "retail_guitar_center_v1"
RETAIL_RUN_ROOT = f"{DATA_ROOT}/data/persona_opt/pgepa_runs/{RETAIL_RUN_TAG}"

# ── Utilities ────────────────────────────────────────────────────────────


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def wait_existing_pids(pids, label, interval=120):
    alive = [p for p in pids if pid_alive(p)]
    if not alive:
        log(f"[{label}] no existing processes to wait for")
        return
    log(f"[{label}] waiting for {len(alive)} existing processes: {alive}")
    while alive:
        time.sleep(interval)
        alive = [p for p in alive if pid_alive(p)]
        if alive:
            log(f"[{label}] still running: {alive}")
    log(f"[{label}] all existing processes done")


def ensure_shim():
    try:
        s = socket.create_connection(("127.0.0.1", VERTEX_SHIM_PORT), timeout=2)
        s.close()
        log(f"vertex shim already running on :{VERTEX_SHIM_PORT}")
        return
    except (ConnectionRefusedError, OSError):
        pass
    log(f"starting vertex shim on :{VERTEX_SHIM_PORT}")
    env = os.environ.copy()
    env["VERTEX_PROXY_TOKEN"] = CRESTA_VERTEX_TOKEN
    shim_log = open("/tmp/vertex_shim_pipeline.log", "w")
    subprocess.Popen(
        [PY, "anthropic_to_vertex_proxy.py", "--port", str(VERTEX_SHIM_PORT)],
        stdout=shim_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            s = socket.create_connection(("127.0.0.1", VERTEX_SHIM_PORT), timeout=2)
            s.close()
            log("vertex shim ready")
            return
        except (ConnectionRefusedError, OSError):
            pass
    log("WARNING: vertex shim may not have started")


def wait_procs(procs, label):
    log(f"[{label}] waiting for {len(procs)} processes...")
    failed = []
    for name, p, logf in procs:
        rc = p.wait()
        if logf:
            logf.close()
        status = "OK" if rc == 0 else f"FAIL(exit {rc})"
        log(f"  [{name}] {status}")
        if rc != 0:
            failed.append(name)
    if failed:
        log(f"[{label}] FAILED: {failed}")
    else:
        log(f"[{label}] all done")
    return failed


# ── GEPA launch ──────────────────────────────────────────────────────────


def launch_gepa(
    islands, domain, iterations, split_file, run_root, per_conc, extra_args=None
):
    os.makedirs(f"{run_root}/logs", exist_ok=True)
    procs = []
    for name, seed in islands:
        run_dir = f"{run_root}/{name}"
        logpath = f"{run_root}/logs/{name}.log"
        cmd = [
            PY,
            "experiments/persona_search/scripts/parallel_gepa.py",
            "--seed-persona",
            seed,
            "--persona-id-base",
            name,
            "--run-dir",
            run_dir,
            "--domains",
            domain,
            "--split-file",
            split_file,
            "--iterations",
            str(iterations),
            "--minibatch",
            str(MB),
            "--min-dissimilarity",
            str(MIN_DISSIM),
            "--max-semantic-similarity",
            str(MAX_SEM_SIM),
            "--embedding-model",
            EMBEDDING_MODEL,
            "--max-mutation-attempts",
            str(MAX_MUT_ATTEMPTS),
            "--max-concurrency",
            str(per_conc),
            "--severe-gate-rate",
            str(SEVERE_GATE),
            "--agent-llm",
            AGENT_LLM,
            "--agent-llm-args",
            AGENT_LLM_ARGS,
            "--user-type",
            USER_TYPE,
            "--user-llm",
            USER_LLM,
            "--effort",
            EFFORT,
            "--mutation-model",
            MUTATION_MODEL,
            "--mutation-effort",
            MUTATION_EFFORT,
            "--judge-model",
            JUDGE_MODEL,
            "--num-trials",
            str(NUM_TRIALS),
            "--val-num-trials",
            str(VAL_NUM_TRIALS),
            "--resume",
            "--no-bedrock",
            "--anthropic-base-url",
            ANTHROPIC_BASE_URL,
            "--anthropic-api-key",
            "dummy",
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
        f = open(logpath, "a")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((name, p, f))
        log(f"  [{name}] pid {p.pid} → {logpath}")
        time.sleep(2)
    return procs


# ── Test eval ────────────────────────────────────────────────────────────


def collect_personas_for_test(seeds, run_root):
    """Build list of (name, persona_text_or_None) for test eval."""
    personas = []
    # Seed personas
    for name, seed_file in seeds:
        text = open(seed_file).read().strip()
        personas.append((f"seed_{name}", text))
    # Optimized personas
    for name, _ in seeds:
        best_f = f"{run_root}/{name}/exploration_best_persona.md"
        if os.path.exists(best_f):
            text = open(best_f).read().strip()
            personas.append((f"opt_{name}", text))
        else:
            log(f"  WARNING: no optimized persona for {name} at {best_f}")
    # No persona
    personas.append(("no_persona", None))
    return personas


def run_test_eval(
    personas, domain, test_ids, save_prefix, batch_size=5, per_conc=5, num_trials=3
):
    log_dir = os.path.join(save_prefix, "logs")
    os.makedirs(log_dir, exist_ok=True)

    all_procs = []
    for batch_start in range(0, len(personas), batch_size):
        batch = personas[batch_start : batch_start + batch_size]
        procs = []
        for name, persona_text in batch:
            save_to = f"gepa_opt/{save_prefix}/test_{name}/{domain}"
            cmd = [
                TAU2,
                "run",
                "--domain",
                domain,
                "--user",
                USER_TYPE,
                "--user-llm",
                USER_LLM,
                "--user-llm-args",
                json.dumps({"reasoning_effort": EFFORT}),
                "--agent-llm",
                AGENT_LLM,
                "--agent-llm-args",
                AGENT_LLM_ARGS,
                "--num-trials",
                str(num_trials),
                "--max-concurrency",
                str(per_conc),
                "--save-to",
                save_to,
                "--auto-resume",
                "--max-steps",
                "120",
                "--task-ids",
                *[str(t) for t in test_ids],
            ]
            if persona_text:
                cmd += [
                    "--user-persona",
                    json.dumps({"custom_guidelines": ROLEPLAY + persona_text}),
                ]
            logpath = os.path.join(log_dir, f"{name}.log")
            f = open(logpath, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append((name, p, f))
            log(f"  [test:{name}] pid {p.pid}")
            time.sleep(1)

        failed = wait_procs(procs, f"test-batch-{batch_start // batch_size}")
        all_procs.extend(procs)

    # Print summary
    log(f"\n{'=' * 60}")
    log(f"Test results — {domain} ({save_prefix})")
    log(f"{'=' * 60}")
    rows = []
    for name, _, _ in all_procs:
        rp = f"{DATA_ROOT}/data/simulations/gepa_opt/{save_prefix}/test_{name}/{domain}/results.json"
        try:
            d = json.load(open(rp))
            sims = d.get("simulations", [])
            if sims:
                avg = sum(s["reward_info"]["reward"] for s in sims) / len(sims)
                rows.append((name, avg, len(sims)))
            else:
                rows.append((name, None, 0))
        except FileNotFoundError:
            rows.append((name, None, -1))

    for name, avg, n in rows:
        if avg is not None:
            log(f"  {name:30s}: {avg:.3f}  (N={n})")
        else:
            log(f"  {name:30s}: {'no sims' if n == 0 else '(no results)'}")

    # Compute seed avg vs opt avg
    seed_rewards = [
        avg for name, avg, _ in rows if name.startswith("seed_") and avg is not None
    ]
    opt_rewards = [
        avg for name, avg, _ in rows if name.startswith("opt_") and avg is not None
    ]
    np_rewards = [
        avg for name, avg, _ in rows if name == "no_persona" and avg is not None
    ]
    if seed_rewards:
        log(f"  {'--- seed avg':30s}: {sum(seed_rewards) / len(seed_rewards):.3f}")
    if opt_rewards:
        log(f"  {'--- opt avg':30s}: {sum(opt_rewards) / len(opt_rewards):.3f}")
    if np_rewards:
        log(f"  {'--- no_persona':30s}: {np_rewards[0]:.3f}")
    if seed_rewards and opt_rewards:
        seed_avg = sum(seed_rewards) / len(seed_rewards)
        opt_avg = sum(opt_rewards) / len(opt_rewards)
        log(f"  {'--- EFFECT (seed-opt)':30s}: {seed_avg - opt_avg:+.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def main():
    log("Full pipeline: airline v3 → test → retail v1 → test")
    log(f"  global concurrency cap: {GLOBAL_MAX_CONC}")

    ensure_shim()

    # Load test IDs
    airline_split = json.load(open(AIRLINE_SPLIT))
    airline_test_ids = sorted(airline_split["test"]["airline"], key=int)
    retail_split = json.load(open(RETAIL_SPLIT))
    retail_test_ids = sorted(retail_split["test"]["retail"], key=int)

    log(f"  airline test: {len(airline_test_ids)} tasks")
    log(f"  retail test:  {len(retail_test_ids)} tasks")

    # ── Phase 1: Airline GEPA resume to 20 iters ────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 1: Airline GEPA v3 → 20 iterations")
    log("=" * 60)

    # Check for existing running airline GEPA processes
    existing_pids = []
    for name, _ in AIRLINE_SEEDS:
        pid_f = f"{AIRLINE_RUN_ROOT}/{name}/pid"
        if os.path.exists(pid_f):
            try:
                pid = int(open(pid_f).read().strip())
                if pid_alive(pid):
                    existing_pids.append(pid)
            except (ValueError, FileNotFoundError):
                pass
    # Also check known PIDs from the current session
    for pid in [79627, 79734, 79761]:
        if pid_alive(pid) and pid not in existing_pids:
            existing_pids.append(pid)

    if existing_pids:
        wait_existing_pids(existing_pids, "airline-wait", interval=120)

    # 4 islands × 6 concurrency = 24 ≤ 25
    airline_per_conc = GLOBAL_MAX_CONC // len(AIRLINE_SEEDS)  # 6
    procs = launch_gepa(
        AIRLINE_SEEDS,
        "airline",
        iterations=20,
        split_file=AIRLINE_SPLIT,
        run_root=AIRLINE_RUN_ROOT,
        per_conc=airline_per_conc,
    )
    failed = wait_procs(procs, "airline-gepa")
    if failed:
        log(
            f"WARNING: airline GEPA failed for {failed}, continuing with available results"
        )

    # ── Phase 2: Airline test eval ───────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 2: Airline test eval (seed + optimized + no_persona)")
    log("=" * 60)

    airline_personas = collect_personas_for_test(AIRLINE_SEEDS, AIRLINE_RUN_ROOT)
    log(f"  {len(airline_personas)} persona runs: {[n for n, _ in airline_personas]}")

    # 9 personas: batches of 5 × 5 concurrency = 25
    run_test_eval(
        airline_personas,
        "airline",
        airline_test_ids,
        save_prefix=AIRLINE_RUN_TAG,
        batch_size=5,
        per_conc=5,
        num_trials=TEST_NUM_TRIALS,
    )

    # ── Phase 3: Retail GEPA ─────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 3: Retail GEPA v1 → 20 iterations")
    log("=" * 60)

    # 5 islands × 5 concurrency = 25
    retail_per_conc = GLOBAL_MAX_CONC // len(RETAIL_SEEDS)  # 5
    procs = launch_gepa(
        RETAIL_SEEDS,
        "retail",
        iterations=20,
        split_file=RETAIL_SPLIT,
        run_root=RETAIL_RUN_ROOT,
        per_conc=retail_per_conc,
    )
    failed = wait_procs(procs, "retail-gepa")
    if failed:
        log(
            f"WARNING: retail GEPA failed for {failed}, continuing with available results"
        )

    # ── Phase 4: Retail test eval ────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 4: Retail test eval (seed + optimized + no_persona)")
    log("=" * 60)

    retail_personas = collect_personas_for_test(RETAIL_SEEDS, RETAIL_RUN_ROOT)
    log(f"  {len(retail_personas)} persona runs: {[n for n, _ in retail_personas]}")

    # 11 personas: batches of 5 × 5 concurrency = 25
    run_test_eval(
        retail_personas,
        "retail",
        retail_test_ids,
        save_prefix=RETAIL_RUN_TAG,
        batch_size=5,
        per_conc=5,
        num_trials=TEST_NUM_TRIALS,
    )

    log("\n" + "=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
