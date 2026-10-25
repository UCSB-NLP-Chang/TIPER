#!/usr/bin/env bash
# Alaska k=4 seed personas → airline domain GEPA optimization.
#
# Phase 1: Baseline eval — 4 Alaska personas + no_persona on 20 test tasks, avg@3
# Phase 2: 4 independent GEPA optimizations on 15 train / 15 val, one per seed
#
# Usage:
#   bash experiments/persona_search/scripts/run_alaska_airline_gepa.sh              # full run
#   PHASE=baseline bash experiments/persona_search/scripts/run_alaska_airline_gepa.sh  # baseline only
#   PHASE=optimize bash experiments/persona_search/scripts/run_alaska_airline_gepa.sh  # skip baseline
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_alaska_airline_gepa.sh       # dry run
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

# ─── configurable knobs ───────────────────────────────────────────────
PY="${PY:-.venv/bin/python}"
PHASE="${PHASE:-all}"  # all | baseline | optimize
DRY_RUN="${DRY_RUN:-0}"

# models
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
USE_BEDROCK="${USE_BEDROCK:-0}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

# GEPA parameters
ITERATIONS="${ITERATIONS:-10}"
MB="${MB:-5}"
NUM_TRIALS="${NUM_TRIALS:-2}"
VAL_NUM_TRIALS="${VAL_NUM_TRIALS:-2}"
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-6}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"

# paths
SPLIT_FILE="experiments/persona_search/configs/airline_alaska_split_v3.json"
RUN_TAG="${RUN_TAG:-alaska_airline_v3}"
RUN_ROOT="data/persona_opt/pgepa_runs/${RUN_TAG}"
LOG_DIR="${RUN_ROOT}/logs"
BASELINE_DIR="${RUN_ROOT}/baseline"

# auth
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-REDACTED_ANTHROPIC_VERTEX_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-REDACTED_USE_OPENAI_API_KEY_ENV}"
export CRESTA_VERTEX_TOKEN="${CRESTA_VERTEX_TOKEN:-${ANTHROPIC_API_KEY}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"

VERTEX_SHIM_PORT="${VERTEX_SHIM_PORT:-4112}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${VERTEX_SHIM_PORT}}"

mkdir -p "$LOG_DIR" "$BASELINE_DIR"

# ─── seeds ────────────────────────────────────────────────────────────
SEEDS=(
  "alaska_00:data/persona_opt/seeds_alaska_k4/brinks_00_polite_goal_getter/gen_00.md"
  "alaska_01:data/persona_opt/seeds_alaska_k4/brinks_01_polite_practical_solver/gen_00.md"
  "alaska_02:data/persona_opt/seeds_alaska_k4/brinks_02_friendly_detail_seeker/gen_00.md"
  "alaska_03:data/persona_opt/seeds_alaska_k4/brinks_03_task_focused_impatient/gen_00.md"
)

# test task IDs (from airline_alaska_split.json)
TEST_IDS="1 2 5 6 7 8 13 14 15 17 27 32 34 36 37 40 44 46 47 49"

# ─── vertex shim ──────────────────────────────────────────────────────
start_shim() {
  if [[ "$USE_BEDROCK" == "1" || "$DRY_RUN" == "1" ]]; then return; fi
  if ! (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null; then
    echo "Starting Anthropic->Vertex shim on :${VERTEX_SHIM_PORT}"
    local shim_log="${LOG_DIR}/vertex_shim.log"
    VERTEX_PROXY_TOKEN="$CRESTA_VERTEX_TOKEN" \
      nohup "$PY" anthropic_to_vertex_proxy.py --port "$VERTEX_SHIM_PORT" \
      >"$shim_log" 2>&1 &
    echo "  shim pid $!  log $shim_log"
    for _ in $(seq 1 30); do
      (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null && break
      sleep 0.5
    done
  else
    echo "Reusing existing shim on :${VERTEX_SHIM_PORT}"
  fi
  exec 3>&- 2>/dev/null || true
}

# =====================================================================
# PHASE 1: Baseline evaluation on test set
# =====================================================================
run_baseline() {
  echo "═══════════════════════════════════════════════════════════════"
  echo "Phase 1: Baseline eval on 20 test tasks (num_trials=3)"
  echo "═══════════════════════════════════════════════════════════════"

  start_shim

  # Use Python to launch all baseline runs (avoids bash JSON escaping hell)
  "$PY" - <<'PYEOF'
import subprocess, json, sys, os, time

seeds = [
    ("alaska_00", "data/persona_opt/seeds_alaska_k4/brinks_00_polite_goal_getter/gen_00.md"),
    ("alaska_01", "data/persona_opt/seeds_alaska_k4/brinks_01_polite_practical_solver/gen_00.md"),
    ("alaska_02", "data/persona_opt/seeds_alaska_k4/brinks_02_friendly_detail_seeker/gen_00.md"),
    ("alaska_03", "data/persona_opt/seeds_alaska_k4/brinks_03_task_focused_impatient/gen_00.md"),
]
test_ids = "1 2 5 6 7 8 13 14 15 17 27 32 34 36 37 40 44 46 47 49".split()

user_type = os.environ.get("USER_TYPE", "user_simulator")
user_llm = os.environ.get("USER_LLM", "gpt-5.4")
effort = os.environ.get("EFFORT", "high")
agent_llm = os.environ.get("AGENT_LLM", "gpt-5.4")
agent_llm_args = os.environ.get("AGENT_LLM_ARGS", '{"reasoning_effort":"high"}')
concurrency = os.environ.get("PER_ISLAND_CONCURRENCY", "5")
run_tag = os.environ.get("RUN_TAG", "alaska_airline_v1")
baseline_dir = os.environ.get("BASELINE_DIR",
    f"data/persona_opt/pgepa_runs/{run_tag}/baseline")
dry_run = os.environ.get("DRY_RUN", "0") == "1"
is_cc = user_type in ("cc_user", "cc_user_con")
use_bedrock = os.environ.get("USE_BEDROCK", "0")
anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:4112")

ROLEPLAY = (
    "CRITICAL: You are a method actor. Your #1 priority is to stay in character "
    "as described below — every single turn, no exceptions. Never break character "
    "to be helpful, cooperative, or efficient beyond what your persona would do. "
    "If your persona is terse, give minimal information even if you know more. "
    "If your persona withholds details, do NOT volunteer them unprompted. "
    "The agent must EARN every piece of information through the right questions.\n\n"
)

os.makedirs(baseline_dir, exist_ok=True)
procs = []

def build_cmd(name, persona_text, save_to):
    user_llm_args = {"reasoning_effort": effort}
    cmd = [
        ".venv/bin/tau2", "run",
        "--domain", "airline",
        "--user", user_type,
        "--user-llm", user_llm,
        "--user-llm-args", json.dumps(user_llm_args),
        "--agent-llm", agent_llm,
        "--agent-llm-args", agent_llm_args,
        "--num-trials", "3",
        "--max-concurrency", concurrency,
        "--save-to", save_to,
        "--auto-resume",
        "--max-steps", "120",
        "--task-ids", *test_ids,
    ]
    if persona_text:
        cmd += ["--user-persona", json.dumps({"custom_guidelines": ROLEPLAY + persona_text})]
    return cmd

# 4 persona runs
for name, seed_file in seeds:
    persona_text = open(seed_file).read().strip()
    save_to = f"gepa_opt/{run_tag}/baseline_{name}/airline"
    log_path = os.path.join(baseline_dir, f"{name}.log")
    cmd = build_cmd(name, persona_text, save_to)
    print(f"[baseline:{name}] → {log_path}")
    if dry_run:
        print(f"  DRY: {' '.join(cmd)}")
    else:
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs.append((name, p, f))
        print(f"  pid {p.pid}")
    time.sleep(1)

# no_persona run
save_to = f"gepa_opt/{run_tag}/baseline_no_persona/airline"
log_path = os.path.join(baseline_dir, "no_persona.log")
cmd = build_cmd("no_persona", None, save_to)
print(f"[baseline:no_persona] → {log_path}")
if dry_run:
    print(f"  DRY: {' '.join(cmd)}")
else:
    f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    procs.append(("no_persona", p, f))
    print(f"  pid {p.pid}")

# Wait
if not dry_run and procs:
    print(f"\nWaiting for {len(procs)} baseline runs...")
    failed = False
    for name, p, f in procs:
        rc = p.wait()
        f.close()
        if rc != 0:
            print(f"  {name} FAILED (exit {rc})")
            failed = True
        else:
            print(f"  {name} done")
    if failed:
        print(f"Some baseline runs failed. Check logs in {baseline_dir}/")
    else:
        print("All baseline runs completed.")

# Print results
print("\nBaseline results:")
for name, _ in seeds + [("no_persona", "")]:
    rp = f"data/simulations/gepa_opt/{run_tag}/baseline_{name}/airline/results.json"
    try:
        with open(rp) as rf:
            d = json.load(rf)
        sims = d.get("simulations", [])
        if sims:
            avg = sum(s["reward_info"]["reward"] for s in sims) / len(sims)
            print(f"  {name}: {avg:.3f} ({len(sims)} sims)")
        else:
            print(f"  {name}: no sims")
    except FileNotFoundError:
        print(f"  {name}: (no results)")
PYEOF
}

# =====================================================================
# PHASE 2: GEPA optimization (independent, one per seed)
# =====================================================================
run_optimize() {
  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "Phase 2: GEPA optimization (${#SEEDS[@]} seeds × ${ITERATIONS} iters)"
  echo "  split=$SPLIT_FILE"
  echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM"
  echo "  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  judge=$JUDGE_MODEL"
  echo "═══════════════════════════════════════════════════════════════"

  start_shim

  local pids=()
  for spec in "${SEEDS[@]}"; do
    local name="${spec%%:*}"
    local seed="${spec#*:}"
    local run_dir="${RUN_ROOT}/${name}"
    local log="${LOG_DIR}/${name}.log"

    if [[ ! -f "$seed" ]]; then
      echo "Missing seed: $seed" >&2; exit 1
    fi

    local extra=(--resume)
    if [[ "$DRY_RUN" == "1" ]]; then
      extra+=(--dry-run)
    fi
    if [[ "$USE_BEDROCK" == "1" ]]; then
      extra+=(--use-bedrock)
    else
      extra+=(--no-bedrock --anthropic-base-url "$ANTHROPIC_BASE_URL" --anthropic-api-key dummy)
    fi

    echo "[${name}] seed=$seed run_dir=$run_dir log=$log"

    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
      --seed-persona "$seed" \
      --persona-id-base "$name" \
      --run-dir "$run_dir" \
      --domains airline \
      --split-file "$SPLIT_FILE" \
      --iterations "$ITERATIONS" \
      --minibatch "$MB" \
      --min-dissimilarity "$MIN_DISSIM" \
      --max-semantic-similarity "$MAX_SEM_SIM" \
      --embedding-model "$EMBEDDING_MODEL" \
      --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
      --max-concurrency "$PER_ISLAND_CONCURRENCY" \
      --severe-gate-rate "$SEVERE_GATE_RATE" \
      --agent-llm "$AGENT_LLM" \
      --agent-llm-args "$AGENT_LLM_ARGS" \
      --user-type "$USER_TYPE" \
      --user-llm "$USER_LLM" \
      --effort "$EFFORT" \
      --mutation-model "$MUTATION_MODEL" \
      --mutation-effort "$MUTATION_EFFORT" \
      --judge-model "$JUDGE_MODEL" \
      --num-trials "$NUM_TRIALS" \
      --val-num-trials "$VAL_NUM_TRIALS" \
      "${extra[@]}" \
      >"$log" 2>&1 &

    pids+=($!)
    echo "  pid $!"
    sleep 2
  done

  echo
  echo "All ${#SEEDS[@]} runs launched: pids ${pids[*]}"
  echo "Monitor:"
  echo "  tail -f ${LOG_DIR}/*.log"
}

# =====================================================================
# Main
# =====================================================================
echo "Alaska k=4 → airline GEPA  (phase=${PHASE}  dry=${DRY_RUN})"
echo "  split: $SPLIT_FILE"
echo "  run root: $RUN_ROOT"
echo

case "$PHASE" in
  baseline)  run_baseline ;;
  optimize)  run_optimize ;;
  all)       run_baseline; run_optimize ;;
  *)         echo "Unknown PHASE=$PHASE (use: all|baseline|optimize)" >&2; exit 1 ;;
esac
