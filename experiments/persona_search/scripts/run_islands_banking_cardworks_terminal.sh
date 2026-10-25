#!/usr/bin/env bash
# Island-model persona search on BANKING with TERMINAL_USE retrieval.
# This is the SAME search config as run_islands_banking_cardworks.sh (the bm25
# v2 run that produced the cw_00/01/03/04 opts), with ONE change: retrieval is
# terminal_use (agentic shell search via @anthropic-ai/sandbox-runtime) instead
# of bm25. Motivation: the terminal test-matrix showed opt-vs-no_p gain holds
# under terminal_use (~+0.02 for cw_00/cw_03), and terminal gives the agent a
# higher baseline ASR (+0.094 vs bm25) -> more headroom for the search to push
# difficulty up. We want to see if optimizing DIRECTLY under terminal retrieval
# finds personas with larger terminal-test gains than reusing bm25-optimized
# personas.
#
# Everything else is identical to the bm25 run:
#   - 4 independent islands (cw_00/01/03/04), own candidate pool + own seen list
#   - agent gpt-5.5 (high), user user_simulator gpt-5.4 (high)
#   - mutation-context=ENRICHED, ITERATIONS=10, MB=5, NUM_TRIALS=2
#   - PER_ISLAND_CONCURRENCY=6 (4x6=24 in-flight), early-stop-gated=6
#   - split=banking_knowledge_split_v2.json (20 train / 20 val / 57 test)
#   - mutation openai/gpt-5.4 xhigh, judge gpt-5.2, lint gpt-5.4-mini
#
# NOTE: terminal_use is ~3-4x slower per sim than bm25 (sandbox shell search),
# so this run will take roughly 3-4x longer wall-clock than the bm25 v2 run and
# proportionally more API spend. Config is intentionally NOT tuned for that —
# per the request, it mirrors the bm25 config exactly.
#
# Usage:
#   bash experiments/persona_search/scripts/run_islands_banking_cardworks_terminal.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_islands_banking_cardworks_terminal.sh
#   ITERATIONS=20 PER_ISLAND_CONCURRENCY=5 bash experiments/persona_search/scripts/run_islands_banking_cardworks_terminal.sh
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"

# --- wait for the current terminal test-matrix (cw_04 seed_terminal) to finish
# so we don't oversubscribe the sandbox/agent pool. ---
echo "[cw-terminal-opt] @ $(date) — 等当前 cw seed terminal test-matrix 退出"
while pgrep -f 'run_testmatrix_cw_seeds' | grep -qv grep; do
  sleep 60
done
echo "[cw-terminal-opt] test-matrix 已退出 @ $(date) — 启动 terminal_use 优化"

# --- search hyperparams (IDENTICAL to bm25 v2 run) ---
ITERATIONS="${ITERATIONS:-10}"
NUM_TRIALS="${NUM_TRIALS:-2}"                   # trials per (persona,task) for minibatch AND val
MB="${MB:-5}"                                   # banking is ~3.5x slower than telecom -> smaller minibatch
MIN_DISSIM="${MIN_DISSIM:-0.30}"
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-8}"
PER_ISLAND_CONCURRENCY="${PER_ISLAND_CONCURRENCY:-6}"   # 4 islands x 6 = 24 total in-flight
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
LINT_MODEL="${LINT_MODEL:-openai/gpt-5.4-mini}"
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"
MUTATION_CONTEXT="${MUTATION_CONTEXT:-enriched}"       # own-lineage outcomes + failure traces
EARLY_STOP_GATED="${EARLY_STOP_GATED:-6}"             # stop an island after 6 consecutive un-gateable mutations
SPLIT_FILE="${SPLIT_FILE:-experiments/persona_search/configs/banking_knowledge_split_v2.json}"
RETRIEVAL="${RETRIEVAL:-terminal_use}"          # <-- ONLY change vs bm25 run

# --- models / routing (identical to bm25 v2) ---
AGENT_LLM="${AGENT_LLM:-gpt-5.5}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

# --- run layout (NEW tag so the bm25 v2 run dir is NOT clobbered) ---
RUN_TAG="${RUN_TAG:-islands_banking_cardworks_v2_terminal}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"

# one independent island per seed: <island-name>:<seed-file>  (same 4 seeds)
SEEDS=(
  "cw_00:data/persona_opt/seeds_cardworks_k5/cw_00_anxious_action_seeker/gen_00.md"
  "cw_01:data/persona_opt/seeds_cardworks_k5/cw_01_rambling_yet_compliant/gen_00.md"
  "cw_03:data/persona_opt/seeds_cardworks_k5/cw_03_wary_step_by_step/gen_00.md"
  "cw_04:data/persona_opt/seeds_cardworks_k5/cw_04_brisk_cooperative_follower/gen_00.md"
)

echo "Launching ${#SEEDS[@]}-island banking CardWorks persona search (INDEPENDENT, mutation-context=$MUTATION_CONTEXT, retrieval=$RETRIEVAL)"
echo "  ITERATIONS=$ITERATIONS NUM_TRIALS=$NUM_TRIALS MB=$MB MIN_DISSIM=$MIN_DISSIM MAX_SEM_SIM=$MAX_SEM_SIM early_stop_gated=$EARLY_STOP_GATED"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY  total in-flight cap=$((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]}))"
echo "  split=$SPLIT_FILE  retrieval=$RETRIEVAL  (no shared seen pool)"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  lint=$LINT_MODEL  judge=$JUDGE_MODEL"
echo "  run_tag=$RUN_TAG  (bm25 v2 run dir preserved separately)"
echo

pids=()
for spec in "${SEEDS[@]}"; do
  name="${spec%%:*}"
  seed="${spec#*:}"
  base="isl_${name}"
  run_dir="${RUN_ROOT}/${base}"
  log="${LOG_DIR}/${base}.log"

  if [[ ! -f "$seed" ]]; then
    echo "Missing seed: $seed" >&2
    exit 1
  fi

  extra=(--resume --no-bedrock)
  [[ "$DRY_RUN" == "1" ]] && extra+=(--dry-run)

  echo "[$name] seed=$seed run_dir=$run_dir log=$log"

  nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py \
    --seed-persona "$seed" \
    --persona-id-base "$base" \
    --run-dir "$run_dir" \
    --domains banking \
    --retrieval-config "$RETRIEVAL" \
    --split-file "$SPLIT_FILE" \
    --mutation-context "$MUTATION_CONTEXT" \
    --early-stop-consecutive-gated "$EARLY_STOP_GATED" \
    --iterations "$ITERATIONS" \
    --minibatch "$MB" \
    --min-dissimilarity "$MIN_DISSIM" \
    --max-semantic-similarity "$MAX_SEM_SIM" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-mutation-attempts "$MAX_MUT_ATTEMPTS" \
    --lint-model "$LINT_MODEL" \
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
    "${extra[@]}" \
    >"$log" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "  pid $pid"
  sleep 2  # small stagger so gen-0 seed evals don't all hit the pool at once
done

echo
echo "All ${#SEEDS[@]} islands launched: pids ${pids[*]}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/isl_*.log"
