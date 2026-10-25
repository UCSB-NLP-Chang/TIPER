#!/usr/bin/env bash
# Island-model persona search on BANKING, seeded from the real-data CardWorks
# personas induced from real_cresta_data/cardworks_clean_no_repeat.json
# (2540 Merrick Bank / CardWorks support transcripts, style-subspace KMeans).
#
# FOUR independent islands (see seeds_cardworks_k5/index.json):
#   cw_00  Anxious Action-Seeker      (306/2540 = 12.0%, HARMS)
#   cw_01  Rambling Yet Compliant     (358/2540 = 14.1%, HARMS)
#   cw_03  Wary Step-By-Step          (1298/2540 = 51.1%, HARMS, dominant)
#   cw_04  Brisk Cooperative Follower (430/2540 = 16.9%, HELPS)
# cw_02 (Long-Winded Disputer, 5.8%) is DROPPED: it is the smallest cluster and
# behaviorally overlaps cw_01 (both wordy/story-driven/dispute-focused).
# cw_04 is a HELPS seed but GEPA maximizes difficulty regardless of seed effect,
# so it is a legitimate distinct starting point (starts low, search pushes up).
#
# INDEPENDENT islands: each keeps its OWN candidate pool and OWN seen list. There
# is NO shared cross-island seen pool — the four seeds are already behaviorally
# distinct real clusters, and cross-island dedup would over-constrain each island
# from exploring its own cluster's neighborhood. Selection uses the held-out VAL
# split; minibatch is sampled from TRAIN; TEST is left untouched for a post-hoc
# generalization check (banking_knowledge_split_v2.json, 20/20/57).
#
# THREE deliberate choices vs the telecom island run:
#   1. agent = gpt-5.5 (NOT gpt-5.4). Banking under gpt-5.4 is already floored
#      (no_persona ASR ~27%), leaving a persona no room to push the agent down.
#      gpt-5.5 has a higher baseline ASR -> real headroom for the search.
#   2. No Anthropic->Vertex shim. The user is the plain user_simulator (gpt-5.4),
#      not Claude, so no shim is needed. gpt-5.x agent/user route through the
#      Cresta Vertex litellm patch; mutation/lint/embedding use OPENAI_API_KEY.
#      All creds load automatically (parallel_gepa imports tau2.utils.utils,
#      which calls load_dotenv() at import time) — do NOT source .env in bash.
#   3. mutation-context = ENRICHED (not basic). Without a shared pool, basic's
#      cross-island persona summaries are moot; enriched instead feeds each island
#      its OWN lineage outcomes (direction + val_diff + accept/reject) plus failure
#      traces — strictly more informative for a solo island, no extra LLM call
#      (twophase would double mutation cost, not worth it on slow/expensive banking).
#
# Retrieval: terminal_use (agentic shell search over the KB via a sandboxed
# @anthropic-ai/sandbox-runtime; ShellSpec allow_writes=False). Trying the
# shell-based retrieval first — heavier than bm25 but closer to real agent use.
#
# Usage:
#   bash experiments/persona_search/scripts/run_islands_banking_cardworks.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_islands_banking_cardworks.sh
#   ITERATIONS=20 PER_ISLAND_CONCURRENCY=5 bash experiments/persona_search/scripts/run_islands_banking_cardworks.sh
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

PY="${PY:-.venv/bin/python}"

# --- search hyperparams (mirror banking bk_00) ---
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
RETRIEVAL="${RETRIEVAL:-bm25}"

# --- models / routing (agent bumped to gpt-5.5; user is plain user_simulator) ---
AGENT_LLM="${AGENT_LLM:-gpt-5.5}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-user_simulator}"
USER_LLM="${USER_LLM:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"
MUTATION_EFFORT="${MUTATION_EFFORT:-xhigh}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

# --- run layout ---
RUN_TAG="${RUN_TAG:-islands_banking_cardworks_v2}"
RUN_ROOT="${RUN_ROOT:-data/persona_opt/pgepa_runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"

# one independent island per seed: <island-name>:<seed-file>
SEEDS=(
  "cw_00:data/persona_opt/seeds_cardworks_k5/cw_00_anxious_action_seeker/gen_00.md"
  "cw_01:data/persona_opt/seeds_cardworks_k5/cw_01_rambling_yet_compliant/gen_00.md"
  "cw_03:data/persona_opt/seeds_cardworks_k5/cw_03_wary_step_by_step/gen_00.md"
  "cw_04:data/persona_opt/seeds_cardworks_k5/cw_04_brisk_cooperative_follower/gen_00.md"
)

echo "Launching ${#SEEDS[@]}-island banking CardWorks persona search (INDEPENDENT, mutation-context=$MUTATION_CONTEXT)"
echo "  ITERATIONS=$ITERATIONS NUM_TRIALS=$NUM_TRIALS MB=$MB MIN_DISSIM=$MIN_DISSIM MAX_SEM_SIM=$MAX_SEM_SIM early_stop_gated=$EARLY_STOP_GATED"
echo "  per-island concurrency=$PER_ISLAND_CONCURRENCY  total in-flight cap=$((PER_ISLAND_CONCURRENCY * ${#SEEDS[@]}))"
echo "  split=$SPLIT_FILE  retrieval=$RETRIEVAL  (no shared seen pool)"
echo "  agent=$AGENT_LLM  user=$USER_TYPE/$USER_LLM  mutate=$MUTATION_MODEL (effort=$MUTATION_EFFORT)  lint=$LINT_MODEL  judge=$JUDGE_MODEL"
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
