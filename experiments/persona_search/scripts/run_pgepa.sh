#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single-run launcher for the serial-Pareto persona search (parallel_gepa.py).
#
# Everything you may need to configure is in the CONFIG block below; every var
# can also be overridden inline, e.g.:
#   ITERATIONS=10 MAX_TASKS=20 bash experiments/persona_search/scripts/run_pgepa.sh
#   DRY_RUN=1 bash experiments/persona_search/scripts/run_pgepa.sh           # $0, no LLM/sims
#   USE_BEDROCK=1 BEDROCK_PROFILE=myprof bash experiments/persona_search/scripts/run_pgepa.sh
#
# Pipeline (one parent -> one child per iteration):
#   gen0 confirm seed -> [select Pareto parent -> mutate (diversity + lint gates,
#   no task-specific) -> minibatch screen -> full valset eval + validity
#   judge (single vote) -> accept iff minibatch passed AND feasible] x ITERATIONS.
#
# SAFETY: refuses to overwrite a non-empty RUN_DIR unless FORCE=1.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
ANTHROPIC_API_KEY=REDACTED_ANTHROPIC_VERTEX_TOKEN
OPENAI_API_KEY=REDACTED_USE_OPENAI_API_KEY_ENV
# ===========================================================================
# CONFIG  (override any of these via environment)
# ===========================================================================
PY="${PY:-.venv/bin/python}"

# --- what to optimize ------------------------------------------------------
SEED="${SEED:-data/persona_opt/seeds_brinks/brinks_01_terse_erratic/gen_00.md}"
PERSONA_ID_BASE="${PERSONA_ID_BASE:-pgepa_brinks01_run1}"   # MUST be unique per concurrent run
RUN_DIR="${RUN_DIR:-data/persona_opt/pgepa_runs/${PERSONA_ID_BASE}}"
DOMAINS="${DOMAINS:-telecom}"        # space-separated, e.g. "telecom banking"
MAX_TASKS="${MAX_TASKS:-50}"         # cap PER domain (0 = all representative tasks)

# --- search loop -----------------------------------------------------------
ITERATIONS="${ITERATIONS:-10}"       # parent->child iterations
MB="${MB:-10}"                      # minibatch pre-screen size (<=0 = full set)
SEED_RNG="${SEED_RNG:-0}"            # rng seed for parent draw / minibatch

# --- diversity gate (layered: lexical Jaccard pre-filter + embedding cosine) -
MIN_DISSIM="${MIN_DISSIM:-0.30}"             # min word-3gram Jaccard dissim vs ALL prior personas
MAX_SEM_SIM="${MAX_SEM_SIM:-0.90}"           # max embedding cosine vs any prior persona
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"   # OpenAI; needs OPENAI_API_KEY
NO_EMBEDDING="${NO_EMBEDDING:-0}"            # 1 = lexical-only (skip semantic layer)
MAX_MUT_ATTEMPTS="${MAX_MUT_ATTEMPTS:-5}"    # retries to clear diversity + lint gates

# --- static persona linter (natural / valid / no-task-specific, pre-sim) ----
LINT="${LINT:-1}"                    # 1 = on (default), 0 = off
LINT_MODEL="${LINT_MODEL:-openai/gpt-5.4-mini}"   # OpenAI; needs OPENAI_API_KEY. empty = reuse MUTATION_MODEL
LINT_MAJ="${LINT_MAJ:-1}"            # linter votes

# --- validity judge --------------------------------------------------------
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
JUDGE_MAJ="${JUDGE_MAJ:-1}"          # single judgment (was 3)
SEVERE_GATE_RATE="${SEVERE_GATE_RATE:-0.05}"   # persona-level validity gate (rate)
JUDGE_VALSET="${JUDGE_VALSET:-1}"    # 1 = judge full-set (failed-only) at gen0/confirm

# --- models / routing ------------------------------------------------------
AGENT_LLM="${AGENT_LLM:-gpt-5.4}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"reasoning_effort\":\"high\"}}"
USER_TYPE="${USER_TYPE:-cc_user_con}"
USER_LLM="${USER_LLM:-claude-opus-4-7}"
EFFORT="${EFFORT:-high}"
MUTATION_MODEL="${MUTATION_MODEL:-openai/gpt-5.4}"   # gpt-5.4 reasoning_effort=high (set in call_mutation_llm); needs OPENAI_API_KEY
MAX_CONCURRENCY="${MAX_CONCURRENCY:-5}"       # within-eval sim fan-out (NOT cross-child)

# --- credentials / backends ------------------------------------------------
# uuid:secret used both as Anthropic key and as the Cresta Vertex token.
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-REDACTED_ANTHROPIC_VERTEX_TOKEN}"
USE_BEDROCK="${USE_BEDROCK:-0}"      # 1 = user-sim via Bedrock; 0 = via local Vertex shim
BEDROCK_PROFILE="${BEDROCK_PROFILE:-}"

# Vertex env for the Anthropic->Vertex user-sim shim (cc_user_con backbone) and
# any vertex_ai/* mutation model. With the OpenAI defaults below it only feeds the
# user-sim shim, but keep it set so claude routing still works if you switch back.
export CRESTA_VERTEX_TOKEN="${CRESTA_VERTEX_TOKEN:-${ANTHROPIC_API_KEY}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-cresta-ml}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"

# OpenAI env for the in-process litellm calls: mutation model (gpt-5.4), validity
# judge ($JUDGE_MODEL), lint model ($LINT_MODEL), and the embedding semantic layer
# (text-embedding-3-small). Export OPENAI_API_KEY (and OPENAI_BASE_URL if you use
# a proxy) in your shell before running; passed through to litellm/OpenAI as-is.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
[[ -n "${OPENAI_BASE_URL:-}" ]] && export OPENAI_BASE_URL

# Local Anthropic-Messages -> Vertex shim for the cc_user_con backbone (only
# used when NOT on Bedrock and NOT a dry run).
VERTEX_SHIM_PORT="${VERTEX_SHIM_PORT:-4112}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${VERTEX_SHIM_PORT}}"

# --- run mode --------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"              # 1 = synthetic scores, no LLM/sims, $0
FOREGROUND="${FOREGROUND:-1}"        # 1 = run in foreground; 0 = nohup background
FORCE="${FORCE:-0}"                  # 1 = allow overwriting a non-empty RUN_DIR
RESUME="${RESUME:-0}"               # 1 = continue from existing pgepa_state.json
CONFIRM_GEN="${CONFIRM_GEN:-}"        # e.g. 9 = skip to full-valset confirm for gen 9
CONFIRM_PERSONA="${CONFIRM_PERSONA:-}"  # optional override skill path for CONFIRM_GEN
CONTINUE_AFTER_CONFIRM="${CONTINUE_AFTER_CONFIRM:-0}"  # 1 = keep iterating after confirm

# ===========================================================================
# Pre-flight
# ===========================================================================
if [[ ! -f "$SEED" ]]; then
  echo "ERROR: seed persona not found: $SEED" >&2
  exit 1
fi
if [[ "$RESUME" == "1" ]]; then
  if [[ ! -s "${RUN_DIR}/pgepa_state.json" ]]; then
    echo "ERROR: RESUME=1 but no ${RUN_DIR}/pgepa_state.json to resume from." >&2
    echo "       (state.json is written only after iteration 1 completes.)" >&2
    exit 1
  fi
elif [[ "$FORCE" != "1" && -s "${RUN_DIR}/pgepa_state.json" ]]; then
  echo "ERROR: ${RUN_DIR}/pgepa_state.json already exists (a previous run is there)." >&2
  echo "       Set RESUME=1 to continue it, FORCE=1 to overwrite, or pick another RUN_DIR." >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

if [[ "$DRY_RUN" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is unset, but the mutation model ($MUTATION_MODEL) now runs" >&2
  echo "       on OpenAI — every iteration's mutation will fail without it. The validity" >&2
  echo "       judge ($JUDGE_MODEL) and lint ($LINT_MODEL) also need it; the embedding" >&2
  echo "       layer ($EMBEDDING_MODEL) would degrade to lexical-only." >&2
  echo "       Export OPENAI_API_KEY (and OPENAI_BASE_URL if proxied) before running." >&2
  exit 1
fi

# Start the Vertex shim if we need it (non-Bedrock, non-dry-run).
if [[ "$USE_BEDROCK" != "1" && "$DRY_RUN" != "1" ]]; then
  if ! (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null; then
    echo "Starting Anthropic->Vertex shim on :${VERTEX_SHIM_PORT}"
    VERTEX_PROXY_TOKEN="$CRESTA_VERTEX_TOKEN" \
      nohup "$PY" anthropic_to_vertex_proxy.py --port "$VERTEX_SHIM_PORT" \
      >"${RUN_DIR}/vertex_shim.log" 2>&1 &
    echo "  shim pid $!  log ${RUN_DIR}/vertex_shim.log"
    for _ in $(seq 1 30); do
      (exec 3<>"/dev/tcp/127.0.0.1/${VERTEX_SHIM_PORT}") 2>/dev/null && break
      sleep 0.5
    done
  else
    echo "Reusing existing shim on :${VERTEX_SHIM_PORT}"
  fi
  exec 3>&- 2>/dev/null || true
fi

# ===========================================================================
# Assemble args
# ===========================================================================
args=(
  --seed-persona "$SEED"
  --persona-id-base "$PERSONA_ID_BASE"
  --run-dir "$RUN_DIR"
  --domains $DOMAINS
  --max-tasks "$MAX_TASKS"
  --iterations "$ITERATIONS"
  --minibatch "$MB"
  --seed "$SEED_RNG"
  --min-dissimilarity "$MIN_DISSIM"
  --max-semantic-similarity "$MAX_SEM_SIM"
  --embedding-model "$EMBEDDING_MODEL"
  --max-mutation-attempts "$MAX_MUT_ATTEMPTS"
  --lint-maj "$LINT_MAJ"
  --judge-model "$JUDGE_MODEL"
  --judge-maj "$JUDGE_MAJ"
  --severe-gate-rate "$SEVERE_GATE_RATE"
  --agent-llm "$AGENT_LLM"
  --agent-llm-args "$AGENT_LLM_ARGS"
  --user-type "$USER_TYPE"
  --user-llm "$USER_LLM"
  --effort "$EFFORT"
  --mutation-model "$MUTATION_MODEL"
  --max-concurrency "$MAX_CONCURRENCY"
)
[[ "$NO_EMBEDDING" == "1" ]] && args+=(--no-embedding)
[[ -n "$LINT_MODEL" ]]       && args+=(--lint-model "$LINT_MODEL")
if [[ "$LINT" == "1" ]]; then args+=(--lint-persona); else args+=(--no-lint-persona); fi
[[ "$JUDGE_VALSET" == "1" ]] && args+=(--judge-valset)
[[ "$DRY_RUN" == "1" ]]      && args+=(--dry-run)
[[ "$RESUME" == "1" ]]       && args+=(--resume)
[[ -n "$CONFIRM_GEN" ]]      && args+=(--confirm-gen "$CONFIRM_GEN")
[[ -n "$CONFIRM_PERSONA" ]]  && args+=(--confirm-persona "$CONFIRM_PERSONA")
[[ "$CONTINUE_AFTER_CONFIRM" == "1" ]] && args+=(--continue-after-confirm)
if [[ "$USE_BEDROCK" == "1" ]]; then
  args+=(--use-bedrock)
  [[ -n "$BEDROCK_PROFILE" ]] && args+=(--bedrock-profile "$BEDROCK_PROFILE")
else
  args+=(--no-bedrock --anthropic-base-url "$ANTHROPIC_BASE_URL" --anthropic-api-key dummy)
fi

# ===========================================================================
# Launch
# ===========================================================================
rollouts=$(( MAX_TASKS + ITERATIONS * (MB + 2 * MAX_TASKS) ))
echo "=== serial-Pareto persona search ==============================="
echo "  seed        $SEED"
echo "  run_dir     $RUN_DIR   (base=$PERSONA_ID_BASE)"
echo "  domains     $DOMAINS   max_tasks=$MAX_TASKS"
echo "  loop        iterations=$ITERATIONS minibatch=$MB"
echo "  diversity   jaccard>=$MIN_DISSIM + semantic<=$MAX_SEM_SIM (embed=$([[ $NO_EMBEDDING == 1 ]] && echo off || echo $EMBEDDING_MODEL))"
echo "  lint        $([[ $LINT == 1 ]] && echo on || echo off) (maj=$LINT_MAJ model=${LINT_MODEL:-$MUTATION_MODEL})"
echo "  judge       $JUDGE_MODEL maj=$JUDGE_MAJ severe_gate=$SEVERE_GATE_RATE judge_valset=$JUDGE_VALSET"
echo "  agent/user  $AGENT_LLM  /  $USER_TYPE:$USER_LLM (bedrock=$USE_BEDROCK)"
echo "  mutate      $MUTATION_MODEL (reasoning_effort=high for gpt/o, xhigh+thinking for claude)"
echo "  dry_run=$DRY_RUN  resume=$RESUME  confirm_gen=${CONFIRM_GEN:-off}  rollout upper-bound ~= $rollouts"
echo "================================================================"

if [[ "$FOREGROUND" == "1" ]]; then
  exec "$PY" experiments/persona_search/scripts/parallel_gepa.py "${args[@]}"
else
  log="${RUN_DIR}/run.log"
  nohup "$PY" experiments/persona_search/scripts/parallel_gepa.py "${args[@]}" >"$log" 2>&1 &
  echo "  pid $!  log $log"
  echo "  monitor: $PY experiments/persona_search/scripts/pgepa_monitor.py --watch 10 --run-dir $RUN_DIR"
fi
