#!/usr/bin/env bash
#
# VitaBench reproduction run — configured to align with the public leaderboard
# (vitabench.github.io) as closely as the OpenAI-only Cresta key allows.
#
# Leaderboard entry being reproduced: "GPT-5.2 (xhigh)" thinking model.
# Paper/leaderboard experimental setup (arXiv:2509.26490):
#   - agent          = evaluated model, temperature 0.0, high/xhigh reasoning, thinking on
#   - user simulator = gpt-4.1-2025-04-14
#   - evaluator      = claude-3.7-sonnet  (deliberately a DIFFERENT family than the
#                      evaluated agent, to avoid self-preference bias)
#   - metric         = Avg@4  => 4 trials per task
#   - language       = chinese (the native task set)
#   - evaluation     = rubric-based sliding window (window=10 turns, overlap=2)
#   - no cap on interaction rounds or tool errors
#
# How this maps to our defaults (deviations noted):
#   agent      = gpt-5.2       reasoning_effort=xhigh  (--enable-think)   [MATCHES]
#   user       = gpt-4.1                                                  [MATCHES]
#   evaluator  = gpt-4.1  <-- MUST be a non-reasoning model. SUBSTITUTE for
#                claude-3.7-sonnet (Cresta key cannot reach Anthropic). A gpt-5.x
#                evaluator gets EVERY judge call rejected with `invalid_prompt`
#                (OpenAI content-moderates reasoning-model INPUT, and the judge prompt
#                embeds the entire conversation) -> 0 tasks score, whole run wasted.
#                gpt-4.1 has no such input moderation AND is the paper's deliberate
#                cross-family judge. See EVAL_INVALID_PROMPT_ISSUE.md for the full
#                write-up + reproduction.
#   trials     = 4  (Avg@4)                                              [MATCHES]
#   language   = chinese                                                 [MATCHES]
#   max_errors = 100000  (effectively "no cap"; num_errors is CUMULATIVE, never
#                reset — a low value wrongly kills long delivery tasks that
#                accumulate address-tool errors. The paper does not terminate
#                on tool errors.)                                        [MATCHES intent]
#   max_steps  = 300     (repo/README default; the paper text says "no limit",
#                but 300 is what the public tool ships with and is rarely hit)
#   max_duration = 600   (per-task wall-clock cap, seconds; NOT in the paper.
#                A task still running at 10 min is a non-converging polling loop
#                that would hit max_steps -> reward 0 anyway, so this changes no
#                score, it only frees the clogged concurrency slot. Set 0 to
#                disable for a fully paper-faithful run.)
#
# Config lives in src/vita/models.yaml; the API key is inline there.
#
# Usage:
#   scripts/run_cross_gpt52.sh                     # delivery, 100 tasks x 4 trials, chinese
#   DOMAIN=instore scripts/run_cross_gpt52.sh      # another single scenario
#   DOMAIN=delivery,instore,ota scripts/run_cross_gpt52.sh   # cross-scenario set
#   NUM_TASKS=3 scripts/run_cross_gpt52.sh         # quick subset (first 3 tasks)
#   NUM_TRIALS=1 scripts/run_cross_gpt52.sh        # cheaper smoke (Avg@1)
#
# Env overrides: DOMAIN, NUM_TASKS, NUM_TRIALS, LANGUAGE (chinese|english),
#   MAX_CONCURRENCY, MAX_STEPS, MAX_ERRORS, MAX_DURATION, AGENT_LLM, USER_LLM,
#   EVALUATOR_LLM, SAVE_TO, CSV_OUT, RESUME.
# Re-running with the same SAVE_TO resumes an interrupted run.

set -euo pipefail

# Resolve repo root (this script lives in <repo>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VITA="$REPO_DIR/.venv/bin/vita"
if [[ ! -x "$VITA" ]]; then
  echo "ERROR: $VITA not found. Create the venv first: uv venv --python 3.12 .venv && uv pip install -e ." >&2
  exit 1
fi

# ---- Leaderboard-aligned defaults ----
AGENT_LLM="${AGENT_LLM:-gpt-5.2}"          # GPT-5.2 (xhigh), thinking
USER_LLM="${USER_LLM:-gpt-4.1}"            # paper's user simulator
EVALUATOR_LLM="${EVALUATOR_LLM:-gpt-4.1}"  # MUST be a non-reasoning model. gpt-5.x
                                           # evaluators get rejected with `invalid_prompt`
                                           # (OpenAI moderates reasoning-model INPUT; the
                                           # judge prompt embeds the whole conversation).
                                           # gpt-4.1 also happens to be the paper's
                                           # cross-family judge choice. See
                                           # EVAL_INVALID_PROMPT_ISSUE.md.
LANGUAGE="${LANGUAGE:-chinese}"            # native task set
NUM_TRIALS="${NUM_TRIALS:-4}"              # Avg@4
MAX_CONCURRENCY="${MAX_CONCURRENCY:-30}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_ERRORS="${MAX_ERRORS:-100000}"         # effectively no cap; num_errors is cumulative (paper does not terminate on tool errors)
MAX_DURATION="${MAX_DURATION:-600}"        # per-task wall-clock cap in seconds (0 = disable, fully paper-faithful)

# DOMAIN can be a single scenario (delivery|instore|ota) or a comma list
# (delivery,instore,ota) which auto-selects the cross_domain task set.
DOMAIN="${DOMAIN:-delivery}"
# Short tag for output filenames (comma -> "cross").
DOMAIN_TAG="${DOMAIN//,/_}"
[[ "$DOMAIN" == *,* ]] && DOMAIN_TAG="cross"

SAVE_TO="${SAVE_TO:-${DOMAIN_TAG}_gpt52xhigh_aligned_${LANGUAGE}.json}"
CSV_OUT="${CSV_OUT:-results_${DOMAIN_TAG}_gpt52xhigh_aligned_${LANGUAGE}.csv}"

# Optional subset for quick validation.
NUM_TASKS_ARG=()
if [[ -n "${NUM_TASKS:-}" ]]; then
  NUM_TASKS_ARG=(--num-tasks "$NUM_TASKS")
fi

echo "=== VitaBench leaderboard-aligned run (domain=$DOMAIN) ==="
echo "agent=$AGENT_LLM (xhigh, --enable-think)  user=$USER_LLM  eval=$EVALUATOR_LLM"
echo "trials=$NUM_TRIALS  lang=$LANGUAGE  max_steps=$MAX_STEPS  max_errors=$MAX_ERRORS  max_duration=${MAX_DURATION}s"
echo "concurrency=$MAX_CONCURRENCY  save_to=$SAVE_TO"
[[ -n "${NUM_TASKS:-}" ]] && echo "num_tasks=$NUM_TASKS (subset)"
echo "=========================================================="

run_vita() {
  "$VITA" run \
    --domain "$DOMAIN" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --evaluator-llm "$EVALUATOR_LLM" \
    --enable-think \
    --num-trials "$NUM_TRIALS" \
    --language "$LANGUAGE" \
    --max-steps "$MAX_STEPS" \
    --max-errors "$MAX_ERRORS" \
    --max-duration "$MAX_DURATION" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --log-level WARNING \
    --save-to "$SAVE_TO" \
    --csv-output "$CSV_OUT" \
    ${NUM_TASKS_ARG[@]+"${NUM_TASKS_ARG[@]}"}
}

# When resuming (save file already exists) vita asks an interactive
# "resume? (y/n)" prompt that would hang a background/nohup run. Auto-answer
# yes in that case so completed (task,trial,seed) runs are skipped and only the
# remaining ones execute. Set RESUME=n to opt out of auto-confirm.
if [[ -f "data/simulations/$SAVE_TO" && "${RESUME:-y}" == "y" ]]; then
  echo "(resuming existing data/simulations/$SAVE_TO — auto-confirming)"
  yes | run_vita
else
  run_vita
fi
