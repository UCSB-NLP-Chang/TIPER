# Adversarial Persona Optimization Pipeline

Automatically search the persona space for **valid** user personas that **minimize** the agent's success rate on tau2-bench. The pipeline is a (μ+λ) elitist evolutionary algorithm: all valid personas persist across rounds, new children are generated via LLM-based reflective mutation, and convergence is tracked with patience-based early stopping.

## Current Configuration

| Parameter | Value |
|---|---|
| Agent | `gpt-5.4`, reasoning_effort=`high` |
| User simulator | `cc_user_con` (persistent Claude Code session) |
| User backbone | `claude-opus-4-7` on Bedrock (`us-west-2-staging_dev`) |
| User effort | `high` |
| Domains | telecom only (23 tasks) |
| Trials per task | 1 |
| Concurrency | 21 total (7 per eval × 3 evals) |
| Per-sim timeout | 600s |
| Mutation model | `bedrock/us.anthropic.claude-opus-4-7` (extended thinking) |
| Children per round | 5 (generate) → 3 (select) |
| Validity threshold | `severe_failure_rate ≤ 0.05` |
| Convergence | max_rounds=20, early stop when untested=0, min_evals=3 |

## Architecture Overview

```
run_loop.py                          # outer loop: round N → N+1 → ... → converge
  └── run_round.py --round N         # one round, fully hands-off
        ├── preflight_bedrock()      # SSO token check (fail fast)
        ├── select.py                # pick 3 personas (2 explore + 1 exploit)
        ├── materialize.py           # write .skill files
        ├── eval_persona.py ×3       # concurrent tau2 run (cc_user_con)
        │     └── tau2 run           # 23 telecom sims, max-concurrency=7
        ├── judge_persona.py ×3      # intent preservation judge (gpt-5.2, maj@3)
        ├── record_results.py        # update lib.jsonl with scores
        ├── converge.py              # check stopping criterion
        └── auto_mutate.py           # LLM authors 5 children → add_children.py dedup
```

## Pipeline Components

### 1. Persona Library (`data/persona_opt/lib.jsonl`)

Single source of truth. One JSON record per line:

```json
{
  "persona_id": "p0007",
  "created_at": "2026-06-03T...",
  "persona_text": "# Persona: Low-Tech & Impatient ...",
  "status": "valid",           // untested | valid | illegal
  "parent_id": "p0005",
  "mutation_direction": "more_impatient_time_sensitive",
  "avg_success_rate": 0.478,   // running mean across evaluations
  "num_evaluations": 1,
  "dedup_checked": true,
  "judge_severe_failure_rate": 0.0,
  "last_round": 3
}
```

### 2. Selection (`select.py`)

Picks N=3 personas per round with explore/exploit balance:

- **2 explore**: randomly sampled from `untested` pool (seed=round_no for reproducibility)
- **1 exploit**: retest the best valid persona (lowest `avg_success_rate`)
- If fewer untested than explore slots, fill with valid retests (ascending score)

```bash
.venv/bin/python scripts/persona_opt/select.py --round 3 --n 3 --n-explore 2
```

### 3. Materialization (`materialize.py`)

Writes persona text from `lib.jsonl` records into `.skill` files at `cc_user_sim/personas/opt_<id>.skill`. These files are loaded by `ClaudeCodeConUserSimulator._load_personas()` at init time and injected into the CC session's system prompt.

### 4. Evaluation (`eval_persona.py`)

Runs `tau2 run` for each persona across the representative task set:

```bash
tau2 run --domain telecom \
  --user cc_user_con --user-llm claude-opus-4-7 \
  --user-llm-args '{"personas":["opt_p0007"],"effort":"high","use_bedrock":true}' \
  --agent-llm gpt-5.4 --agent-llm-args '{"reasoning_effort":"high"}' \
  --num-trials 1 --max-concurrency 7 \
  --save-to persona_opt/round_03/p0007/telecom --auto-resume \
  --max-steps 200 --timeout 600 \
  --task-ids <23 telecom task IDs>
```

- Uses `--auto-resume` for idempotent restart
- Infrastructure-broken sims are retried up to 2 times, then dropped from scoring
- `persona_score = mean(per-task reward)` over all non-infra tasks

Three evals run concurrently via `ThreadPoolExecutor(max_workers=3)` in `run_round.py`, each with `--max-concurrency 7` (total 21 sims).

### 5. Judge (`judge_persona.py`)

Wraps `scripts/judge_user_intent_preservation.py` to determine persona validity:

- Uses `gpt-5.2` with majority vote `maj@3`
- Only judges **failed** sims (reward < 1.0), capped at 10 per persona
- Counts severe failures (fabrication, goal deviation, etc.)
- **`severe_failure_rate = severe_count / total_scored_sims`** (full denominator, not just judged count)
- Status = `illegal` if `severe_failure_rate > 0.05`, else `valid`

Illegal personas stay in the library for dedup reference but are excluded from selection and mutation.

### 6. Record Results (`record_results.py`)

Folds eval + judge results into `lib.jsonl`:

- Updates `avg_success_rate` via running mean: `new_avg = (old_avg × old_n + score) / (old_n + 1)`
- Increments `num_evaluations`
- Sets `status` from judge verdict
- Writes `round_record.json` with per-persona round details

### 7. Convergence (`converge.py`)

Simple avg-based strategy:

- **Best persona** = lowest `avg_success_rate` among valid personas with `>= min_evals` (default 3) evaluations
- **Early stop**: only when `untested == 0` (mutation exhausted — all children dedup-rejected)
- **Hard cap**: `max_rounds` (default 20)
- **Final output**: persona with lowest avg, regardless of convergence
- Writes `convergence_state.json` with per-round history

### 8. Mutation (`auto_mutate.py`)

LLM-based reflective mutation (inspired by GEPA). After eval+judge+record+converge, if not converged:

**Inputs to the LLM:**
1. **Mutation context** — valid parents (all historical, not just this round), score trends, direction usage, illegal patterns
2. **Severe failure details** — from ALL rounds' judge outputs, showing what behavior triggered rejection
3. **Reflective traces** (GEPA-inspired) — conversation snippets from sims where the agent **succeeded** (persona was too easy), showing the mutator WHY and WHERE the persona failed to be challenging
4. **Full persona texts** — entire library for semantic dedup awareness
5. **Constraints** — invariant block requirements, output format

**Parent selection:** Fitness-proportionate, preferring parents with the lowest `avg_success_rate` (hardest = most promising for further mutation).

**Output:** JSON array of `{parent_id, mutation_direction, persona_text, reasoning}`. Each child is a ~30% rewrite of its parent.

**Dedup:** Children pass through `add_children.py`:
- Embedding cosine similarity vs entire library
- Hard reject if `≥ 0.92` similarity
- Hard reject if similar to an illegal persona (`≥ 0.85`)
- Rejected children trigger a retry with rejection reasons in the prompt (up to 3 retries)

```bash
.venv/bin/python scripts/persona_opt/auto_mutate.py --round 3 \
  --n-children 5 --model bedrock/us.anthropic.claude-opus-4-7 --effort xhigh
```

### 9. Loop Driver (`run_loop.py`)

Chains `run_round.py` repeatedly until convergence:

```bash
AWS_PROFILE=us-west-2-staging_dev \
  .venv/bin/python scripts/persona_opt/run_loop.py \
    --start-round 3 --max-rounds 10 \
    --total-sim-concurrency 21 --mutate-n-children 5
```

- Checks `convergence_state.json` after each round
- Stops if `should_stop=true` or max rounds reached
- Includes Bedrock SSO preflight check before each round

### 10. Status Dashboard (`status.py`)

Live monitoring of the optimization loop:

```bash
.venv/bin/python scripts/persona_opt/status.py --watch 30
```

Shows: loop process status, library summary, convergence state, top valid personas, per-round phase with sim completion counts, log tail.

## cc_user_con: How it Works

`ClaudeCodeConUserSimulator` (persistent-session mode) differs from `cc_user` (per-turn subprocess):

| | cc_user | cc_user_con |
|---|---|---|
| CC lifecycle | New subprocess per turn | One persistent session per sim |
| Tool calling | Text-based `<tool_call>` in user msg | Native MCP via in-process relay bridge |
| System prompt | CC adds its own boilerplate | `--bare` + `--exclude-dynamic-system-prompt-sections` |
| Skill injection | Every turn (prepended to user msg) | First message only (`_with_static_preamble`) |
| Context | Full transcript re-sent each turn | CC maintains conversation state internally |
| Failure recovery | N/A (stateless) | Restart + replay (up to `max_restarts=1`) |

**MCP Bridge Architecture:**
1. `_start_bridge()` creates an in-process FastMCP server (streamable-http on localhost)
2. CC connects to the bridge via `--mcp-config`
3. When CC calls `mcp__ccbridge__<tool>`, the bridge relays it as a `ToolCall` to the orchestrator
4. The orchestrator executes against the real environment DB and returns results
5. CC receives the tool result via the MCP protocol and continues

**Persona loading:** `_load_personas()` reads `cc_user_sim/personas/opt_<id>.skill` → `_persona_cache` → injected into CC's `--system-prompt` via `_persona_block()`.

## On-Disk Layout

```
data/persona_opt/
  goal.json                          # objective + constraints
  lib.jsonl                          # persona library (source of truth)
  convergence_state.json             # stopping state
  rounds/round_NN/
    selected.json                    # which personas were picked
    skill_manifest.json              # persona_id → skill file mapping
    eval/<id>/persona_eval.json      # per-persona eval summary
    judge/<id>/judge_status.json     # validity verdict
    judge/<id>/<domain>/per_sim.jsonl # per-sim judge details
    round_record.json                # round summary
    mutation_log.json                # mutation outcome
    logs/                            # eval/judge/mutate stdout logs

data/simulations/persona_opt/
  round_NN/<id>/<domain>/results.json # raw tau2 simulation results

cc_user_sim/personas/
  opt_<id>.skill                     # materialized persona files (derived)

scripts/persona_opt/
  _common.py          # shared paths, utilities
  lib.py              # library CLI (show, query)
  seed.py             # initialize lib with seed personas
  select.py           # round selection
  materialize.py      # write .skill files
  eval_persona.py     # run tau2 evaluation
  judge_persona.py    # intent preservation judge
  record_results.py   # fold results into library
  mutation_context.py # dump context for mutation
  auto_mutate.py      # LLM-based mutation
  add_children.py     # dedup + persist children
  converge.py         # convergence check
  report.py           # final report
  run_round.py        # single round driver
  run_loop.py         # continuous loop driver
  status.py           # live dashboard
```

## Running the Pipeline

### Prerequisites

```bash
uv sync --all-extras
cp .env.example .env  # OPENAI_API_KEY for judge
aws sso login --profile us-west-2-staging_dev  # Bedrock access
```

### Start the loop

```bash
AWS_PROFILE=us-west-2-staging_dev \
  nohup .venv/bin/python scripts/persona_opt/run_loop.py \
    --start-round 3 --max-rounds 10 \
    --total-sim-concurrency 21 --mutate-n-children 5 \
    > /tmp/persona_opt_loop.log 2>&1 &
```

### Monitor

```bash
.venv/bin/python scripts/persona_opt/status.py --watch 30
```

### Generate final report

```bash
.venv/bin/python scripts/persona_opt/report.py
```

## Evolution History

See the live library state with:

```bash
.venv/bin/python scripts/persona_opt/status.py
.venv/bin/python scripts/persona_opt/lib.py show --ranked
```

The full per-round audit trail is in `data/persona_opt/rounds/round_NN/round_record.json`.
