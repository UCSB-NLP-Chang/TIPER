# Persona Optimization — adversarial persona search

An optimization loop that searches the space of **user personas** for ones that are
**valid** (realistic, task-consistent, intent-preserving — they don't cheat) yet
**minimize the agent's tau2-bench success rate**. Lower `avg_success_rate` ⇒ harder persona.

The loop runs fully automatically via `run_loop.py`. Each component is a tiny,
composable CLI (JSON-in/JSON-out, atomic writes, idempotent). **Mutation is LLM-authored**
by `auto_mutate.py` (Bedrock Opus-4.7 with extended thinking) — it reads the library
context and score trends, writes child personas, and `add_children.py` dedups them.

---

## Models & config (plan §2)

| role | model | settings |
|------|-------|----------|
| agent (system under test) | `gpt-5.4` | `reasoning_effort: high` |
| user simulator | `cc_user_con` + `claude-opus-4-7` | `effort: high`, `use_bedrock: true` |
| judge | `gpt-5.2` | OpenAI Batch API, maj@3 |
| mutation LLM | `bedrock/us.anthropic.claude-opus-4-7` | extended thinking, `effort: xhigh` |
| dedup embeddings | `text-embedding-3-large` | cosine |

Locked parameters: **num_trials = 1**; persona is **illegal if `severe_failure_rate > 0.05`**;
convergence: run up to **20 rounds**, early stop only when **untested = 0**.

Task set: **telecom 23 tasks** (from `data/tau2/representative_tasks_agg_min.json`).

---

## The `goal.json`

`data/persona_opt/goal.json` parameterizes *what kind* of persona to hunt. Claude drafts it;
**the user sets** `target_flavor` (and optionally `seed_personas`). The objective, models, and
thresholds are pre-filled to match the locked decisions. Claude uses `target_flavor` to steer
the personas it authors at the mutation step.

---

## Library (`data/persona_opt/lib.jsonl`)

One JSON object per line — the single source of truth. `.skill` files are derived from it.

```json
{"persona_id":"p0007","created_at":"...","persona_text":"...",
 "status":"untested|valid|illegal","parent_id":null,"mutation_direction":null,
 "avg_success_rate":null,"num_evaluations":0,"dedup_checked":true,
 "judge_severe_failure_rate":null,"last_round":null}
```

**Status lifecycle:** `untested` → (eval + judge) → `valid` or `illegal`.
- `untested` personas are selected **by status, never by a placeholder score** — `avg_success_rate` stays `null` until evaluated.
- `illegal` personas are excluded from selection and from being mutation parents, but **stay in the library** for dedup.

---

## Directory layout

```
data/persona_opt/
  goal.json                      # objective + constraints (user sets target_flavor)
  lib.jsonl                      # source of truth
  convergence_state.json         # global-best history + patience counter
  final_report.{json,md}         # ranked output (plan §10)
  rounds/round_NN/
    selected.json                # the 3 personas picked this round
    skill_manifest.json          # persona_id -> opt_<id> skill name
    mutation_context.json        # what Claude reads to author children
    children_in.json             # children Claude authored (input to dedup)
    children.json                # accepted children (new ids)
    mutation_log.json            # rejects, gray-zone flags, shortfall
    round_record.json            # per-iteration record (scores, statuses)
    eval/<id>/persona_eval.json  # per-task rewards + persona_score
    eval/<id>/{telecom,banking}/ # raw tau2 run save dirs are under data/simulations/
    judge/<id>/judge_status.json # valid/illegal + severe_rate
    judge/<id>/{telecom,banking}/per_sim.jsonl,summary.json
cc_user_sim/personas/opt_<id>.skill   # derived, materialized per round
```

---

## Component reference

All run as `.venv/bin/python scripts/persona_opt/<name>.py ...`.

| component | purpose | key args | output |
|-----------|---------|----------|--------|
| `run_round.py` | **one-command round**: select→materialize→eval→judge→record→converge | `--round N [--n 3 --eval-concurrency 3 --force --dry-run]` | everything below + a summary |
| `lib.py` | inspect the library | `show [--status S] [--ranked]`, `get --persona-id`, `stats` | stdout |
| `seed.py` | bootstrap round-0 untested personas | `--from-skills <stems...>` / `--text` | lib.jsonl |
| `select.py` | pick N personas (plan §3) | `--round N [--n 3]` | `selected.json` |
| `materialize.py` | write `opt_<id>.skill` files | `--round N` | skill files + `skill_manifest.json` |
| `eval_persona.py` | run both domains, aggregate score (§4) | `--persona-id --round [--max-tasks --dry-run]` | `persona_eval.json` |
| `judge_persona.py` | judge → valid/illegal (§5) | `--persona-id --round [--maj 3 --threshold 0.05 --dry-run]` | `judge_status.json` |
| `record_results.py` | fold eval+judge into lib (§1,4,5) | `--round N` | lib.jsonl + `round_record.json` |
| `mutation_context.py` | dump context for Claude to author children (§6,8) | `--round N` | `mutation_context.json` |
| `add_children.py` | dedup + persist authored children (§8) | `--round N [--sim-method embed\|difflib]` | lib.jsonl + `children.json`, `mutation_log.json` |
| `converge.py` | convergence check / stop (§9) | `--round N [--max-rounds 20 --min-evals 3]` | `convergence_state.json` |
| `report.py` | final ranked output (§10) | — | `final_report.{json,md}` |

Reused existing code: `scripts/judge_user_intent_preservation.py` (judge), the
`tau2 run` CLI (proven in `scripts/run_banking_all_presets.sh`),
`claude_code_user_simulator.py` (persona-by-name loading; `effort`/`use_bedrock` in llm_args),
`tau2.knowledge.embedders.openai_embedder.OpenAIEmbedder` (dedup cosine).

---

## The iteration loop (one round)

```
select  →  materialize  →  eval_persona (×3)  →  judge_persona (×3)  →  record_results
        →  converge  →  auto_mutate (LLM authors children → add_children dedup)
```

### Fully automated: `run_round.py`

Everything is chained by `run_round.py` — select, materialize, evaluate (concurrently with
infra-retry), judge, record, converge check, and auto-mutate via LLM. No human intervention.

```bash
# env must carry OPENAI_API_KEY (judge) + AWS_PROFILE (Bedrock user sim)
AWS_PROFILE=us-west-2-staging_dev \
  .venv/bin/python scripts/persona_opt/run_round.py --round N
```

`run_round.py` is **idempotent**: a persona whose `persona_eval.json` / `judge_status.json`
already exists is skipped (re-run safe after an interruption). `--force` recomputes; evals
still use tau2 `--auto-resume`, so even a forced re-run only fills missing sims. Per-persona
logs land in `rounds/round_NN/logs/{eval,judge}_<id>.log`.

### Manual (component-by-component)

The same thing, if you want to drive each step yourself:

```bash
PY=.venv/bin/python

# 0. (once, before round 1) seed the library
$PY scripts/persona_opt/seed.py --from-skills impatient-rushing suspicious-withholding distracted-rambler

$PY scripts/persona_opt/select.py --round N
$PY scripts/persona_opt/materialize.py --round N
for p in <ids from selected.json>; do $PY scripts/persona_opt/eval_persona.py --persona-id $p --round N; done
for p in <ids>; do $PY scripts/persona_opt/judge_persona.py --persona-id $p --round N; done   # sync mode
$PY scripts/persona_opt/record_results.py --round N
$PY scripts/persona_opt/mutation_context.py --round N      # then Claude writes children_in.json
$PY scripts/persona_opt/add_children.py --round N
$PY scripts/persona_opt/converge.py --round N              # prints should_stop
```

When `converge` prints `should_stop=True`, run `report.py`.

### Infrastructure errors are retried, not just dropped

A transient cc_user backbone failure can either terminate a sim as `infrastructure_error` or
leak the raw error string ("API Error: ...") into a customer message. `_common.is_infra_broken`
detects both. `eval_persona.py` **re-runs** such sims (drops them from `results.json`, then tau2
`--auto-resume` regenerates exactly those tasks) up to `--max-infra-retries` (default 2); only
sims still broken afterwards are dropped from the score. The judge applies the same filter to
its denominator and selection, so a flaky turn can never falsely mark a persona `illegal`.

### Mutation (`auto_mutate.py`)

Fully automated LLM-based reflective mutation. After eval+judge+record+converge, if not
converged, `auto_mutate.py` calls Bedrock Opus-4.7 (extended thinking) to author 5 children:

**Inputs to the LLM prompt:**
1. All valid parents (sorted by score ascending — hardest first), with full persona texts
2. Score trends across all rounds (from `round_record.json`)
3. Severe failure details from ALL rounds (why past personas were judged illegal)
4. Reflective traces — conversation snippets where the agent succeeded (persona was too easy)
5. Direction pool + usage counts (prefer underrepresented directions)
6. Full library persona texts (for dedup awareness)
7. Constraints and invariant-writing best practices

**Parent selection guidance:** ~60% of children from the lowest-scoring parent, ~40% spread.
Each child uses a **distinct** `mutation_direction`, is a ~30% rewrite of its parent, and
must include an `## Invariants` section with anti-pattern guards.

Children are deduped via `add_children.py` (OpenAI embeddings cosine). Rejected children
trigger a retry with rejection reasons appended to the prompt (up to 3 retries).

---

## Evaluation details

`eval_persona.py` runs **two** `tau2 run` subprocesses (telecom, then banking_knowledge with
`--retrieval-config terminal_use`), each `--num-trials 1 --auto-resume`. It reads each
`data/simulations/<save_to>/results.json`, **drops infra-error sims**
(`termination_reason == infrastructure_error`, mirroring `compute_metrics`), takes the per-task
mean reward (keys prefixed `telecom::` / `banking::`), and sets
`persona_score = mean over all evaluated tasks`. `--auto-resume` makes re-running a round safe.

---

## Judge & validity (plan §5)

`judge_persona.py` calls the judge on **both** results.json, merges the per-sim verdicts, and
sets `status = illegal if severe_rate > 0.05`. The judge never sees or alters the success score.

**Failed-only pruning (default).** A *severe failure* (the persona cheated the agent into
failing) can only occur in a sim the agent actually FAILED — a fully successful sim
(`reward == 1.0`) is "ok" by construction. So by default the judge only scores the **failed**
sims (`reward < --success-threshold`, default 0.999), worst-reward first, capped at
`--max-failures` (default 10) per persona. The rate uses the **full** scored-sim count as the
denominator: `severe_rate = severe / n_scored` (all non-infra sims), so the 0.05 threshold keeps
its original meaning while we pay to judge only where severe failures can live (~3–4× fewer judge
calls). Pass `--judge-all` to judge every sim. When a persona has more failures than the cap,
`judge_status.json.truncated = true` flags that some failed sims went unjudged.

---

## Mutation & dedup (plan §8)

`add_children.py` checks each authored child against the **whole** library:
- reject if max cosine to any persona `>= 0.92` (exact / near / trivial / semantic duplicate);
- reject if cosine to any **illegal** persona `>= 0.85` (minor variant of an illegal one);
- otherwise accept; flag **gray-zone** (`>= 0.85`) with the nearest neighbor for Claude to sanity-check.

On rejection Claude regenerates with a different direction (up to 5 retries across re-runs); if
fewer than 3 unique children result, it adds what it has and the shortfall is logged. Accepted
children are appended as `status=untested, avg_success_rate=null, num_evaluations=0,
dedup_checked=true`. Use `--sim-method difflib` for offline runs (no embeddings API).

---

## Convergence (plan §9)

`converge.py` uses a simple avg-based strategy:

- **Best persona** = lowest `avg_success_rate` among valid personas with `>= min_evals` (default 3) evaluations
- **Early stop**: only when `untested == 0` (mutation has exhausted its ability to produce novel children — all dedup-rejected)
- **Hard cap**: `max_rounds` (default 20)
- **Final output**: persona with lowest avg, regardless of whether convergence was reached

No statistical significance testing — run enough rounds and let the running mean converge naturally via the exploit-retest slot.

---

## Per-iteration records

Every round leaves a self-contained audit trail under `rounds/round_NN/`: which personas were
`selected`, the `skill_manifest`, each persona's `persona_eval.json` (per-task rewards) and
`judge_status.json`, the consolidated `round_record.json` (scores + statuses), the
`mutation_context.json` Claude read, the `children_in.json` Claude authored, and
`mutation_log.json` (accepts/rejects/gray-zone/shortfall). The library and
`convergence_state.json` carry cumulative state across rounds.

---

## Smoke test (cheap → full)

```bash
PY=.venv/bin/python
# wiring only, zero API cost (synthetic results + verdicts, offline dedup):
$PY scripts/persona_opt/seed.py --from-skills impatient-rushing suspicious-withholding distracted-rambler
$PY scripts/persona_opt/select.py --round 0
$PY scripts/persona_opt/materialize.py --round 0
for p in p0001 p0002 p0003; do $PY scripts/persona_opt/eval_persona.py  --persona-id $p --round 0 --dry-run; done
for p in p0001 p0002 p0003; do $PY scripts/persona_opt/judge_persona.py --persona-id $p --round 0 --dry-run; done
$PY scripts/persona_opt/record_results.py --round 0
$PY scripts/persona_opt/mutation_context.py --round 0
# author children_in.json, then:
$PY scripts/persona_opt/add_children.py --round 0 --sim-method difflib
$PY scripts/persona_opt/converge.py --round 0
$PY scripts/persona_opt/report.py

# one real task per domain (exercises Bedrock + terminal_use shell), still cheap:
$PY scripts/persona_opt/eval_persona.py --persona-id p0001 --round 0 --max-tasks 1
$PY scripts/persona_opt/judge_persona.py --persona-id p0001 --round 0 --max-sims 2 --maj 1
```

Reset to a clean slate: delete `data/persona_opt/{lib.jsonl,rounds,convergence_state.json,final_report.*}`
and `cc_user_sim/personas/opt_*.skill`.

---

## Tunable parameters

| param | default | where |
|-------|---------|-------|
| num_trials | 1 | `eval_persona --num-trials` |
| max_concurrency | 3 | `eval_persona --max-concurrency` |
| validity threshold | 0.05 | `judge_persona --threshold` |
| judge maj | 3 | `judge_persona --maj` |
| judge max_failures | 10 | `judge_persona --max-failures` |
| judge success cutoff | 0.999 | `judge_persona --success-threshold` |
| judge all sims | off | `judge_persona --judge-all` |
| dedup cosine hard | 0.92 | `add_children --cos-threshold` |
| dedup illegal | 0.85 | `add_children --illegal-threshold` |
| max_rounds | 20 | `converge --max-rounds` |
| min_evals | 3 | `converge --min-evals` |
| n per round | 3 | `select --n` |
| bedrock profile | none | `eval_persona --bedrock-profile` |
| infra retries | 2 | `eval_persona --max-infra-retries`, `run_round --max-infra-retries` |
| eval concurrency | 3 | `run_round --eval-concurrency` |
| judge concurrency | 16 | `run_round --judge-concurrency`, `judge_persona --concurrency` |
```
