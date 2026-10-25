# GEPA Integration for Persona Optimization

## Context

The current persona optimization loop (`lineage_loop.py`) runs 3 independent (1+1)-ES lineages,
each evaluating 50 reprep tasks per round with custom mutation/selection logic.
Cost: ~$498/lineage × 3 = ~$1,500. Weaknesses: noisy single-trial scoring (SE≈0.07),
conflation of mutation signal with evaluation, slow iteration (30 min/round), overfitting
risk (same 50 tasks every round).

GEPA (Generative Evolutionary Prompt Adaptation, v0.1.1, already installed) replaces the
custom loop. Its per-task-instance Pareto front replaces 3 independent lineages. Its
evaluation cache prevents re-evaluating the same persona on the same task. Its reflective
mutation + minibatch acceptance + full valset eval pipeline is a more principled version
of our current mutate/evaluate/accept/revert logic.

Goal: validity as a soft multi-objective (alongside difficulty) tracked via Pareto front,
GEPA managing mutation/selection, our existing eval pipeline for scoring.

---

## Architecture

```
gepa_runner.py  (CLI entry point)
  └─> gepa.optimize(adapter=TauPersonaAdapter(...), ...)
        ├─> adapter.evaluate(batch, candidate)  → tau2 subprocess → scores + traces
        ├─> adapter.make_reflective_dataset()   → bookend success traces + severe failures
        └─> custom_candidate_proposer()         → domain-specific mutation prompt (reused)
```

### GEPA Flow Per Iteration

1. Select parent from Pareto front
2. Sample 10 tasks from trainset (50 reprep, epoch-shuffled)
3. Evaluate parent on 10 tasks (cached from prior iteration)
4. Extract traces → build reflective dataset
5. Custom mutation: domain-specific prompt → new persona text
6. Evaluate new candidate on 10 tasks
7. Acceptance: `sum(new_scores) > sum(old_scores)` on minibatch?
   - Yes → full valset eval (50 tasks, cached where possible) → add to Pareto front
   - No → reject, next iteration
8. Stop when budget exhausted (`max_metric_calls=500`)

---

## Files to Create

### A. `scripts/persona_opt/tau_persona_adapter.py` (~250 lines)

Implements `GEPAAdapter[TauTask, TauTrajectory, TauOutput]`.

#### Types

```python
@dataclass
class TauTask:
    task_id: str  # e.g. "[mms_issue]airplane_mode_on|..."
    domain: str  # "telecom" or "banking"


@dataclass
class TauTrajectory:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
    messages: list[dict]
    judge_verdict: str | None
    judge_violations: list[dict]


@dataclass
class TauOutput:
    task_id: str
    domain: str
    reward: float | None
    termination_reason: str
```

#### `evaluate(batch, candidate, capture_traces)`

1. Materialize `candidate["persona_text"]` → single `.skill` file (`gepa_eval.skill`)
2. Group batch by domain → run tau2 subprocess per domain
   - Reuse `_build_cmd` and `_run_domain_with_retry` from `eval_persona.py` (direct import)
   - Each call gets a unique run label via monotonic counter: `gepa_opt/eval_{N:04d}/{domain}`
3. Parse `results.json` → per-task scores: `1 - reward` (higher = harder = better for GEPA)
   - Timeout/max_steps → `0.0` (discourages timeout-farming)
   - Infra error → `0.0`
4. If `capture_traces=True`:
   - Run judge → populate `TauTrajectory.judge_verdict`
   - Set `objective_scores = [{"difficulty": 1-reward, "validity": 0.0 if severe else 1.0}]`
5. If `capture_traces=False`:
   - Skip judge, set `validity=1.0` for all tasks (post-hoc filtering later)

#### `make_reflective_dataset(candidate, eval_batch, components_to_update)`

For `"persona_text"` component:
- Extract success traces (score < 1.0 = agent succeeded → persona failed) using bookend
  format (first 4 + last 8 messages) — same pattern as `collect_success_traces` in lineage_loop.py
- Extract severe failures from judge verdicts
- Format per GEPA protocol:
  ```json
  {
    "Inputs": {"task_id": "...", "domain": "...", "difficulty_score": 0.0},
    "Generated Outputs": {"conversation_snippet": "...bookend..."},
    "Feedback": "Agent SUCCEEDED. Persona failed to create enough difficulty because..."
  }
  ```

#### Custom Mutation (via `custom_candidate_proposer`)

Passed to `gepa.optimize()` as `custom_candidate_proposer`. Wraps the domain-specific
mutation prompt from `lineage_loop.py`:

```python
def custom_proposer(candidate, reflective_dataset, components_to_update):
    persona_text = candidate["persona_text"]
    records = reflective_dataset.get("persona_text", [])

    # Extract success traces + severe failures from reflective dataset records
    success_traces = [r for r in records if r["Inputs"]["difficulty_score"] < 1.0]
    severe_failures = [r for r in records if "severe" in (r.get("Feedback") or "")]

    # Reuse _build_mutation_prompt from lineage_loop.py
    prompt = _build_mutation_prompt(
        persona_text, score, traces, history, "gepa", severe_failures=severe_failures
    )
    result = call_mutation_llm(prompt, mutation_model)
    return {"persona_text": result["persona_text"]}
```

This keeps all domain-specific knowledge: validity constraints, effective techniques
(wrong mental model, emotional fixation), timeout avoidance, severe failure anti-patterns.

### B. `scripts/persona_opt/gepa_runner.py` (~150 lines)

CLI entry point:

```bash
python scripts/persona_opt/gepa_runner.py \
  --seed-persona cc_user_sim/personas/opt_p0001.skill \
  --max-metric-calls 500 \
  --minibatch-size 10 \
  --run-dir data/persona_opt/gepa_runs/run_001
```

Flow:
1. Load 50 reprep tasks → `TauTaskDataLoader` (implements GEPA's `DataLoader` protocol
   with string task_id keys)
2. Load seed persona text → `{"persona_text": text}`
3. Create `TauPersonaAdapter` with model/infra config
4. Call `gepa.optimize(seed_candidate, trainset=loader, valset=loader, adapter=adapter, ...)`
5. Post-hoc: run full judge on Pareto front candidates, filter for validity ≥ 0.95,
   write best personas to disk

### `TauTaskDataLoader`

GEPA's `DataLoader[str, TauTask]` with string IDs for readable Pareto tracking and cache keys:

```python
class TauTaskDataLoader:
    def __init__(self, tasks: list[TauTask]):
        self._tasks = {t.task_id: t for t in tasks}
        self._ordered_ids = [t.task_id for t in tasks]

    def all_ids(self) -> list[str]:
        return list(self._ordered_ids)

    def fetch(self, ids: Sequence[str]) -> list[TauTask]:
        return [self._tasks[tid] for tid in ids]

    def __len__(self) -> int:
        return len(self._tasks)
```

---

## Code Reuse Map

| Source | What | How |
|--------|------|-----|
| `_common.py` | `REPO_ROOT`, `SIMULATIONS_DIR`, `PERSONAS_DIR`, `load_rep_tasks`, `is_infra_broken`, `read_json` | Direct import |
| `eval_persona.py` | `_build_cmd`, `_run_domain_with_retry`, `_aggregate_results` | Direct import |
| `lineage_loop.py` | `_build_mutation_prompt`, `call_mutation_llm`, `materialize`, `collect_success_traces` | Direct import |
| `judge_persona.py` | Judge subprocess (sync mode) | Subprocess call for `capture_traces=True` evals |

**Replaced by GEPA** (not reused): `run_one_lineage`, `run_lineage_loop`, state management,
convergence/patience logic, keep/revert logic, `_locked_write_json`.

---

## GEPA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `trainset` / `valset` | Same 50 reprep tasks | Limited task pool; trainset=valset OK |
| `reflection_minibatch_size` | 10 | Enough for 3-8 success traces |
| `max_metric_calls` | 500 | ~$150 in sims, ~10 full valset evals |
| `candidate_selection_strategy` | `"pareto"` | Replaces 3 independent lineages |
| `frontier_type` | `"instance"` | Per-task Pareto front (diversity across tasks) |
| `cache_evaluation` | `True` | Never re-evaluate same persona on same task |
| `skip_perfect_score` | `False` | Still mutate even if minibatch is fully difficult |
| `perfect_score` | `1.0` | Max difficulty score |
| `raise_on_exception` | `False` | Individual tau2 failures shouldn't crash loop |
| `use_merge` | `False` | Start simple; enable later if Pareto front is diverse |
| `batch_sampler` | `"epoch_shuffled"` | Rotating task samples prevent overfitting |
| `reflection_lm` | `"bedrock/us.anthropic.claude-opus-4-7"` | Same as current mutation model |

---

## Cost Comparison

```
GEPA (single run, 500 eval budget):
  Sims:     500 × $0.30/sim  = $150
  Judge:    ~100 × $0.08     = $8   (only on capture_traces=True minibatches)
  Mutation: ~15 × $0.50      = $8
  Post-hoc: ~200 × $0.08     = $16  (judge on Pareto front candidates)
  ─────────────────────────────────
  Total: ~$180

Current system (3 lineages):
  Total: ~$1,500

Savings: ~8x cheaper
```

---

## Key Design Decisions

### 1. Score Polarity
GEPA uses higher=better. Our score = `1 - reward` → higher = harder = better.
Agent success → score 0 (bad for our objective). Agent failure → score 1 (good).

### 2. Timeout Handling
Score `0.0` for timeout/max_steps sims. This penalizes timeout-farming personas — they
get low difficulty scores on timeout tasks, so the Pareto front won't favor them.

### 3. Validity as Multi-Objective
`objective_scores[i] = {"difficulty": 1-reward, "validity": 0.0 if severe else 1.0}` per task.
GEPA's Pareto front logs these. Post-hoc filtering picks candidates with mean validity ≥ 0.95.
This replaces the hard "illegal → revert" constraint with a Pareto-tracked soft objective.

### 4. Judge Cost Optimization
Only run judge during reflective mutation evals (`capture_traces=True`), not during
acceptance/valset evals. Post-hoc full judge on Pareto front candidates. Reduces judge
calls from ~500 to ~100 + post-hoc ~200.

### 5. Single Persona ID
Use `gepa_eval` as persona_id for all GEPA evaluations. GEPA is single-threaded for
evaluations, so no race condition on the `.skill` file.

### 6. Custom Mutation Over GEPA Default
GEPA's generic `InstructionProposalSignature` doesn't know about our validity constraints,
effective persona techniques, or timeout avoidance. Our `_build_mutation_prompt` is reused
via `custom_candidate_proposer` to keep all domain expertise.

---

## Implementation Sequence

1. **Types + DataLoader** — `TauTask`, `TauTrajectory`, `TauOutput` dataclasses;
   `TauTaskDataLoader` implementing GEPA's `DataLoader[str, TauTask]` protocol
2. **`evaluate()` method** — materialize + tau2 subprocess + result parsing + scoring
3. **`make_reflective_dataset()`** — bookend trace extraction, severe failure collection
4. **Custom proposer** — wrap `_build_mutation_prompt` + `call_mutation_llm`
5. **`gepa_runner.py`** — CLI, wire up `gepa.optimize()`, post-hoc analysis
6. **Judge integration** — judge on capture_traces evals, post-hoc on Pareto front

---

## Verification

1. **Smoke test** (`--dry-run`): Mock `subprocess.run` with fake results. Validate GEPA
   engine starts/iterates/stops, DataLoader works, adapter methods are called correctly.
2. **Minimal live** (`--max-metric-calls 4 --minibatch-size 2`): 2 tasks, ~$1. Validates
   real tau2 subprocess + parsing.
3. **Full run** (`--max-metric-calls 500`): Compare to lineage_loop baseline — expect
   comparable difficulty scores at ~8x lower cost.
4. **Resumability**: GEPA state saves to `run_dir`; restart with same `--run-dir` to
   resume from cache.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GEPA assumes cheap evals | Our evals are 3-30 min each | `cache_evaluation=True` + limited budget (500 calls). Wall time ~8h. |
| No SubsetEvaluationPolicy | Full valset (50 tasks) on every acceptance = ~$15/acceptance | Acceptable within budget (~10 acceptances = $150) |
| `lineage_loop.py` private function imports | `_build_mutation_prompt` etc. are `_` prefixed | Same package directory, add comment noting dependency |
| Persona `.skill` file race condition | Other processes could corrupt it | Use dedicated `gepa_eval` persona_id that nothing else uses |
| GEPA's generic reflective dataset format | May lose domain-specific context | Custom `make_reflective_dataset` includes bookend traces, judge verdicts, domain info |
