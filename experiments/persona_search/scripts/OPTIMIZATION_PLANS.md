# Persona Optimization Loop: Plan A vs Plan B

## Overview

Both plans share the same objective: evolve adversarial user personas via (1+1)-ES
to minimize agent success rate while staying valid (no cheating). The key difference
is how they allocate compute between **mutation signal** (finding loopholes) and
**evaluation accuracy** (reliable scoring).

---

## Plan A: Full-Eval-Per-Round (Current Implementation)

### Loop Design

```
for each lineage (p0001, p0002, p0003) independently:
  for round = 1 .. max_rounds:
    1. Materialize current persona → .skill file
    2. Evaluate on 50 reprep tasks (35 telecom + 15 banking), 1 trial
    3. Judge all sims → valid / illegal / timeout_attack
    4. If illegal or timeout_attack → revert to last valid gen, re-mutate with feedback
    5. If valid → compare score to best_score
       - If best_score - score >= MIN_IMPROVEMENT_DELTA → accept as new best
       - Else → stalled_rounds += 1
    6. Collect success traces (first 4 + last 8 messages from agent-success sims)
    7. Feed traces + history to mutation LLM → generate next persona
    8. If stalled_rounds >= patience → converge

  Every 10 rounds: generalization check on full 211 tasks, 2 trials
  On convergence/exit: final generalization check if not already done
```

### Mechanism Details

#### 2. Evaluation (50 tasks, 1 trial)

- **Task set**: 35 telecom (top by discrimination index > 0.248) + 15 banking
  (disc > 0.567), selected from main matrix results. Stored in
  `representative_tasks_50mixed.json`.
- **Scoring**: `persona_score = mean(per_task_reward)` over scored sims.
  Timeout and max_steps sims are EXCLUDED from scoring (they don't reflect
  genuine agent failure). Infrastructure-error sims are also excluded.
- **Why 50 tasks**: Balances statistical power with compute cost. With binary
  rewards at p≈0.5, SE ≈ sqrt(0.25/50) ≈ 0.071.
- **Why 1 trial**: Cost constraint. 2 trials would halve SE but double compute.

#### 3. Judge

- Runs on ALL sims from the round (both telecom and banking).
- Uses `judge_user_intent_preservation.py` with 3-vote majority.
- `severe_failure_rate > 0.05` → illegal. Persona is reverted.
- Also checks `timeout_rate > 0.25` → treated as timeout_attack, same handling.

#### 5. Improvement Detection

- `MIN_IMPROVEMENT_DELTA = 0.08` (≈4 task net-flips out of 50).
- Rationale: With 1 trial on 50 tasks, paired comparison shows ~74-84%
  concordant tasks (same result regardless of persona). Only discordant pairs
  carry signal. 4 net-flips exceeds the expected noise from random variation.
- Problem: The best_score itself may be noise (single measurement, wide CI).

#### 6-7. Mutation

- Success traces use bookend format: first 4 messages (problem setup) + last 8
  messages (resolution / where persona gave in).
- Mutation LLM (opus-4.7) sees: current persona, score, history, traces,
  accumulated severe failure examples, timeout_rate.
- Must rewrite ~30% of persona. Constrained to cognitive friction techniques
  (wrong mental model, genuine confusion, emotional fixation). Must NOT
  fabricate facts or create impossible preconditions.

#### 8. Convergence

- `patience = 3`: stop after 3 consecutive rounds without improvement
  (delta >= 0.08).

#### Generalization Check

- Every 10 rounds (GEN_CHECK_INTERVAL = 10) + final.
- Full task set: 114 telecom (base split) + 97 banking = 211 tasks.
- 2 trials, retry 3rd if |trial1 - trial2| > 5%.
- Purpose: detect overfitting to reprep subset.

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| reprep tasks | 50 (35T + 15B) | Balance signal vs cost |
| trials per round | 1 | Cost constraint |
| MIN_IMPROVEMENT_DELTA | 0.08 | 4 net-flips, exceeds SE≈0.07 |
| patience | 3 | Allow recovery from stalled rounds |
| max_rounds | 15 | Budget constraint |
| GEN_CHECK_INTERVAL | 10 | Cost: full eval ≈ $90 per check |
| TIMEOUT_RATE_THRESHOLD | 0.25 | Catch friction-based strategies |
| max_severe_retries | 3 | Allow recovery from illegal mutations |
| max_success_traces | 12 | Enough for mutation LLM, not overwhelming |
| mutation rewrite % | ~30% | Enough change to close loopholes, preserve identity |

### Cost Model (per lineage)

```
Per round:      50 tasks × $0.30/sim (weighted avg) = $15
15 rounds:      15 × $15 = $225
Gen checks:     ~2 × 211 × 2 trials × $0.30 = $253
Judge+mutation: ~$20
────────────────────────────
Total per lineage: ~$498
Total (3 lineages): ~$1,494  +  baseline $122  =  ~$1,616
Wall time per round: ~30 min
```

### Strengths

1. Every round produces a statistically meaningful (if noisy) score.
2. Keep/revert decision is immediate — bad mutations are caught within 1 round.
3. Simple control flow: one score per round drives all decisions.

### Weaknesses

1. **Conflates two goals**: 50-task eval serves both mutation signal AND
   evaluation, but mutation only needs a few traces while evaluation needs
   many tasks. Compute is wasted on tasks that don't inform mutation.
2. **Slow iteration**: 30 min per round means only 15 rounds in ~8 hours.
   Mutation search space is explored slowly.
3. **Noisy best_score**: Single measurement with SE=0.07 means best_score
   has 95% CI of ±0.14. A "best" persona might just be a lucky roll.
4. **Overfitting risk**: Same 50 tasks every round. Mutation LLM sees
   the same traces and optimizes for the same loopholes.

---

## Plan B: Lightweight Mutation + Periodic Evaluation (Proposed)

### Core Insight

The mutation LLM doesn't need 50 tasks to find loopholes — it only looks at
success traces (capped at 12). The 50-task score is too noisy for reliable
per-round decisions anyway (SE=0.07, single trial). Better to:

- **Explore fast**: Small task samples per round for traces → more mutation rounds
- **Evaluate reliably**: Periodic checkpoints with full reprep set → robust scoring

### Loop Design

```
for each lineage independently:
  checkpoint_best_score = None
  checkpoint_best_gen = initial_gen

  for round = 1 .. max_rounds:
    1. Sample 10 tasks from reprep pool (random, no replacement within epoch)
    2. Evaluate on 10 tasks, 1 trial
    3. Quick-judge on 10 sims → catch obviously illegal personas early
    4. Collect success traces from the 10 sims
    5. Feed traces + history to mutation LLM → generate next persona
       (No keep/revert here — always move forward)

    Every CHECKPOINT_INTERVAL (5) rounds:
      6. Evaluate current gen on full 50 reprep tasks, 2 trials (+retry)
      7. Full judge on all sims → valid / illegal
      8. If illegal → revert to checkpoint_best_gen, re-mutate
      9. If valid → compare to checkpoint_best_score
         - If improved (delta >= 0.05) → update checkpoint_best
         - If not improved → revert to checkpoint_best_gen
      10. Reset mutation direction (avoid path dependency)

  Final: evaluate checkpoint_best_gen on full 211 tasks, 2 trials
```

### Mechanism Details

#### 1. Task Sampling (10 tasks per round)

- **Why 10**: Enough to produce 3-8 success traces for mutation signal.
  Too few (3-5) might miss important task categories. Too many wastes compute
  on evaluation that won't be used.
- **Sampling strategy**: Random sample from 50-task reprep pool, WITHOUT
  replacement within a 5-round epoch. Over 5 rounds, each task is seen once
  (50 tasks / 10 per round = 5 rounds = 1 epoch). This ensures:
  - No task is over-represented in mutation signal
  - Full coverage over each checkpoint interval
  - Different traces each round → diverse mutation pressure
- **Domain balance**: Maintain ratio 7T:3B per sample (7 telecom + 3 banking)
  to match reprep composition.

#### 3. Quick-Judge (10 sims)

- Purpose: Early detection of clearly illegal personas. Don't want to
  mutate 5 more times on top of an illegal base.
- Threshold: `severe_count >= 2` out of 10 sims (20%) → flag as likely
  illegal, revert immediately and re-mutate.
- This is a SOFT check. The checkpoint's full 50-sim judge is the real
  validity gate.
- Uses same judge pipeline but with lower statistical confidence (acceptable
  because it's just a safety catch, not the final verdict).

#### 5. Free-Running Mutation (between checkpoints)

- **Key difference from Plan A**: No keep/revert based on per-round score.
  The 10-task score is too noisy (SE ≈ 0.16) for any meaningful comparison.
- Instead, always accept the mutation and move forward. The mutation LLM
  sees what the agent succeeded at and closes those loopholes — no score
  needed for this.
- History fed to mutation LLM includes per-round traces but NOT per-round
  scores (which would be misleading noise). Only checkpoint scores are
  shown as "score history".
- This enables aggressive exploration: 5 consecutive mutations can compound
  to a substantially different persona, which is then evaluated at checkpoint.

#### 6-9. Checkpoint Evaluation (every 5 rounds)

- **50 reprep tasks, 2 trials**: SE ≈ sqrt(0.25/100) ≈ 0.05. Much more
  reliable than Plan A's single-trial 50-task eval (SE ≈ 0.07).
- **Retry logic**: If |trial1 - trial2| > 5%, run trial 3, take mean of all 3.
- **MIN_IMPROVEMENT_DELTA = 0.05**: Lower than Plan A's 0.08 because
  2-trial eval has lower variance. 0.05 ≈ 1 SE, conservative enough to
  filter noise while allowing gradual improvement.
- **Revert on no improvement**: If the 5 rounds of free-running mutation
  didn't improve over checkpoint_best, revert to checkpoint_best_gen and
  try a different mutation direction in the next epoch.
- **Judge**: Full judge on all 100 sims (50 tasks × 2 trials). `severe_rate
  > 0.05` → illegal, revert to checkpoint_best_gen.

#### Convergence

- `patience = 2` checkpoints (= 10 rounds) without improvement.
- Lower patience than Plan A because each checkpoint is more reliable
  (2 trials vs 1).

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| tasks per round | 10 (7T + 3B) | Enough traces for mutation, fast iteration |
| trials per round | 1 | Mutation signal only, no scoring decision |
| CHECKPOINT_INTERVAL | 5 | 1 epoch of task coverage before evaluation |
| checkpoint tasks | 50 (full reprep) | Same as Plan A reprep set |
| checkpoint trials | 2 (+retry) | SE ≈ 0.05, much more reliable |
| MIN_IMPROVEMENT_DELTA | 0.05 | ~2.5 task net-flips, matches SE of 2-trial eval |
| patience | 2 checkpoints | = 10 rounds, each checkpoint is high-confidence |
| max_rounds | 30 | Double Plan A: rounds are 3x cheaper |
| quick_judge_threshold | 2/10 severe | Soft catch for obviously illegal, not final |
| TIMEOUT_RATE_THRESHOLD | 0.25 | Same as Plan A, checked at checkpoint |
| max_success_traces | 12 | Same as Plan A |

### Cost Model (per lineage)

```
Per round:      10 tasks × $0.30 = $3
30 rounds:      30 × $3 = $90
Checkpoints:    6 × 50 tasks × 2 trials × $0.30 = $180
Final gen check: 211 × 2 × $0.30 = $127
Judge+mutation: ~$30
────────────────────────────
Total per lineage: ~$427
Total (3 lineages): ~$1,281  +  baseline $122  =  ~$1,403
Wall time per round: ~6 min (vs Plan A's 30 min)
Wall time per checkpoint: ~40 min (2 trials)
```

### Strengths

1. **2x more mutation rounds** (30 vs 15) for similar cost — explores
   persona space much more aggressively.
2. **More reliable evaluation**: 2-trial checkpoint (SE≈0.05) vs single-trial
   per-round (SE≈0.07). Reduces risk of "lucky noise" being crowned best.
3. **Task diversity**: Rotating 10-task samples prevent overfitting to
   specific tasks. Each epoch covers all 50 reprep tasks.
4. **Clean separation of concerns**: Mutation is driven by traces (qualitative),
   evaluation is driven by scores (quantitative). No conflation.
5. **Faster iteration**: 6 min per round enables rapid exploration. Bad
   directions are explored cheaply and pruned at checkpoint.

### Weaknesses

1. **Delayed feedback**: Bad mutations aren't caught for up to 5 rounds
   (unless obviously illegal). Could waste 5 rounds heading in a wrong direction.
   Mitigation: quick-judge catches the worst cases immediately.
2. **Compound drift**: 5 consecutive unchecked mutations could drift far
   from a good persona. Mitigation: revert to checkpoint_best on failure.
3. **10-task sample may miss important loopholes**: If a persona fails on
   specific task types not in the sample, mutation won't address them.
   Mitigation: rotating samples ensure full coverage per epoch.
4. **More complex control flow**: Two-tier evaluation (per-round quick +
   checkpoint full) is harder to implement and debug than Plan A's uniform
   per-round eval.

---

## Side-by-Side Comparison

| Dimension | Plan A | Plan B |
|-----------|--------|--------|
| Tasks per round | 50 | 10 |
| Round purpose | Eval + mutation signal | Mutation signal only |
| Score per round | Yes (noisy, SE≈0.07) | No (too noisy at 10 tasks) |
| Keep/revert | Per round (delta≥0.08) | Per checkpoint (delta≥0.05) |
| Checkpoint eval | Every 10 rounds, full set | Every 5 rounds, reprep 50×2 trials |
| Eval reliability | Low (1 trial, SE≈0.07) | High (2 trials, SE≈0.05) |
| Rounds possible | 15 | 30 |
| Cost per lineage | ~$498 | ~$427 |
| Wall time per round | ~30 min | ~6 min |
| Overfitting risk | Higher (same 50 tasks) | Lower (rotating samples) |
| Feedback latency | 1 round | 5 rounds |
| Implementation | Current code | Requires refactor |

## Task Sampling Detail (Plan B)

```
reprep_pool = 50 tasks (35 telecom + 15 banking)

epoch = 5 rounds (= CHECKPOINT_INTERVAL)
per_epoch:
  shuffle(reprep_pool)
  round 1: tasks[0:10]   (7T + 3B)
  round 2: tasks[10:20]  (7T + 3B)
  round 3: tasks[20:30]  (7T + 3B)
  round 4: tasks[30:40]  (7T + 3B)
  round 5: tasks[40:50]  (7T + 3B)
  → checkpoint: evaluate ALL 50 tasks, 2 trials
```

Each task appears exactly once per epoch. Over one epoch + checkpoint,
every task is seen 3 times total (1 in mutation round + 2 at checkpoint).

## Recommendation

**Plan B is preferred** for this use case because:

1. The mutation LLM's primary input is success traces, not scores. It doesn't
   need 50 tasks to identify patterns — 10 tasks producing 3-8 traces is sufficient.
2. The per-round score in Plan A (SE≈0.07) is too noisy for reliable keep/revert
   decisions. Plan B acknowledges this by deferring scoring to checkpoints.
3. Double the mutation rounds means faster exploration and more chances to
   find effective adversarial strategies.
4. 2-trial checkpoint evaluation gives a more trustworthy best_score, reducing
   the "lucky noise" problem identified in the current run (p0001 gen 2).
5. Rotating task samples reduce overfitting to the reprep subset.

The main risk (delayed feedback) is mitigated by the quick-judge safety net
and by keeping the checkpoint interval short (5 rounds = ~30 min of mutation
before a reliable evaluation).
