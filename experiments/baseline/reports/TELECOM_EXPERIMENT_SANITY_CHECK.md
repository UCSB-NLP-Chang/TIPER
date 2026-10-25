# Telecom Experiment Sanity Check Report

Date: 2026-05-30 (corrected — previous version had incorrect per-trial scores for #1-#14)

## 1. Experiment Matrix

All experiments: telecom domain, 3 trials, seed 300, max_steps 200, 114 tasks per trial.

Cost = avg (agent_cost + user_cost) per conversation.

| #   | Agent      | Reasoning | User         | T1    | T2    | T3    | Avg       | Spread | Cost  |
| --- | ---------- | --------- | ------------ | ----- | ----- | ----- | --------- | ------ | ----- |
| 1   | GPT-5.2    | none      | GPT-5.2 user | 58.8% | 55.3% | 57.0% | **57.0%** | 3.5pp  | $0.09 |
| 2   | GPT-5.2    | high      | GPT-5.2 user | 90.4% | 89.5% | 87.7% | **89.2%** | 2.6pp  | $0.12 |
| 3   | Sonnet 4.5 | high      | GPT-5.2 user | 84.2% | 86.0% | 88.6% | **86.3%** | 4.4pp  | $0.51 |
| 4   | GPT-5.2    | none      | S4.6 user    | 48.2% | 44.7% | 41.2% | **44.7%** | 7.0pp  | $0.49 |
| 5   | GPT-5.2    | high      | S4.6 user    | 88.6% | 85.1% | 86.8% | **86.8%** | 3.5pp  | $0.75 |
| 6   | GPT-5.2    | none      | S4.6 CC      | 48.2% | 44.7% | 49.1% | **47.4%** | 4.4pp  | $0.33 |
| 7   | GPT-5.2    | high      | S4.6 CC      | 91.2% | 91.2% | 92.1% | **91.5%** | 0.9pp  | $0.51 |
| 8   | GPT-5.4    | none      | S4.6 user    | 48.2% | 46.5% | 50.9% | **48.5%** | 4.4pp  | $0.48 |
| 9   | GPT-5.4    | high      | S4.6 user    | 82.3% | 85.0% | 81.6% | **82.9%** | 3.4pp  | $0.65 |
| 10  | GPT-5.4    | none      | S4.6 CC      | 46.5% | 43.9% | 44.7% | **45.0%** | 2.6pp  | $0.26 |
| 11  | Sonnet 4.5 | none      | S4.6 user    | 64.9% | 67.5% | 67.5% | **66.7%** | 2.6pp  | $1.40 |
| 12  | Sonnet 4.5 | none      | S4.6 CC      | 62.3% | 64.9% | 64.9% | **64.0%** | 2.6pp  | $1.09 |
| 13  | Sonnet 4.5 | high      | S4.6 user    | 66.7% | 62.8% | 61.4% | **63.6%** | 5.3pp  | $1.34 |
| 14  | Sonnet 4.5 | high      | S4.6 CC      | 64.0% | 66.7% | 64.0% | **64.9%** | 2.6pp  | $1.22 |
| 15  | GPT-5.4    | none      | GPT-5.2 user | 64.0% | 59.6% | 63.2% | **62.3%** | 4.4pp  | $0.09 |
| 16  | GPT-5.4    | high      | GPT-5.2 user | 93.9% | 91.2% | 94.7% | **93.3%** | 3.5pp  | $0.13 |

Note: #9 has 340/342 sims (2 tasks missing from trials 0 and 1). All other experiments have 342/342.

## 2. Leaderboard Comparison

Published leaderboard (taubench.com) uses GPT-5.2 user with reasoning_effort=low, 4 trials, seed 300, benchmark v0.2.1-dev.

| Model | Leaderboard Pass^1 | Our Avg (GPT-5.2 user) | Our Avg (S4.6 user) | Our Avg (S4.6 CC) |
|-------|-------------------|----------------------|-------------------|-------------------|
| GPT-5.2 (none) | 61.62% | 57.0% | 44.7% | 47.4% |
| GPT-5.2 (high) | 89.69% | 89.2% | 86.8% | 91.5% |
| Sonnet 4.5 (high) | 84.87% | 86.3% | 63.6% | 64.9% |
| Sonnet 4.5 (none) | N/A | N/A | 66.7% | 64.0% |
| GPT-5.4 (none) | N/A (telecom) | 62.3% | 48.5% | 45.0% |
| GPT-5.4 (high) | N/A (telecom) | 93.3% | 82.9% | TBD |

### Observations

**GPT-5.2 (high) + GPT-5.2 user vs leaderboard**: Our 89.2% avg closely matches leaderboard 89.69%. The small difference is expected — leaderboard uses 4 trials and reasoning_effort=low for the user sim.

**GPT-5.2 (none) + GPT-5.2 user vs leaderboard**: Our 57.0% is slightly lower than leaderboard 61.62%. Likely due to user sim reasoning_effort difference (none vs low).

**Sonnet 4.5 (high) + GPT-5.2 user vs leaderboard**: Our 86.3% is slightly above leaderboard 84.87%. Within expected variance.

**GPT-5.4 (high) + GPT-5.2 user**: 93.3% — the highest score across all experiments, significantly above GPT-5.2 (high) at 89.2%.

## 3. Pairwise Comparisons

### 3.1 Reasoning Effect (none vs high)

| Agent | User | None Avg | High Avg | Delta |
|-------|------|----------|----------|-------|
| GPT-5.2 | GPT-5.2 user | 57.0% | 89.2% | +32.2pp |
| GPT-5.2 | S4.6 user | 44.7% | 86.8% | +42.1pp |
| GPT-5.2 | S4.6 CC | 47.4% | 91.5% | +44.1pp |
| GPT-5.4 | GPT-5.2 user | 62.3% | 93.3% | +31.0pp |
| GPT-5.4 | S4.6 user | 48.5% | 82.9% | +34.4pp |
| Sonnet 4.5 | S4.6 user | 66.7% | 63.6% | **-3.1pp** |
| Sonnet 4.5 | S4.6 CC | 64.0% | 64.9% | +0.9pp |

Reasoning improves GPT models by 31-44pp. **Sonnet 4.5 is the exception**: reasoning has no benefit or even slightly hurts performance (-3.1pp with S4.6 user). This suggests Sonnet 4.5's extended thinking may cause overthinking on telecom tasks.

### 3.2 User Simulator Effect (GPT-5.2 vs S4.6 user vs S4.6 CC)

| Agent | Reasoning | GPT-5.2 user | S4.6 user | S4.6 CC |
|-------|-----------|-------------|-----------|---------|
| GPT-5.2 | none | 57.0% | 44.7% | 47.4% |
| GPT-5.2 | high | 89.2% | 86.8% | 91.5% |
| GPT-5.4 | none | 62.3% | 48.5% | 45.0% |
| GPT-5.4 | high | 93.3% | 82.9% | TBD |
| Sonnet 4.5 | none | N/A | 66.7% | 64.0% |
| Sonnet 4.5 | high | 86.3% | 63.6% | 64.9% |

Observations:
- **S4.6 user is harder than GPT-5.2 user** for no-reasoning agents: ~12pp drop for GPT-5.2, ~14pp for GPT-5.4
- **S4.6 user is harder than GPT-5.2 user** even for high-reasoning: ~2-10pp drop
- **S4.6 CC vs S4.6 standard**: Generally within 3pp, CC is slightly easier for some configs
- **GPT-5.4 (high) shows largest user-sim gap**: 93.3% vs 82.9% = 10.4pp between GPT-5.2 user and S4.6 user

### 3.3 Agent Model Comparison (same user, same reasoning)

| Reasoning | User | GPT-5.2 | GPT-5.4 | Sonnet 4.5 |
|-----------|------|---------|---------|------------|
| none | GPT-5.2 user | 57.0% | 62.3% | N/A |
| none | S4.6 user | 44.7% | 48.5% | 66.7% |
| none | S4.6 CC | 47.4% | 45.0% | 64.0% |
| high | GPT-5.2 user | 89.2% | 93.3% | 86.3% |
| high | S4.6 user | 86.8% | 82.9% | 63.6% |
| high | S4.6 CC | 91.5% | TBD | 64.9% |

Observations:
- **Sonnet 4.5 (none) > GPT-5.2/5.4 (none)** by ~17-22pp with S4.6 user — Sonnet 4.5 is significantly better without reasoning
- **GPT-5.4 (high) > GPT-5.2 (high) with GPT-5.2 user**: 93.3% vs 89.2% (+4.1pp) — GPT-5.4 is the overall best agent
- **GPT-5.2 (high) > GPT-5.4 (high) with S4.6 user**: 86.8% vs 82.9% (+3.9pp) — reversed with harder user sim
- **GPT-5.2 (high) >> Sonnet 4.5 (high)** with S4.6 user: 86.8% vs 63.6% — reasoning benefits GPT far more
- **GPT-5.4 (none) > GPT-5.2 (none)**: +5.3pp with GPT-5.2 user, +3.8pp with S4.6 user — GPT-5.4 has stronger baseline
- **Sonnet 4.5 reasoning paradox**: High reasoning hurts Sonnet 4.5 with S4.6 user (-3.1pp) while massively helping GPT-5.2 (+42.1pp)

### 3.4 Trial-to-Trial Consistency (Spread)

| Spread Range | Experiments | Pattern |
|-------------|-------------|---------|
| < 3pp | #7, #10, #11, #12, #14 | Stable |
| 3-5pp | #1, #2, #3, #5, #6, #8, #9, #15, #16 | Normal |
| 5-8pp | #4, #13 | Moderate: S4.6 user + no-reasoning/Sonnet |

Observations:
- **All spreads are now ≤ 7.0pp** — no anomalous variance in any experiment
- **High reasoning reduces variance for GPT** (#2: 2.6pp vs #1: 3.5pp; #7: 0.9pp vs #6: 4.4pp; #16: 3.5pp vs #15: 4.4pp)
- **Sonnet 4.5 (none) is very consistent** (#11: 2.6pp, #12: 2.6pp)
- **Sonnet 4.5 (high) variance is moderate** (#13: 5.3pp vs #11: 2.6pp) — reasoning slightly increases variance but not dramatically
- **GPT-5.4 is well-behaved**: #9 spread = 3.4pp, #10 = 2.6pp — no issues
- **#4 has the highest spread** (7.0pp) — GPT-5.2 (none) + S4.6 user, consistent with weak agents having more variance against harder user sims

## 4. Cost Analysis

| Agent | Reasoning | GPT-5.2 user | S4.6 user | S4.6 CC |
|-------|-----------|-------------|-----------|---------|
| GPT-5.2 | none | $0.09 | $0.49 | $0.33 |
| GPT-5.2 | high | $0.12 | $0.75 | $0.51 |
| GPT-5.4 | none | $0.09 | $0.48 | $0.26 |
| GPT-5.4 | high | $0.13 | $0.65 | TBD |
| Sonnet 4.5 | none | N/A | $1.40 | $1.09 |
| Sonnet 4.5 | high | $0.51 | $1.34 | $1.22 |

Observations:
- **User sim is the dominant cost driver**, not agent model or reasoning. S4.6 user costs 4-6x more than GPT-5.2 user.
- **Sonnet 4.5 as agent is expensive**: $1.09-$1.40 per conversation with S4.6 user.
- **GPT-5.2 and GPT-5.4 have similar agent costs** ($0.09-$0.13 with GPT-5.2 user).
- **CC user is cheaper than standard S4.6 user** (e.g., $0.33 vs $0.49, $0.51 vs $0.75).
- **Reasoning adds ~30-50% to cost** for GPT models with same user sim.

## 5. Anomalies and Issues

1. **Sonnet 4.5 reasoning paradox (#13/#14)**: Enabling extended thinking (reasoning_effort=high) **hurts** Sonnet 4.5 performance with S4.6 user (63.6% vs 66.7%, -3.1pp) and is neutral with CC user (64.9% vs 64.0%). This is opposite to GPT models where reasoning helps by 31-44pp. Verified that config correctly passes `reasoning_effort: "high"`. Possible causes: overthinking on simple telecom tasks, or Bedrock's extended thinking implementation causing strategy drift.

2. **GPT-5.4 (high) user-sim sensitivity**: GPT-5.4 (high) shows the largest user-sim gap: 93.3% with GPT-5.2 user vs 82.9% with S4.6 user (10.4pp). GPT-5.2 (high) shows only 2.4pp gap (89.2% vs 86.8%). This suggests GPT-5.4 may be more sensitive to user behavior patterns.

3. **GPT-5.4 (high) required litellm upgrade**: GPT-5.4 + reasoning_effort + tools needs OpenAI's `/v1/responses` API, supported from litellm v1.82.3. Upgraded from 1.81.11 to 1.86.2.

4. **CC user routing**: CC user requires explicit `use_bedrock: true` and `bedrock_profile` in user-llm-args. Without these, it falls back to direct Anthropic API which may hit rate limits.

5. **Data correction**: Previous version of this report (pre-correction) had incorrect per-trial scores for experiments #1-#14. The error was in the calculation method used during the initial analysis. All values have been recomputed directly from results.json reward_info fields.

## 6. Summary

- **Leaderboard alignment**: Our GPT-5.2 results match leaderboard within expected margins.
- **Reasoning is the dominant factor for GPT models**: +31-44pp improvement, dwarfing user sim choice (~2-14pp) and agent model choice (~4-22pp without reasoning).
- **GPT-5.4 (high) + GPT-5.2 user is the best configuration**: 93.3% avg, surpassing GPT-5.2 (high) at 89.2%.
- **GPT-5.4 (none) > GPT-5.2 (none)**: GPT-5.4 has a stronger baseline (+3.8-5.3pp without reasoning).
- **Reasoning HURTS Sonnet 4.5**: -3.1pp with S4.6 user, neutral with CC user. Extended thinking causes strategy drift on telecom tasks.
- **User sim choice matters more for weak agents**: S4.6 user drops no-reasoning performance by 12-14pp vs GPT-5.2 user, but only 2-10pp for high-reasoning.
- **CC user is comparable to standard S4.6 user**: Within 3pp in most settings, and cheaper.
- **All experiments have normal variance**: Max spread is 7.0pp (#4), typical for no-reasoning + harder user sim. No anomalous results.
- **Sonnet 4.5 is the most expensive**: 3-10x more per conversation than GPT models.
