# Mutation Context 完整规格

本文档描述 persona 优化中 mutation prompt 的完整构建流程和各组成部分。

## 总览：Twophase 模式

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: 4 个并行 enriched mutation prompt                       │
│   每个看不同的 trace 子集，独立生成一个完整 persona proposal        │
│   model: --reflect-model (default=openai/gpt-5.4)               │
│   effort: --reflect-effort (default=medium)                     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: 1 个 aggregate prompt                                   │
│   把 4 个 proposal 综合成一个最终 persona                          │
│   model: --mutation-model (default=openai/gpt-5.4)              │
│   effort: --mutation-effort (default=xhigh)                     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
              diversity gate + lint gate
              (最多重试 8 次，Phase 1 缓存复用)
```

---

## Phase 1 Prompt 构成

每个 Phase 1 prompt = `_build_mutation_prompt()` + edit_pct override + `GEPA_MODE_OVERRIDE`

### 1. System Role & Task

```
You are an expert adversarial persona designer for a customer-support simulation benchmark.

## TASK
Modify the persona below IN-PLACE to make it HARDER for the support agent,
while keeping it VALID (realistic, in-character, no cheating).
```

### 2. Current Persona + Score

```
## CURRENT PERSONA (gepa)
Agent success rate: 79.0% (lower = harder = better)

```persona_text```
```

- `score` = agent_success_rate，从 traces 中统计 success / (success + genuine_failure)
- persona_text = parent candidate 的完整 persona markdown

### 3. Score History

```
## SCORE HISTORY (recent generations)
[最近 5 条 history entries，JSON 格式]
```

每条 entry 包含：
- `kind`: "minibatch_diagnostic" 或 "twophase_diagnostic"
- `batch_size`: trace 总数
- `noisy_agent_success_rate`: 当次 minibatch 上的 agent 成功率
- `n_success_traces`, `n_failure_traces`, `n_severe`
- `direction`: mutation_direction
- （enriched 模式）`hypotheses`: 3 个假设 + `selected_hypothesis`
- （twophase 模式）`proposal_directions`: Phase 1 的 4 个方向

**注意**：history 是 proposer 闭包内的列表变量，不持久化。Resume 后 history 为空。

### 4. Success Traces（Agent 成功的对话）

```
## CONVERSATION TRACES WHERE AGENT SUCCEEDED
These are conversations where the agent SOLVED the problem despite the persona.
Each trace shows the OPENING (first 4 messages) and the RESOLUTION (last 8 messages)
where the agent broke through the persona's resistance.

Study the RESOLUTION section carefully — that's where the persona failed:
- What agent strategy overcame the persona's resistance?
- At what point did the persona "give in" or provide the key info?
- What subtle behavior change would cause the agent to take the WRONG action instead?

[JSON array of trace objects]
```

每条 trace：
```json
{
  "task_id": "telecom_task_042",
  "n_turns": 0,
  "conversation": "OPENING (first 4 messages):\n[user] ...\n[assistant] ...\n\nRESOLUTION (last 8 messages):\n[user] ...\n[assistant] ..."
}
```

- **来源**：Plan C 后改为从 train minibatch 的 disk rollouts 收集（`_collect_train_traces()`），不再使用 val traces
- **数量**：twophase Phase 1 用 4 success + 2 failure，round-robin 分成 4 组，每组 2 success + 1 failure
- **snippet 格式**：`_format_snippet()` 生成，包含 opening（前 4 条消息）+ resolution（后 8 条消息），中间省略

### 5. Failure Traces（Agent 失败的对话，仅 enriched/twophase）

```
## CONVERSATION TRACES WHERE AGENT FAILED (persona was EFFECTIVE — PRESERVE these strategies)
Study what worked: the persona's misdirection/ambiguity caused the agent to take
the WRONG action or give up. Your mutation should KEEP these effective techniques
while fixing the weaknesses shown in the success traces above.

[JSON array of failure trace objects]
```

- 仅 enriched/twophase 模式包含
- 每组分到 1 条 failure trace

### 6. Severe Failures（历史非法行为）

```
## SEVERE FAILURES TO AVOID (past mutations triggered these judge violations)
The following behaviors were flagged as ILLEGAL by the judge. Your mutation MUST NOT
reproduce these patterns.

[JSON array of severe failure objects]
```

每条：
```json
{
  "task_id": "...",
  "domain": "telecom",
  "violations": [],
  "conversation_snippet": "前500字符的对话片段"
}
```

- 累积所有历史 gen 中被 judge 判定为 ILLEGAL_USER_BEHAVIOR 的 case
- `accumulated_severe` 在 proposer 闭包内累积，不持久化

### 7. Timeout Rate 警告

```
## TIMEOUT RATE: 12%
Conversations that hit the time/step limit are EXCLUDED from scoring — they do NOT count
as agent failures. A persona that "wins" by pure friction (stalling, endless tangents,
refusing to engage) produces timeouts, NOT genuine agent failures.

Do NOT add behaviors that:
- Create circular loops with no resolution path
- Add layer upon layer of delays that push conversations past the step limit
- ...

Instead, create difficulty through MISDIRECTION, AMBIGUITY, and SUBTLE REQUIREMENTS...
```

- 当 timeout_rate > 15% 时追加 ⚠️ WARNING 前缀

### 8. Mutation Rules

```
## MUTATION RULES
1. Rewrite {mutation_edit_pct} of the persona — be BOLD...
2. Target SPECIFIC agent strategies visible in the traces...
3. Make it harder for the agent to extract info or complete the task...
```

- `mutation_edit_pct` 默认 "40-50%"，通过 `--mutation-edit-pct` 配置

### 9. Validity Constraints

```
## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- When DIRECTLY asked for known info (name, DOB, account number), provide it accurately
- Never abandon the call while the problem is unresolved
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts
- Never fabricate facts not grounded in the scenario or tool results
- Never switch to a completely different goal mid-conversation
- Never accept outcomes contradicting explicit criteria
- Never make the conversation MEANINGLESS
```

### 10. Output Format（Enriched 模式的 hypothesis-driven）

```
## HYPOTHESIS-DRIVEN MUTATION
Before writing the mutation, reason step by step:
1. Analyze the success traces and failure traces to identify patterns
2. Propose exactly 3 distinct hypotheses for mutation directions
3. For each hypothesis, briefly predict how it would change agent behavior
4. Select the single most promising hypothesis and implement it fully

## OUTPUT FORMAT
Return ONLY a JSON object:
{
  "hypotheses": [3 个假设],
  "selected_hypothesis": 0|1|2,
  "mutation_direction": "snake_case",
  "reasoning": "2-3 sentences",
  "persona_text": "完整 persona markdown"
}
```

### 11. GEPA_MODE_OVERRIDE（追加在 prompt 末尾）

```
## GEPA MODE OVERRIDE (serial Pareto search)
The numeric score above is a noisy diagnostic — do NOT overfit to it.

Mutation constraints for this mode:
- Substantially rewrite {edit_pct} of the persona...
- Commit to ONE clear adversarial direction this time...
- The persona MUST stay a GENERAL, reusable behavioral profile...
- Preserve ALL validity-critical constraints...
```

### 12. Few-shot Examples（GEPA_MODE_OVERRIDE 的一部分）

```
## Few-shot: GENERAL (good) vs TASK-SPECIFIC (forbidden) rules
GOOD: "When the agent asks you to run a diagnostic, say you 'already tried that'..."
BAD:  "When the agent mentions order #A1003, insist the $49.99 refund never arrived..."
```

### 13. Invalid Cases Registry（GEPA_MODE_OVERRIDE 的一部分）

```
## Invalid cases observed in THIS run (do NOT reproduce)
- [lint] instructs illegal info-withholding
  """
  persona snippet (max 600 chars)
  """
```

- 运行中被 lint 或 judge 拒绝的 persona 片段，最多展示最近 4 条

### 14. Already Explored Personas（GEPA_MODE_OVERRIDE 的一部分）

```
## ALREADY EXPLORED personas (do NOT rehash these directions)
- gen1 dir=assumption_driven_shorthand_paraphrasing val_diff=0.240 (idx1, ACCEPTED)
  Summary: cooperative but compresses details into shorthand...
  [Core reporting habit] Lead with conclusion not evidence | [Where ambiguity comes from] If more than one item... | ...
- gen2 dir=household_proxy_identifier_collapse mb=0.180(parent=0.244) REJECTED: below_parent
  Summary: organizes problem around household context...
  [Core blind spot] Think in real-life categories | [How ambiguity appears] Lead with who uses it... | ...
```

每条包含：
- **header**: gen 编号 + mutation_direction + 分数/结果（ACCEPTED 显示 val_diff，REJECTED 显示 minibatch 分数 + 原因）
- **Summary**: 从 persona_text 提取的结构化摘要
  - opening：第一个实质性非标题行（≤100 chars）
  - modules：前 4 个 `##` section 的名称 + 首条 bullet（≤60 chars/bullet），跳过 Invariants 节
  - 用 `|` 分隔各模块

### 15. Retry Hint（diversity/lint gate 失败时）

```
- RETRY 2: the previous attempt was rejected because too similar to an existing persona
  (semantic_near_duplicate, lexical_dissim=93%)
  Most similar existing persona (DIFFERENTIATE from this):
  ```
  前 300 chars 的最相似 persona
  ```
  Produce a DISTINCT, valid, GENERAL persona that fixes this.
```

---

## Phase 2 Prompt 构成

Phase 2 = `build_aggregate_prompt()` + `GEPA_MODE_OVERRIDE`

### 1. System Role

```
You are an expert adversarial persona designer for a customer-support simulation benchmark.
```

### 2. Parent Persona

```
I provided an assistant with the following current persona instruction (gepa):
```parent_persona```
```

### 3. Proposals Section

```
4 parallel processes each independently proposed an updated persona based on
a different subset of evaluation traces. Here are the proposed updates:

### Proposal 1: terminology_confusion_symptom_first_reporting
**Reasoning**: In the success traces, the agent won because...
```proposal_1_persona_text```

### Proposal 2: partial_compliance_ambiguous_done
**Reasoning**: ...
```proposal_2_persona_text```
...
```

- 包含每个 Phase 1 proposal 的 direction + reasoning + 完整 persona_text

### 4. Synthesis Instruction

```
Your task is to synthesize these proposed persona updates into a single, comprehensive persona.
Incorporate the key improvements and specific insights from all proposals.
When proposals offer complementary guidance, include all relevant details.
When they conflict, prefer more specific and task-relevant guidance.
```

### 5. Validity Constraints + Output Format

与 Phase 1 相同的 validity constraints，但 output format 简化（无 hypothesis）：
```json
{
  "mutation_direction": "snake_case",
  "reasoning": "2-3 sentences: which proposal elements you combined and why",
  "persona_text": "完整 persona markdown"
}
```

### 6. GEPA_MODE_OVERRIDE

同 Phase 1，追加在末尾。

---

## Gate Pipeline（mutation 之后）

```
mutate_valid_diverse() 循环 (max 8 次):
  ├─ _mutate_once()           → 生成 candidate persona
  ├─ diversity_check()        → lexical Jaccard ≥ 0.30 + semantic cosine ≤ 0.90
  ├─ _struct_check()          → 检查是否有多个 # Persona 头
  ├─ lint_persona()           → LLM 静态检查 (natural/valid/no_task_specific)
  └─ 失败 → 构建 retry hint, 重跑 (twophase 复用 Phase 1 cache, 只重跑 Phase 2)
```

---

## 数据流总览

```
train tasks (30) ──sample──→ minibatch (5 tasks)
                              │
                    _collect_train_traces()
                    从 disk 加载 parent 在这些 task 上的历史 rollout
                              │
                    make_reflective_dataset()
                    把 trajectories 转成 {Inputs, Generated Outputs, Feedback} records
                              │
                    _extract_traces_and_metrics()
                    分类：success_traces / failure_traces / severe / timeout
                              │
                 ┌─────────── twophase ────────────┐
                 │  _distribute_traces_to_groups() │
                 │  4 success + 2 failure          │
                 │  → 4 groups (2s+1f each)        │
                 │                                 │
                 │  Phase 1: 4× enriched prompt    │
                 │  Phase 2: aggregate prompt      │
                 └─────────────────────────────────┘
                              │
                    diversity + lint gate (≤8 attempts)
                              │
                    minibatch eval (5 tasks × 2 trials)
                    child_mb ≥ parent_mb → pass
                              │
                    val confirm (20 tasks × 2 trials)
                    → Pareto front update
```
