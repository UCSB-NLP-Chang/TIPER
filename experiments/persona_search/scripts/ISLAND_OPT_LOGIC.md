# Island-Model Persona Optimization — 当前逻辑说明

## 目标

对抗性 persona 搜索：找到让 agent (gpt-5.4) 在 telecom 任务上**失败率最高**的 user persona，同时保持 persona 行为的**合法性**（不违反评测规则）。

## 架构：5 岛并行

5 个独立的 `parallel_gepa.py` 进程，每个以一个 brinks seed 为起点（brinks_00..04），各自维护**隔离的 candidate pool**，但共享一个**文件锁 seen pool** 确保跨岛 persona 唯一（embedding cosine + lexical 3-gram Jaccard dedup）。

总并发 = 5 岛 × `--max-concurrency 5` = 25 个 sim 同时跑。

## 数据划分

114 个 telecom 任务按 (issue_type × PERSONA_tag) 分层切分，seed=42：

| Split | 数量 | 用途 |
|-------|------|------|
| **train** | 30 | minibatch 采样源（预筛用） |
| **val** | 30 | selection / Pareto frontier（决定是否接受 child） |
| **test** | 54 | 搜索期间不碰，事后泛化评估 |

三 split 共享同一 issue/难度 mix（分层均衡）。val→test 落差是纯泛化信号。test 故意做大(~47%) 求低方差泛化裁决。

## 模型配置

| 角色 | 模型 | 备注 |
|------|------|------|
| Agent (SUT) | gpt-5.4 | reasoning_effort=high |
| User simulator | cc_user_con / claude-opus-4-7 | persistent claude subprocess, MCP tool relay |
| Mutation (proposer) | openai/gpt-5.4 | reasoning_effort=xhigh |
| Judge (validity) | gpt-5.2 | failed-only judging, maj1 |
| Lint (预检) | gpt-5.4-mini | maj1 |
| Embedding (dedup) | text-embedding-3-small | dim=1536 |

认证路由：user simulator 走 Anthropic→Vertex shim (:4112)；mutation/lint/embedding 走 OpenAI API。

## Gen 0：Seed 评估

1. 在 **val split (30 task)** 上评估 seed persona → `subscores`（Pareto key）、`difficulty_subscores`、`validity_subscores`
   - 含 judge + trace capture
2. 在 **train split (30 task)** 上评估 seed persona → `train_subscores`（minibatch 预筛基线）
   - 只跑 difficulty，无 judge
   - seed 作为所有后续 child 的第一个 parent，需要完整 train coverage
3. 跨 run 复用（`--seed-reuse-glob`）：如果有旧 run 的 seed 结果（同一 persona），直接复用命中 task 的 sim，只跑缺失的
   - 当前命中：brinks_01=26/60, brinks_03=14/60, brinks_04=14/60, brinks_00/02=0
   - 复用的 sim 会合并回 results.json，judge 也能看到
4. 将 seed 写入 shared seen pool（跨岛 diversity gate 能看到所有 seed）

**Difficulty 定义**：`1 − agent_reward`。Agent 成功 = reward 1.0 = difficulty 0.0；agent 失败 = difficulty > 0。

## 每代 Iteration (g=1..20)

### Step 1: 选 Parent

从 val subscores 的 **Pareto frontier** 上采样一个 parent（多目标：difficulty × validity × timeout_safety）。

### Step 2: 变异

GPT-5.4 (xhigh reasoning) 基于 parent persona + 反思数据（失败/成功 trajectory 摘要）生成 child persona。

变异前注入：
- 本岛已探索的 persona（diversity gate 用）
- 他岛 persona 的 2 行摘要（最多 40 条，from shared seen pool）
- 已知的 bad persona 案例（few-shot negative examples）

**Diversity gate**：child 必须与所有已见 persona（本岛 + 他岛）满足：
- lexical 3-gram Jaccard dissimilarity ≥ 0.30
- embedding cosine similarity ≤ 0.90

最多 8 次变异尝试（`--max-mutation-attempts 8`）。通过 diversity gate 后立即写入 shared seen pool（不等接受/拒绝，防并发重复）。

### Step 3: Lint

GPT-5.4-mini 对 child persona 做合法性预检（格式、明显违规）。不通过则记录为 invalid case（用作后续 few-shot），重试变异。

### Step 4: Minibatch 预筛

从 **train split (30)** 中按域比例抽 **10 task** 组成 minibatch，在 child persona 上跑 sim。

与 parent 在**完整 minibatch (10 task)** 上比较 `child_mb_mean ≥ parent_mb_mean`：
- Parent 已有 score 的 task → 直接复用缓存
- Parent 缺失 score 的 task → **按需补评 parent**（只跑缺失的那几个 task），结果缓存到 `parent.train_subscores` 供后续 gen 复用
- 这样预筛永远在完整 10 task 上公平比较

结果：
- 通过（child ≥ parent）→ 进入 val confirm
- 未通过 → 本代结束（child 被拒）

### Step 5: Val Confirm

在 **val split (30 task)** 上跑 child 的完整 sim + judge：

1. **Difficulty eval**：30 task 全跑，得到 per-instance difficulty（无 judge，无 trace）
2. **Validity judge**：只对 failed sim 跑 judge（worst-reward-first，最多 15 个），检测是否有非法 user 行为
3. **Feasibility gate**：
   - severe_rate > 5% → persona invalid，拒绝
   - timeout_rate ≥ 阈值 → 拒绝

通过 → child 加入 candidate pool，其 val subscores 进入 Pareto frontier。

### Step 6: 记录与 State 更新

- Child 的 `train_subscores` = minibatch 已有的 10 个 task 结果（不做额外全量补评；将来做 parent 时按需补评）
- 更新 `pgepa_state.json`（candidates、gen_logs、best_idx）
- 如被拒（feasibility gate 未通过），记录为 invalid case（few-shot）

## 一代的 rollout 成本

| 阶段 | Task 数 | 是否必跑 | 备注 |
|------|---------|---------|------|
| Parent 按需补评 | 0~10 | 有缺失时 | 平均 ~6-7 个（首代 seed 为 0） |
| Minibatch (child) | 10 | 必跑 | 预筛 |
| Val confirm (child) | 30 | 预筛通过时 | difficulty + judge |
| **总计** | 10~50 | | 被预筛拒 = 最少 10；全通过 = 最多 ~50 |

对比旧逻辑：旧逻辑每个 accepted child 额外补评 20 个 train task（固定），新逻辑省掉这部分。

## 关键设计决策

### 按需补评 vs 全量补评 train

旧逻辑：child 被接受后**一次性跑 20 个 train task**（30 − minibatch 10），补全 `train_subscores`。每个 accepted child 额外 ~15min。

新逻辑（当前）：**不预补评 child**。child 的 `train_subscores` 只保留 minibatch 已有的 10 个结果。当它将来做 parent 时，如果新 minibatch 抽到了它没 score 的 task，**按需补评 parent 缺失的那几个 task**，结果缓存复用。

理由：
- 省掉 ~20 rollout/accepted child 的固定成本
- 实际每代只需补评 parent 缺失的几个 task（平均 ~6-7 个），比一次性 20 个少
- 被拒的 child 完全不产生 train 补评成本
- Seed 做第一个 parent 时有完整 30 task 的 train scores（gen0 跑过），不需要补评

### 跨 run 复用（gen0 warm-start）

Gen 0 的 seed eval 支持从旧 run 的 `results.json` 复用（`--seed-reuse-glob`）。

- 只复用 task_id 匹配且 reward 非空的 sim
- 复用的 sim 合并回当前 run 的 `results.json`（judge 也能看到）
- 验证前提：旧 run 用的是同一个 seed persona 文件（通过时间线确认：seed 文件 mtime 早于旧 run）
- 仅作用于 gen0 seed，绝不用于 mutated children

### Resume

Launcher 默认带 `--resume`。从 `pgepa_state.json` 恢复：
- candidates（含 subscores、train_subscores、persona_text）
- seen_personas（本岛已见 persona 文本列表）
- invalid_cases（few-shot 用的 bad persona）
- gen_logs（每代日志）
- start_iter（从最后完成的 gen + 1 继续）

被 kill 中断的 gen 会从头重跑（minibatch 重新抽样 + eval）。sim 层有 `--auto-resume`，已完成的 task 不会重跑。

### Shared Seen Pool

`island_seen_pool.py` 实现的 JSONL + `fcntl.flock` 文件锁：
- **写**：独占锁，child 产出后立即写入（不等接受/拒绝）；gen0 seed 也写入
- **读**：共享锁，容忍半写尾行（JSON parse 失败跳过）
- **用途**：
  - `.texts(exclude_island)` → 折入本岛 diversity gate（与他岛 persona 比 dissimilarity）
  - `.summaries(limit)` → 注入变异 prompt（"ALREADY EXPLORED by parallel searches"）
- 防止跨岛生成相似 persona，但不做跨岛 Pareto 融合（各岛独立选择）

## 文件结构

```
scripts/persona_opt/
├── parallel_gepa.py          # 主搜索循环（serial-Pareto: 1 parent → 1 child/gen）
├── tau_persona_adapter.py    # tau2 sim/judge 适配层（含 reuse_cache warm-start）
├── island_seen_pool.py       # 跨岛 file-locked JSONL pool
├── lineage_loop.py           # 变异 LLM 调用（call_mutation_llm）+ persona materialization
├── run_islands_telecom.sh    # 5 岛启动器（含 --resume、--seed-reuse-glob、shim 管理）
├── monitor_islands.py        # 聚合监控（gen/best/阶段/pool 一览）
├── pgepa_monitor.py          # 单岛详细监控
├── show_island_settings.py   # 配置速览（解析 launcher 默认值 + env override）
└── ISLAND_OPT_LOGIC.md       # 本文档

data/persona_opt/
├── telecom_island_split.json # train/val/test 划分（分层均衡，seed=42）
├── seeds_brinks/             # 5 个 seed persona（brinks_00..04）
└── pgepa_runs/islands_telecom_v1/
    ├── isl_brinks_0X/        # 每岛 run_dir
    │   ├── pgepa_state.json  # 完整 state（candidates + gen_logs + config）
    │   └── candidates.json   # 搜索完成后的最终 candidates
    ├── shared_seen_pool.jsonl # 跨岛 dedup pool
    └── logs/                 # 每岛日志 + vertex shim 日志
```

## 监控

```bash
# 聚合速览（显示实际 gen、best persona 名、当前 eval 阶段 + 进度）
.venv/bin/python scripts/persona_opt/monitor_islands.py --watch 15

# 展开所有岛的详细视图
.venv/bin/python scripts/persona_opt/monitor_islands.py --full

# 配置确认
.venv/bin/python scripts/persona_opt/show_island_settings.py
```

Monitor 阶段标记：
- `mb` = minibatch 预筛（10 task from train）
- `val` = val confirm（30 task from val split）
- gen 显示的是**实际在跑的代数**（从活跃 eval dir 名推断），不是滞后的 state

## 操作手册

```bash
# 首次启动
bash scripts/persona_opt/run_islands_telecom.sh

# 暂停（kill 所有岛）
pkill -f "parallel_gepa.py"

# 恢复（launcher 默认 --resume）
bash scripts/persona_opt/run_islands_telecom.sh

# 查看某岛的最新日志
tail -f data/persona_opt/pgepa_runs/islands_telecom_v1/logs/isl_brinks_01.log

# 查看 shared pool 状态
wc -l data/persona_opt/pgepa_runs/islands_telecom_v1/shared_seen_pool.jsonl
```

## 搜索完成后

20 代结束后每岛输出：
- `exploration_best_persona.md` — best persona 文本
- `candidates.json` — 所有 accepted candidates
- `pgepa_state.json` — 完整 state

下一步：在 **test split (54 task)** 上评估所有岛的 best persona，验证泛化性。
