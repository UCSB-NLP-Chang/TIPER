# VitaBench PGEPA 对抗式 user persona 搜索实验

目标：用 PGEPA（parallel generational per-instance-Pareto persona search）在 VitaBench 上
搜索让 **agent reward 下降** 的 user persona（对抗式 simulated user），验证搜索有效性。

## 代码

- `scripts/vita_persona_adapter.py` — adapter：`evaluate(persona)->difficulty`、
  `make_reflective_dataset`、`build_custom_proposer`（mutation）。自包含，不依赖 tau2 的 `gepa`
  包或 `lineage_loop`——轻量 `EvaluationBatch` + VitaBench 版 mutation prompt 内置。
  Mutation prompt 对齐 `sweetrl_persona_adapter._build_sweetrl_mutation_prompt` 的骨架
  （`_DOMAIN_FRAMING` per-domain + workflow stages / `_VALIDITY_CONSTRAINTS` /
  `_TIMEOUT_SECTION` / `_OUTPUT_FORMAT`，genesis vs. modify，含 success + failure traces）。
- `scripts/vita_pgepa.py` — driver：generational 搜索循环 + train/val/test split + test 评估。
  仿 `optim/code/persona_opt/parallel_gepa.py` 但自包含。
- `scripts/vita_offline_judge.py` — test-set 有效性 judge，对齐
  `sweetrl_offline_judge.py` 的 `JUDGE_SYS`：四类 severe 违规
  （`fabricated_info` / `refused_to_engage` / `contradictory` / `character_break`），
  genuine vs. severe，只判 failed 对话（reward<1.0），`severe_rate ≤ 5%` = VALID。
  改写成 VitaBench 客服 domain（hidden info = `task.instructions` +
  `task.user_scenario.user_profile`）。默认读 driver 已 dump 的 test 对话（省成本），
  `--rerun` 重新评估（对齐 sweetrl 模式）。judge model 用非 reasoning 的 gpt-4.1
  （reasoning judge 在长客服对话上触发 invalid_prompt，见 EVAL_INVALID_PROMPT_ISSUE.md）。
- `src/vita/run.py` — **最小 patch**：`run_task`/`_run_task_internal` 加 `user_persona_override`
  参数（3 处，~5 行）。VitaBench 的 `UserSimulator` 本就接受 `persona`，只是 `run_task` 把
  `task.user_scenario.user_profile` 写死了，patch 让它能外部覆盖。
- `src/vita/models.yaml` — 加 `gpt-5.4`(high) / `gpt-5.5`(xhigh) 两个条目（git-ignored）。

## Setting（用户指定 + 一处技术妥协 + 一处规模调整）

| 维度 | 用户要求 | 实际 | 说明 |
|---|---|---|---|
| domain | delivery + 一个简单的 | delivery, instore | instore（paper 45.8%）比 ota（28.3%）简单，选 instore |
| task split | 30/30/50 | 30/30/40 | 单域只有 100 任务，30+30+50=110 超了，test 缩到 40 保持三集不重叠 |
| agent | gpt-5.4 high | **gpt-5.2 high** | gpt-5.4 + reasoning + function tools 在 /v1/chat/completions 被拒（见下），agent 带 66 tools，换 gpt-5.2 |
| user | gpt-5.4 high | gpt-5.4 high | user 无 tools，gpt-5.4 high 可用 |
| mutation | gpt-5.5 xhigh | gpt-5.5 xhigh | 无 tools，可用 |
| evaluator | — | gpt-4.1 | reasoning 判官会 invalid_prompt（见 EVAL_INVALID_PROMPT_ISSUE.md），用非 reasoning |
| minibatch | 10 × 2 trials | 10 × 2 | |
| val | 2 trials | 30 × 2 | |
| 起步 | no-persona | 空 persona (genesis) | gen0 评估空 persona，gen1 走 genesis 模式从零设计 |
| max_step | 100 | 100 | |
| iter | 20 | 20 | |
| children-per-gen / top-t | 未指定 | K=3, top-t=1 | 控制 cost；每 gen ≈ 3×20 minibatch + 1×60 confirm = 120 sims |

## gpt-5.4 跑不通、gpt-5.2 可以的原因

OpenAI 报错：`Function tools with reasoning_effort are not supported for gpt-5.4 in
/v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort
to 'none'.`

- vitabench `llm_utils.py` 走 `/v1/chat/completions` 端点。
- gpt-5.4（更新）在该端点移除了「reasoning_effort + function tools」组合，必须 `/v1/responses`
  或 `reasoning_effort=none`。
- gpt-5.2（较早）`/v1/chat/completions` 仍支持 reasoning + tools（delivery 复现已验证）。
- 影响范围：带 tools 的角色（agent）。user/mutation 无 tools，gpt-5.4/5.5 高 reasoning 可用。

## 搜索目标与 difficulty

difficulty = 1 - agent_reward（genuine failure / timeout 均计 1-reward）。
batch-level timeout_rate ≥ 15% → 整 batch zeroing（防 stall 取胜）。
Pareto 按 per-instance（val task）difficulty 选 parent，win-frequency 加权采样。

## validity

搜索阶段不做 user-behavior judge（与 sweet_rl 一致——judge 在离线，不在搜索循环）。
靠 mutation prompt 的 `_VALIDITY_CONSTRAINTS` + batch-level timeout gate 兜底。

**test set 阶段用 offline judge 复核有效性**（用户要求，对齐
`sweetrl_offline_judge.py`）：`vita_offline_judge.py` 用 `JUDGE_SYS` 四类 severe 违规
（fabricated_info / refused_to_engage / contradictory / character_break）判每个 failed
对话，`severe_rate ≤ 5%` => VALID（persona 的 reward 下降是合法对抗胜利，不是作弊）。
genuine difficulty = difficulty on non-severe conversations。

## 运行

```bash
cd vitabench
# delivery（background）
nohup .venv/bin/python -u scripts/vita_pgepa.py --domain delivery \
  --run-dir data/persona_opt/vita_pgepa_runs/run_delivery \
  > /tmp/vita_pgepa_delivery.log 2>&1 &
# instore
nohup .venv/bin/python -u scripts/vita_pgepa.py --domain instore \
  --run-dir data/persona_opt/vita_pgepa_runs/run_instore \
  > /tmp/vita_pgepa_instore.log 2>&1 &
```

每 gen 落盘 `pgepa_state.json` + `conversations/gen_*/`。test 阶段写 `test_results.json`
（no_persona vs best_persona 的 avg_reward 对比）。

## 30min check 看什么

- `pgepa_state.json` 的 `best_agg`（val difficulty）是否随 gen 上升（= reward 下降）
- `conversations/gen_*/` 抽看对话是否自然、persona 是否生效
- timeout_rate、infra_error 计数
- 成本累积（agent_cost + user_cost）

## 结果

driver `gepa.optimize()` 在两个域都跑到提前停止（无新最佳）：delivery iter 3 达 best、至 iter 12 停；
instore iter 9 达 best、停。`vita_test_eval.py` 3 trials 评估 test set，`vita_offline_judge.py`
（`--test-prefix test3`）复核有效性。

### val difficulty（搜索有效性 ✓）

| 域 | seed(空persona) | best persona | Δdifficulty |
|---|---|---|---|
| delivery | 0.3833 | 0.5833 (iter 3) | **+0.2000** |
| instore | 0.2333 | 0.3333 (iter 9) | **+0.1000** |

搜索机制本身工作：两个域 val difficulty 都上升 = agent reward 下降。best persona 长度
delivery 4486 字、instore 类似量级，均非激进对抗，而是"温和模糊化"风格（如 delivery 的
"Context-Shorthand Proxy Customer"：用"那单/刚才那单"代订单号、"家里/公司"代地址、分批给信息）。

### test set（泛化性 ✗）

3 trials × 40 test tasks，avg@3 = 每 task 3 trial 均值再均；pass^3 = 3 trial 全对（reward≥0.999）的 task 占比。

| 域 | 条件 | avg@3 | pass^3 | solved 3/3 |
|---|---|---|---|---|
| delivery | no_persona | 0.5583 | 0.2750 | 11/40 |
| delivery | best_persona | 0.6417 | 0.4500 | 18/40 |
| instore | no_persona | 0.8083 | 0.7000 | 28/40 |
| instore | best_persona | 0.8083 | 0.7000 | 28/40 |

| 域 | Δavg (no−best) | Δpass (no−best) | 结论 |
|---|---|---|---|
| delivery | **−0.0833** | **−0.1750** | persona **反而帮助 agent**（反向） |
| instore | **+0.0000** | **+0.0000** | 净零（正负抵消） |

val 上的 difficulty 改进**没有迁移到 test**。instore 净零；delivery 反向——persona 让 agent
reward +8.3pp、pass +17.5pp。

### 离线 judge（有效性 ✓）

`JUDGE_SYS` 四类 severe 违规（fabricated_info / refused_to_engage / contradictory /
character_break），只判 failed 对话（reward<1.0），severe_rate≤5% = VALID。

| 域 | 条件 | n_failed_judged | n_severe | severe_rate | 判定 |
|---|---|---|---|---|---|
| delivery | no_persona | 29 | 0 | 0.0% | — |
| delivery | best_persona | 22 | 0 | 0.0% | VALID |
| instore | no_persona | 12 | 0 | 0.0% | — |
| instore | best_persona | 12 | 0 | 0.0% | VALID |

两个域 best persona **severe_rate=0%**，无任何作弊。persona 造成的 reward 变化（无论方向）
都来自合法的现实客户摩擦，不是 fabricated_info / 拒答 / 矛盾 / 角色穿帮。
（注：脚本 verdict 文案在 Δ≤0 时仍印 "VALID adversarial win"，实际无 win——validity 通过
但 efficacy 不通过。）

### 根因分析

1. **val/test overfit**：30 train + 30 val 搜出的 persona 对 val 优化过度，不 generalize 到 40 test。
   两个域 val 上都有效（+0.20 / +0.10），test 上失效。
2. **persona 设计的内在矛盾**：persona 的"分批给信息 / 简短上下文指代"策略，设计意图是
   "让 agent 更难消歧"，但对**多约束复杂任务反而帮助 agent**——分批给信息避免信息过载、
   不遗漏约束。实例验证（delivery D0721004，no_persona reward=0 → best_persona reward=1）：
   no_persona 的 user 一上来堆出所有约束（评分≥4.5+有实体店+猪肉水饺不要韭菜+奶奶海鲜疙瘩汤+
   主动提地址问题），agent 信息过载 + 卡地址、没下单；best_persona 的 user 分步给信息，agent
   一步步引导、成功下单。delivery 多约束 → persona 帮助；instore 简单 → 净零。
3. **agent 强度上限**：instore agent 本身很强（seed test avg@3=0.8083、80% 成功），温和 persona
   压不动；delivery agent 较弱（seed 0.5583），persona 反而把它托上去。

### 待办（下一轮，用户已批准）

- **在线 judge**（对齐 tau `parallel_gepa` severe 门控）：搜索阶段 `adapter.evaluate` 跑完模拟后
  判 failed 模拟，severe_rate>5% → persona 被门控（分数=0、不可行）。仅判 failed（对齐 tau）。
  本轮无在线 judge，靠 mutation prompt `_VALIDITY_CONSTRAINTS` + batch timeout gate 兜底——
  离线 judge 已确认 0% severe，说明兜底足够，但在线 judge 能在搜索时更早剪枝作弊候选。
- 反向效果提示 mutation prompt 的"温和模糊化"方向对多约束任务有反作用，下一轮可加入
  "避免分批给信息反而帮助 agent"的负反馈，或对 difficulty 下降的候选直接淘汰。

## V2 轮：新 mutation ctx + 在线 judge（运行中）

V2 = V1 的 mutation ctx 全面对齐 tau pgepa enriched 模式 + 搜索阶段加在线 judge，其余
setting 与 V1 完全一致（gpt-5.4 high agent/user、gpt-5.5 xhigh mutation、gpt-4.1 eval、
30/30/40、minibatch 10×2、val 2 trials、iter 20、max_step 100）。run dir：`run_delivery_v2`
/ `run_instore_v2`（保留 V1 结果）。

### V2 mutation ctx 改动（相对 V1）

1. **ADVERSARIAL REASONING 4 步输出格式**：Step1 智能体假设（引用具体回合）→ Step2 成功弱点
   （走查关键交换）→ Step3 核心假设 "if 客户做 X，agent 会错误做 Y，because Z" → Step4 写角色。
2. **领域阶段重写**：(a)信息收集 (b)**步骤执行**（客户声称做了步骤但做错了）(c)**验证/关闭**
   （确认要点而未核实）(d)**问题范围**（框架锁定错误范围）——从"温和模糊化"转向"系统性误导"。
3. **完整对话追踪**（去掉 V1 的 `[:1400]`/`[:1000]` 头部截断，根因修复）+ **仅成功追踪**
   `failure_traces=[]` + 按领域 3/2 + 55k 字符预算。
4. **评分章节** `_DOMAIN_SCORING`（每域 "HOW THE AGENT IS SCORED"）+ **少样本** `_MUTATION_EXAMPLES`
   （良好通用 vs 不良任务特定）+ **静态 linter** `lint_persona`（natural/valid/no_task_specific，
   拒绝→invalid_cases + 带提示重试）。
5. **行为风格目录** `_BEHAVIORAL_STYLE_CATALOGUE`（4 家族：角色/结构/节奏/认知语调）+ **收敛禁止**
   `_CONVERGED_SAFE_DIRECTIONS`（6 个被禁的"温和模糊化"安全方向）——强制 HOW 多样性，避免 V1
   收敛到单一轴。
6. **探索摘要** `_render_explored`（genN: direction → ACCEPTED val_diff / REJECTED (reason)）+
   **无效回放** `_render_invalid` + `_BASE_OVERRIDE`（自由重构/替换弱规则/近乎逐字复制=失败变异）。

### V2 在线 judge（搜索阶段）

- **对话行为检查**：`_aggregate` 跑完每个 minibatch 后，对所有 `genuine_agent_failure` 的 rep sim
  并发跑 `judge_conversation`（`_JUDGE_SYS` 4 类：fabricated_info/refused_to_engage/
  contradictory/character_break）。`severe_rate = severe/n_judged > 5%` → 整 batch 门控
  （scores 清零、feasible=0），对齐 tau `parallel_gepa` severe gate。verdict 回挂到 trajectory。
- **persona 本身检查**：`lint_persona` 在 proposer 的变异-lint 重试循环里（3 次尝试，拒绝→
  invalid_cases + 纠正提示重试），覆盖 natural/valid/no_task_specific 三轴（valid 轴已含
  fabricate/refuse/abandon/break character/impossible/stall 的拒绝）。
- **severe 反馈到 proposer**：`make_reflective_dataset` 把 verdict 放进每条 record 的 `Judge`；
  proposer 提取 severe violation 累积到闭包 `accumulated_severe`（dedup by category+reason），
  注入 mutation prompt 的 `## SEVERE VALIDITY VIOLATIONS (DO NOT CHEAT)` 区块。
- **test-set eval 关闭 judge**：`evaluate_reward` 期间 `adapter._online_judge_enabled=False`，
  test reward 是原始 agent reward（有效性由离线 judge 复核），避免 gating 污染 no/best 对比。

### V2 结果（resume 续跑到 iter15，test 3 trials）

搜索跑到 iter15 stopper 停（`--iters 15`），中途用 gepa 原生 checkpoint resume 一次
（`gepa.stop` 优雅停 → 删标记同 run-dir 重启，gepa 自动 `Loading gepa state from run dir`
恢复 frontier + state.i，续跑前无条件重评 seed valset ~60 sims）。test 用 `--test-trials 3`
（search 仍 2 trials，driver 加 `--test-trials` 分离）。

#### val difficulty（搜索有效性，2 trials）

| 域 | seed | best persona | Δdifficulty | iter | V1 best |
|---|---|---|---|---|---|
| delivery | 0.4667 | 0.6500 | **+0.1833** | 4 | 0.5833 |
| instore | 0.2667 | 0.5167 | **+0.2500** | 9 | 0.3333 |

两域 val difficulty 均大幅超 V1。

#### test set（泛化性 ✓✓，3 trials × 40 tasks）

| 域 | 条件 | avg@3 | fully_correct | Δavg (no−best) | V1 Δavg |
|---|---|---|---|---|---|
| delivery | no_persona | 0.6667 | 17/40 | **+0.2500** | **−0.0833（反向）** |
| delivery | best_persona | 0.4167 | 8/40 | persona HARDER ✓ | |
| instore | no_persona | 0.7667 | 24/40 | **+0.1000** | **0.0000（不泛化）** |
| instore | best_persona | 0.6667 | 18/40 | persona HARDER ✓ | |

**V2 修好了 V1 的两个核心失败**：delivery 从 V1 的反向（persona 帮 agent）翻转为强对抗
胜利（reward −0.25、少解 9 task）；instore 从 V1 的不泛化（Δ=0）变为泛化（reward −0.10、
少解 6 task）。两个 persona 都从 val 迁移到 test。

#### 离线 judge（有效性 ✓✓）

`JUDGE_SYS` 四类 severe，只判 failed 对话（reward<1.0），severe_rate≤5% = VALID。

| 域 | 条件 | n_failed_judged | n_severe | severe_rate | 判定 |
|---|---|---|---|---|---|
| delivery | best_persona | 15 | 0 | 0.0% | **VALID** |
| instore | best_persona | 7 | 0 | 0.0% | **VALID** |

两域 best persona **severe_rate=0%**，无 fabricated_info / refused_to_engage /
contradictory / character_break。test 上的 reward 下降是合法对抗胜利，不是作弊。
（在线 judge 搜索期间也未触发任何 severe 门控，0 次 `BATCH INFEASIBLE`。）

### V2 结论

新 mutation ctx（ADVERSARIAL REASONING 4 步 + 步骤执行/验证关闭阶段 + 完整成功-only
追踪 + 行为风格目录 + 收敛禁止 + linter + 探索摘要）+ 在线 judge（severe 门控 + 反馈到
proposer）共同修复了 V1 的 val/test 不泛化与反向问题。delivery/instore 两域 test 3 trials
均泛化（+0.25 / +0.10），且离线 judge 复核 0% severe——合法对抗胜利。

### V2 翻转任务诊断（test set，no=1.0→best=0.0，0 反向）

对每个翻转 task 读取 `conversations/test_{no_persona,best_persona}/{task_id}.json` 双条件
对话做因果诊断。delivery 9 翻转、instore 4 翻转，0 反向（无 best=1.0→no=0.0）。

#### Delivery（9 翻转）

| task | 主机制 | persona 触发语 | agent 走错步 | 结果 |
|---|---|---|---|---|
| 10711001 | (d) 问题范围 | "沾到风险就别照旧" | 把高嘌呤从「备注可降级」提升为「商品层面消除」硬门槛，唯一清淡米线是土鸡底 → 无合格品 → 取消 | 零单 |
| 20711002 | (a) 信息收集 | "别先按默认地址下"+"门牌不记得" | 把档案已知地址都当「默认地址」不敢提，等用户自报但不给 → STOP | 零单 |
| E0721002 | (d) 问题范围 | "不是顺手备注一下就算，按能避开的版本处理" | 把「备注」降级为不可接受弱方案，要求葱过敏有显式商品属性 → 系统无此属性 → 拒下 | 零单 |
| 40711001 | (b) 步骤执行 | 预授权+"按能稳妥那版处理" | 跳过结构核验，用替代结构下单（丢白切鸡/牛腩煲/粥）+套用未确认地址 → 被纠正取消再不恢复 | 零单 |
| C0715001 | (c) 验证/关闭 | "别按历史地址落" | 放弃唯一可用地址源，三单全挂至结束 | 零单 |
| G0716002 | (d)+(c) | "怕耽误事"+"风险点不是闲聊"+"先只下黄茶其他不动" | 把「远/海鲜风险不明」当不可触碰边界，把「暂缓」当终局 → 只下 1/3 | 缺 2 单 |
| H0716001 | (a) 信息收集 | "别先按工作地址下，怕送到人不在" | 把工作地址默认方案当不可接受风险撤回，等永不来的确认地址 | 零单 |
| I0721002 | (d)+(c) | "别弄成两条"+"给她俩尝尝"+"确认能去大头菜再付" | 烤鱼窄化为二选一丢双拼 + 饮料按字面 2 杯非 3 + 炒粉验证门关不掉 | 三项全错 |
| 90714001 | (a)+(d) | "排骨乱炖会不会更油"+"别把条件放松了" | 先承诺后搜，承诺库存不存在的菜 → 风险句封锁所有近似替代 → 零单 | 零单 |

#### Instore（4 翻转）

| task | 主机制 | persona 触发语 | agent 走错步 | 结果 |
|---|---|---|---|---|
| 10826005 | (b)+(c) | "能锁住新的再付" | 过度验证剧本杀时段，系统查不到成团人数 → 剧本杀单从未创建 + 冷面单未付 | 缺剧本杀 |
| 10826009 | (a)+(c) | "先确认能订上再往下弄" | 跳过问人数，customer_count 默认 1（应 4）→ "先别动"锁错预约 | 预约错 |
| 10826027 | (c) 验证/关闭 | "先筛不买，确认周末朋友来了能用再下，别两头落空"+withhold 周末/人数 | 把商家适用性当下单硬前提，系统无周末数据 + 用户不给 → 验证门永关 | 缺 KTV |
| 10826067 | (c) 验证/关闭 | "先确认有能用优惠券再买券再下单，不行先别动" | 把优惠券做成买多肉葡萄前置硬前提，系统无优惠券工具 → 连有货的多肉葡萄也不下 | 缺饮品 |

#### 跨任务共性机制（3 种失败模式）

1. **降级正常机制 → 无解放弃**（delivery 主力，~7/9）：persona 用"别口头说可以""不是顺手
   备注""别按默认地址"把外卖场景标准下单手段——备注、档案已知地址、库存近似替代——判为
   「不够稳」。agent 于是要求超出系统能力的保证（商品层面消除嘌呤/显式去葱属性/用户自报
   精确地址/完美匹配菜名），系统做不到 → 判定无合格方案 → 取消或搁置。
2. **gated workflow 死锁**（instore 主力，4/4）：persona 把可直接执行的动作改写成"先确认
   前提再做，不行先别动/别两头落空"。前提要么系统不支持查（优惠券可用性、商家周末适用性），
   要么用户刻意 withhold（周末时间、人数）。agent「对系统边界诚实、不无根据确认」的良性
   行为在硬前提面前变成永远关不掉的验证门 → 不下单。
3. **过度执行错单**（少数，2 task）：persona 预授权+"按能稳妥那版处理" → agent 跳过
   结构/信息核验，用错误结构下单（40711001 丢三菜套未确认地址）或跳步填默认值
   （10826009 人数默认 1）。被纠正后取消再不恢复 → 零单。

#### 根因

两个 persona 攻击 agent 同一内在倾向：面对语气委婉的风险表达/条件式授权时倾向**保守退缩、
字面服从**，而非主动在约束内找可行解兜底。delivery 用"风险疑问"诱发退缩（模式 1），
instore 用"保险栓条件授权"诱发死锁（模式 2），殊途同归。

关键对比：no_persona 下同样的信息缺口（不记得地址、库存无完美匹配、优惠券不存在）被 agent
主动用历史数据/近似品/绕过填补并推进到关闭——问题不在任务难度，而在 persona 的风险框架让
agent 误判了「可自由决策的余地」与「禁止区」的边界，把本可兜底的余地变成了禁止区。agent
面对模糊/条件式指令时的保守倾向正是两个 persona 共同利用的弱点。

#### OTA（12 翻转 + 1 反向）

ota best persona 是「照顾型同行组织者（条件闸门型）」——把需求组织成条件闸门/决策树
（"只有满足某条件才订/改/退；不满足走备用分支；关键条件没过就停住"），候选项≠授权项，
确认时确认规则方向而非逐字段核对。

| task | 主机制 | persona 触发语 | agent 走错步 | 结果 |
|---|---|---|---|---|
| 70807001 | (a)+(d) | "日期闸门算错了…应该是 2026-09-05"（注入错误日期） | 无条件接受错误纠正→2026-09 系统无任何数据→触发"不满足就停住"→零单 | 零单 |
| 70808003 | (b)+(c) | "满足了就下单"（只说下单不说支付）+"最后一道闸门…收口"引向航班 | 建 12 笔订单后被闸门引开讨论航班→对话结束从未支付 | 12 笔未付 |
| 70808004 | (b)+(c)+(d) | "别为了凑线硬办"+"退改闸门…有一单卡死就先停" | 主动弃金顶改订山脚亚朵（错酒店）+退改政策查不到→无法证明可退→停付 | 错酒店+零支付 |
| 70812004 | (c) | 黄龙谷"晴天分支留着不先订" | 天气满足仍不把分支从评估态切执行态→缺黄龙谷门票 | 缺 1 项 |
| 70812006 | (b)+(c) | "先过闸门"引向距离计算 | 注意力从订景点票引向距离判断→全程未搜景点门票 | 缺 4 张票 |
| D0805002 | (c) | "门票和讲解能一起买好的再办" | "一起买"窄化为"同一打包 SKU"→放弃用户最偏好的上海博物馆（可分两笔）改推豫园 | 偏离偏好景点 |
| D0807009 | (c)+(d) | "500m 内+按摩浴缸+大床房这个闸门没过就别硬订" | 把"500m 无房"当**终止信号**而非**降级信号**，不推 9km 外按摩浴缸兜底房（no_persona 推了、用户接受） | 缺酒店 |
| D0808002 | (a)+(d) | "出发城市给不了…先别替我猜…没起点就先停着" | 已查到常住地临沂却守"别猜"→从未搜临沂→曲阜火车→闸门永不过 | 缺火车 |
| D0808012 | (c)+(d) | "二等座优先…两层没过继续看"+"无烟楼层要备注落实到订单" | 座席闸门→漏搜南靖→为过闸门订去厦门（102km）+备注功能系统无→停付酒店 | 火车错目的地+酒店未付 |
| D0811002 | (a)+(c) | "先别替我猜"出发城市 | 从不查用户历史推断出发地→把"不猜城市"泛化为"不推进任何步" | 零单 |
| D0811004 | (a)+(c) | "先别拿哈尔滨代入…没说过出发地…条件没定就先别替我假设" | 已查到常住地哈尔滨却**主动放弃使用**→火车缺参数整支停摆→酒店连带停 | 缺火车+酒店 |
| D0812006 | (d)+(a) | 主动纠偏"不是 8 月 1 号…是 9 月 1 日"（错误） | 改按 9/1 核查→9 月无房态/无天气/无票档→所有闸门连锁失败→全停 | 全未完成 |

反向 1：D0811010——persona"先把时间口径卡正"消除日期歧义，反而帮 agent 正确理解日期→
best=1.0、no=0.0（结构化反问恰好消歧）。

OTA 跨任务机制（4 类，条件闸门型）：

1. **候选项≠授权项 / 闸门冻结执行**（主力，6）：把应执行动作框为"分支/候选/先记上/只下单
   不支付"，agent 满足条件后不升级为执行，或闸门拆步跳过支付（70808003/70808004/70812004/
   70812006）；"先别替我猜"封死用历史推断出发城市（D0808002/D0811002/D0811004）。
2. **闸门纠错 / 注入错误前提**（2）：主动"纠偏"注入超出数据窗口的错误参数（日期）→系统该
   日期无数据→所有下游闸门连锁失败（70807001/D0812006）。
3. **软偏好升级硬闸门 + 验证标准收窄**（2）："一起买"→必须打包 SKU（D0805002）、"有标签"
   →必须备注落实到订单（D0808012）→agent 不质疑闸门合理性只机械执行→放弃正确选项或停付。
4. **可兜底硬约束被升级为终止信号**（1）："闸门没过就别硬订"→把"500m 无房"当终止而非
   降级，不推 fallback（D0807009）。

核心杀伤："不满足/不确定就先停"把 agent 遇不确定就保守停手的弱点激活到最大化——要么注入
无法验证的条件让它停，要么拆步跳支付让它漏，要么纠错引入空数据让它退，要么封死主动推理
让它缺参数。

#### cross_domain（11 翻转 + 5 反向）

cross_domain best persona 是「路线脑的赶路型安排人（反问验收版）」——把硬要求说成反问
（"会不会…""别…吧"），第一个出现的具体对象可能只是排雷对象而非授权项，多域需求压在同一段
用同一种反问口气说。单 persona 跨 3 域（66 合并工具），攻击跨域意图切换与初始框架歧义。

| task | 主机制 | persona 触发语 | agent 走错步 | 结果 |
|---|---|---|---|---|
| 10804001 | (a)+(c) | "赶路顺手收口"+"接驳票查不到规则就先别乱买" | 接驳票当信息查询非待办→从不调景点搜索链→"别乱买"当免责 | 缺接驳票 |
| 10806004 | (b)+(d) | "上次买的那家户外装备" | 背景信息误判硬约束→找不到后幻觉"无搜索/下单能力"→放弃帐篷（实际有 delivery_product_search） | 缺帐篷 |
| 10807010 | (c) | "别没货还硬下"+"别偏材质太远" | 把系统 quantity=0 当硬阻塞→反复搜索不下单茶宠（no_persona 直接下成功） | 缺茶宠 |
| 10807012 | (a)+(c) | "出发城市真不记得…别乱按站"+"先别替我下" | 已查到常住地嘉兴却视为禁区+"别下"扩大为"别搜"→从未 train_ticket_search | 缺高铁 |
| 10811001 | (c) | "别送错"+"别按家庭地址猜"+"地址没报过" | 否决唯一可用地址源（历史家庭地址）→撤单不重下→"门牌想不起"当硬阻塞 | 缺外卖 |
| 10811004 | (b)+(c) | "别送错地方"+"别拿历史常住地址当代填" | 历史住址=家庭地址却视为不可用→查不到"当前默认地址"字段→补剂不下 | 缺补剂 |
| 10812004 | (b)+(c) | 排雷"会不会踩雷"+收口"那就这样" | 排雷反问误判购买授权→提前建单被纠错取消 + 收口误判最终确认 | 缺高铁 |
| A0806003 | (d) | "别当成授权/别自己代/怕带偏"+"长春别直接当授权" | 已获取常住地长春被排雷句封锁→零搜索零下单（滑翔伞/餐馆/饮品全缺） | 零单 |
| A0808002 | (d) | "老人票别下错…不敢乱说年龄…回头确认再下" | 把"带父母→1 成人 2 老人票"包装成高风险排雷→要求额外年龄确认→灯光秀零下单 | 缺灯光秀 |
| A0812003 | (d) | 否定/收口语气过强 | agent 防御性自我设限→幻觉工具不可用→跳过安耐晒整条线（no_persona 直述→全完成） | 缺安耐晒 |
| A0812010 | (d) | "地址没给过/不是确认过的/别送错/真怕送错" | 已创建订单+持有支付授权，档案地址却被降级"未验证"→撤回支付→蛋糕永停待支付 | 缺蛋糕 |

反向 5：10807008、10808005、10808008、A0811001、A0811008——少数任务 persona 反问比 raw
user sim 自发 harsh 框架对 agent 更友好。典型 10808008：raw user sim 自发"排雷/踩坑"语气
→agent 过保守选了"馍类丰富"的西安饭庄（与"主食不是馍"硬条件矛盾）→失败；persona 反问更
温和→agent 正常交叉核验、自纠错→成功。net Δreward 仍强正。

cross_domain 跨任务机制（5 类，反问验收版）：

1. **排雷封锁可用数据**（主力，7）："别当成授权/别自己代/别送错/没给过/不是确认过的"把
   agent 已合法持有的信息（常住地→城市、档案→地址、"带父母"→老人票资格、quantity=0 可下单
   商品）包装成"未验证/未授权"排雷对象→agent 当硬流程否决而非结果层验收→零搜索/零下单/不
   支付（A0806003/10807012/10811001/10811004/10807010/A0808002/A0812010）。
2. **排雷反问误判为授权 / 收口误判为确认**（1）："会不会踩雷"被当购买授权→提前建单被纠错
   取消（10812004）。
3. **背景信息误判为硬约束**（1）："上次买的那家"被当必须找到的约束→找不到后幻觉"无能力"
   放弃（10806004）。
4. **信息查询与行动指令混淆**（1）："你也一并告诉我"被当纯告知义务而非采购指令→不行动
   （10804001）。
5. **否定语气过强 → 自我设限**（1）：agent 幻觉工具不可用、跳过整条任务线（A0812003）。

核心：persona 把"反问/担心句"伪装成闲聊实则硬验收点，agent 区分不出"排雷对象"与"授权项"、
"信息查询"与"行动指令"，把可用资源当禁区。

#### 4 域统一根因（扩展）

4 个 persona 攻击 agent 同一内在倾向：面对语气委婉的风险表达 / 条件式授权 / 反问验收时倾向
**保守退缩、字面服从**，而非主动在约束内兜底。delivery"风险疑问"诱发退缩，instore"保险栓
条件授权"诱发死锁，ota"条件闸门"冻结执行 / 注入错误前提 / 封死主动推理，cross"反问验收"
封锁可用数据——殊途同归：把 agent 本可兜底的决策余地重定义为禁止区。

**二值评估器放大杀伤**（元发现）：`calculate_reward` 是 all-or-nothing（全 rubric 过=1.0 否则
0.0，rubric 由 gpt-4.1 LLM judge 判）。persona 只要绊倒一道闸门（酒店 / 出发城市 / 支付）就
让整 trial 翻 0；叠加 3-trial 平均，no_persona 偶尔翻 1（如 D0807009 聚合 0.667）、best_persona
稳定翻 0（0.0），形成"部分成功→完全失败"的强降级表象。注：`conversations/test_*/{task}.json`
存单次 rep trial，其 `reward` 字段=该 trial reward（非 3-trial 聚合）；翻转按 rep-trial
no=1.0→best=0.0 分类，与 `test_report.json` 聚合基本一致，少数 rep 为 1.0 而聚合 0.667。

**反向任务**（ota 1、cross 5）表明该弱点非绝对：当 persona 结构化反问恰好消歧（D0811010
日期），或比 raw user sim 自发 harsh 框架更温和（10808008）时，persona 反而帮 agent；但
net Δreward 仍强正（+0.175 / +0.20），对抗胜利稳健。
