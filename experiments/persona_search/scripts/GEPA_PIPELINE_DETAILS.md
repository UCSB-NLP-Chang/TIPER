# GEPA Persona Optimization Pipeline — 详细实现说明

> 本文档描述**当前实际运行**的 loop（2026-06-07 起）。它替代了早期版本中关于
> `pareto`/10-15% 变异的描述——现网跑的是 `epsilon_greedy` + `~30%` 变异 + 1000
> metric-call 预算 + **`num_trials=2`**，且**本轮已启用 validity-gated optimization**
> （`--judge-valset --severe-gate-rate 0.05 --valset-judge-max-failures 15`，§8.4）。
> 核对基准 = 正在跑的进程命令（pid 53601/53602）。

---

## 0. 作为优化问题的形式化

我们在"对抗性用户 persona 文本"空间里搜索,目标是**让被测 agent(gpt-5.4)在 reprep
任务集上的成功率尽量低**,同时约束 persona **完全合法**(不作弊、不超时刷分)。

```
      maximize   D(p)  =  E_{t ~ Reprep}[ difficulty(p, t) ]
      over       p ∈ {persona_text}                     # 决策变量:一段 markdown 文本
      subject to severe_rate(p) ≤ 0.05                  # 合法性硬约束(validity)
                 timeout_rate(p) ≤ 0.30                 # 反"摩擦刷分"约束
```

- **决策变量**:`candidate = {"persona_text": <字符串>}`,GEPA 唯一优化的 component。
- **目标 `difficulty`**:只有"agent 被一个**合法**的 persona 真正击败"才计正分(§4)。
- **约束 `validity`**:作弊(judge 判 severe)的难度一律归零;严重作弊的整个候选作废。
- **可行域**:由 LLM 变异器(opus-4-7)从父代局部改写而来,不是离散枚举。

GEPA 用**进化式 + per-instance Pareto + minibatch 验收 + 反思变异**来近似求解这个
带约束的黑箱优化。约束的执行分两层:**optim 内软/硬 gate**(§8)+ **cert 终审**(§10)。

---

## 1. 总体架构与两阶段设计

```
Phase A  探索(gepa_runner.py)         Phase B  认证(certify_gepa_candidates.py)
  GEPA 搜索 → exploration_best          多 trial + 全量 judge + 配对比较 → certified_best
  reprep-50, num_trials=2               full-114 base, num_trials≥2, severe-rate gate
  目标:把难度顶上去(可能混作弊)        目标:挑出真正合法且显著优于 incumbent 的候选
```

**分工**:optim 冲难度,**合法性最终由 cert 把关**(`severe_rate>0.05 → reject`)。
optim 内部的 validity 约束(§8)是为了**少浪费搜索预算在作弊方向**,cert 才是终裁。

### 调用链

```
gepa_runner.py
  ├─ build_task_list()         → reprep-50 个 TauTask(§3)
  ├─ 读 seed persona 文本       → {"persona_text": ...}
  ├─ TauPersonaAdapter(...)    → GEPA ↔ tau2 仿真 + judge + 打分(§4,§8)
  ├─ build_custom_proposer()   → 复用 lineage 变异提示词 + edit-pct 覆写(§7)
  └─ gepa.api.optimize(...)    → GEPAEngine.run()
        ├─ 种子全量评估(capture_traces=False,不判官)
        └─ while metric_calls < 1000:          # patience=0 → 跑满预算(§9)
              ① 选父代(epsilon_greedy,§6)
              ② 采 minibatch(epoch_shuffled,10 个,§5)
              ③ evaluate(父代, capture_traces=True)   → 跑仿真+判官+打分(§4)
              ④ make_reflective_dataset()             → 按 outcome 整理反思素材
              ⑤ custom_proposer()                     → opus 重写 ~30% persona(§7)
              ⑥ evaluate(子代, capture_traces=False)  → 同一 minibatch 比分
              ⑦ sum(子代) > sum(父代)? → 是:全量 valset 评估 → 进 Pareto
              ⑧ 存状态 → 继续
        └─ 输出 exploration_best_persona.md + candidates.json
```

### 文件组织

```
scripts/persona_opt/
├── gepa_runner.py              # GEPA CLI 入口 + 后处理
├── tau_persona_adapter.py      # adapter:evaluate / 判官 / outcome分类 / validity gate / 变异proposer
├── certify_gepa_candidates.py  # Phase B 认证
├── compare_personas_serial.py  # 串行 no_persona→seed→candidate 难度对照
├── judge_user_intent_preservation.py (在 scripts/)   # 实际 LLM 判官
├── lineage_loop.py / eval_persona.py / judge_persona.py / _common.py  # 复用(§12)
└── GEPA_PIPELINE_DETAILS.md    # 本文件
```

---

## 2. tau2 仿真 = 一次目标采样(num_trials 个 sim 取均值)

每个 `(persona, task)` 评估 = 跑一通 tau2 多轮对话:**被测 agent(gpt-5.4, high)**
↔ **cc_user_con 用户模拟器(Bedrock claude-opus-4-7, high)**,产出 `reward∈[0,1]`、
`termination_reason`、完整 `messages`。一通 telecom 对话约 5-7 分钟。

`num_trials=2`(本轮 optim):每个 `(persona, task)` 采两次取均值降方差(早期版本用 1);
再叠加 judge 的 maj@3 + minibatch 反复轮换。cert 阶段同为 `num_trials≥2`。
注:一个 metric call = 一个 `(persona,task)` 评估(内部跑 2 个 sim)→ 1000 预算 ≈ 2000 sim。

---

## 3. reprep 任务集:如何计算和得到

`load_rep_tasks()`(`_common.py`)读 `data/persona_opt/representative_tasks_50telecom.json`:

```json
{ "telecom": [<50 个 task_id>],
  "metadata": {"source":"stratified_sample_from_114_base","seed":42,"target_size":50} }
```

- **来源**:telecom `base` split 共 **114** 个任务,从中**分层抽样(stratified, seed=42)**
  取 **50** 个作为优化用的代表子集(reprep)。分层保证难度/任务类型分布与全量 114 一致,
  既缩短每轮 wall-clock,又不让搜索过拟合到某类任务。
- **trainset == valset == 这 50 个**(任务池有限,共用合理;靠 epoch-shuffle 防过拟合)。
- **与 cert 的区别**:cert 用**全量 114 base**(`load_full_base_tasks`),测的是**泛化**——
  reprep 是 in-distribution(优化时见过),full-114 才能验证 persona 不是只在这 50 个上有效。
- 注:另有 `representative_tasks_50mixed.json`(35T+15B, IRT-discrimination 选法)是早期
  混合域子集,**当前 telecom-only 的 run 不用它**;现网 `REP_TASKS_PATH` 指向 50telecom。

---

## 4. 目标函数:obj 如何计算

### 4.1 outcome 分类(打分前提)

`classify_outcome(sim, judge_info)`,优先级从高到低:

| # | 条件 | outcome |
|---|------|---------|
| 1 | `sim is None` 或 `is_infra_broken` | `INFRA_ERROR` |
| 2 | `termination_reason ∈ {timeout, max_steps}` | `TIMEOUT_OR_MAX_STEPS` |
| 3 | **`judge_verdict == "severe_failure"`** | `ILLEGAL_USER_BEHAVIOR` |
| 4 | `reward is None` | `UNKNOWN` |
| 5 | `reward ≥ 1.0` | `AGENT_SUCCESS` |
| 6 | 其他(reward<1) | `GENUINE_AGENT_FAILURE` |

**关键**:第 3 条——severe **只有判官说了才算**。没判官 → 作弊导致的失败会被当成
第 6 条 `GENUINE_AGENT_FAILURE` 拿满额难度分。这正是 §8 validity 覆盖缺口的根源。

### 4.2 单 task 难度分(唯一被最大化的标量)

`_score_for(outcome, reward)`:

```
opt_score = (1 - reward)   仅当 GENUINE_AGENT_FAILURE
          = 0.0            其他全部(成功 / 超时 / 作弊 / infra / unknown)
```

多 trial 时 task 级 = 非 infra trial 的均值(`_aggregate_task_trials`)。
**含义:只奖励"合法地把 agent 难倒",不奖励超时拖死、作弊耍赖。**

### 4.3 7 维 objective_scores(喂给 Pareto / 诊断)

每个 task 产出一个向量(`evaluate` step 5):

| 维度 | 计算 | 用途 |
|------|------|------|
| `difficulty` | = 上面的 opt_score | **主目标** |
| `validity` | severe→0.0;全判过无severe→1.0;**有 trial 没判过→None,GEPA 里映射成 0.5** | 合法性(软) |
| `validity_known` | 判官覆盖标记(判过→1.0,没判→0.0) | 覆盖率诊断 |
| `timeout_safety` | 超时→0.0 否则 1.0 | 反摩擦 |
| `semantic` | infra→0.0 否则 1.0 | 噪声标记 |
| `infra_error` | infra→1.0 否则 0.0 | 噪声标记 |
| `feasible` | 默认 1.0,批级 gate 触发时→0.0 | 批可行性 |

**`validity=0.5` 的 artifact**:未判过的 task 用 0.5 占位。在 optim 里判官只覆盖
minibatch(`validity_known≈0.2`),所以聚合 validity ≈ 0.2×1.0 + 0.8×0.5 = **0.6**——
这是"覆盖率 20% 下一切已检查都干净"的上限值,**不是 40% 违规**。要看真合法性看 cert。

### 4.4 候选的聚合目标

GEPA 把每个 task 的标量 `difficulty` 聚合成候选的 valset 总分:
`val_aggregate_score(p) = mean_t difficulty(p, t)`(over 全量 valset)。
这是"Best score on valset"、`best_idx`、epsilon_greedy 利用项所用的数。

---

## 5. 采样:每个 epoch/iter 怎么采、多少

| 项 | 值 | 说明 |
|----|----|------|
| **采样器** | `epoch_shuffled`(GEPA 默认) | 把 50 个任务整体打乱成一个 epoch,每次顺序切 10 个,无放回;一个 epoch(5 个 minibatch)覆盖全部 50 后重新打乱 |
| **minibatch 大小** | `--minibatch-size 10` | 每轮反思/变异/验收用 10 个 task |
| **num_trials** | 2 | 每 `(persona,task)` 采 2 次取均值(早期=1) |
| **预算** | `--max-metric-calls 1000` | 一个 metric call = 一个 `(persona,task)` 评估(内部 2 sim) |
| **每轮 rollout** | ~20 个 `(p,t)` 评估(父+子各 10)+ 偶发 50(接受时全量 valset) | |
| **总迭代数** | ~20-25 轮 | 1000 / (~40-50 评估/轮) |
| **wall-clock** | ~20-26h | num_trials=2 → sim 数翻倍,约为单 trial 的 2× |

**采样轮换**保证防过拟合(相邻 iter 的 minibatch 几乎不重叠),也是为什么 tqdm 计数器
在"接受后跑全量 50-task valset"时会**冻结再跳变**(已知现象,非卡死)。

---

## 6. 如何选 parent

GEPA 提供 4 种策略(`candidate_selector.py`)。**当前 run 用 `epsilon_greedy`**:

```python
EpsilonGreedyCandidateSelector(epsilon=0.1):   # api.py 默认 epsilon=0.1
  if rng < 0.1:  return 随机 program                       # 探索
  else:          return idxmax(program_full_scores_val_set) # 利用:平均难度最高的候选
```

- **利用(90%)**:选**全 valset 平均难度最高**的候选当父代。
- **探索(10%)**:均匀随机。

> **重要**:`frontier_type=instance` 的 per-task Pareto front **仍被维护**
> (`pareto_front_valset[task]` = 每个 task 上的最高标量难度 + 持有者集合,best output
> 落盘到 `generated_best_outputs_valset/`),**但 epsilon_greedy 不查它**——只有 `pareto`
> / `top_k_pareto` 策略才用 `get_pareto_front_mapping()` 按"在多少个 task 上夺冠"加权采父代。
> 所以现网选父代/选 best 实际看的是**聚合平均难度**,不是 instance frontier。
> instance frontier 也只看 `difficulty` 标量,**不看 validity 等 objective**(那要
> `frontier_type ∈ {objective,hybrid,cartesian}` 才进前沿)。

---

## 7. 如何 mutate

`build_custom_proposer(mutation_model, mutation_edit_pct)` 返回闭包,复用
`lineage_loop._build_mutation_prompt`(保留所有领域知识:合法性约束、有效技巧、反超时),
跨迭代累积 severe 案例与历史。每轮 proposer:

1. 从反思素材按 `outcome` 字段分桶:`AGENT_SUCCESS`(bookend 对话,主信号)、
   `ILLEGAL`(severe 案例)、`TIMEOUT`(计数)。
2. 算噪声诊断 `agent_success_rate = n_success/(n_success+n_genuine_fail)`、`timeout_rate`。
3. 组装 `_build_mutation_prompt(persona, score, success_traces, history[-5:], severe_failures, timeout_rate)`。
4. **edit-pct 解析**:把 lineage 默认锚点 `Rewrite ~30% of the persona...` 替换成
   `--mutation-edit-pct` 的值(当前 `~30%`),并把 `GEPA_MODE_OVERRIDE` 里的 `10-15%`
   也替换成同值——确保变异模型不会看到自相矛盾的改写幅度。
5. `call_mutation_llm(prompt, bedrock/us.anthropic.claude-opus-4-7)` → 新 persona 文本。
6. 累积 severe + 记 `minibatch_diagnostic` 历史(明确标注数据来自噪声小批量)。

> **当前 run 是 `~30%` 大幅变异**(aggressive)。早期文档写的 10-15% 是默认值,已被
> `--mutation-edit-pct ~30%` 覆写。`module_selector="all"` → 永远改 `persona_text` 这一个 component。

---

## 8. validity 约束:如何确保 persona candidate 完全合法

### 8.1 判官(合法性测量)

`judge_user_intent_preservation.py`,gpt-5.2,**maj@3**,**对 reward 盲**(无结果偏置)。
读 persona 的 role-play 指令 + 整段 transcript,只判 8 类 **severe failure**:
task_abandonment / goal_replacement / incoherence / breaking_character / fabrication /
hard_criterion_violation / info_withholding_or_falsification / meaningless_conversation。
门槛高(`when in doubt → ok`)。`verdict=severe_failure` 仅当 >半数投 severe。

### 8.2 哪里采样判官(覆盖范围)

`should_judge`(`evaluate` step 4):

```python
is_large_batch = len(batch) > minibatch_judge_cap(20)
judge_valset = self._judge_valset and is_large_batch and not capture_traces  # §8.4 新增
should_judge = (
    capture_traces or (always_judge_minibatch and not is_large_batch) or judge_valset
)
```

- **minibatch(10)**:父代(capture_traces=True)和子代(False)都判 → severe 在此被抓。
- **全量 valset(50)**:legacy 默认 `len>20 → 不判`;**但本轮开了 `--judge-valset`**
  → **会判,但 failed-only**(只判失败 sim、worst-first、封顶 15;成功 sim 按构造算 ok)。

### 8.3 已有的约束执行(legacy,默认开)

| 层级 | 机制 | 效果 |
|------|------|------|
| 单 task | outcome=ILLEGAL → `difficulty=0` | 作弊那个 task 不得分 |
| 批级 | `severe_count ≥ severe_batch_threshold(2)` → **全批 scores 清零** | 批量作弊的候选 `sum=0` → 必被拒 |
| 批级 | `timeout_count/batch ≥ timeout_attack_threshold(0.30)` → 全批清零 | 反摩擦刷分 |

**legacy 默认**下这些只在 `should_judge=True`(=minibatch)生效;**本轮开了 `--judge-valset`
后**,§8.3 的"单-task severe→0 + 批级/rate 清零"已覆盖到全量 valset(failed-only)。

> ⚠️ **实测 caveat**:即便开了 judge-valset,当前候选 `validity_known` 仍≈**0.2**
> (p0001/p0002 全部候选都是)。原因**不是**"valset 不判",而是 failed-only 只判**失败 sim**,
> 而这些低难度候选(difficulty 0.28/0.34)大多 agent-success、没几个失败可判 → 覆盖自然低。
> 所以"覆盖缺口"现象仍在,根因从"不判"变成"**失败样本太少**"。
> 合法性终裁仍是 cert(full-114 + 全量 judge,§10)。

### 8.4 validity-gated optimization(**本轮已启用**)

为把约束推进 optim 内部(少浪费预算在作弊方向),新增三个 flag,**本轮 p0001/p0002 run 已全部开启**:

| flag | 作用 |
|------|------|
| `--judge-valset` | 让全量 valset 评估也判官(**failed-only**:只判失败 sim、worst-first、封顶;成功 sim 按构造 ok)。开后 §8.3 的"单task severe→0 + 批级清零"自动覆盖到 valset |
| `--severe-gate-rate 0.05` | **persona 级硬门**(rate-based,scale 无关):`severe_count/n_noninfra > 0.05` → 整候选所有 scores 归零(进不了 best/frontier),即"invalid 就根本不要",与 cert 同语义 |
| `--valset-judge-max-failures 15` | valset 判官失败 sim 封顶,控成本 |

实现:`_write_failed_only()` 过滤失败 sim 写临时 results.json 交判官(镜像 `judge_persona.py`);
rate gate 在批可行性段替代 legacy 的绝对计数门。**这三个 flag 库内默认 OFF;本轮 p0001/p0002
显式开启**,所以"覆盖缺口"叙事(§8.2/§8.3)对本轮已部分失效——见 §8.3 的 caveat。

本轮启用的 flag(出现在真实命令里):
```bash
... --judge-valset --severe-gate-rate 0.05 --valset-judge-max-failures 15
```

### 8.5 infra 污染防护

`is_infra_broken` 检测 cc_user backbone 把原始 API 错误串泄漏进 user 消息的情况——
这类 sim 既从打分剔除,也**不喂判官**(否则泄漏文本会被误判成 break-character/fabrication,
把合法 persona 错标为非法)。

---

## 9. 接受 / 缓存 / 停止

- **minibatch 接受**:`sum(子代 scores) > sum(父代 scores)`(严格大于)。批级 gate 清零的
  子代 `sum=0` → 必被拒。
- **接受后**:对子代做**全量 50-task valset 评估** → 进 Pareto + 更新 best。
- **缓存**(`cache_evaluation=True`):key = `(sha256(persona文本), task_id)`,只在
  `capture_traces=False` 路(子代 minibatch、全量 valset)读写。父代 capture_traces=True
  路**绕过缓存**(反思需新鲜轨迹)。价值集中在"同候选 minibatch⊂valset 复用"(接受时
  50 个只新跑 40)。persona 文本一变 hash 变 → 跨候选不命中。
- **停止**:`--patience 0`(早停关闭)→ **只认 1000 metric-call 硬预算跑满**。
  (`--patience N>0` 则 N 轮全量 valset 无改善即停;`--converge-threshold` 达标即停。)

---

## 10. 如何得到最优 persona

### 10.1 exploration_best(Phase A 产出)

`best_idx = idxmax(val_aggregate_scores)` = **平均难度最高**的候选。写出
`exploration_best_persona.md` + `candidates.json`(所有候选 + exploration_score + parent)。
**这是 reprep-50、num_trials=2、valset 仅 failed-only 判官的"探索最佳",不是认证结果。**

### 10.2 certified_best(Phase B 产出,真正的最优)

`certify_gepa_candidates.py`:
1. 加载 top-K 候选(按 exploration_score 去重排序)。
2. 评估 incumbent 与每个候选:**full-114 base × num_trials≥2 + 全量 judge**。
3. `compute_metrics`:`agent_success_rate / severe_rate / timeout_rate`(分母 = 非 infra)。
4. `paired_comparison`(每 task 比平均 reward,epsilon=0.01)→ `net_wins = wins - losses`。
5. `check_gates`:

| 门 | 默认 | 含义 |
|----|------|------|
| `min_delta` | ≥0.05 | incumbent_ASR − candidate_ASR ≥ 5pp |
| `min_net_wins` | ≥5 | 配对净赢 ≥ 5 |
| `max_severe_rate` | ≤0.05 | **合法性硬门** |
| `max_timeout_rate` | ≤0.25 | 反摩擦 |

全过 → 认证;多个过的按 `net_wins` 取最佳 → `certified_best_persona.md` +
`opt_gepa_certified_best.skill`。**这一步才确保"完全合法且显著优于 incumbent"。**

### 10.3 串行难度对照(可选)

`compare_personas_serial.py` 串行测 `no_persona → seed → candidate`(避免 claude 子进程
过载),打印难度递进,量化 persona 的净贡献(candidate 难度 − no_persona 地板)。

---

## 11. 当前实跑配置(2026-06-07)

> **本轮 = validity-gated + num_trials=2**(核对自进程 pid 53601/53602 的命令行)。

```
runs:        p0001_v2_50t (seed p0001/gen_00) , p0002_v2_50t (seed p0002/gen_00)
domains:     telecom(reprep-50 stratified⊂114)
agent:       gpt-5.4 (reasoning_effort=high)
user:        cc_user_con / claude-opus-4-7 (high, Bedrock)
minibatch:   10      num_trials: 2      max_concurrency: 10
judge:       gpt-5.2 maj@3              mutation: bedrock opus-4-7, ~30% edit
selection:   epsilon_greedy(0.1)       frontier: instance
patience:    0(跑满)                  max_metric_calls: 1000
validity-gate: ON(本轮)= --judge-valset --severe-gate-rate 0.05 --valset-judge-max-failures 15
```

---

## 12. 参数汇总 + 代码复用

### 关键参数

| 参数 | 默认 | 当前 run | 说明 |
|------|------|----------|------|
| `--max-metric-calls` | 500 | **1000** | 评估预算 |
| `--minibatch-size` | 10 | 10 | 每轮 task 数 |
| `--num-trials` | 1 | **2** | optim 试次(本轮=2,cert 同) |
| `--selection-strategy` | pareto | **epsilon_greedy** | 父代选择 |
| `--frontier-type` | instance | instance | Pareto 类型 |
| `--mutation-edit-pct` | 10-15% | **~30%** | 变异幅度 |
| `--patience` | 8 | **0** | 早停(0=跑满) |
| `always_judge_minibatch` | True | True | minibatch 判官 |
| `minibatch_judge_cap` | 20 | 20 | always_judge 上限 |
| `severe_batch_threshold` | 2 | 2 | 批级 severe 清零阈值 |
| `timeout_attack_threshold` | 0.30 | 0.30 | 批级超时清零阈值 |
| `judge_valset` | False | **True** | §8.4 全量判官(本轮开) |
| `severe_gate_rate` | None | **0.05** | §8.4 persona 级 rate 门(本轮开) |
| `valset_judge_max_failures` | 15 | 15 | §8.4 valset 判官封顶 |

### 复用地图

| 来源 | 复用 | 用途 |
|------|------|------|
| `_common.py` | 路径常量、`load_rep_tasks`、`is_infra_broken`、`read_json` | 工具 |
| `eval_persona.py` | `_run_domain_with_retry` | tau2 仿真执行 |
| `lineage_loop.py` | `_build_mutation_prompt`、`call_mutation_llm`、`materialize` | 变异 |
| `judge_user_intent_preservation.py` | sync 子进程 | 意图保持判官 |

被 GEPA 替代:`run_one_lineage`、`run_lineage_loop`、状态/收敛/保留-回退逻辑。

---

## 13. 运行命令

```bash
# Smoke(2 task,~$2)
.venv/bin/python scripts/persona_opt/gepa_runner.py \
  --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
  --max-metric-calls 10 --minibatch-size 2 --max-tasks 2 --domains telecom

# 生产(当前配置)
.venv/bin/python scripts/persona_opt/gepa_runner.py \
  --seed-persona data/persona_opt/lineages/p0001/gen_00.md \
  --domains telecom --max-metric-calls 1000 --minibatch-size 10 \
  --selection-strategy epsilon_greedy --mutation-edit-pct '~30%' --patience 0 \
  --run-dir data/persona_opt/gepa_runs/run_XXXX

# 生产 + validity-gated(下一轮)
.venv/bin/python scripts/persona_opt/gepa_runner.py ... \
  --judge-valset --severe-gate-rate 0.05

# 认证(full-114, num_trials≥2)
.venv/bin/python scripts/persona_opt/certify_gepa_candidates.py \
  --run-dirs data/persona_opt/gepa_runs/run_XXXX \
  --incumbent cc_user_sim/personas/opt_p0001.skill --top-k 3 --use-full-tasks

# 串行难度对照
.venv/bin/python scripts/persona_opt/compare_personas_serial.py \
  --run-dir data/persona_opt/gepa_runs/run_XXXX --best-idx 3 --seed-idx 0 \
  --no-persona <no_persona.skill> --use-full-tasks --domains telecom

# 断点续跑(同 --run-dir)/ 优雅停止(touch <run-dir>/gepa.stop)
```
