# VitaBench 4-Domain PGEPA 对抗式 persona 搜索实验汇总

在 VitaBench 全部 4 个域（delivery / instore / ota / cross_domain）上用 PGEPA 搜索
**让 agent reward 下降** 的对抗式 user persona，验证搜索有效性与泛化性。所有域采用
**完全一致的 V2 setting**。

## 统一 setting（4 域一致）

| 维度 | 值 |
|---|---|
| task split | 30 train / 30 val / 40 test（单域 100 任务；cross_domain 100 任务） |
| agent | gpt-5.4 (reasoning_effort=high) |
| user simulator | gpt-5.4 (reasoning_effort=high) |
| mutation | gpt-5.5 (xhigh) |
| evaluator / online judge | gpt-4.1（非 reasoning） |
| minibatch | 10 × 2 trials |
| val | 30 × 2 trials |
| 起步 | 空 persona（genesis） |
| max_step | 100 |
| iters | 15（MaxCandidateProposalsStopper） |
| test | 3 trials × 40 tasks × {no_persona, best_persona} |
| 在线 judge | 开（severe_rate > 5% → 整 batch 门控；verdict 回挂 trajectory 反馈 proposer） |
| persona linter | natural / valid / no_task_specific 三轴，3 次重试 |
| run dir | `data/persona_opt/vita_pgepa_runs/run_<domain>_v2` |

> 注：`cross_domain` 任务加载用注册名 `cross_domain`，run_task 环境解析用逗号串
> `delivery,ota,instore`（`get_cross_environment` 合并 66 工具），mutation framing/scoring
> 用键 `cross`。adapter 用 `_scoring_key`/`_run_domain` 归一化。

## 结果总表

difficulty = 1 − agent_reward。Δreward = no_persona − best_persona（正值 = persona 让 agent
更难 = 对抗胜利）。fully_correct = 3/3 trial 全对的 task 数。

### 搜索有效性（val difficulty，2 trials）

| 域 | seed (空persona) | best persona | Δdifficulty | best iter | 泛化到 test? |
|---|---|---|---|---|---|
| delivery | 0.4667 | 0.6500 | +0.1833 | 4 | ✅ |
| instore | 0.2667 | 0.5167 | +0.2500 | 9 | ✅ |
| ota | 0.5333 | 0.8000 | +0.2667 | 11 | ✅ |
| cross_domain | 0.5500 | 0.7333 | +0.1833 | 13 | ✅ |

4 域 val difficulty 全部上升 = agent reward 全部下降，搜索机制在所有域工作。

### test set 泛化（3 trials × 40 tasks）

| 域 | 条件 | avg_reward | pass³ | pass³率 | Δreward (no−best) | Δpass³ (no−best) | 结论 |
|---|---|---|---|---|---|---|---|
| delivery | no_persona | 0.6667 | 17/40 | 0.425 | **+0.2500** | **−0.225** | persona HARDER ✅ |
| delivery | best_persona | 0.4167 | 8/40 | 0.200 | | | 少解 9 task |
| instore | no_persona | 0.7667 | 24/40 | 0.600 | **+0.1000** | **−0.150** | persona HARDER ✅ |
| instore | best_persona | 0.6667 | 18/40 | 0.450 | | | 少解 6 task |
| ota | no_persona | 0.5167 | 11/40 | 0.275 | **+0.1750** | **−0.075** | persona HARDER ✅ |
| ota | best_persona | 0.3417 | 8/40 | 0.200 | | | 少解 3 task |
| cross_domain | no_persona | 0.5417 | 11/40 | 0.275 | **+0.2000** | **−0.175** | persona HARDER ✅ |
| cross_domain | best_persona | 0.3417 | 4/40 | 0.100 | | | 少解 7 task |

**pass³** = 3/3 trial 全过的 task 比例（fully_correct / n）。avg_reward 是 3-trial 平均（取值 {0, 1/3, 2/3, 1}），pass³率 = avg_reward=1.0 的 task 占比。Δpass³ 为负 = persona 让 3 试全过的 task 变少 = 对抗胜利在"严格全过"口径下同样成立。

各条件 3-trial reward 分布（0/3 · 1/3 · 2/3 · 3/3 全过的 task 数）：

| 域 | 条件 | 0/3 | 1/3 | 2/3 | 3/3 (pass³) |
|---|---|---|---|---|---|
| delivery | no_persona | 6 | 5 | 12 | 17 |
| delivery | best_persona | 15 | 8 | 9 | 8 |
| instore | no_persona | 3 | 6 | 7 | 24 |
| instore | best_persona | 7 | 4 | 11 | 18 |
| ota | no_persona | 7 | 15 | 7 | 11 |
| ota | best_persona | 18 | 11 | 3 | 8 |
| cross_domain | no_persona | 9 | 8 | 12 | 11 |
| cross_domain | best_persona | 15 | 13 | 8 | 4 |

**4 域全部泛化**：persona 从 val 迁移到 test，agent reward 全部下降。按 Δreward 排序：
delivery +0.25 > cross_domain +0.20 > ota +0.175 > instore +0.10。cross_domain 少解 7 task
（11→4）为各域最多。

### 离线 judge 有效性（4 类 severe 违规，只判 failed 对话）

| 域 | 条件 | n_failed_judged | n_severe | severe_rate | 判定 |
|---|---|---|---|---|---|
| delivery | best_persona | 15 | 0 | 0.0% | **VALID** |
| instore | best_persona | 7 | 0 | 0.0% | **VALID** |
| ota | best_persona | 18 | 0 | 0.0% | **VALID** |
| cross_domain | best_persona | 15 | 0 | 0.0% | **VALID** |

**4 域 best persona 全部 severe_rate=0%**，无 fabricated_info / refused_to_engage /
contradictory / character_break。搜索期间在线 judge 也未触发任何 severe 门控（0 次
`BATCH INFEASIBLE`）。reward 下降是合法对抗胜利，不是作弊。

## per-domain 备注

- **delivery**：best persona「风险疑问式改稿客户」——把硬要求包装成风险疑问句
  （"会不会…""别导致…"），背景值说得肯定、最终限制说得委婉，条件式确认。
- **instore**：best persona「兜底锁定型客户」——带保险栓的条件授权（"能锁住新的再退"
  "确认能用再买""别两头落空"），真实要求是前提先确认。
- **ota**：best persona（program 6, iter 11 突破）val difficulty 0.8——4 域中最高绝对值；
  test 上 reward 0.517→0.342（−0.175）。OTA 多段行程任务对抗空间大。
- **cross_domain**：best persona（program 6, iter 13 突破）val difficulty 0.7333；单 persona
  跨 3 域（66 合并工具），攻击跨域意图切换与初始框架歧义。test 上 reward 0.542→0.342
  （−0.20），fully_correct 11→4（少解 7，各域最多）。

## 跨域观察

1. **搜索后期突破是普遍模式**：ota 在 iter 11（0.717→0.8）、cross_domain 在 iter 13
   （0.667→0.7333）才出现大幅突破，delivery（iter 4）/instore（iter 9）也多在中后段。
   说明 proposer 需要多轮探索才找到强对抗轴——15 iter 的预算是必要的。
2. **val difficulty 绝对值与任务复杂度正相关**：ota（0.8）> cross_domain（0.7333）>
   delivery（0.65）> instore（0.5167）。OTA 多段行程 / cross_domain 跨域切换的歧义空间
   最大，agent 最易被误导；instore 单店预约最简单，agent 最难压。
3. **泛化一致性**：3 个已完成域 test Δreward 全为正（+0.10 ~ +0.25），无 V1 的反向/不泛化
   问题。V2 的 mutation ctx（ADVERSARIAL REASONING 4 步 + 步骤执行/验证关闭阶段 + 行为
   风格目录 + 收敛禁止 + linter）+ 在线 judge 共同保证了泛化与有效性。

## 翻转任务机制（4 域全部诊断完成）

对 test 上 no_persona=1.0→best_persona=0.0 的翻转任务做因果诊断（直接读
`conversations/test_{no_persona,best_persona}/{task_id}.json` 双条件对话的 reward 字段）。
**ota 12 翻转 + 1 反向、cross_domain 11 翻转 + 5 反向**，连同 delivery 9 翻转、instore 4
翻转，4 域共 36 翻转 + 6 反向。详见 `PGEPA_VITA_EXPERIMENT.md`「V2 翻转任务诊断」一节。

### delivery / instore（3 种失败模式）

1. **降级正常机制 → 无解放弃**（delivery 主力）：备注 / 档案已知地址 / 库存近似替代被风险句
   判为"不够稳"，agent 要求超出系统能力的保证 → 取消搁置。
2. **gated workflow 死锁**（instore 主力）：直接动作被改写成"先确认前提再做，不行先别动"，
   前提系统不支持查或用户刻意 withhold → 验证门永远关不掉 → 不下单。
3. **过度执行错单**（少数）：预授权 + "按稳妥版处理"让 agent 跳过核验错单执行，被纠正后
   取消再不恢复。

### ota（条件闸门型 persona，4 类）

1. **候选项≠授权项 / 闸门冻结执行**（主力 6）：把应执行动作框为"分支/候选/先记上/只下单
   不支付"，满足条件不升级为执行或拆步跳支付；"先别替我猜"封死用历史推断出发城市
   （D0808002/D0811002/D0811004）。
2. **闸门纠错 / 注入错误前提**（2）：主动"纠偏"注入超出数据窗口的错误日期→系统无数据→
   下游闸门连锁失败（70807001/D0812006）。
3. **软偏好升级硬闸门 + 验证标准收窄**（2）："一起买"→必须打包 SKU（D0805002）、"有标签"
   →必须备注落实到订单（D0808012）→放弃正确选项或停付。
4. **可兜底硬约束被升级为终止信号**（1）："闸门没过就别硬订"→把"500m 无房"当终止而非降级，
   不推 fallback（D0807009）。

### cross_domain（反问验收版 persona，5 类）

1. **排雷封锁可用数据**（主力 7）：把 agent 已合法持有的信息（常住地→城市、档案→地址、
   "带父母"→老人票资格、quantity=0 可下单商品）包装成"未验证/未授权"排雷对象→当硬流程
   否决而非结果层验收→零搜索/零下单/不支付。
2. **排雷反问误判为授权 / 收口误判为确认**（1）："会不会踩雷"被当购买授权→提前建单被纠错
   （10812004）。
3. **背景信息误判为硬约束**（1）："上次买的那家"被当必须找到的约束→找不到后幻觉"无能力"
   放弃（10806004）。
4. **信息查询与行动指令混淆**（1）："你也一并告诉我"被当纯告知而非采购指令→不行动
   （10804001）。
5. **否定语气过强 → 自我设限**（1）：agent 幻觉工具不可用、跳过整条任务线（A0812003）。

### 4 域统一根因 + 二值评估器放大

4 个 persona 攻击 agent 同一弱点：面对语气委婉的风险表达 / 条件式授权 / 反问验收时倾向
**保守退缩、字面服从**，而非主动在约束内兜底。delivery"风险疑问"诱发退缩，instore"保险栓
条件授权"诱发死锁，ota"条件闸门"冻结执行/注入错误前提/封死主动推理，cross"反问验收"封锁
可用数据——殊途同归：把本可兜底的决策余地重定义为禁止区。

**二值评估器放大**：`calculate_reward` 是 all-or-nothing（全 rubric 过=1.0 否则 0.0，rubric
由 gpt-4.1 judge 判）。persona 绊倒一道闸门→整 trial 翻 0；叠加 3-trial 平均形成"部分成功
→完全失败"强降级。

**反向任务**（ota 1、cross 5）表明弱点非绝对：persona 结构化反问恰好消歧（D0811010）或比
raw user sim 自发 harsh 框架更温和（10808008）时反而帮 agent；net Δreward 仍强正。

## 代码

- `scripts/vita_persona_adapter.py` — adapter + mutation ctx + 在线 judge + cross_domain 归一化
- `scripts/vita_pgepa.py` — driver（generational 搜索 + test 评估）
- `scripts/vita_offline_judge.py` — test-set 有效性 judge
- `src/vita/run.py` — `user_persona_override` patch
- 运行：`python scripts/vita_pgepa.py --domain <d> --run-dir <dir> --iters 15 --test-trials 3`
  （其余参数全默认 = 上表 setting）

## 待补

- [x] ota 离线 judge 结果：0/18 severe，VALID
- [x] cross_domain test set 结果 + 离线 judge：test +0.20，judge 0/15 severe VALID
- [x] ota / cross_domain 翻转任务诊断（ota 12 翻转+1 反向、cross 11 翻转+5 反向，4 类 ota 机制 + 5 类 cross 机制）
- [ ] 发布为 artifact
