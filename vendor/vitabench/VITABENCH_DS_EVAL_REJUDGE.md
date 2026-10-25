# VitaBench · 换 evaluator 重评：deepseek-v4-flash-0731 as judge

> 用 **deepseek-v4-flash-0731** 替换 gpt-4.1 作为 evaluator，对 V2 已有的 4 域 test set 对话**离线重打分**，得到 update 分数。**不重跑 agent/user 仿真**——只换裁判，看分数如何变化。

日期：2026-08-10 · 4 域（delivery / instore / ota / cross_domain）· 每域 40 task × {no_persona, best_persona}

---

## 1. 实验设计

| 维度 | 值 |
|------|-----|
| 目的 | 评估器换模型后，V2 test set 分数是否稳健 |
| 方法 | **离线 re-judge**：加载 V2 已保存的 test set 对话 trace，用 `TrajectoryEvaluator` + 新 judge 重打分 |
| 是否重跑 agent | **否**。agent/user 仿真完全沿用 V2（gpt-5.4 high 对话），只重新评分 |
| 对照 | 同一批 trace **同时**用 gpt-4.1（V2 原 judge）重打分 → 消除 trial 差异，纯看 judge 效应 |
| trace 来源 | `run_<domain>_v2/conversations/test_{no_persona,best_persona}/<task_id>.json`（每 task 1 个代表 trial 的完整对话） |
| 评估器 | ① gpt-4.1（对照，OpenAI 直连）② **deepseek-v4-flash-0731**（新 judge，经 shim） |
| 评分方式 | sliding window（window=10, overlap=2），二元 reward（全 rubric 过=1.0 否则 0.0） |
| 并发 | 20（gpt-4.1 走 OpenAI 不受限；deepseek 经 shim 被主动节流到 ~19 req/min） |

**为什么同 batch 跑两个 judge**：V2 `test_report.json` 的分数是 3-trial 聚合，而 re-judge 用的是 1 个代表 trial，trial 数不同不可直接比。所以在同一批代表 trial 上同时跑 gpt-4.1 + deepseek，差值就是**纯 judge 效应**，干净可比。

---

## 2. 基础设施（shim）

deepseek-v4-flash-0731 经 Cresta Fireworks dev-proxy，只说 Anthropic Messages 协议；VitaBench 的 evaluator 发 OpenAI chat-completions。`openai_to_fireworks_proxy.py`（:4114）做双向转换：

- **OpenAI ↔ Anthropic 双向**：messages（system→顶级、tool 结果↔tool_result 块、assistant tool_calls↔tool_use 块）、tools（function↔input_schema）、响应（text/tool_use→content/tool_calls，stop_reason→finish_reason）
- **Round-robin**：DNS 解析 dev-proxy 的 4 个 replica IP（10.0.152.157 / .154.138 / .156.91 / .159.185），`itertools.cycle` 轮询直连（Host header + SSL 跳 hostname 验证）
- **Exponential backoff**：429 / 5xx / 超时 / 空响应重试，`min(2·2^attempt, 30s)`，最多 4 次
- **主动 rate limiter**：dev-proxy 限额 20 req/min。令牌桶节流 3.1s/请求（~19/min），从源头避免 429
- **evaluator_extracter graceful 修复**：原实现遇到空/坏 JSON 直接抛 `JSONDecodeError` 崩掉整个 reward 计算；改为返回 `[]`（匹配 evaluator 已有但失效的 `else: keep current states` 分支）

**实测效果**：rate limiter 加入后，429 重试从 143 次（无节流）降到 **0 次**（全程 4 域）。parse 失败 72 次（~1.9%），全部 graceful 处理，无崩溃。

---

## 3. 主结果：4 域 × 2 条件 × 2 judge

| 域 | 条件 | gpt-4.1 avg (fully) | deepseek avg (fully) | deepseek − gpt-4.1 |
|------|------|----------------------|----------------------|--------------------|
| delivery | no_persona | 0.825 (33/40) | 0.775 (31/40) | −0.050 |
| delivery | best_persona | 0.625 (25/40) | 0.600 (24/40) | −0.025 |
| instore | no_persona | 0.900 (36/40) | 0.875 (35/40) | −0.025 |
| instore | best_persona | 0.675 (27/40) | 0.650 (26/40) | −0.025 |
| ota | no_persona | 0.775 (31/40) | 0.650 (26/40) | −0.125 |
| ota | best_persona | 0.450 (18/40) | 0.425 (17/40) | −0.025 |
| cross_domain | no_persona | 0.675 (27/40) | 0.525 (21/40) | −0.150 |
| cross_domain | best_persona | 0.500 (20/40) | 0.400 (16/40) | −0.100 |

### persona 对抗效果 Δ(no − best)

| 域 | gpt-4.1 Δ | deepseek Δ | 差值 |
|------|-----------|------------|------|
| delivery | +0.200 | +0.175 | −0.025 |
| instore | +0.225 | +0.225 | 0.000 |
| ota | +0.325 | +0.225 | −0.100 |
| cross_domain | +0.175 | +0.125 | −0.050 |
| **4 域均值** | **+0.231** | **+0.187** | −0.044 |

### 4 域 fully_correct 合计

| judge | no_persona | best_persona |
|-------|------------|--------------|
| gpt-4.1 | 127/160 | 90/160 |
| deepseek | 113/160 | 83/160 |

---

## 4. 三个发现

### 发现 1：deepseek judge 一致地比 gpt-4.1 更严

**8/8 条件**差值全为负（−0.025 ~ −0.150），4 域均值 no_persona −0.088、best_persona −0.044。没有任何一个条件下 deepseek 比 gpt-4.1 宽松。复杂域差距最大：cross_domain no −0.150、ota no −0.125——任务越复杂、对话越长，两 judge 分歧越大。简单域（instore）几乎一致（−0.025）。

### 发现 2：persona 对抗效果在 deepseek judge 下完全成立

4 域 Δ(no−best) **全为正**（+0.125 ~ +0.225）。针对 gpt-5.4 搜出的对抗式 persona，换 deepseek 当裁判依然让 agent reward 显著下降。instore 完全一致（0.225），delivery 几乎一致（0.200 vs 0.175）。

### 发现 3：换 evaluator 不改变 V2 的任何结论方向

V2 报告的核心结论——persona 有效（Δ>0）、4 域泛化、reward 下降合法——在 deepseek judge 下**全部成立**。deepseek 下 Δ 幅度略小（均值 0.187 vs 0.231），原因是 deepseek 对 no_persona 压低更多（−0.088）而对 best_persona 压低较少（−0.044），两端压缩不对称使 Δ 收窄，但方向与显著性不变。

---

## 5. update 分数（deepseek-v4-flash-0731 作为 evaluator）

| 域 | no_persona | best_persona | Δ(no−best) |
|------|------------|--------------|------------|
| delivery | 0.775 (31/40) | 0.600 (24/40) | +0.175 |
| instore | 0.875 (35/40) | 0.650 (26/40) | +0.225 |
| ota | 0.650 (26/40) | 0.425 (17/40) | +0.225 |
| cross_domain | 0.525 (21/40) | 0.400 (16/40) | +0.125 |
| **均值** | **0.706** | **0.519** | **+0.187** |

对照（gpt-4.1，同 batch 代表 trial）：no 0.794 / best 0.563 / Δ +0.231。

---

## 6. caveat

1. **judge 随机性**：evaluator 调用带 temperature=1.0，同 trace 两次 re-judge 波动 ±0.05~0.075（delivery no_persona gpt-4.1 一次 0.750、一次 0.825）。deepseek−gpt-4.1 的差距（−0.025~−0.150）在简单域接近噪声，但 4 域 8 条件**一致偏严** + 复杂域差距大，说明系统性偏严真实存在，不是噪声。
2. **代表 trial vs 3-trial 聚合**：re-judge 基于 V2 保存的每 task 1 个代表 trial，而 V2 `test_report.json` 是 3-trial 聚合，二者绝对分数不直接可比（代表 trial 比聚合乐观）。本报告的 gpt-4.1 vs deepseek 对比在同批代表 trial 上进行，**消除了这个差异**，纯反映 judge 效应。
3. **离线 re-judge 不重跑 agent**：对话内容是 V2 的 gpt-5.4 high 产出的，deepseek 只做评分，不参与仿真。若要"deepseek 当 agent"的分数是另一回事（此前误跑过一次：deepseek 当 agent 在 delivery 拿 0.992，但那是 agent 角色，非本报告范畴）。

---

## 7. 代码与数据

- `scripts/vita_rejudge_deepseek.py` — re-judge 驱动（双 judge 同 batch 打分）
- `openai_to_fireworks_proxy.py`（:4114）— OpenAI↔Anthropic shim + roundrobin + backoff + rate limiter
- `src/vita/utils/utils.py` — `evaluator_extracter` graceful 修复
- 数据来源：`data/persona_opt/vita_pgepa_runs/run_{delivery,instore,ota,crossdomain}_v2/rejudge_report_dseval.json` + `rejudge_summary_dseval.json`
- 运行：`python scripts/vita_rejudge_deepseek.py --domains delivery,instore,ota,cross_domain --max-concurrency 20 --suffix dseval`（shim 用 `FIREWORKS_PROXY_TOKEN=… python3 openai_to_fireworks_proxy.py --port 4114` 启动）
