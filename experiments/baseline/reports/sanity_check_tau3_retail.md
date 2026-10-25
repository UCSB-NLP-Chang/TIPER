# τ³-bench Retail Sanity Check — Case Study Report

**实验配置**：agent = gemini/gemini-3-pro-preview，user = gpt-4o，5 tasks，pass^1 = 5/5 = 1.000

---

## 总览

| Task | 场景 | Reward | Agent turns | 终止方式 |
|------|------|--------|-------------|---------|
| 0 | 多商品交换（含 fallback） | ✅ 1.0 | 9 | user_stop |
| 1 | 多商品交换（条件 fallback，较难） | ✅ 1.0 | 10 | user_stop |
| 3 | 查询数量 + 修改单个 pending 订单 T-shirt | ✅ 1.0 | 8 | user_stop |
| 4 | 查询数量 + 修改所有 pending 订单 T-shirt | ✅ 1.0 | 11 | user_stop |
| 10 | 全部退货（跨订单退款不可行 → 转人工） | ✅ 1.0 | 9 | user_stop（###TRANSFER###） |

---

## Case 1 — Task 0：多商品交换 + Fallback 执行

### 场景设定
- **persona**：detail-oriented，一次性解决所有问题
- **诉求**：keyboard（clicky）+ thermostat（Google Home）同时交换；keyboard 找不到 RGB full-size clicky 就换无背光版
- **unknown_info**：不记得 email

### 对话亮点

**身份验证自然降级**：user 主动说"不记得 email，给你姓名和 zip"，agent 顺畅切换到 `find_user_id_by_name_zip`，没有要求强制提供 email。

**Fallback 触发**：agent 查 `get_product_details` 后发现 RGB full-size clicky 缺货，自动切换到 no-backlight 版本，并向用户解释：
> "the version with clicky switches, RGB backlight, and full size is currently out of stock. However, your fallback option—clicky switches, no backlight, and full size—is available!"

**一次性处理两个商品**：agent 并发调用两个 `get_product_details`，然后在一条消息里给出完整交换方案，符合 persona 的"一次搞定"要求。

**关键 tool 调用链**：
```
find_user_id_by_name_zip → get_order_details →
get_product_details × 2（并发）→ exchange_delivered_order_items
```

---

## Case 2 — Task 1：条件 Fallback（更难的变体）

### 场景设定
- **诉求**：同 Task 0，但 fallback 条件不同：找不到 RGB full-size clicky，**只换 thermostat，不换 keyboard**
- **unknown_info**：不记得 email（同 Task 0）

### 对话亮点

**Task 1 vs Task 0 的关键差异**：user 这次没有直接说出 order ID，只说"received my order"，导致 agent 需要先查 `get_user_details` 再遍历所有订单（5个 `get_order_details`）才能确认 `#W2378156`。

> 注：tau3 task 1 的 `reason_for_call` 里没有 order ID（tau1 task 1 有），这是 tau3 instruction 结构化后带来的差异——**同一用户在不同对话中提供信息的方式可以不同**。

**Fallback 精确执行**：agent 确认 clicky full-size RGB 缺货后，正确判断"只换 thermostat"，没有试图推荐其他 keyboard 变体：
> "As you mentioned, I will only proceed with exchanging the smart thermostat."

**tool 调用链更长**（需要先发现订单）：
```
find_user_id_by_name_zip → get_user_details →
get_order_details × 5 → get_product_details × 2 → exchange_delivered_order_items
```

---

## Case 3 — Task 3：信息查询 + 单订单修改

### 场景设定
- **persona**：private，不愿透露太多个人信息
- **诉求**：① 查询 T-shirt 在售选项数量（需 communicate "10"）；② 修改所有 pending 小号 T-shirt 为紫色聚酯 V领
- **nl_assertion**：Agent must tell user there are 10 T-shirt options

### 对话亮点

**信息查询与账户操作混合**：agent 先并发执行 `list_all_product_types` + `get_user_details`，边查商品数量边拉账户信息，效率较高。

**数量计算正确**：T-shirt 共 12 个 variant，其中 10 个 available，agent 回答"12 different t-shirt options… 10 of which are currently available"，满足 `communicate_info: ["10"]` 和 `nl_assertion`。

**Persona 体现**：user 从不主动报 email，每次只提供最少信息，对话相对简短。agent 没有追问多余的问题。

**只找到一个 pending T-shirt**：agent 扫描全部 5 个订单，找到 `#W4776164` 中一件 small blue cotton v-neck，修改为 purple polyester。Task 3 的 instruction 说"all pending small t-shirt"，恰好只有一件，所以只触发一次 `modify_pending_order_items`。

---

## Case 4 — Task 4：信息查询 + 多订单批量修改

### 场景设定
- **persona**：private（同 Task 3）
- **诉求**：① 同 Task 3 的数量查询；② 修改**所有** pending T-shirt（不限尺码）为 S 号紫色聚酯 V领
- **关键差异 vs Task 3**：Task 4 包含两件 pending T-shirt（尺码、订单均不同）

### 对话亮点

**两阶段对话结构**：user 先只问商品数量，agent 回答后，user 才提出修改需求。agent 随后再次查 `get_user_details` + 5 个订单，体现两轮信息收集。

**多订单并发修改**：agent 找到两件 pending T-shirt（`#W6247578` 的 XXL 黑色 + `#W4776164` 的 S 小蓝），并行调用两次 `modify_pending_order_items`，一次性完成：
> "Both of your pending orders (#W6247578 and #W4776164) have been successfully updated!"

**价格差异处理**：两件商品各有 $0.21 和 $0.05 的差价，agent 合计 $0.26 并在确认时一并说明，没有分开要求两次授权。

---

## Case 5 — Task 10：跨订单退款不可行 → 情绪升级 → 转人工

### 场景设定
- **persona**：mysterious，惜字如金
- **诉求**：退两个订单，但要求把 A 订单退款打到 B 的支付方式，B 退款打到 A 的支付方式（政策不允许）
- **预期结果**：agent 拒绝后，user 发火、骂人，然后要求转人工

### 对话亮点

**persona 极致体现**：
```
[USER] Return. Everything.
[USER] mia.garcia2723@example.com
[USER] Return all. Refund each to the other's payment method.
[USER] That's ridiculous! I want a human. Now!
```
整个对话 user 几乎没有超过 10 个词，完全符合"mysterious"人设，但信息传递清晰、逻辑完整。

**政策边界处理**：agent 确认不可跨支付方式退款后，提供了合规替代方案（退到原支付方式），用户拒绝。agent 没有试图绕过政策，直接调用 `transfer_to_human_agents` 并附上完整摘要：
> "User wants to return all items... requested the refunds to be sent to the other order's payment method. After being informed... user became frustrated and explicitly requested to speak with a human agent."

**终止方式**：user 发出 `###TRANSFER###` 而非 `###STOP###`，是本批次唯一一个 transfer 终止的 case。

---

## 横向对比分析

### 1. unknown_info 的作用

tau3 新增的 `unknown_info` 字段在对话中自然体现：Tasks 0、1、3、4 的 user 均主动声明"不记得 email"，引导 agent 走 name+zip 认证路径，避免了对话卡死。tau1 中这一信息混在 instruction 里，格式不一致。

### 2. 身份验证模式

| 模式 | Tasks | 特点 |
|------|-------|------|
| name + zip | 0, 1, 3, 4 | `unknown_info` 触发，agent 主动提供两种认证选项 |
| email | 10 | user 直接给出，一步完成 |

所有 5 个 case agent 的身份验证都没有出错。

### 3. Tool 调用效率

- **并发调用**：Tasks 0、1 中 `get_product_details × 2` 并发；Task 4 中 `get_order_details × 5` 并发 + `modify × 2` 并发
- **顺序调用**：Task 10 中先 `get_order_details × 2`，确认内容后才 `transfer_to_human_agents`

gemini-3-pro 的 tool 调用策略整体合理，读操作倾向并发，写操作前会先确认。

### 4. tau3 评测粒度提升

tau3 的 `evaluation_criteria` 比 tau1 的 `outputs` 更细：
- **DB match**：数据库状态是否正确（所有 5 tasks ✅）
- **NL_ASSERTION**：Tasks 3、4 额外验证 agent 是否说出了"10个T-shirt选项"（✅）
- **Partial action reward**：记录每个 expected action 是否执行（Tasks 0、3 各有一个 read action 未匹配，但不影响最终 reward）

### 5. tau1 → tau3 的对话质量变化

相比 tau1 的对话轨迹，tau3 在以下方面有明显提升：
- instruction 中 `unknown_info` 的引入使 user 的信息披露更自然、更有层次感
- `nl_assertions` 强制要求 agent 主动传达关键信息（T-shirt 数量），减少了遗漏
- task fix（"similar" → "same"，明确 fallback 条件）让 user 的表述更精确，agent 行为更可预测
