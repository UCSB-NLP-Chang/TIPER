# Evaluator `invalid_prompt` 事故复盘（reasoning 模型不能当判官）

**日期**：2026-07-13
**影响**：一次完整的单域 delivery 放量（2 trials，200 runs）跑了 44 分钟，
**45 个任务全部失败、0 条落盘**。

## 一句话结论

**OpenAI 的 evaluator（判官）绝对不能用 gpt-5.x / o 系列 reasoning 模型。**
这些模型对**输入 prompt** 做严格内容审核，而 evaluator 会把整段对话塞进判分 prompt，
触发 `invalid_prompt`（"prompt flagged as potentially violating usage policy"）被 API 拒绝。
判官换成非 reasoning 的 **gpt-4.1** 即可（也正好是 paper 的跨家族判官选择）。

## 现象

放量配置（脚本 `scripts/run_cross_gpt52.sh` 当时的默认）：

| 角色 | 模型 | reasoning |
|---|---|---|
| agent | gpt-5.2 | xhigh + `--enable-think` |
| user | gpt-4.1 | 无 |
| **evaluator** | **gpt-5.2-eval**（gpt-5.2 底模，model_id 别名） | **low** |

跑 44 分钟后 `data/simulations/delivery_gpt52xhigh_final.json` 里 **0/200 落盘**。
日志 `/tmp/vita_delivery_final.log` 有 149 次 `invalid_prompt`，45 个 distinct 任务全部
`failed on attempt N/4`，0 个 `FINISHED SIMULATION`。

## 根因

API 返回的 body（`llm_utils.generate` 打印的 "Full response"）：

```json
{
  "error": {
    "message": "Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_prompt"
  }
}
```

注意报错链接指向 **reasoning** 指南 —— 这是 gpt-5.x/o 系列 reasoning 模型**专属**的输入审核。

失败发生在 **EVAL 阶段，不是 agent 阶段**。日志时间线固定为：

```
Processing window 1/3 with 10 messages     <- evaluator 在跑滑窗（evaluator_traj.py 的 print）
    "code": "invalid_prompt"               <- 判官 generate() 被拒
Task 30711001 failed on attempt 1/4: 'NoneType' object has no attribute 'content'
```

链路：
1. orchestrator 正常跑完 agent↔user，产出完整轨迹（所以有 `Processing window`）。
2. evaluator 用滑窗把**整段对话 + rubric** 拼进判分 prompt，调 `generate(gpt-5.2-eval, ...)`。
3. gpt-5.2 对这个输入触发内容审核 → 返回 `invalid_prompt`。
4. `llm_utils.generate` 解析 `response['choices']` → `KeyError` → 函数**返回 None**
   （generate 的 except 分支吞掉异常返回 None）。
5. 上层拿 None 去取 `.content` → `'NoneType' object has no attribute 'content'`。
6. `run_task` 对该异常**重试 4 次**，但 `invalid_prompt` 是**确定性**的（同一 prompt 必被拒），
   4 次全废 → 任务被丢弃，不落盘。

为什么之前 smoke 用 gpt-5 / gpt-5.4-mini 当 eval 能通？偶然：短轨迹、不同模型版本审核阈值不同。
一旦放量、轨迹变长、覆盖到会被审核的内容，reasoning 判官必然大面积翻车。

## 排查过程（可复现的隔离实验）

1. **最简 reasoning 调用正常** —— 排除"xhigh 本身坏了"：
   ```python
   from vita.utils.llm_utils import generate
   from vita.data_model.message import SystemMessage, UserMessage
   m = generate('gpt-5.2',
                [SystemMessage(role='system', content='You are a helpful assistant.'),
                 UserMessage(role='user', content='你好，帮我查一下订单。')],
                enable_think=True)
   print('OK' if m else 'FAILED')   # -> OK
   ```

2. **真实 agent 首轮（system prompt + 66 工具 + 中/英文）也正常** —— 排除"system prompt / 工具 schema / 语言"：
   ```python
   from vita.registry import registry
   from vita.run import get_tasks
   from vita.utils.llm_utils import generate
   from vita.data_model.message import SystemMessage, UserMessage
   tasks = get_tasks('delivery', ['10711001'], None, 'chinese')
   env = registry.get_env_constructor('delivery')(tasks[0].environment, 'chinese')
   AC = registry.get_agent_constructor('llm_agent')
   agent = AC(tools=env.get_tools(), domain_policy=env.get_policy(), llm='gpt-5.2',
              llm_args=None, time=env.tools.db.time, enable_think=True, language='chinese')
   m = generate('gpt-5.2',
                [SystemMessage(role='system', content=str(agent.system_prompt)),
                 UserMessage(role='user', content='你好')],
                tools=env.get_tools(), enable_think=True)
   print('OK' if m else 'FAILED')   # -> OK
   ```

3. **换 gpt-4.1 判官后 smoke 通过** —— 确认修复。2 任务、agent gpt-5.2 xhigh、
   `--max-duration 180`（快速截轨迹）、`--evaluator-llm gpt-4.1`：
   ```bash
   .venv/bin/vita run --domain delivery \
     --agent-llm gpt-5.2 --user-llm gpt-4.1 --evaluator-llm gpt-4.1 \
     --enable-think --num-trials 1 --language chinese \
     --max-steps 300 --max-errors 100000 --max-duration 180 \
     --max-concurrency 2 --num-tasks 2 --log-level WARNING \
     --save-to smoke_evalfix.json --csv-output smoke_evalfix.csv
   ```
   结果：落盘出分（task 10711002 reward 0.0，`evaluation_method: sliding_window`，2 windows），
   日志 **0 次 `invalid_prompt`、0 次 NoneType 失败**。

## 修复

`scripts/run_cross_gpt52.sh` 默认 `EVALUATOR_LLM` 由 `gpt-5.2-eval` 改为 **`gpt-4.1`**。
`gpt-5.2-eval` 条目保留在 `models.yaml` 但**不要用作 evaluator**（会翻车）。

**通用规则：VitaBench 的 evaluator 必须用非 reasoning 模型（gpt-4.1 / gpt-4o）。**

## 遗留的代码脆弱点（未修，仅记录）

`llm_utils.generate` 在任何 API 失败时（包括 `invalid_prompt`）都**静默返回 None**，
上层直接 `.content` 崩，再靠**对确定性错误无意义的 4 次重试**兜底。更健壮的做法是：
判官某个 window 判分失败时降级处理（跳过/记为未满足）而不是让整个任务失败；
或至少对 `invalid_prompt` 这类**不可重试**的 4xx 直接 fail-fast 并明确报错。
本次靠"换非 reasoning 判官"从源头绕开，未改这段代码。
