# VitaBench 复现（OpenAI-only）

用父 repo `.env` 的 `OPENAI_API_KEY`（Cresta sandbox key）复现 VitaBench。

## 环境

- 独立 venv：`vitabench/.venv`（`uv venv --python 3.12 .venv && uv pip install -e .`）
- 运行用 `.venv/bin/vita ...`（工作目录 = `vitabench/`）
- key 已内联在 `src/vita/models.yaml`（git-ignored，勿提交）

## 必要的代码修复（`src/vita/utils/llm_utils.py`）

原始 `generate()` 把整个 `data`（含 `base_url`/`headers`/`max_input_tokens`/
`cost_1m_token_dollar`/`name`）当 JSON body POST 给 OpenAI → 400
`Unrecognized request arguments supplied`。真实 OpenAI 端点根本跑不通。

新增 `sanitize_payload()` 在 POST 前清洗 body：
- 剥离上述非 API 记账字段 + 所有值为 None 的字段（如空 tools）
- 对推理模型（`gpt-5*` / `o1/o3/o4*`）：`max_tokens` → `max_completion_tokens`、
  去掉 `temperature`（这些模型只接受默认值）、去掉 Anthropic 风格的 `thinking`

## 模型配置（models.yaml）

当前主力（cross-scenario 复现）：

| 角色 | 模型条目 | 实际模型 | reasoning | 说明 |
|---|---|---|---|---|
| agent | `gpt-5.2` + `--enable-think` | gpt-5.2 | **xhigh** | 前沿推理 |
| user | `gpt-5.2-user` | gpt-5.2 | **none** | 便宜 user，`model_id` 别名同底模 |
| evaluator | `gpt-5.4-mini` | gpt-5.4-mini | **low** | 便宜 judge |

`model_id` 别名机制：yaml 条目 `name` 可与真实模型不同，加 `model_id: <真实id>`
即可让同一底模（如 gpt-5.2）配出多个不同 reasoning 的角色条目。请求体里 `model`
POST 的是 `model_id`。

reasoning_effort 取值（gpt-5.x）：`none/low/medium/high/xhigh`（**不支持 minimal**）。
非 think 模式下 `kwargs_adapter` 会尊重条目里显式写的 reasoning_effort。

其它可选：`gpt-5`、`gpt-4.1`、`gpt-4o`、`gpt-5-mini`。

## 冒烟验证（已通过）

- 全 gpt-4.1，delivery 单任务：reward 1.0，$0.16，0.45min
- gpt-5.2 agent + gpt-5 评估器：端到端通，$0.15，1.63min（gpt-5 参数适配正确）

注意：task-id 是 8 位字符串（如 `10711001`），不是 `1`。用 `--num-tasks N` 跑前 N 个。

## 放量命令（之后随时用）

**推荐用脚本** `scripts/run_cross_gpt52.sh`（cross-scenario 100 任务 × 2 trials，
agent xhigh / user none / eval gpt-5.4-mini low）：
```bash
cd vitabench
scripts/run_cross_gpt52.sh                 # 全 100 任务 × 2 trials
NUM_TASKS=3 scripts/run_cross_gpt52.sh     # 先跑前 3 个任务验证
MAX_CONCURRENCY=8 scripts/run_cross_gpt52.sh
LANGUAGE=chinese scripts/run_cross_gpt52.sh
```
可覆盖的环境变量：`NUM_TASKS / NUM_TRIALS / LANGUAGE / MAX_CONCURRENCY / MAX_STEPS /
AGENT_LLM / USER_LLM / EVALUATOR_LLM / SAVE_TO / CSV_OUT`。同名 `SAVE_TO` 可断点续跑。

等价手写命令：
```bash
.venv/bin/vita run --domain delivery,instore,ota \
  --agent-llm gpt-5.2 --user-llm gpt-5.2-user --evaluator-llm gpt-5.4-mini \
  --enable-think --num-trials 2 --language english \
  --max-concurrency 6 --log-level WARNING \
  --save-to cross_gpt52xhigh.json --csv-output results_cross.csv
```

单域全跑：把 `--domain` 换成 `delivery` / `instore` / `ota`（单域不加逗号）。

查看结果：`.venv/bin/vita view --file data/simulations/cross_gpt52xhigh_english.json`
结果 JSON 存于 `data/simulations/`。
