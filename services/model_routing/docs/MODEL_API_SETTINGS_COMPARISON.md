# Tau2 模型API设置对比

本文档展示了在Tau2中使用的主要模型（GPT-5.2, GPT-5.4, Claude Sonnet 4.5, Claude Sonnet 4.6）的实际API请求参数，以及Claude Code User Simulator的特殊配置。

## 快速对比表

### Agent/标准User Simulator

| 模型 | Provider | 推理参数 | Temperature | Seed | 特殊配置 |
|------|----------|----------|-------------|------|----------|
| **GPT-5.2** | OpenAI | `reasoning_effort: "high"` (可选) | 1.0 | ✓ | - |
| **GPT-5.4** | OpenAI | `reasoning_effort: "high"` (默认) | 1.0 | ✓ | - |
| **Sonnet 4.5** | Bedrock | `thinking: {budget_tokens: 4096}` | 1.0 | ✓ | `allowed_openai_params`, `aws_region_name` |
| **Sonnet 4.6** | Bedrock | 无 (可配置) | 1.0 | ✓ | `aws_region_name` |

### Claude Code User Simulator (特殊系统)

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **模型** | Sonnet 4.6 | 通过`claude` CLI调用 |
| **API调用数/轮** | 2次 | 标题生成 + 实际回复 |
| **推理模式** | `thinking: {type: "adaptive"}` | **自动启用**，不可关闭 |
| **Temperature** | 省略→默认1.0 | CLI不暴露此参数 |
| **Max Tokens** | 32000 | 固定值 |
| **Output Effort** | "high" | 固定值 |
| **系统提示** | 3块 (2样板+1实际) | 自动添加SDK标识 |
| **消息包装** | `<system-reminder>` + `<session>` | 自动添加日期和标签 |

---

## 1. GPT-5.2 API设置

### 配置1: 基础配置 (无推理增强)
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "gpt-5.2",
    "completion_model": "gpt-5.2",
    "tool_choice": "auto",
    "n_tools": 15,
    "params": {
      "temperature": 1.0,
      "seed": 361454,
      "num_retries": 3
    }
  },
  "request": {
    "model": "gpt-5.2",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "temperature": 1.0,
    "seed": 361454,
    "num_retries": 3
  }
}
```

**使用场景**: 标准agent响应，不需要额外推理能力

### 配置2: 高推理模式
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "gpt-5.2",
    "completion_model": "gpt-5.2",
    "tool_choice": "auto",
    "n_tools": 15,
    "params": {
      "reasoning_effort": "high",
      "temperature": 1.0,
      "seed": 373753,
      "num_retries": 3
    }
  },
  "request": {
    "model": "gpt-5.2",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "reasoning_effort": "high",
    "temperature": 1.0,
    "seed": 373753,
    "num_retries": 3
  }
}
```

**使用场景**: 复杂推理任务（如banking_knowledge domain）

**关键差异**:
- ✅ 添加 `reasoning_effort: "high"`
- 🔍 GPT-5.2的推理能力是**可选的**，可以根据任务开启或关闭

---

## 2. GPT-5.4 API设置

### 配置1: 默认高推理模式
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "gpt-5.4",
    "completion_model": "gpt-5.4",
    "tool_choice": "auto",
    "n_tools": 15,
    "params": {
      "reasoning_effort": "high",
      "temperature": 1.0,
      "seed": 361454,
      "num_retries": 3
    }
  },
  "request": {
    "model": "gpt-5.4",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "reasoning_effort": "high",
    "temperature": 1.0,
    "seed": 361454,
    "num_retries": 3
  }
}
```

**使用场景**: 大多数任务的默认配置

### 配置2: 标准模式 (无推理增强)
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "gpt-5.4",
    "completion_model": "gpt-5.4",
    "tool_choice": "auto",
    "n_tools": 15,
    "params": {
      "temperature": 1.0,
      "seed": 361454,
      "num_retries": 3
    }
  },
  "request": {
    "model": "gpt-5.4",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "temperature": 1.0,
    "seed": 361454,
    "num_retries": 3
  }
}
```

**关键差异**:
- ⚠️ GPT-5.4**默认启用**高推理模式（与GPT-5.2不同）
- 💡 可以手动关闭推理模式以提高速度

---

## 3. Claude Sonnet 4.5 API设置 (Bedrock)

### 配置: 扩展思考模式
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "completion_model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "tool_choice": "auto",
    "n_tools": 13,
    "params": {
      "thinking": "{'type': 'enabled', 'budget_tokens': 4096}",
      "temperature": 1.0,
      "allowed_openai_params": "['thinking']",
      "aws_region_name": "us-west-2",
      "seed": 361454,
      "num_retries": 3
    }
  },
  "request": {
    "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "thinking": {
      "type": "enabled",
      "budget_tokens": 4096
    },
    "allowed_openai_params": ["thinking"],
    "temperature": 1.0,
    "aws_region_name": "us-west-2",
    "seed": 361454,
    "num_retries": 3
  }
}
```

**使用场景**: 需要深度推理的复杂任务

**关键特性**:
- 🧠 `thinking`: 扩展思考配置
  - `type: "enabled"`: 启用思考模式
  - `budget_tokens: 4096`: 思考token预算（对应 `reasoning_effort: "high"`）
- 🔐 `allowed_openai_params: ["thinking"]`: 防止LiteLLM的`drop_params`删除此参数
- 🌎 `aws_region_name: "us-west-2"`: Bedrock区域

**Thinking Budget映射**:
```python
reasoning_effort → thinking.budget_tokens
"minimal"  → 128
"low"      → 1024
"medium"   → 2048
"high"     → 4096
```

---

## 4. Claude Sonnet 4.6 API设置 (Bedrock)

### 配置1: 标准模式 (无扩展思考)
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "completion_model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "tool_choice": "auto",
    "n_tools": 4,
    "params": {
      "temperature": 1.0,
      "seed": 626729,
      "num_retries": 3
    }
  },
  "request": {
    "model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "temperature": 1.0,
    "seed": 626729,
    "num_retries": 3
  }
}
```

**使用场景**: 标准任务，不需要扩展思考

### 配置2: 扩展思考模式 (可手动启用)
```json
{
  "setting": {
    "call_name": "agent_response",
    "model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "completion_model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "tool_choice": "auto",
    "n_tools": 15,
    "params": {
      "thinking": "{'type': 'enabled', 'budget_tokens': 4096}",
      "temperature": 1.0,
      "allowed_openai_params": "['thinking']",
      "aws_region_name": "us-west-2",
      "seed": 361454,
      "num_retries": 3
    }
  }
}
```

**关键差异**:
- ✅ Sonnet 4.6**默认不启用**扩展思考（与4.5不同）
- 💡 可以手动添加 `reasoning_effort` 参数来启用

---

## 详细参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 生成随机性控制 (0.0-2.0) |
| `seed` | int | 随机 | 可重现性种子 |
| `num_retries` | int | 3 | API调用失败重试次数 |
| `tool_choice` | str | "auto" | 工具调用策略: "auto", "required", "none" |

### OpenAI特定参数

| 参数 | 类型 | 可用模型 | 说明 |
|------|------|----------|------|
| `reasoning_effort` | str | GPT-5.2, GPT-5.4, o1, o3 | 推理强度: "minimal", "low", "medium", "high" |
| `seed` | int | 所有模型 | OpenAI原生支持seed参数 |

### Bedrock/Anthropic特定参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `thinking` | dict | `{"type": "enabled", "budget_tokens": 4096}` |
| `allowed_openai_params` | list | `["thinking"]` - 防止参数被过滤 |
| `aws_region_name` | str | Bedrock区域，如 "us-west-2", "us-east-1" |

### Thinking参数详解

```python
thinking = {
    "type": "enabled",  # 或 "disabled"
    "budget_tokens": 4096,  # 思考token预算 (128-8192)
}
```

**预算建议**:
- 128: 简单任务，快速响应
- 1024: 中等复杂度
- 2048: 复杂推理
- 4096: 深度推理（对应 `reasoning_effort: "high"`）

---

## 在Tau2中的CLI使用

### GPT-5.2 (标准模式)
```bash
uv run tau2 run \
  --domain telecom \
  --agent-llm "gpt-5.2" \
  --agent-llm-args '{"temperature": 1.0}'
```

### GPT-5.2 (高推理模式)
```bash
uv run tau2 run \
  --domain banking_knowledge \
  --agent-llm "gpt-5.2" \
  --agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
```

### GPT-5.4 (默认高推理)
```bash
uv run tau2 run \
  --domain banking_knowledge \
  --agent-llm "gpt-5.4" \
  --agent-llm-args '{"temperature": 1.0}'
  # reasoning_effort: "high" 自动添加
```

### Sonnet 4.5 (扩展思考)
```bash
uv run tau2 run \
  --domain banking_knowledge \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0" \
  --agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
  # 自动转换为 thinking: {budget_tokens: 4096}
```

### Sonnet 4.6 (标准模式)
```bash
uv run tau2 run \
  --domain telecom \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --agent-llm-args '{"temperature": 1.0}'
```

### Sonnet 4.6 (扩展思考)
```bash
uv run tau2 run \
  --domain banking_knowledge \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
```

---

## 代码中的自动转换

### reasoning_effort → thinking 转换

位置: `src/tau2/utils/llm_utils.py:518-530`

```python
if model.startswith("claude-") and not completion_model.startswith("openai/"):
    effort = completion_kwargs.pop("reasoning_effort", None)
    if effort and effort != "none":
        budget_map = {"minimal": 128, "low": 1024, "medium": 2048, "high": 4096}
        if effort in budget_map:
            completion_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_map[effort],
            }
    if "thinking" in completion_kwargs:
        allowed = list(completion_kwargs.get("allowed_openai_params", []))
        allowed.append("thinking")
        completion_kwargs["allowed_openai_params"] = allowed
```

### 关键点
1. 只对 `claude-*` 模型进行转换
2. 不影响通过本地OpenAI代理的Claude模型
3. 自动添加 `allowed_openai_params` 防止参数被删除

---

## 实际请求对比

### 完整请求示例

#### GPT-5.2 (高推理)
```json
{
  "model": "gpt-5.2",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I need help with..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "shell",
        "description": "Execute a shell command...",
        "parameters": {...}
      }
    }
  ],
  "tool_choice": "auto",
  "reasoning_effort": "high",
  "temperature": 1.0,
  "seed": 373753,
  "num_retries": 3
}
```

#### Claude Sonnet 4.5 (扩展思考)
```json
{
  "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I need help with..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "shell",
        "description": "Execute a shell command...",
        "parameters": {...}
      }
    }
  ],
  "tool_choice": "auto",
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4096
  },
  "allowed_openai_params": ["thinking"],
  "temperature": 1.0,
  "aws_region_name": "us-west-2",
  "seed": 361454,
  "num_retries": 3
}
```

**关键差异**:
- GPT-5.2: `reasoning_effort: "high"`
- Sonnet 4.5: `thinking: {type: "enabled", budget_tokens: 4096}`
- Sonnet 4.5: 额外需要 `allowed_openai_params`
- Sonnet 4.5: Bedrock特定的 `aws_region_name`

---

## 性能与成本考虑

### 推理模式影响

| 模型 | 标准模式 | 高推理模式 | 延迟增加 | Token成本增加 |
|------|----------|------------|----------|---------------|
| GPT-5.2 | ✓ | ✓ | ~2-3x | ~1.5x |
| GPT-5.4 | 无 | 默认 | N/A | N/A |
| Sonnet 4.5 | ✓ | ✓ | ~2-4x | ~1.5-2x |
| Sonnet 4.6 | ✓ | ✓ | ~2-4x | ~1.5-2x |

### 推荐配置

**简单任务** (mock, telecom基础场景):
```bash
--agent-llm-args '{"temperature": 1.0}'
# 不启用推理增强
```

**复杂任务** (banking_knowledge):
```bash
--agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
# GPT: 自动添加reasoning_effort
# Claude: 自动转换为thinking
```

**调试/快速迭代**:
```bash
--agent-llm-args '{"temperature": 1.0, "reasoning_effort": "minimal"}'
# 最小推理，最快响应
```

---

## 调试技巧

### 1. 查看实际发送的请求
```bash
ls data/api_first_requests/
cat data/api_first_requests/agent_response__gpt-5.2__*.json | jq '.setting.params'
```

### 2. 验证thinking转换
```python
from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage

generate(
    model="bedrock/us.anthropic.claude-sonnet-4-6",
    messages=[SystemMessage(content="Test")],
    reasoning_effort="high",  # 会自动转换为thinking
)
```

### 3. 强制查看LiteLLM调用
```python
import litellm

litellm._turn_on_debug()  # 打印详细日志
```

---

## 常见问题

### Q1: GPT-5.2 vs GPT-5.4 推理模式的区别?
- **GPT-5.2**: 推理模式是**可选的**，需要显式指定 `reasoning_effort`
- **GPT-5.4**: 推理模式是**默认的**，自动包含 `reasoning_effort: "high"`

### Q2: 为什么Sonnet需要 `allowed_openai_params`?
LiteLLM默认启用 `drop_params=True`，会删除不认识的参数。由于 `thinking` 是Anthropic特有的，需要明确允许。

### Q3: Bedrock的 `aws_region_name` 从哪来?
从环境变量 `AWS_REGION` 或 `AWS_DEFAULT_REGION` 读取，默认 `us-west-2`。

### Q4: 如何关闭GPT-5.4的默认推理模式?
不在 `llm_args` 中包含 `reasoning_effort`，或显式设置为 `"none"`：
```bash
--agent-llm-args '{"temperature": 1.0, "reasoning_effort": "none"}'
```

### Q5: thinking budget应该设置多少?
- **快速响应**: 128-1024
- **标准任务**: 2048
- **复杂推理**: 4096
- **极限思考**: 8192 (很少需要)

---

## 5. Claude Code User Simulator API设置 (特殊系统)

### 背景

`ClaudeCodeUserSimulator` 不是直接调用LLM API，而是通过 `claude` CLI子进程。这导致实际发送给API的请求与标准方式**完全不同**。

### 捕获的配置 (发送给本地代理)

#### 标准模式
```json
{
  "setting": {
    "call_name": "capture_raw",
    "model": "anthropic/claude-sonnet-4-6",
    "completion_model": "anthropic/claude-sonnet-4-6",
    "tool_choice": null,
    "n_tools": 0,
    "params": {
      "temperature": 1.0,
      "api_key": "dummy",
      "api_base": "http://127.0.0.1:4215",
      "num_retries": 0
    }
  },
  "request": {
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [
      {
        "role": "system",
        "content": "You are role-playing as a customer contacting customer service.\n<scenario>\nYou want to check your account balance.\n</scenario>"
      },
      {
        "role": "user",
        "content": "Hi, this is customer service. How can I help?"
      }
    ],
    "temperature": 1.0,
    "api_key": "dummy",
    "api_base": "http://127.0.0.1:4215"
  }
}
```

**注意**: 这只是Tau2**期望**发送的格式，但Claude Code CLI会进行大量转换！

### Claude Code实际发送的请求

#### 每轮2次API调用

**REQ-1: 标题生成**
```json
{
  "model": "claude-sonnet-4-6",
  "system": [
    {"type": "text", "text": "x-anthropic-billing-header: ..."},
    {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    {"type": "text", "text": "Generate a concise, sentence-case title..."}
  ],
  "messages": [
    {"role": "user", "content": "<session>...</session>"}
  ],
  "temperature": 1.0,
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {"type": "object", "properties": {"title": {"type": "string"}}}
    }
  }
}
```

**REQ-2: 实际回复**
```json
{
  "model": "claude-sonnet-4-6",
  "system": [
    {"type": "text", "text": "x-anthropic-billing-header: ..."},
    {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    {"type": "text", "text": "You are role-playing as a customer...\n<scenario>..."}
  ],
  "messages": [
    {
      "role": "user",
      "content": "<system-reminder>\nCurrent date: 2026-06-02\n</system-reminder>\n\n<session>\nHi, this is customer service. How can I help?\n</session>"
    }
  ],
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "high"},
  "max_tokens": 32000,
  "context_management": {"strategy": "clear_thinking_20251015"}
}
```

### 关键转换对比

| 特性 | Tau2期望发送 | Claude Code实际发送 |
|------|-------------|-------------------|
| **API调用数/轮** | 1次 | **2次** (title-gen + reply) |
| **thinking** | 无 (或手动指定) | **`{type: "adaptive"}` 自动启用** |
| **temperature** | 显式1.0 | REQ-1: 1.0, REQ-2: 省略 |
| **max_tokens** | 默认4096 | **32000** |
| **output_config** | 无 | **`{effort: "high"}`** |
| **系统提示** | 1块 | **3块** (billing + SDK标识 + 实际) |
| **用户消息** | 原始内容 | `<system-reminder>` + `<session>` 包装 |
| **消息结构** | 多轮对话 | 单个扁平化块 |
| **API端点** | `/v1/messages` | `/v1/messages?beta=true` |
| **context_management** | 无 | `clear_thinking_20251015` |

### 无法控制的参数

`claude` CLI **不暴露**以下参数，无法通过命令行配置：
- ❌ `temperature` (REQ-2中省略)
- ❌ `thinking` (固定为`adaptive`)
- ❌ `max_tokens` (固定32000)
- ❌ `output_config.effort` (固定`high`)
- ❌ `context_management` (固定策略)

### 实验影响

⚠️ **关键警告**: 在Tau2中使用 `ClaudeCodeUserSimulator` 不等同于使用标准 `UserSimulator` + Sonnet 4.6：

1. **成本**: 每轮2倍API调用（标题生成是纯overhead）
2. **推理能力**: Adaptive thinking自动启用，可能提升质量但增加成本
3. **不可比性**: 与其他用户模拟器不是"同模型对比"，而是"不同系统对比"
4. **系统提示污染**: 额外的SDK标识块可能改变行为
5. **格式差异**: `<system-reminder>` 和 `<session>` 标签影响token计数和模型理解

### 使用场景

✅ **适合使用Claude Code User Simulator**:
- 想要最真实的"Claude Code作为用户"体验
- 需要测试与Claude Code的集成
- 接受额外成本和复杂性

❌ **不适合使用Claude Code User Simulator**:
- 需要公平对比不同模型
- 需要精确控制temperature/thinking等参数
- 成本敏感的实验
- 需要可复现的结果

### CLI使用

```bash
# 使用Claude Code User Simulator
uv run tau2 run \
  --domain telecom \
  --agent-llm "gpt-5.2" \
  --user cc_user \
  --user-llm "bedrock/us.anthropic.claude-sonnet-4-6"

# 标准User Simulator (对比)
uv run tau2 run \
  --domain telecom \
  --agent-llm "gpt-5.2" \
  --user user_simulator \
  --user-llm "bedrock/us.anthropic.claude-sonnet-4-6"
```

**重要**: 这两个命令使用**不同的系统**，结果不可直接比较！

---

## 参考文件

实际API请求文件位置: `data/api_first_requests/`

### 按模型/系统分类
```bash
# GPT-5.2
agent_response__gpt-5.2__*.json
user_simulator_response__gpt-5.2__*.json

# GPT-5.4
agent_response__gpt-5.4__*.json

# Sonnet 4.5
agent_response__bedrock_us.anthropic.claude-sonnet-4-5*.json

# Sonnet 4.6
agent_response__bedrock_us.anthropic.claude-sonnet-4-6*.json
user_simulator_response__bedrock_us.anthropic.claude-sonnet-4-6*.json

# Claude Code User Simulator
capture_raw__anthropic_claude-sonnet-4-6__*.json
```

### 代码参考
- 参数转换: `src/tau2/utils/llm_utils.py:474-620`
- 配置定义: `src/tau2/config.py:14-30`
- CLI参数解析: `src/tau2/cli.py:49-229`
- Claude Code User: `src/tau2/user/claude_code_user_simulator.py`

### 文档参考
- **API_LAYERS_EXPLAINED.md**: 完整的API调用层次解析
- **CC_USER_BACKBONE_API.md**: Claude Code backbone详细分析
