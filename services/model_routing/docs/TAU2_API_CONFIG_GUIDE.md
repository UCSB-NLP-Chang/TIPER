# Tau2 API配置与Tool Calling详解

## 目录
1. [默认配置](#默认配置)
2. [LiteLLM参数传递机制](#litellm参数传递机制)
3. [Tool Calling实现](#tool-calling实现)
4. [OpenAI vs Bedrock API差异](#openai-vs-bedrock-api差异)
5. [实际请求示例](#实际请求示例)

---

## 默认配置

### 核心配置文件
所有默认值定义在 `src/tau2/config.py`，这是**单一真实来源** (single source of truth)。

### 主要默认设置

#### 模拟参数
```python
DEFAULT_MAX_STEPS = 200  # 最大对话轮数
DEFAULT_MAX_ERRORS = 10  # 允许的最大连续工具错误
DEFAULT_SEED = 300  # 随机种子
DEFAULT_MAX_CONCURRENCY = 3  # 最大并发任务数
DEFAULT_NUM_TRIALS = 1  # 每个任务运行次数
```

#### LLM默认配置
```python
# Agent (系统被测试方)
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_LLM_AGENT = "gpt-4.1-2025-04-14"
DEFAULT_LLM_TEMPERATURE_AGENT = 1.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": 1.0}

# User Simulator (用户模拟器)
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_LLM_TEMPERATURE_USER = 1.0
DEFAULT_LLM_ARGS_USER = {"temperature": 1.0}

# Evaluation
DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"  # 自然语言断言评估
DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1-2025-04-14"  # 环境接口
```

#### 基础设施配置
```python
DEFAULT_MAX_RETRIES = 3  # API调用重试次数
DEFAULT_RETRY_MIN_WAIT = 1.0  # 重试最小等待时间(秒)
DEFAULT_RETRY_MAX_WAIT = 10.0  # 重试最大等待时间(秒)

# 缓存 (默认禁用)
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"  # 或 "local"

# 可观测性
USE_LANGFUSE = False  # Langfuse追踪
```

---

## LiteLLM参数传递机制

### 核心函数: `llm_utils.generate()`

位置: `src/tau2/utils/llm_utils.py`

#### 签名
```python
def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> AssistantMessage:
```

#### 参数流程

```
1. CLI参数 (cli.py)
   ├─> --agent-llm "gpt-5.2"
   ├─> --agent-llm-args '{"temperature": 0.7}'
   └─> --user-llm "bedrock/us.anthropic.claude-sonnet-4-6"

2. RunConfig对象 (data_model/simulation.py)
   ├─> TextRunConfig(llm_agent="gpt-5.2", llm_args_agent={...})
   └─> 传递到 runner/build.py

3. Agent/User构造 (build_agent/build_user)
   ├─> LLMAgent(llm="gpt-5.2", llm_args={...})
   └─> UserSimulator(llm="bedrock/...", llm_args={...})

4. LLM调用 (llm_utils.generate)
   ├─> 转换Tau2 messages → LiteLLM messages
   ├─> 转换Tau2 Tools → OpenAI tool schema
   ├─> 调用 litellm.completion()
   └─> 解析响应 → AssistantMessage
```

### 关键转换逻辑

#### 1. 消息格式转换 (`to_litellm_messages`)

```python
# Tau2 Message → LiteLLM Message
UserMessage(content="...") 
  → {"role": "user", "content": "..."}

AssistantMessage(content="...", tool_calls=[...])
  → {"role": "assistant", 
     "content": "...",
     "tool_calls": [{"id": "...", "name": "...", "function": {...}}]}

ToolMessage(id="...", content="...")
  → {"role": "tool", "tool_call_id": "...", "content": "..."}
```

#### 2. 工具名称清理 (Bedrock兼容)

```python
def _sanitize_tool_name(name: str) -> str:
    """Replace chars that violate Bedrock's toolUse.name constraint."""
    return _VALID_TOOL_NAME_RE.sub("_", name)


# 例如: "get-user-info" → "get_user_info"
```

#### 3. 模型特定处理

##### Claude模型 (Anthropic/Bedrock)
```python
if model.startswith("claude-"):
    # reasoning_effort → thinking budget
    effort = kwargs.pop("reasoning_effort", None)
    if effort == "minimal":
        thinking_budget = 128
    if effort == "low":
        thinking_budget = 1024
    if effort == "medium":
        thinking_budget = 2048
    if effort == "high":
        thinking_budget = 4096

    kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    # 防止LiteLLM的drop_params删除thinking参数
    kwargs["allowed_openai_params"] = ["thinking"]
```

##### OpenAI Realtime模型
```python
if model == "gpt-realtime-1.5":
    # 音频原生模式的特殊配置
    kwargs["modalities"] = ["audio", "text"]
    kwargs["voice"] = "alloy"
```

#### 4. 本地代理处理

```python
def _local_proxy_openai_model_kwargs(model: str):
    """将本地LiteLLM代理的Claude模型路由到OpenAI适配器"""
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url and model.startswith("claude-") and is_localhost(base_url):
        # 使用OpenAI格式调用本地代理
        return f"openai/{model}", {"api_base": base_url, "api_key": "dummy"}
    return model, {}
```

---

## Tool Calling实现

### 工具定义架构

#### 1. Python函数 → Tool对象

位置: `src/tau2/environment/tool.py`

```python
from tau2.environment.tool import as_tool


def create_task(user_id: str, title: str, description: str = None) -> str:
    """Create a new task for a user.

    Args:
        user_id: The ID of the user creating the task
        title: The title of the task
        description: Optional description of the task

    Returns:
        A confirmation message
    """
    # 实现...
    return "Task created successfully"


# 转换为Tool
tool = as_tool(create_task)
```

#### 2. Tool → OpenAI Schema

```python
tool.openai_schema
# 输出:
{
    "type": "function",
    "function": {
        "name": "create_task",
        "description": "Create a new task for a user.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "title": "User Id",
                    "description": "The ID of the user creating the task",
                },
                "title": {
                    "type": "string",
                    "title": "Title",
                    "description": "The title of the task",
                },
                "description": {
                    "type": "string",
                    "title": "Description",
                    "description": "Optional description of the task",
                    "default": null,
                },
            },
            "required": ["user_id", "title"],
            "title": "parameters",
        },
    },
}
```

### Tool Calling流程

```
1. Agent接收用户消息
   UserMessage(content="Create a task called 'Important Meeting'")

2. Agent调用generate()
   generate(
       model="gpt-5.2",
       messages=[system, assistant, user],
       tools=[create_task, get_users, ...],
       tool_choice="auto"
   )

3. LiteLLM发送请求
   litellm.completion(
       model="gpt-5.2",
       messages=[...],
       tools=[{openai_schema}, ...],
       tool_choice="auto"
   )

4. 模型返回tool_calls
   {
       "role": "assistant",
       "content": null,
       "tool_calls": [{
           "id": "call_abc123",
           "type": "function",
           "function": {
               "name": "create_task",
               "arguments": '{"user_id": "user_1", "title": "Important Meeting"}'
           }
       }]
   }

5. 解析为AssistantMessage
   AssistantMessage(
       content=None,
       tool_calls=[
           ToolCall(
               id="call_abc123",
               name="create_task",
               arguments={"user_id": "user_1", "title": "Important Meeting"}
           )
       ]
   )

6. Environment执行工具
   result = environment.execute_tool_call(tool_call)
   → ToolMessage(id="call_abc123", content="Task created successfully")

7. 继续对话
   AssistantMessage → User sees result
```

### Tool Choice选项

```python
# 1. Auto (默认): 模型决定是否调用工具
tool_choice = "auto"

# 2. Required: 强制模型调用工具 (用于Solo Agent)
tool_choice = "required"

# 3. None: 禁用工具调用
tool_choice = None  # tools参数为None时自动设置
```

---

## OpenAI vs Bedrock API差异

### 模型命名

```python
# OpenAI
model = "gpt-4.1-2025-04-14"
model = "gpt-5.2"

# Bedrock (通过LiteLLM)
model = "bedrock/us.anthropic.claude-sonnet-4-6"
model = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# 直接Anthropic API
model = "claude-sonnet-4-6"
model = "claude-opus-4-5"
```

### 认证

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Bedrock (需要AWS凭证)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# 或使用AWS SSO
aws sso login --profile your-profile
export AWS_PROFILE=your-profile

# Anthropic API
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 特定参数

#### OpenAI特有
```python
{
    "seed": 42,  # 可重现性
    "reasoning_effort": "high",  # o1/o3模型的推理强度
    "response_format": {...},  # 结构化输出
}
```

#### Bedrock/Anthropic特有
```python
{
    "thinking": {  # 扩展思考
        "type": "enabled",
        "budget_tokens": 4096,
    },
    "allowed_openai_params": ["thinking"],  # 防止LiteLLM删除
}
```

### Tool Schema差异

两者都遵循OpenAI的function calling格式，但Bedrock对工具名称有更严格的要求:

```python
# ✅ Bedrock接受
"tool_name", "get_user_info", "create_task_v2"

# ❌ Bedrock拒绝 (需要清理)
"get-user-info" → "get_user_info"
"create:task" → "create_task"
```

Tau2通过 `_sanitize_tool_name()` 自动处理此问题。

---

## 实际请求示例

### 示例1: Bedrock Claude Sonnet 4.6 (Agent)

**配置:**
```bash
uv run tau2 run \
  --domain mock \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --agent-llm-args '{"temperature": 1.0}' \
  --seed 626729
```

**实际发送到API的请求** (保存在 `data/api_first_requests/`):
```json
{
  "model": "bedrock/us.anthropic.claude-sonnet-4-6",
  "messages": [
    {
      "role": "system",
      "content": "<instructions>\nYou are a customer service agent...\n</instructions>\n<policy>\n# Mock Domain Policy\n...\n</policy>"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you today?",
      "tool_calls": null
    },
    {
      "role": "user",
      "content": "Hi! I need to create a new task..."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_task",
        "description": "Create a new task for a user.",
        "parameters": {
          "type": "object",
          "properties": {
            "user_id": {"type": "string", "description": "..."},
            "title": {"type": "string", "description": "..."},
            "description": {"type": "string", "default": null}
          },
          "required": ["user_id", "title"]
        }
      }
    },
    // ... 更多工具
  ],
  "tool_choice": "auto",
  "temperature": 1.0,
  "seed": 626729,
  "num_retries": 3
}
```

### 示例2: OpenAI GPT-5.2 (Agent)

**配置:**
```bash
uv run tau2 run \
  --domain banking_knowledge \
  --agent-llm "gpt-5.2" \
  --agent-llm-args '{"temperature": 1.0}' \
  --seed 361454
```

**关键差异:**
- 模型名称: `"gpt-5.2"` (vs `"bedrock/..."`)
- 相同的tool schema格式
- 相同的messages格式
- 额外参数: `seed` (OpenAI支持，Bedrock忽略)

### 示例3: Claude Code User Simulator

**配置:**
```python
ClaudeCodeUserSimulator(
    llm="bedrock/us.anthropic.claude-sonnet-4-6",
    llm_args={
        "temperature": 1.0,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    },
)
```

**特殊处理:**
- 通过子进程调用 `claude --print`
- MCP服务器通过SSE暴露用户工具
- 工具调用从MCP日志中提取，然后重放到真实环境

---

## 调试技巧

### 1. 查看实际API请求

每个唯一的配置组合会自动保存首次请求:

```bash
ls data/api_first_requests/
# agent_response__gpt-5.2__18cd2b5c.json
# agent_response__bedrock_us.anthropic.claude-sonnet-4-6__0484f578.json
```

### 2. 启用详细日志

```bash
uv run tau2 run \
  --domain telecom \
  --verbose-logs \
  --llm-log-mode all  # 或 "latest"
```

保存位置: `data/simulations/<save_to>/llm_logs/`

### 3. 检查LiteLLM转换

```python
from tau2.utils.llm_utils import to_litellm_messages, generate
from tau2.data_model.message import UserMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    UserMessage(content="Hello!"),
]

litellm_messages = to_litellm_messages(messages)
print(litellm_messages)
```

### 4. 测试工具schema

```python
from tau2.environment.tool import as_tool


def my_tool(arg1: str, arg2: int = 5) -> str:
    """Tool description.

    Args:
        arg1: First argument
        arg2: Second argument with default
    """
    return f"{arg1}-{arg2}"


tool = as_tool(my_tool)
print(tool.openai_schema)
```

---

## 常见问题

### Q1: 为什么Bedrock模型名称这么长?

Bedrock需要完整的ARN格式:
- `us.anthropic.claude-sonnet-4-6` = 区域前缀 + 模型族 + 版本
- LiteLLM自动添加 `bedrock/` 前缀进行路由

### Q2: `thinking` 参数为什么需要 `allowed_openai_params`?

LiteLLM默认启用 `drop_params=True`，会删除未知参数。Claude 4.x的 `thinking` 参数未被LiteLLM <1.82.7识别，需要明确允许。

### Q3: 如何混合使用OpenAI和Bedrock?

```bash
uv run tau2 run \
  --domain telecom \
  --agent-llm "gpt-5.2" \              # Agent用OpenAI
  --user-llm "bedrock/us.anthropic.claude-sonnet-4-6"  # User用Bedrock
```

### Q4: `reasoning_effort` vs `thinking` 的区别?

- `reasoning_effort`: OpenAI o1/o3模型的原生参数 (值: "minimal", "low", "medium", "high")
- `thinking`: Anthropic扩展思考参数 (需要budget_tokens)
- Tau2会自动转换: `reasoning_effort="high"` → `thinking={"budget_tokens": 4096}` (仅针对Claude模型)

### Q5: 为什么有时工具名称被修改了?

Bedrock要求工具名称只能包含 `[a-zA-Z0-9_-]`。Tau2自动清理不符合的名称:
```python
"get-user-info" → "get_user_info"  # 保持兼容
```

---

## 参考

### 关键文件
- `src/tau2/config.py` - 所有默认配置
- `src/tau2/utils/llm_utils.py` - LLM调用核心逻辑
- `src/tau2/environment/tool.py` - 工具定义和schema生成
- `src/tau2/agent/llm_agent.py` - Agent实现
- `src/tau2/user/user_simulator.py` - User simulator实现

### LiteLLM文档
- https://docs.litellm.ai/docs/providers
- https://docs.litellm.ai/docs/providers/bedrock
- https://docs.litellm.ai/docs/providers/openai

### API文档
- OpenAI: https://platform.openai.com/docs/api-reference
- Anthropic: https://docs.anthropic.com/en/api
- AWS Bedrock: https://docs.aws.amazon.com/bedrock/
