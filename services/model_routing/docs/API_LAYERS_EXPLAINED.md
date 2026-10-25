# Tau2 API调用层次详解

## 问题：这些配置是直接发给OpenAI API的吗？

**简短答案**: ❌ **不完全是**。有3层转换。

**完整答案**: 
1. 你看到的是**传给LiteLLM的参数**
2. LiteLLM会再次转换成**各个Provider的原生格式**
3. 最终发送给**OpenAI/Bedrock/Anthropic API**

---

## 三层架构

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Tau2 Config                                    │
│ (CLI参数, TextRunConfig)                                │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: LiteLLM Unified Format                        │
│ (OpenAI-like格式, tau2保存的api_first_requests)         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Provider Native API                           │
│ (实际HTTP请求到OpenAI/Bedrock/Anthropic)                │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Tau2 Config → LiteLLM

### 代码位置
`src/tau2/utils/llm_utils.py:474-566`

### 输入 (Tau2格式)
```python
generate(
    model="bedrock/us.anthropic.claude-sonnet-4-6",
    messages=[
        SystemMessage(content="You are a helpful agent"),
        UserMessage(content="Hello"),
    ],
    tools=[create_task_tool, get_users_tool],
    tool_choice="auto",
    temperature=1.0,
    reasoning_effort="high",  # Tau2统一参数
    seed=626729,
)
```

### 转换过程
```python
# 1. 消息格式转换
litellm_messages = to_litellm_messages(messages)
# Tau2 Message → LiteLLM dict

# 2. 工具schema转换
tools_schema = [tool.openai_schema for tool in tools]
# Tau2 Tool → OpenAI function schema

# 3. 模型特定参数转换
if model.startswith("claude-"):
    # reasoning_effort → thinking
    effort = kwargs.pop("reasoning_effort", None)
    if effort == "high":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        kwargs["allowed_openai_params"] = ["thinking"]

# 4. 调用LiteLLM
completion_kwargs = {
    "temperature": 1.0,
    "thinking": {"type": "enabled", "budget_tokens": 4096},
    "allowed_openai_params": ["thinking"],
    "seed": 626729,
    "num_retries": 3,
}

response = litellm.completion(
    model="bedrock/us.anthropic.claude-sonnet-4-6",
    messages=litellm_messages,
    tools=tools_schema,
    tool_choice="auto",
    **completion_kwargs,
)
```

### 输出 (保存在api_first_requests/)
这就是你看到的JSON文件内容：
```json
{
  "model": "bedrock/us.anthropic.claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools": [
    {"type": "function", "function": {...}}
  ],
  "tool_choice": "auto",
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "allowed_openai_params": ["thinking"],
  "temperature": 1.0,
  "seed": 626729,
  "num_retries": 3
}
```

**⚠️ 重要**: 这是**传给LiteLLM的格式**，不是最终的API格式！

---

## Layer 2: LiteLLM → Provider API

### LiteLLM的职责

LiteLLM是一个统一的LLM调用库，它：
1. 接受**OpenAI风格的参数**
2. 根据 `model` 前缀识别Provider
3. 转换为**各Provider的原生格式**
4. 发送HTTP请求

### 转换示例

#### OpenAI (直通，几乎无转换)
```python
# LiteLLM接收
{
    "model": "gpt-5.2",
    "messages": [...],
    "tools": [...],
    "temperature": 1.0,
    "reasoning_effort": "high",
    "seed": 626729
}

# 发送给 api.openai.com
POST https://api.openai.com/v1/chat/completions
{
    "model": "gpt-5.2",
    "messages": [...],
    "tools": [...],
    "temperature": 1.0,
    "reasoning_effort": "high",  # OpenAI原生支持
    "seed": 626729
}
```
✅ **几乎1:1映射**

#### Bedrock/Anthropic (需要显著转换)
```python
# LiteLLM接收
{
    "model": "bedrock/us.anthropic.claude-sonnet-4-6",
    "messages": [
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": "Hello"}
    ],
    "tools": [
        {"type": "function", "function": {"name": "create_task", ...}}
    ],
    "thinking": {"type": "enabled", "budget_tokens": 4096},
    "temperature": 1.0
}

# LiteLLM转换并发送给 bedrock-runtime.us-west-2.amazonaws.com
POST /model/us.anthropic.claude-sonnet-4-6/converse
{
    "modelId": "us.anthropic.claude-sonnet-4-6",
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": "You are..."},  # system合并到user
                {"text": "Hello"}
            ]
        }
    ],
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "create_task",
                    "description": "...",
                    "inputSchema": {...}
                }
            }
        ],
        "toolChoice": {"auto": {}}
    },
    "inferenceConfig": {
        "temperature": 1.0
    },
    "additionalModelRequestFields": {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4096
        }
    }
}
```

**重要转换**:
1. `messages` → Bedrock `converse` API格式
2. `tools` → `toolConfig.tools` (不同schema)
3. `tool_choice: "auto"` → `toolChoice: {auto: {}}`
4. `temperature` → `inferenceConfig.temperature`
5. `thinking` → `additionalModelRequestFields.thinking`
6. `system` 消息合并到第一个 `user` 消息

---

## 实际HTTP请求对比

### OpenAI API (真实请求)

```http
POST https://api.openai.com/v1/chat/completions HTTP/1.1
Host: api.openai.com
Authorization: Bearer sk-...
Content-Type: application/json

{
  "model": "gpt-5.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant"
    },
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_task",
        "description": "Create a task",
        "parameters": {
          "type": "object",
          "properties": {
            "title": {"type": "string"}
          },
          "required": ["title"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 1.0,
  "reasoning_effort": "high",
  "seed": 626729
}
```

### Bedrock Converse API (真实请求)

```http
POST /model/us.anthropic.claude-sonnet-4-6/converse HTTP/1.1
Host: bedrock-runtime.us-west-2.amazonaws.com
X-Amz-Date: 20260602T120000Z
Authorization: AWS4-HMAC-SHA256 ...
Content-Type: application/json

{
  "modelId": "us.anthropic.claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "You are a helpful assistant"
        },
        {
          "text": "Hello"
        }
      ]
    }
  ],
  "toolConfig": {
    "tools": [
      {
        "toolSpec": {
          "name": "create_task",
          "description": "Create a task",
          "inputSchema": {
            "json": {
              "type": "object",
              "properties": {
                "title": {"type": "string"}
              },
              "required": ["title"]
            }
          }
        }
      }
    ],
    "toolChoice": {
      "auto": {}
    }
  },
  "inferenceConfig": {
    "temperature": 1.0
  },
  "additionalModelRequestFields": {
    "thinking": {
      "type": "enabled",
      "budget_tokens": 4096
    }
  }
}
```

**完全不同的格式！**

---

## 参数在各层的对应关系

### 示例：temperature参数

| Layer | 格式 | 位置 |
|-------|------|------|
| **Tau2** | `temperature=1.0` | 函数参数 |
| **LiteLLM输入** | `"temperature": 1.0` | 顶层字段 |
| **OpenAI API** | `"temperature": 1.0` | 顶层字段 |
| **Bedrock API** | `"temperature": 1.0` | `inferenceConfig.temperature` |

### 示例：reasoning_effort参数

| Layer | 格式 | 说明 |
|-------|------|------|
| **Tau2** | `reasoning_effort="high"` | 统一参数 |
| **LiteLLM输入 (OpenAI)** | `"reasoning_effort": "high"` | 直接传递 |
| **LiteLLM输入 (Claude)** | `"thinking": {"type": "enabled", "budget_tokens": 4096}` | 已转换 |
| **OpenAI API** | `"reasoning_effort": "high"` | 原生支持 |
| **Bedrock API** | `"additionalModelRequestFields": {"thinking": {...}}` | 嵌套字段 |

### 示例：tools参数

| Layer | 格式 |
|-------|------|
| **Tau2** | `Tool` 对象列表 |
| **LiteLLM输入** | OpenAI function schema |
| **OpenAI API** | `tools: [{type: "function", function: {...}}]` |
| **Bedrock API** | `toolConfig.tools: [{toolSpec: {...}}]` |

---

## 如何查看真实API请求

### 方法1: LiteLLM调试模式

```python
import litellm

litellm._turn_on_debug()  # 打印详细日志

from tau2.utils.llm_utils import generate
from tau2.data_model.message import UserMessage

generate(model="gpt-5.2", messages=[UserMessage(content="Hello")], temperature=1.0)

# 输出会包含:
# POST Request Sent from LiteLLM:
# https://api.openai.com/v1/chat/completions
# {"model": "gpt-5.2", "messages": [...], ...}
```

### 方法2: HTTP代理抓包

```bash
# 使用mitmproxy抓包
mitmproxy -p 8080

# 设置代理
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080

# 运行tau2
uv run tau2 run --domain mock --agent-llm "gpt-5.2"

# 在mitmproxy中查看实际HTTP请求
```

### 方法3: LiteLLM日志文件

```python
import litellm
litellm.set_verbose = True

# 或在环境变量中设置
export LITELLM_LOG=DEBUG
```

---

## 常见误解与澄清

### ❌ 误解1: api_first_requests是真实API格式
**真相**: 这是**LiteLLM输入格式**，不是发送给API的格式。

### ❌ 误解2: num_retries会发送给OpenAI
**真相**: `num_retries` 是**LiteLLM参数**，控制LiteLLM的重试逻辑，不会发送给API。

### ❌ 误解3: seed在所有模型上都有效
**真相**: 
- ✅ OpenAI: 原生支持 `seed`
- ❌ Bedrock: 不支持 `seed`（LiteLLM会忽略）
- ❌ Anthropic API: 不支持 `seed`

### ❌ 误解4: thinking参数直接发给Bedrock
**真相**: LiteLLM会将 `thinking` 嵌套在 `additionalModelRequestFields` 中。

---

## 总结对比表

| 方面 | api_first_requests (Layer 2) | 真实API请求 (Layer 3) |
|------|----------------------------|---------------------|
| **格式** | OpenAI风格统一格式 | Provider原生格式 |
| **作用** | LiteLLM的输入 | 发送给API的HTTP body |
| **messages** | OpenAI格式 | OpenAI保持，Bedrock转换 |
| **tools** | OpenAI function schema | OpenAI保持，Bedrock转换为toolSpec |
| **system消息** | 独立role | OpenAI保持，Bedrock合并到user |
| **temperature** | 顶层字段 | OpenAI顶层，Bedrock在inferenceConfig |
| **thinking** | 顶层字段 | Bedrock在additionalModelRequestFields |
| **num_retries** | LiteLLM控制 | ❌ 不发送给API |
| **allowed_openai_params** | LiteLLM控制 | ❌ 不发送给API |

---

## 实际示例：完整链路

### 输入 (Tau2代码)
```python
from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.environment.tool import as_tool


def create_task(title: str) -> str:
    """Create a task"""
    return f"Task '{title}' created"


generate(
    model="bedrock/us.anthropic.claude-sonnet-4-6",
    messages=[
        SystemMessage(content="You are helpful"),
        UserMessage(content="Create task"),
    ],
    tools=[as_tool(create_task)],
    tool_choice="auto",
    temperature=1.0,
    reasoning_effort="high",
    seed=123,
)
```

### Layer 2: 保存到api_first_requests
```json
{
  "model": "bedrock/us.anthropic.claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Create task"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_task",
        "description": "Create a task",
        "parameters": {
          "type": "object",
          "properties": {"title": {"type": "string"}},
          "required": ["title"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "allowed_openai_params": ["thinking"],
  "temperature": 1.0,
  "seed": 123,
  "num_retries": 3
}
```

### Layer 3: 实际发送给Bedrock
```json
{
  "modelId": "us.anthropic.claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": [
        {"text": "You are helpful"},
        {"text": "Create task"}
      ]
    }
  ],
  "toolConfig": {
    "tools": [
      {
        "toolSpec": {
          "name": "create_task",
          "description": "Create a task",
          "inputSchema": {
            "json": {
              "type": "object",
              "properties": {"title": {"type": "string"}},
              "required": ["title"]
            }
          }
        }
      }
    ],
    "toolChoice": {"auto": {}}
  },
  "inferenceConfig": {
    "temperature": 1.0
  },
  "additionalModelRequestFields": {
    "thinking": {
      "type": "enabled",
      "budget_tokens": 4096
    }
  }
}
```

**注意删除的参数**:
- ❌ `seed` (Bedrock不支持)
- ❌ `num_retries` (LiteLLM内部参数)
- ❌ `allowed_openai_params` (LiteLLM内部参数)

---

## 结论

**你在 `data/api_first_requests/` 中看到的配置是:**

✅ 传给**LiteLLM的参数** (Layer 2)  
❌ **不是**直接发给OpenAI/Bedrock API的格式 (Layer 3)

**为什么Tau2保存Layer 2而不是Layer 3?**
1. **统一格式**: 所有Provider使用相同格式，便于对比
2. **可复现**: 可以用这些参数重新调用LiteLLM
3. **抽象层**: 不需要关心各Provider的具体实现细节

**如果需要查看Layer 3 (真实API请求):**
- 使用 `litellm._turn_on_debug()`
- 使用HTTP代理 (mitmproxy, Charles)
- 查看LiteLLM源码中的Provider实现

---

## 特殊案例：Claude Code的API请求

### 什么是Claude Code User Simulator?

`ClaudeCodeUserSimulator` 是Tau2中的特殊用户模拟器，它通过调用 `claude` CLI子进程来生成用户响应，而不是直接调用LLM API。

### 架构差异

```
普通 UserSimulator:
Tau2 → LiteLLM → Anthropic API
     ↓          ↓
   Layer 2   Layer 3

ClaudeCodeUserSimulator:
Tau2 → claude CLI → Claude Code内部处理 → Anthropic API
     ↓            ↓                      ↓
  捕获的参数   Claude Code特殊处理      真实请求
```

### 捕获的参数 (发送给本地代理)

这是Tau2发送给 `claude --print` 子进程时的配置（通过本地代理捕获）：

#### 配置1: 标准模式
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
        "content": "You are role-playing as a customer contacting customer service.\n[ZZ_BACKBONE_CAPTURE_MARKER_ZZ] Follow the scenario exactly.\n<scenario>\nYou want to check your account balance.\n</scenario>"
      },
      {
        "role": "user",
        "content": "Hi, this is customer service. How can I help?"
      }
    ],
    "tools": null,
    "tool_choice": null,
    "temperature": 1.0,
    "api_key": "dummy",
    "api_base": "http://127.0.0.1:4215",
    "num_retries": 0
  }
}
```

#### 配置2: 高推理模式
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
      "reasoning_effort": "high",
      "api_key": "dummy",
      "api_base": "http://127.0.0.1:4199",
      "num_retries": 0
    }
  },
  "request": {
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [
      {
        "role": "system",
        "content": "You are role-playing as a customer...\n<scenario>..."
      },
      {
        "role": "user",
        "content": "Hi, this is customer service. How can I help?"
      }
    ],
    "temperature": 1.0,
    "reasoning_effort": "high",
    "api_key": "dummy",
    "api_base": "http://127.0.0.1:4199",
    "num_retries": 0
  }
}
```

### Claude Code实际发送给Anthropic API的请求

根据实际抓包 (参见 `CC_USER_BACKBONE_API.md`)，Claude Code会进行大量转换：

#### 关键发现

**1. 每次用户响应 = 2次API调用**
```
REQ-1: 生成会话标题 (title-gen)
├─ output_config.format = json_schema{title}
├─ temperature = 1
└─ system = "Generate a concise, sentence-case title..."

REQ-2: 实际用户回复
├─ thinking = {"type": "adaptive"}  ← 自动启用！
├─ output_config.effort = "high"
├─ max_tokens = 32000
├─ temperature = 省略 (默认1.0)
└─ context_management = {clear_thinking_20251015}
```

**2. 系统提示被改写为3块**
```json
[
  {
    "type": "text",
    "text": "x-anthropic-billing-header: ..."
  },
  {
    "type": "text", 
    "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."
  },
  {
    "type": "text",
    "text": "<你的实际系统提示>"
  }
]
```

**3. 用户消息被包装**
```xml
<system-reminder>
Current date: 2026-06-02
</system-reminder>

<session>
Hi, this is customer service. How can I help?
</session>
```

**4. 消息结构变化**
```
Tau2发送: 多轮对话 [system, user, assistant, user, ...]
Claude Code发送: 单个扁平化的user消息块
```

**5. API端点差异**
```
Tau2直接调用: POST /v1/messages
Claude Code: POST /v1/messages?beta=true
```

### 完整对比：Tau2 vs Claude Code

| 维度 | Tau2 UserSimulator | Claude Code UserSimulator | 相同？ |
|------|-------------------|---------------------------|--------|
| **API调用数** | 1次/轮 | 2次/轮 (title-gen + reply) | ❌ |
| **Extended Thinking** | OFF (默认) | **`{"type": "adaptive"}` ON** | ❌ |
| **Temperature** | 显式1.0 | 省略→API默认1.0 | ≈ |
| **max_tokens** | LiteLLM默认 | **32000** | ❌ |
| **output_config.effort** | 无 | **"high"** | ❌ |
| **系统提示** | 单块 | **3块 (2个样板+1个实际)** | ❌ |
| **用户消息** | 原始内容 | `<system-reminder>` + `<session>` 包装 | ❌ |
| **消息结构** | 多轮 (user/assistant交替) | 单个扁平化user块 | ❌ |
| **API端点** | `/v1/messages` | `/v1/messages?beta=true` | ❌ |
| **context_management** | 无 | `clear_thinking_20251015` | ❌ |

### 示例：同一个用户场景的实际请求对比

#### Tau2直接调用 (标准UserSimulator)
```http
POST https://api.anthropic.com/v1/messages
Content-Type: application/json
x-api-key: sk-ant-...

{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": "You are role-playing as a customer...\n<scenario>You want to check your account balance.</scenario>\n\nHi, this is customer service. How can I help?"
    }
  ],
  "temperature": 1.0,
  "max_tokens": 4096
}
```

#### Claude Code调用 (ClaudeCodeUserSimulator)

**REQ-1: 标题生成**
```http
POST https://api.anthropic.com/v1/messages?beta=true
Content-Type: application/json
x-api-key: sk-ant-...

{
  "model": "claude-sonnet-4-6",
  "system": [
    {"type": "text", "text": "x-anthropic-billing-header: ..."},
    {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    {"type": "text", "text": "Generate a concise, sentence-case title for this conversation..."}
  ],
  "messages": [
    {
      "role": "user",
      "content": "<session>...</session>"
    }
  ],
  "temperature": 1.0,
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"]
      }
    }
  }
}
```

**REQ-2: 实际回复**
```http
POST https://api.anthropic.com/v1/messages?beta=true
Content-Type: application/json
x-api-key: sk-ant-...

{
  "model": "claude-sonnet-4-6",
  "system": [
    {"type": "text", "text": "x-anthropic-billing-header: ..."},
    {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    {"type": "text", "text": "You are role-playing as a customer contacting customer service.\n<scenario>\nYou want to check your account balance.\n</scenario>"}
  ],
  "messages": [
    {
      "role": "user",
      "content": "<system-reminder>\nCurrent date: 2026-06-02\n</system-reminder>\n\n<session>\nHi, this is customer service. How can I help?\n</session>"
    }
  ],
  "thinking": {
    "type": "adaptive"
  },
  "output_config": {
    "effort": "high"
  },
  "max_tokens": 32000,
  "context_management": {
    "strategy": "clear_thinking_20251015"
  }
}
```

### 关键差异总结

1. **成本差异**: Claude Code每轮消耗**2倍**API调用（标题生成是纯overhead）
2. **推理能力**: Claude Code**自动启用** `thinking: adaptive`，即使未指定
3. **不可控性**: `claude` CLI**不暴露** temperature/thinking/max_tokens参数
4. **系统提示污染**: 2个样板块自动添加，可能影响模型行为
5. **格式差异**: `<system-reminder>` 和 `<session>` 标签会影响token计数

### 实验影响

⚠️ **重要**: 在Tau2实验中，使用 `ClaudeCodeUserSimulator` 与使用标准 `UserSimulator` (相同模型) **不是**公平对比：

- Claude Code是一个**不同的系统**，不只是"相同模型的不同包装"
- Adaptive thinking可能提升用户模拟质量，但也增加成本
- 系统提示的额外块可能改变模型行为
- 无法直接控制temperature/thinking等参数

### 如何复现抓包

1. 运行抓包脚本:
```bash
uv run python scripts/capture_cc_backbone.py
```

2. 手动抓包:
```bash
# 设置本地代理
export ANTHROPIC_BASE_URL=http://localhost:8888

# 运行Claude Code
claude --print --bare --model claude-sonnet-4-6 \
  --system-prompt "You are a customer" \
  "Hello"
```

3. 查看捕获的请求:
```bash
cat data/api_first_requests/capture_raw__anthropic_claude-sonnet-4-6__*.json
```

### 参考文档

- **CC_USER_BACKBONE_API.md**: Claude Code backbone详细分析
- **claude CLI版本**: 2.1.158
- **测量模型**: claude-sonnet-4-6

---

## 延伸阅读

- LiteLLM文档: https://docs.litellm.ai/
- LiteLLM Bedrock实现: https://github.com/BerriAI/litellm/blob/main/litellm/llms/bedrock.py
- OpenAI API文档: https://platform.openai.com/docs/api-reference
- Bedrock Converse API: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
