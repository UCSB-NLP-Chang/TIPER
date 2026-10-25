# Tau2 API配置总结

> 本文档汇总了Tau2框架中各个模型和系统的API配置关键信息。

## 📚 相关文档

1. **TAU2_API_CONFIG_GUIDE.md** - 完整的配置指南和tool calling详解
2. **API_LAYERS_EXPLAINED.md** - API调用的三层架构详解
3. **MODEL_API_SETTINGS_COMPARISON.md** - 各模型实际配置对比
4. **CC_USER_BACKBONE_API.md** - Claude Code backbone详细分析

---

## 🎯 核心问题解答

### Q: api_first_requests中的配置是直接发给OpenAI/Bedrock API的吗？

**A: ❌ 不是！**

你看到的是**传给LiteLLM的统一格式** (Layer 2)，不是最终发给API的格式 (Layer 3)。

```
Layer 1: Tau2 Config
   ↓ (Tau2转换)
Layer 2: LiteLLM统一格式 ← api_first_requests/保存的内容
   ↓ (LiteLLM转换)
Layer 3: Provider原生API ← 实际HTTP请求
```

**关键差异**:
- **OpenAI**: Layer 2 ≈ Layer 3 (几乎1:1映射)
- **Bedrock**: Layer 2 完全不同于 Layer 3 (大量转换)
  - `messages` 格式不同
  - `tools` 嵌套在 `toolConfig.tools`
  - `thinking` 放在 `additionalModelRequestFields`
  - `temperature` 放在 `inferenceConfig`

---

## 📊 模型配置快速对比

### GPT-5.2
```json
// 标准模式
{"temperature": 1.0, "seed": 361454}

// 高推理模式
{"reasoning_effort": "high", "temperature": 1.0, "seed": 373753}
```
- ✅ 推理模式**可选**
- 💡 根据任务复杂度决定

### GPT-5.4
```json
// 默认配置（自动包含高推理）
{"reasoning_effort": "high", "temperature": 1.0, "seed": 361454}
```
- ⚠️ **默认启用**高推理
- 💡 主要与GPT-5.2的区别

### Claude Sonnet 4.5 (Bedrock)
```json
{
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "temperature": 1.0,
  "allowed_openai_params": ["thinking"],
  "aws_region_name": "us-west-2",
  "seed": 361454
}
```
- 🧠 使用 `thinking` 代替 `reasoning_effort`
- 🔐 需要 `allowed_openai_params` 防止参数被过滤
- 🌎 Bedrock特定的区域配置

### Claude Sonnet 4.6 (Bedrock)
```json
// 标准模式（默认）
{"temperature": 1.0, "seed": 626729}

// 扩展思考模式（手动启用）
{
  "thinking": {"type": "enabled", "budget_tokens": 4096},
  "temperature": 1.0,
  "allowed_openai_params": ["thinking"]
}
```
- ✅ **默认不启用**扩展思考
- 💡 与Sonnet 4.5的主要区别

### Claude Code User Simulator (固定配置)
```json
// 无论llm_args如何配置，都是：
{
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "high"},
  "max_tokens": 32000
}
```
- 🔒 **固定配置**，无法通过`llm_args`修改
- ❌ **不支持** `reasoning_effort`参数（会被忽略）
- ❌ **不支持** `temperature`参数（会被忽略）
- 💡 `claude` CLI不暴露这些参数

---

## ⚠️ Claude Code User Simulator特殊情况

### 与标准User Simulator的对比

| 特性 | 标准UserSimulator | Claude Code UserSimulator |
|------|-------------------|---------------------------|
| **实现方式** | 直接调用LiteLLM | 通过`claude` CLI子进程 |
| **API调用数/轮** | 1次 | **2次** (title-gen + reply) |
| **Extended Thinking** | OFF | **`adaptive` 自动启用** |
| **Temperature** | 显式1.0 | 省略→默认1.0 |
| **max_tokens** | 4096 | **32000** |
| **系统提示** | 1块 | **3块** (billing + SDK + 实际) |
| **用户消息** | 原始 | `<system-reminder>` + `<session>` |
| **可控性** | ✅ 完全可控 | ❌ CLI不暴露参数 |
| **reasoning_effort支持** | ✅ 支持 | ❌ **不支持**（被忽略） |

### 关键警告

🚨 **ClaudeCodeUserSimulator ≠ UserSimulator + Sonnet 4.6**

这是**两个不同的系统**，不能直接比较：
1. 成本2倍（每轮2次调用）
2. Adaptive thinking**强制启用**（无法关闭）
3. 系统提示有额外样板块
4. **无法控制** thinking/temperature/max_tokens/reasoning_effort

**重要**: `llm_args` 中的 `reasoning_effort` 参数会被**完全忽略**！
详见: `CLAUDE_CODE_USER_REASONING_EFFORT.md`

### 何时使用

✅ **适合**:
- 测试Claude Code集成
- 需要真实的"Claude Code作为用户"体验

❌ **不适合**:
- 公平对比实验
- 成本敏感场景
- 需要精确控制参数

---

## 🔧 参数映射表

### reasoning_effort → thinking转换

| reasoning_effort | thinking.budget_tokens |
|------------------|------------------------|
| "minimal" | 128 |
| "low" | 1024 |
| "medium" | 2048 |
| "high" | 4096 |

**自动转换条件**:
- 模型名称以 `claude-` 开头
- 不是通过OpenAI代理调用
- 代码位置: `src/tau2/utils/llm_utils.py:518-530`

### 参数在各层的位置

| 参数 | Layer 2 (LiteLLM) | Layer 3 (OpenAI) | Layer 3 (Bedrock) |
|------|-------------------|------------------|-------------------|
| `temperature` | 顶层 | 顶层 | `inferenceConfig.temperature` |
| `reasoning_effort` | 顶层 | 顶层 | N/A (转为thinking) |
| `thinking` | 顶层 | N/A | `additionalModelRequestFields.thinking` |
| `tools` | 顶层 | 顶层 | `toolConfig.tools` (不同schema) |
| `seed` | 顶层 | 顶层 | ❌ 不支持 (会被忽略) |

---

## 💡 常见误解

### ❌ 误解1: api_first_requests是真实API格式
**✅ 真相**: 这是LiteLLM输入格式，不是最终HTTP请求

### ❌ 误解2: num_retries会发送给OpenAI
**✅ 真相**: 这是LiteLLM的重试参数，不会发送给API

### ❌ 误解3: seed在所有模型上都有效
**✅ 真相**: 
- OpenAI: ✅ 原生支持
- Bedrock: ❌ 不支持（会被忽略）
- Anthropic API: ❌ 不支持

### ❌ 误解4: GPT-5.2和GPT-5.4的区别只是版本
**✅ 真相**: GPT-5.4**默认启用**高推理模式，GPT-5.2需要手动指定

### ❌ 误解5: Sonnet 4.5和4.6配置相同
**✅ 真相**: 4.5默认启用扩展思考，4.6默认关闭

---

## 📝 CLI使用示例

### 标准Agent配置

```bash
# GPT-5.2 标准
uv run tau2 run --domain telecom \
  --agent-llm "gpt-5.2" \
  --agent-llm-args '{"temperature": 1.0}'

# GPT-5.2 高推理
uv run tau2 run --domain banking_knowledge \
  --agent-llm "gpt-5.2" \
  --agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'

# GPT-5.4 (默认高推理)
uv run tau2 run --domain banking_knowledge \
  --agent-llm "gpt-5.4" \
  --agent-llm-args '{"temperature": 1.0}'

# Sonnet 4.6 标准
uv run tau2 run --domain telecom \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --agent-llm-args '{"temperature": 1.0}'

# Sonnet 4.6 扩展思考
uv run tau2 run --domain banking_knowledge \
  --agent-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
```

### User Simulator配置

```bash
# 标准User Simulator
uv run tau2 run --domain telecom \
  --agent-llm "gpt-5.2" \
  --user user_simulator \
  --user-llm "bedrock/us.anthropic.claude-sonnet-4-6" \
  --user-llm-args '{"temperature": 1.0}'

# Claude Code User Simulator
uv run tau2 run --domain telecom \
  --agent-llm "gpt-5.2" \
  --user cc_user \
  --user-llm "bedrock/us.anthropic.claude-sonnet-4-6"
  # 注意: cc_user的参数无法通过llm-args控制！
```

---

## 🔍 调试技巧

### 1. 查看LiteLLM输入 (Layer 2)
```bash
cat data/api_first_requests/agent_response__gpt-5.2__*.json | jq '.setting.params'
```

### 2. 查看真实API请求 (Layer 3)
```python
import litellm

litellm._turn_on_debug()  # 打印完整HTTP请求
```

### 3. 使用HTTP代理抓包
```bash
# 启动mitmproxy
mitmproxy -p 8080

# 设置代理
export HTTPS_PROXY=http://localhost:8080

# 运行tau2
uv run tau2 run --domain mock --agent-llm "gpt-5.2"
```

### 4. 验证thinking转换
```python
from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage

generate(
    model="bedrock/us.anthropic.claude-sonnet-4-6",
    messages=[SystemMessage(content="Test")],
    reasoning_effort="high",  # 会自动转换为thinking
)
```

---

## 📈 性能与成本对比

### 推理模式影响

| 模型 | 标准模式 | 高推理模式 | 延迟增加 | 成本增加 |
|------|----------|-----------|----------|----------|
| GPT-5.2 | ✓ | ✓ | ~2-3x | ~1.5x |
| GPT-5.4 | 无 | 默认 | N/A | N/A |
| Sonnet 4.5 | ✓ | ✓ | ~2-4x | ~1.5-2x |
| Sonnet 4.6 | ✓ | ✓ | ~2-4x | ~1.5-2x |
| **CC User** | N/A | 固定adaptive | ~2-4x | **2x (双调用)** |

### 推荐配置策略

**简单任务** (mock, telecom基础):
```bash
--agent-llm-args '{"temperature": 1.0}'
# 不启用推理增强
```

**复杂任务** (banking_knowledge):
```bash
--agent-llm-args '{"temperature": 1.0, "reasoning_effort": "high"}'
```

**调试/快速迭代**:
```bash
--agent-llm-args '{"temperature": 1.0, "reasoning_effort": "minimal"}'
```

---

## 🗂️ 文件位置速查

### 配置相关
- `src/tau2/config.py` - 所有默认值
- `src/tau2/cli.py` - CLI参数解析
- `src/tau2/utils/llm_utils.py` - API调用核心逻辑

### 实际请求示例
- `data/api_first_requests/agent_response__gpt-5.2__*.json`
- `data/api_first_requests/agent_response__gpt-5.4__*.json`
- `data/api_first_requests/agent_response__bedrock_us.anthropic.claude-sonnet-4-5*.json`
- `data/api_first_requests/agent_response__bedrock_us.anthropic.claude-sonnet-4-6*.json`
- `data/api_first_requests/capture_raw__anthropic_claude-sonnet-4-6__*.json` (Claude Code)

### Agent/User实现
- `src/tau2/agent/llm_agent.py` - LLM Agent
- `src/tau2/user/user_simulator.py` - 标准User Simulator
- `src/tau2/user/claude_code_user_simulator.py` - Claude Code User Simulator

---

## ⚖️ 实验设计建议

### ✅ 公平对比原则

**可以对比**:
- GPT-5.2 vs GPT-5.4 (同推理等级)
- Sonnet 4.5 vs Sonnet 4.6 (同推理等级)
- 不同模型的标准UserSimulator

**不能直接对比**:
- ClaudeCodeUserSimulator vs 标准UserSimulator
- 不同推理等级的同模型配置

### 📋 报告要求

使用ClaudeCodeUserSimulator时**必须声明**:
1. 每轮2次API调用（标题生成+回复）
2. Adaptive thinking自动启用
3. 系统提示包含SDK标识样板
4. 这是**不同的系统**，不是"相同模型"

### 🔬 控制变量

如果需要公平对比:
1. 使用相同的user实现（都用标准或都用CC）
2. 使用相同的推理等级
3. 固定temperature和seed
4. 记录实际API调用数和成本

---

## 🎓 学习路径

1. **入门**: 先读 `TAU2_API_CONFIG_GUIDE.md` 了解基础
2. **深入**: 读 `API_LAYERS_EXPLAINED.md` 理解三层架构
3. **对比**: 读 `MODEL_API_SETTINGS_COMPARISON.md` 看实际配置
4. **特殊**: 读 `CC_USER_BACKBONE_API.md` 了解Claude Code特性
5. **实践**: 运行实验，查看 `data/api_first_requests/` 中的实际请求

---

## 📞 参考链接

- LiteLLM文档: https://docs.litellm.ai/
- OpenAI API文档: https://platform.openai.com/docs/api-reference
- Anthropic API文档: https://docs.anthropic.com/en/api/messages
- Bedrock Converse API: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- Tau2 GitHub: (项目仓库链接)
