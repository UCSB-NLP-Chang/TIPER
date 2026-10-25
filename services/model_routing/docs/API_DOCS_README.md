# Tau2 API配置文档索引

本目录包含关于Tau2框架中LLM API配置、参数传递和模型设置的完整文档。

## 📚 文档列表

### 1. API_CONFIGURATION_SUMMARY.md ⭐ **从这里开始**
**快速参考和总结文档**

- 🎯 所有关键问题的简短解答
- 📊 模型配置快速对比表
- ⚠️ Claude Code特殊情况警告
- 💡 常见误解澄清
- 📝 CLI使用示例
- 🔍 调试技巧

**适合**: 快速查询、新手入门、实验配置

---

### 2. TAU2_API_CONFIG_GUIDE.md
**完整的配置指南和Tool Calling详解**

内容:
- 默认配置详解
- LiteLLM参数传递机制
- Tool Calling完整实现流程
- OpenAI vs Bedrock API差异
- 实际请求示例分析

**适合**: 深入理解配置系统、开发新功能、调试问题

---

### 3. API_LAYERS_EXPLAINED.md
**API调用三层架构详解**

内容:
- Layer 1: Tau2 Config → LiteLLM
- Layer 2: LiteLLM统一格式
- Layer 3: Provider原生API
- 各层参数转换详解
- Claude Code特殊处理流程
- 完整的HTTP请求对比

**适合**: 理解API调用流程、排查转换问题、了解Provider差异

**核心问题**: "api_first_requests中的配置是直接发给API的吗？"
**答案**: ❌ 不是！是传给LiteLLM的格式，还需要再次转换。

---

### 4. MODEL_API_SETTINGS_COMPARISON.md
**各模型实际配置对比**

内容:
- GPT-5.2配置详解
- GPT-5.4配置详解
- Claude Sonnet 4.5配置详解
- Claude Sonnet 4.6配置详解
- Claude Code User Simulator特殊配置
- 推理模式对比
- 性能与成本分析

**适合**: 选择模型、配置实验、对比性能

**快速对比**:
- GPT-5.2: 推理模式可选
- GPT-5.4: 推理模式默认启用
- Sonnet 4.5: 扩展思考默认启用
- Sonnet 4.6: 扩展思考默认关闭
- CC User: Adaptive thinking强制启用 + 每轮2次调用

---

### 5. CC_USER_BACKBONE_API.md
**Claude Code Backbone详细分析**

内容:
- ClaudeCodeUserSimulator实现原理
- 每轮2次API调用详解
- 系统提示改写分析
- 消息包装机制
- 与标准UserSimulator的11维度对比
- 控制性问题和限制

**适合**: 使用Claude Code User、对比实验设计、理解成本差异

**关键警告**: ClaudeCodeUserSimulator ≠ UserSimulator + Sonnet 4.6

---

## 🗺️ 使用导航

### 场景1: 我是新手，想快速了解配置
👉 先读: **API_CONFIGURATION_SUMMARY.md**  
👉 然后: 查看实际请求示例 `data/api_first_requests/`

### 场景2: 我要配置实验，选择模型
👉 先读: **MODEL_API_SETTINGS_COMPARISON.md**  
👉 参考: **API_CONFIGURATION_SUMMARY.md** 中的CLI示例

### 场景3: 我的配置没生效，想调试
👉 先读: **API_LAYERS_EXPLAINED.md** 理解转换流程  
👉 然后: **TAU2_API_CONFIG_GUIDE.md** 查看调试技巧  
👉 使用: `litellm._turn_on_debug()` 查看真实请求

### 场景4: 我要使用Claude Code User Simulator
👉 **必读**: **CC_USER_BACKBONE_API.md**  
👉 警告: **API_CONFIGURATION_SUMMARY.md** 中的特殊情况说明  
👉 对比: **MODEL_API_SETTINGS_COMPARISON.md** 第5节

### 场景5: 我想理解Tool Calling工作原理
👉 先读: **TAU2_API_CONFIG_GUIDE.md** 第3节  
👉 深入: **API_LAYERS_EXPLAINED.md** 查看完整流程  
👉 实践: 查看 `src/tau2/environment/tool.py`

### 场景6: 我要对比OpenAI和Bedrock
👉 先读: **API_LAYERS_EXPLAINED.md** 中的请求对比  
👉 参考: **TAU2_API_CONFIG_GUIDE.md** 中的差异表  
👉 验证: 查看 `data/api_first_requests/` 中的实际示例

---

## 🔑 关键概念速查

### API调用三层架构
```
Layer 1: Tau2 Config (CLI参数, TextRunConfig)
   ↓
Layer 2: LiteLLM统一格式 (api_first_requests/保存的)
   ↓
Layer 3: Provider原生API (实际HTTP请求)
```

### 推理模式对比
| 模型 | 默认推理 | 参数名 | 转换 |
|------|---------|--------|------|
| GPT-5.2 | ❌ OFF | `reasoning_effort` | 无 |
| GPT-5.4 | ✅ ON | `reasoning_effort` | 无 |
| Sonnet 4.5 | ✅ ON | `thinking` | `reasoning_effort` → `thinking` |
| Sonnet 4.6 | ❌ OFF | `thinking` | `reasoning_effort` → `thinking` |

### reasoning_effort → thinking映射
```python
"minimal" → 128 tokens
"low"     → 1024 tokens
"medium"  → 2048 tokens
"high"    → 4096 tokens
```

### Claude Code特殊性
- **API调用**: 每轮2次 (title + reply)
- **Thinking**: 强制 `adaptive`
- **Max tokens**: 强制 32000
- **系统提示**: 3块 (billing + SDK + 实际)
- **可控性**: ❌ CLI不暴露参数

---

## 📂 相关代码文件

### 配置与转换
- `src/tau2/config.py` - 所有默认值
- `src/tau2/cli.py` - CLI参数解析
- `src/tau2/utils/llm_utils.py` - API调用核心逻辑
  - `to_litellm_messages()` - 消息格式转换
  - `generate()` - 主调用函数
  - `_sanitize_tool_name()` - Bedrock工具名清理

### Agent与User
- `src/tau2/agent/llm_agent.py` - LLM Agent实现
- `src/tau2/user/user_simulator.py` - 标准User Simulator
- `src/tau2/user/claude_code_user_simulator.py` - Claude Code User

### 工具系统
- `src/tau2/environment/tool.py` - Tool定义和schema生成
- `src/tau2/domains/*/tools.py` - 各domain的工具定义

---

## 📊 实际请求示例位置

```bash
data/api_first_requests/
├── agent_response__gpt-5.2__*.json           # GPT-5.2 Agent
├── agent_response__gpt-5.4__*.json           # GPT-5.4 Agent
├── agent_response__bedrock_*_sonnet-4-5*.json # Sonnet 4.5 Agent
├── agent_response__bedrock_*_sonnet-4-6*.json # Sonnet 4.6 Agent
├── user_simulator_response__gpt-5.2__*.json  # GPT-5.2 User
├── user_simulator_response__bedrock_*_4-6*.json # Sonnet 4.6 User
└── capture_raw__anthropic_claude-sonnet-4-6__*.json # Claude Code User
```

### 查看示例
```bash
# GPT-5.2高推理模式
cat data/api_first_requests/agent_response__gpt-5.2__1c160381.json | jq '.setting.params'

# Sonnet 4.5扩展思考
cat data/api_first_requests/agent_response__bedrock*4-5*.json | jq '.setting.params' | head -20

# Claude Code捕获
cat data/api_first_requests/capture_raw__*.json | jq '.request'
```

---

## ⚠️ 重要警告

### 1. api_first_requests不是真实API格式
这是**LiteLLM输入** (Layer 2)，不是最终HTTP请求 (Layer 3)。

**OpenAI**: Layer 2 ≈ Layer 3  
**Bedrock**: Layer 2 完全不同于 Layer 3

### 2. Claude Code User不等于标准User + Sonnet 4.6
这是**完全不同的系统**:
- 每轮2次调用（成本2倍）
- Adaptive thinking强制启用
- 系统提示有额外样板
- 无法控制thinking/temperature/max_tokens

**实验报告时必须声明这些差异！**

### 3. 推理模式默认值不同
- GPT-5.2: 默认**关闭**
- GPT-5.4: 默认**开启**
- Sonnet 4.5: 默认**开启** (实验配置)
- Sonnet 4.6: 默认**关闭**

### 4. Bedrock不支持seed
`seed` 参数在Bedrock上会被忽略，无法保证可重现性。

---

## 🔧 常用调试命令

### 查看LiteLLM输入
```bash
# 查看所有GPT-5.2的参数配置
ls data/api_first_requests/agent_response__gpt-5.2__*.json | \
  xargs -I {} sh -c 'jq ".setting.params" "{}"'

# 对比标准User vs Claude Code User
diff \
  <(jq '.request' data/api_first_requests/user_simulator_response__bedrock*4-6*.json | head -50) \
  <(jq '.request' data/api_first_requests/capture_raw__*.json | head -50)
```

### 启用LiteLLM调试
```python
import litellm

litellm._turn_on_debug()

from tau2.utils.llm_utils import generate
# 会打印完整的HTTP请求
```

### 抓包真实请求
```bash
# 方法1: mitmproxy
mitmproxy -p 8080
export HTTPS_PROXY=http://localhost:8080
uv run tau2 run --domain mock --agent-llm "gpt-5.2"

# 方法2: 捕获Claude Code
uv run python scripts/capture_cc_backbone.py
```

---

## 📖 扩展阅读

### LiteLLM
- 官方文档: https://docs.litellm.ai/
- Bedrock实现: https://github.com/BerriAI/litellm/blob/main/litellm/llms/bedrock.py
- OpenAI实现: https://github.com/BerriAI/litellm/blob/main/litellm/llms/openai.py

### Provider API文档
- OpenAI API: https://platform.openai.com/docs/api-reference
- Anthropic API: https://docs.anthropic.com/en/api/messages
- Bedrock Converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

### Tau2相关
- CLAUDE.md - Tau2项目说明
- AGENTS.md - 架构和模式指南

---

## 🎯 快速答案

### Q: 配置文件在哪？
A: `src/tau2/config.py`

### Q: 如何修改Agent的推理模式？
A: `--agent-llm-args '{"reasoning_effort": "high"}'`

### Q: 为什么我的thinking参数没生效？
A: 检查是否添加了 `"allowed_openai_params": ["thinking"]`

### Q: GPT-5.2和5.4有什么区别？
A: GPT-5.4默认启用高推理模式，GPT-5.2需要手动指定

### Q: 如何查看真实API请求？
A: `litellm._turn_on_debug()` 或使用mitmproxy抓包

### Q: Claude Code User和标准User有什么区别？
A: 完全不同的系统！CC每轮2次调用，adaptive thinking强制启用

### Q: Bedrock支持seed吗？
A: ❌ 不支持，参数会被忽略

### Q: api_first_requests中的是真实API请求吗？
A: ❌ 不是，是LiteLLM输入格式，还需要再次转换

---

## 📝 更新日志

- **2026-06-02**: 创建完整的API配置文档体系
  - API_CONFIGURATION_SUMMARY.md (总结)
  - TAU2_API_CONFIG_GUIDE.md (完整指南)
  - API_LAYERS_EXPLAINED.md (三层架构)
  - MODEL_API_SETTINGS_COMPARISON.md (模型对比)
  - 更新CC_USER_BACKBONE_API.md分析
  - 创建本索引文档

---

**建议阅读顺序**: Summary → Comparison → Layers → Guide → CC_USER
