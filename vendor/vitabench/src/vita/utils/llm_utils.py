import json
import re
import time
from typing import Any, Optional

from loguru import logger
import requests


from vita.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add interleaved thinking content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                reasoning_content = message.raw_data["message"].get("reasoning_content")
                if reasoning_content:
                    messages_formatted[-1]["reasoning_content"] = reasoning_content
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            raw_message = original_msg.raw_data.get("message", {})

            # Extract reasoning content and signature
            reasoning_content = raw_message.get("reasoning_content") or raw_message.get("reasoning")
            signature = raw_message.get("signature")
            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content
            if signature:
                messages_formatted[i]["signature"] = signature

    except Exception as e:
        import traceback
        traceback.print_exc()

    return messages_formatted


def to_deepseek_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            reasoning_content = original_msg.raw_data.get("message", {}).get("reasoning_content", None) or \
                               original_msg.raw_data.get("message", {}).get("reasoning", None)

            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content

    except Exception as e:
        import traceback
        traceback.print_exc()

    return messages_formatted


# Keys stored in models.yaml for bookkeeping/transport that are NOT valid
# OpenAI chat-completions request parameters. They must be stripped from the
# JSON body before POSTing, otherwise the OpenAI API returns 400
# "Unrecognized request arguments supplied".
_NON_API_KEYS = {
    "base_url",
    "headers",
    "max_input_tokens",
    "cost_1m_token_dollar",
    "name",
    "model_id",
}


def _is_openai_reasoning_model(model: str) -> bool:
    """gpt-5*/o1/o3/o4* are reasoning models with stricter param rules."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def sanitize_payload(data: dict) -> dict:
    """Build a clean OpenAI-compatible request body from ``data``.

    - drop non-API bookkeeping keys (base_url/headers/cost/etc.)
    - drop keys whose value is None (e.g. empty tools/tool_choice)
    - for reasoning models: rename max_tokens -> max_completion_tokens,
      drop temperature (only default is supported) and Anthropic-style thinking.
    """
    model = data.get("model", "")
    payload = {
        k: v for k, v in data.items() if k not in _NON_API_KEYS and v is not None
    }
    if _is_openai_reasoning_model(model):
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        payload.pop("temperature", None)
        payload.pop("thinking", None)
    return payload


def kwargs_adapter(data: dict, enable_think: False, messages: list) -> dict:
    if "claude" in data["model"]:
        if not enable_think:
            data["thinking"] = {"type": "disabled"}
        else:
            data["messages"] = to_claude_think_official(data["messages"], messages)
    elif "deepseek" in data["model"]:
        if enable_think:
            data["messages"] = to_deepseek_think_official(data["messages"], messages)
    else:
        if not enable_think:
            # Non-think mode. Respect an explicit reasoning_effort from the model
            # config; only fall back / strip when none was configured.
            # NOTE: gpt-5.x reject "minimal" (they support none/low/medium/high/xhigh),
            # so only the exact "gpt-5" model gets "minimal"; newer gpt-5.x default
            # to "none".
            if not data.get("reasoning_effort"):
                model_id = data.get("model", "")
                if model_id == "gpt-5":
                    data["reasoning_effort"] = "minimal"
                elif model_id.startswith("gpt-5"):
                    data["reasoning_effort"] = "none"
                elif "reasoning_effort" in data:
                    data.pop("reasoning_effort")
    return data


def _format_responses_input(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """Convert vitabench Message history into Responses-API `input` items.

    Returns (instructions, input_items). SystemMessage -> instructions; user ->
    {role:user}; assistant text -> {role:assistant}; assistant tool_calls ->
    {type:function_call, call_id, name, arguments}; ToolMessage (tool result) ->
    {type:function_call_output, call_id, output}.
    """
    instructions = None
    items: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            instructions = m.content
        elif isinstance(m, UserMessage):
            items.append({"role": "user", "content": m.content or ""})
        elif isinstance(m, AssistantMessage):
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            if m.is_tool_call():
                for tc in (m.tool_calls or []):
                    items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        if not isinstance(tc.arguments, str) else tc.arguments,
                    })
        elif isinstance(m, ToolMessage):
            items.append({
                "type": "function_call_output",
                "call_id": m.id,
                "output": m.content or "",
            })
    return instructions, items


def generate_responses(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> AssistantMessage | None:
    """Call OpenAI's /v1/responses endpoint. Required for models that reject
    reasoning_effort + function tools on /v1/chat/completions (gpt-5.4).

    reasoning_effort comes from the model's models.yaml entry; max_tokens ->
    max_output_tokens; tool results are passed back as function_call_output items.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        cfg = models[model]
        base_url = cfg["base_url"]  # .../v1/responses
        headers = cfg["headers"]
        real_model = cfg.get("model_id", model)

        instructions, input_items = _format_responses_input(messages)

        resp_tools = None
        if tools:
            resp_tools = []
            for t in tools:
                s = t.openai_schema  # chat-completions shape: {type:function, function:{...}}
                fn = s.get("function", s)
                resp_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        reasoning = None
        re_effort = cfg.get("reasoning_effort")
        if re_effort:
            reasoning = {"effort": re_effort}

        body: dict = {"model": real_model, "input": input_items}
        if instructions:
            body["instructions"] = instructions
        if resp_tools:
            body["tools"] = resp_tools
            if tool_choice is None:
                body["tool_choice"] = "auto"
            else:
                body["tool_choice"] = tool_choice
        if reasoning:
            body["reasoning"] = reasoning
        if cfg.get("max_tokens"):
            body["max_output_tokens"] = cfg["max_tokens"]

        max_retries = 3
        retry_delay = 1
        response = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(base_url, json=body, headers=headers, timeout=(10, 600))
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable:
                    response = resp.json()
                    break
                if attempt < max_retries:
                    logger.warning(f"API returned {resp.status_code}, attempt {attempt + 1} retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    logger.warning(f"Request exception, attempt {attempt + 1}: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

        if response is None:
            raise ValueError("no response from /v1/responses")

        # parse output items: collect assistant text + function_calls
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in response.get("output", []) or []:
            t = item.get("type")
            if t == "message":
                for c in item.get("content", []) or []:
                    if c.get("type") in ("output_text", "text"):
                        content_parts.append(c.get("text", "") or "")
            elif t == "function_call":
                args_raw = item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
                tool_calls.append(ToolCall(
                    id=item.get("call_id"),
                    name=item.get("name"),
                    arguments=args,
                ))
        content = "".join(content_parts) if content_parts else None
        tool_calls = tool_calls or None

        u = response.get("usage") or {}
        usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
        }
        cost = get_response_cost(usage, model)
        return AssistantMessage(
            role="assistant", content=content, tool_calls=tool_calls,
            cost=cost, usage=usage, raw_data=response,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)
        return None


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    # Models that require the /v1/responses endpoint (e.g. gpt-5.4 rejects
    # reasoning_effort + function tools on /v1/chat/completions). Route them to
    # generate_responses; otherwise use the stock chat-completions path.
    if models.get(model, {}).get("endpoint") == "responses":
        return generate_responses(model, messages, tools, tool_choice, enable_think, **kwargs)
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        try:
            data = {
                "model": model,
                "messages": messages_formatted,
                "stream": False,
                "temperature": kwargs.get("temperature"),
                "tools": tools,
                "tool_choice": tool_choice,
            }
            data.update(models[model])
            # Allow a config entry to alias a real OpenAI model via `model_id`
            # (e.g. two "gpt-5.2" variants with different reasoning_effort). The
            # POSTed `model` and all downstream detection use the real id.
            data["model"] = models[model].get("model_id", model)
            data = kwargs_adapter(data, enable_think, messages)
            headers = models[model]["headers"]
            base_url = data["base_url"]
            data = sanitize_payload(data)

            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(base_url, json=data, headers=headers, timeout=(10, 600))

                    # Retry on rate limits (429) and any server error (5xx);
                    # everything else (2xx / 4xx) is returned as-is.
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if not retryable:
                        response = response.json()
                        break

                    if attempt < max_retries:
                        logger.warning(f"API returned {response.status_code}, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        response.raise_for_status()

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        except Exception as e:
            logger.error(e)
            raise e
        usage = get_response_usage(response)
        cost = get_response_cost(usage, model)
        try:
            response = response['choices'][0]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Full response: {json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, dict) else response}")
            raise ValueError(f"Invalid API response format: {e}") from e
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        content = response['message'].get('content')
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=response,
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost
