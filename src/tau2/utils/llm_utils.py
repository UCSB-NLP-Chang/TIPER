import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

# Must be imported before litellm so the google.auth patch is in place first.
import tau2.utils.cresta_vertex  # noqa: F401
from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.utils import DATA_WORKSPACE

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to type mismatches between streaming and non-streaming response types
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Configure httpx connection limits for LiteLLM
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
litellm.client_session = httpx.Client(limits=httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")

# litellm._turn_on_debug()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

if USE_LANGFUSE:
    litellm.success_callback = ["langfuse"]
else:
    litellm.success_callback = []

litellm.drop_params = True

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        # Non-fatal: litellm's completion_cost strips provider prefixes (e.g. the
        # `anthropic/` on `anthropic/accounts/fireworks/models/glm-5p2`) and raises
        # BadRequestError "Provider NOT provided". The sim itself succeeds; only the
        # per-call cost accounting is unavailable. Demoted to debug to keep logs clean.
        logger.debug(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
        # Anthropic prompt-caching stats (0 when caching is off / unsupported).
        # Recorded so future sims/optimization runs can quantify cache hit rate
        # and real input-token cost (cache_read is billed at ~10% of input price).
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0)
        or 0,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


_VALID_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    """Replace chars that violate Bedrock's toolUse.name constraint."""
    return _VALID_TOOL_NAME_RE.sub("_", name)


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": _sanitize_tool_name(tc.name),
                        "function": {
                            "name": _sanitize_tool_name(tc.name),
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def _local_proxy_openai_model_kwargs(model: str) -> tuple[str, dict[str, Any]]:
    """Route local Anthropic-compatible aliases through the OpenAI proxy adapter.

    Some experiments expose Claude-shaped model names from a local LiteLLM proxy
    while routing them to OpenAI models. Calling those aliases through LiteLLM's
    Anthropic adapter can corrupt tool_choice/tool schemas on the second hop.
    Using the OpenAI adapter against the proxy preserves standard OpenAI tool
    calling for regular tau2 user/agent simulations.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not base_url or not model.startswith("claude-"):
        return model, {}

    parsed = urlparse(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return model, {}

    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"

    return f"openai/{model}", {
        "api_base": api_base,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", "dummy"),
    }


def validate_message(message: Message) -> None:
    """
    Validate the message.
    """

    def has_text_content(message: Message) -> bool:
        """
        Check if the message has text content.
        """
        return message.content is not None and bool(message.content.strip())

    def has_content_or_tool_calls(message: ParticipantMessageBase) -> bool:
        """
        Check if the message has content or tool calls.
        """
        return message.has_content() or message.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: list[Message]) -> None:
    """
    Validate the message history.
    """
    for message in messages:
        validate_message(message)


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.

    Args:
        messages: List of litellm message dictionaries

    Returns:
        Modified message list with content split into lines for readability
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            # Split content on newlines for better readability
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    Behavior depends on the current log mode:
    - "all": Saves every LLM call
    - "latest": Only keeps the most recent call of each call_name type

    Args:
        request_data: Dictionary containing request information
        response_data: Dictionary containing response information
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message")
    """
    log_dir = llm_log_dir.get()

    if log_dir is None:
        # No log directory set, skip logging
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get current logging mode
    current_log_mode = llm_log_mode.get()

    # If mode is "latest" and call_name is provided, remove existing files with the same call_name
    if current_log_mode == "latest" and call_name:
        # Find and remove existing files with this call_name
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                # File might have been removed by another thread, ignore
                pass

    # Create a new file for this LLM call
    call_id = str(uuid.uuid4())[:8]  # Use short UUID for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

    # Include call_name in filename if provided
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    # Create complete JSON structure with both request and response
    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Write to file with indentation
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


# Tracks which (call_name, model, params) settings have already been logged, so
# we emit the full outgoing API parameters exactly once per distinct setting.
_LOGGED_API_SETTINGS: set[str] = set()


# Directory where the first full API request for each distinct setting is saved.
API_FIRST_REQUEST_DIR = Path(
    os.environ.get(
        "TAU2_API_FIRST_REQUEST_DIR",
        DATA_WORKSPACE / "artifacts" / "api_first_requests",
    )
)


def _log_api_settings_once(
    call_name: Optional[str],
    model: str,
    completion_model: str,
    tool_choice: Optional[str],
    tools_schema: Optional[list],
    completion_kwargs: dict,
    litellm_messages: list,
) -> None:
    """Save the COMPLETE first API request for each distinct setting, once.

    The dedup signature is the configuration that stays constant across calls
    (model, reasoning_effort/thinking, temperature, tool_choice, #tools, etc.),
    so each unique setting fires exactly once — on its first call. For that first
    call we persist the full outgoing request (including the complete, untruncated
    messages actually sent to the API) to a local JSON file.
    """
    safe_params = {
        k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
        for k, v in completion_kwargs.items()
    }
    setting = {
        "call_name": call_name,
        "model": model,
        "completion_model": completion_model,
        "tool_choice": tool_choice,
        "n_tools": len(tools_schema) if tools_schema else 0,
        "params": safe_params,
    }
    sig = json.dumps(setting, sort_keys=True, default=str)
    if sig in _LOGGED_API_SETTINGS:
        return
    _LOGGED_API_SETTINGS.add(sig)

    # Build a readable, collision-resistant filename for this setting.
    short_hash = hashlib.sha1(sig.encode()).hexdigest()[:8]
    raw_name = f"{call_name or 'call'}__{model}__{short_hash}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)

    # The complete first request exactly as sent to litellm.completion().
    full_request = {
        "setting": setting,
        "request": {
            "model": completion_model,
            "messages": litellm_messages,
            "tools": tools_schema,
            "tool_choice": tool_choice,
            **completion_kwargs,
        },
    }
    try:
        API_FIRST_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        out_path = API_FIRST_REQUEST_DIR / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(full_request, f, indent=2, default=str)
    except Exception as e:  # never let logging break a run
        logger.warning(f"Failed to save first API request for {safe_name}: {e}")
        out_path = "<save failed>"

    print(
        f"[LLM API first request saved -> {out_path}]\n"
        + json.dumps(setting, indent=2, sort_keys=True, default=str),
        file=sys.stderr,
        flush=True,
    )


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    validate_message_history(messages)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES
    if kwargs.get("retry_strategy") is None:
        kwargs["retry_strategy"] = "exponential_backoff_retry"

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"
    completion_model, proxy_kwargs = _local_proxy_openai_model_kwargs(model)
    completion_kwargs = {**proxy_kwargs, **kwargs}

    # Workaround: litellm <1.82.7 doesn't recognize Claude 4.x as supporting
    # reasoning, so reasoning_effort/thinking get silently dropped. Convert
    # reasoning_effort to the Anthropic thinking param and whitelist it via
    # allowed_openai_params so drop_params=True doesn't strip it.
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

    # Anthropic prompt caching: mark the (large, constant across calls) system
    # prompt and the tools schema as ephemeral cache breakpoints so repeated
    # calls within a sim and across sims of the same domain hit cache_read
    # (~10% of input price) instead of re-billing full input tokens. Both
    # Cresta dev-proxy routes (/proxy/fireworks, /proxy/vertex) honor
    # cache_control. Skipped for OpenAI-routed aliases (no cache_control) and
    # non-Claude models (OpenAI auto-caches; others may reject the marker).
    if (
        "claude-" in model or model.startswith("claude-")
    ) and not completion_model.startswith("openai/"):
        for m in litellm_messages:
            if m.get("role") == "system" and "cache_control" not in m:
                # Use the content-block list form so litellm's vertex_ai/anthropic
                # transform preserves the cache_control breakpoint on the system
                # prompt (the sibling-key form on a string content gets dropped
                # when tools are present).
                m["content"] = [
                    {
                        "type": "text",
                        "text": m["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                m["cache_control"] = {"type": "ephemeral"}  # marker to skip re-wrap
        if tools_schema:
            last_tool = tools_schema[-1]
            if "cache_control" not in last_tool:
                last_tool["cache_control"] = {"type": "ephemeral"}
        # Third breakpoint on the last conversation message: the growing
        # history (msg1..msgN-1) becomes a cacheable prefix, so the next
        # call reads it at ~10% input price and only the newest turn is
        # billed at 1x. The breakpoint moves each turn. A top-level
        # cache_control sibling is placed correctly by litellm's
        # vertex/anthropic transform on tool_result and user text blocks
        # (verified by direct transform_request probes); it is dropped
        # for assistant messages that carry only tool_calls (content is
        # None), so those are skipped (rare as a trailing message anyway,
        # since the loop normally ends on a tool_result awaiting the next
        # agent step).
        last_msg = None
        for m in reversed(litellm_messages):
            if m.get("role") != "system":
                last_msg = m
                break
        if (
            last_msg is not None
            and "cache_control" not in last_msg
            and last_msg.get("role") in ("tool", "user")
        ):
            last_msg["cache_control"] = {"type": "ephemeral"}

    # Prepare request data for logging
    formatted_messages = _format_messages_for_logging(litellm_messages)
    request_data = {
        "model": model,
        "completion_model": completion_model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in completion_kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    # Log the full outgoing API parameters once per distinct setting.
    _log_api_settings_once(
        call_name=call_name,
        model=model,
        completion_model=completion_model,
        tool_choice=tool_choice,
        tools_schema=tools_schema,
        completion_kwargs=completion_kwargs,
        litellm_messages=litellm_messages,
    )

    start_time = time.perf_counter()
    try:
        response = completion(
            model=completion_model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **completion_kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time
    cost = get_response_cost(response)
    usage = get_response_usage(response)

    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in raw_tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
        generation_time_seconds=generation_time_seconds,
    )

    # Log complete LLM call (request + response)
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    # Add timestamp to request data
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    return message


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


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
        usage["cache_read_input_tokens"] += (
            message.usage.get("cache_read_input_tokens", 0) or 0
        )
        usage["cache_creation_input_tokens"] += (
            message.usage.get("cache_creation_input_tokens", 0) or 0
        )
    return usage


def extract_json_from_llm_response(response: str) -> str:
    """
    Extract JSON from an LLM response, handling markdown code blocks.
    """
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content between first { and last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    # Return original response as fallback
    return response
