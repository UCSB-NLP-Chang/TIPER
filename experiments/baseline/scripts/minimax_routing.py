"""RoutingClient: 根据模型名路由 OpenAI 请求到两个后端。

- fireworks 模型（minimax-m3 等）→ 本地 :4114 openai→fireworks shim（Bearer dummy，
  shim 自换 FIREWORKS_PROXY_TOKEN；注入 reasoning_effort）。
- 其他模型（gpt-5.4 等）→ 标准 OpenAI（OPENAI_API_KEY，默认 base_url）。

这样同一进程内 agent(minimax) 与 user/human(gpt-5.4) 用不同后端，无需单一
OPENAI_BASE_URL。两 bench（sweet_rl / lost_in_conversation）只调
`client.chat.completions.create(model=..., ...)`，RoutingClient 实现该接口即可
drop-in 替换 OpenAI 实例。
"""

import os

import openai

_RealOpenAI = openai.OpenAI

# :4114 = openai_to_fireworks_proxy.py（openai /v1/chat/completions → fireworks）
_FW_BASE = os.environ.get("MINIMAX_FW_BASE", "http://127.0.0.1:4114/v1")


def is_fireworks_model(model: str) -> bool:
    m = (model or "").lower()
    # "claude"/"sonnet"/"vertex" routes to the vertex shim (MINIMAX_FW_BASE env
    # points at :4113 oai→vertex when running claude-sonnet-5 as agent).
    return (
        "fireworks" in m
        or "minimax" in m
        or "claude" in m
        or "sonnet" in m
        or "vertex" in m
        or "gpt-oss" in m
        or "qwen3p7" in m
        or "muse-glimmer" in m
    )


class _RoutingCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model=None, **kwargs):
        client = self.parent.fw if is_fireworks_model(model) else self.parent.oai
        return client.chat.completions.create(model=model, **kwargs)


class _RoutingChat:
    def __init__(self, parent):
        self.parent = parent

    @property
    def completions(self):
        return _RoutingCompletions(self.parent)


class RoutingClient:
    """Drop-in 替换 `OpenAI(...)`：根据 model 名路由到 :4114 或标准 OpenAI。"""

    def __init__(
        self,
        fw_base: str = _FW_BASE,
        oai_api_key: str | None = None,
        fw_api_key: str = "dummy",
    ):
        self.fw = _RealOpenAI(api_key=fw_api_key, base_url=fw_base)
        self.oai = _RealOpenAI(api_key=oai_api_key or os.environ["OPENAI_API_KEY"])

    @property
    def chat(self):
        return _RoutingChat(self)


def patch_model_openai():
    """替换 lost_in_conversation/model_openai.py 的模块单例 client。
    覆盖 assistant(generate) + user(generate_json) + system(generate_json)。
    用于 run_demo.py 子进程顶部。
    """
    import model_openai

    model_openai.model.client = RoutingClient()
    return model_openai.model.client


def patch_openai():
    """Monkey-patch `openai.OpenAI` 使任意 `OpenAI(...)` 调用返回 RoutingClient。

    RoutingClient 内部 sub-client 用 patch 前已保存的 `_RealOpenAI`，无递归。
    用于进程内 bench（sweetrl gepa_runner_sweetrl）：adapter/proposer 内的
    `OpenAI(api_key=..., base_url=...)` 一并路由——minimax→:4114，gpt-5.x→标准。

    必须在 bench 模块 `from openai import OpenAI` 之前调用，这样绑定的名字就是
    patched 后的 factory。
    """
    import openai

    # Idempotent: a second call (e.g. cert script calls patch_openai() then
    # imports gepa_runner_sweetrl whose top-level calls it again) must NOT
    # re-capture openai.OpenAI — by then it is the factory from the first
    # call, so re-saving would make _RealOpenAI the factory → infinite
    # recursion in RoutingClient.__init__. _RealOpenAI was already captured
    # at module import (line 15, the real class); keep it.
    if getattr(openai.OpenAI, "_minimax_routing_patched", False):
        return openai.OpenAI

    def _factory(*a, **kw):
        return RoutingClient()

    _factory._minimax_routing_patched = True
    openai.OpenAI = _factory
    return openai.OpenAI
