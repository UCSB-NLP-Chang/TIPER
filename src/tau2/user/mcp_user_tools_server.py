"""MCP server that exposes KnowledgeUserTools for Claude Code user simulator."""

import asyncio
import inspect
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.tools import KnowledgeUserTools
from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DB_PATH


def _build_tool_schema(method) -> dict:
    """Build JSON schema from method signature."""
    sig = inspect.signature(method)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        if ann == inspect.Parameter.empty:
            t = "string"
        elif ann == str:
            t = "string"
        elif ann == float:
            t = "number"
        elif ann == int:
            t = "integer"
        elif ann == bool:
            t = "boolean"
        else:
            t = "string"
        props[name] = {"type": t}
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def build_server(user_toolkit: KnowledgeUserTools) -> Server:
    server = Server("tau2-user-tools")

    # Collect all @is_tool methods (non-discoverable, always available)
    from tau2.environment.toolkit import DISCOVERABLE_ATTR

    tool_methods = {
        name: getattr(user_toolkit, name)
        for name in user_toolkit._func_tools.keys()
        if not getattr(getattr(user_toolkit, name), DISCOVERABLE_ATTR, False)
    }
    # Also include skill methods (discoverable but user-facing)
    skill_methods = {
        name: getattr(user_toolkit, name) for name in user_toolkit.get_skills().keys()
    }
    all_methods = {**tool_methods, **skill_methods}

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []
        for name, method in all_methods.items():
            doc = (method.__doc__ or "").strip().split("\n")[0]
            schema = _build_tool_schema(method)
            tools.append(Tool(name=name, description=doc, inputSchema=schema))
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name not in all_methods:
            return [TextContent(type="text", text=f"Error: unknown tool '{name}'")]
        try:
            result = all_methods[name](**arguments)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


async def main(db_json_path: str | None = None):
    if db_json_path:
        db = TransactionalDB.load(db_json_path)
    else:
        db = TransactionalDB.load(str(KNOWLEDGE_DB_PATH))

    user_toolkit = KnowledgeUserTools(db)
    server = build_server(user_toolkit)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(db_path))
