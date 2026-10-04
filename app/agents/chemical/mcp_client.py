"""MCP session helpers: tool discovery and execution."""

from __future__ import annotations

import logging
import time

from mcp import Client

from .models import _ToolResult

logger = logging.getLogger(__name__)

_MAX_LOGGED_ARG_CHARS = 40


_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def _tool_schema(tool) -> dict:
    """JSON schema of an MCP tool.

    mcp 2.x exposes input_schema; 1.x (and test doubles) use inputSchema.
    """
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or _EMPTY_SCHEMA


async def _discover_tools(client: Client) -> list[dict]:
    """Fetch tool schemas from the MCP server as OpenAI function definitions."""
    tools_response = await client.list_tools()
    openai_tools = []

    # Loops through each tool.
    # Converts it into a dictionary that matches the format OpenAI expects for function calling:
    # "type": "function" → tells OpenAI this is a callable function.
    # "name" → the tool’s name.
    # "description" → either the tool’s description or a fallback string.

    # "parameters" → the schema of inputs the tool accepts.
    for tool in tools_response.tools:
        schema = _tool_schema(tool)
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": schema,
            },
        })

    logger.debug(
        "[ChemicalAgent] Discovered %d MCP tools: %s",
        len(openai_tools),
        [t["function"]["name"] for t in openai_tools],
    )
    return openai_tools


def _extract_output(result) -> str:
    """Flatten an MCP call result into plain text."""
    if result.structured_content:
        return str(result.structured_content)
    if result.content:
        texts = [item.text for item in result.content if hasattr(item, "text")]
        return "\n".join(texts) if texts else "[Tool returned empty content]"
    return "[Tool returned no content]"


async def _execute_mcp_tool(client: Client, tool_name: str, arguments: dict) -> _ToolResult:
    """Call one MCP tool; never raises — failures come back as success=False."""
    t0 = time.monotonic()
    # Truncate long values so clinical SMILES are not fully logged
    safe_args = {
        k: (v[:_MAX_LOGGED_ARG_CHARS] + "…" if isinstance(v, str) and len(v) > _MAX_LOGGED_ARG_CHARS else v)
        for k, v in arguments.items()
    }
    logger.info("[ChemicalAgent] Calling MCP tool=%s args=%s", tool_name, safe_args)

    try:
        result = await client.call_tool(tool_name, arguments)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info("[ChemicalAgent] Tool=%s SUCCESS latency=%.0fms", tool_name, latency_ms)
        return _ToolResult(tool_name, arguments, _extract_output(result), success=True)

    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.error(
            "[ChemicalAgent] Tool=%s FAILED latency=%.0fms error=%s",
            tool_name, latency_ms, exc, exc_info=True,
        )
        return _ToolResult(
            tool_name, arguments,
            f"[Tool error — {tool_name}]: {type(exc).__name__}: {exc}",
            success=False,
        )
