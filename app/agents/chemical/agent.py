"""
app.agents.chemical.agent
~~~~~~~~~~~~~~~~~~~~~~~~~
LLM-driven ChemicalAgent for the AILIXIR Scientific Operating System.

Architecture:
    User request (natural language)
        → LLM (via Groq / OpenAI-compatible API)
        → MCP tool discovery (dynamic, from server)
        → LLM selects tool(s) + generates arguments
        → MCP Client executes tool call(s)
        → Chemical MCP Server → underlying AI services
        → LLM synthesizes final answer
        → User

Design decisions:
  * One MCP session per ``agent.run()`` call — avoids global state while
    minimising reconnects within a single request.
  * The LLM is the sole routing / argument-generation mechanism — no keyword rules.
  * Errors are classified (connection / protocol / LLM / validation) and
    returned as clean user-facing messages; full tracebacks go to the logger.
  * Backward-compatible: ``run(intent, entities)`` still works because the
    orchestrator passes those; the natural-language query is reconstructed.

Module layout (this file keeps the orchestration only):
    prompts.py     system prompt          models.py      _ToolResult / _NoToolCallError
    mcp_client.py  discovery + execution  llm.py         selection + synthesis
    messages.py    user-facing strings    query.py       legacy query reconstruction
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from mcp import Client

from app.config import settings

from . import messages
from .llm import _build_llm_client, _llm_synthesize, _llm_tool_selection
from .mcp_client import _discover_tools, _execute_mcp_tool
from .models import _NoToolCallError, _ToolResult
from .prompts import CHEMICAL_AGENT_SYSTEM_PROMPT
from .query import resolve_query

logger = logging.getLogger(__name__)

__all__ = [
    "CHEMICAL_AGENT_SYSTEM_PROMPT",
    "ChemicalAgent",
    "_NoToolCallError",
    "_ToolResult",
    "_build_llm_client",
    "_discover_tools",
    "_execute_mcp_tool",
    "_llm_synthesize",
    "_llm_tool_selection",
]

_DISCOVERY_TIMEOUT_S = 15.0
_SELECTION_TIMEOUT_S = 35.0
_TOOL_TIMEOUT_S = 90.0
_SYNTHESIS_TIMEOUT_S = 60.0


class _PipelineStop(Exception):
    """Internal early exit: ``message`` is returned to the user as the final answer."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ChemicalAgent:
    """LLM-driven Chemical Agent.

    Public interface (backward-compatible with the existing Orchestrator)::

        await agent.run(intent, entities)            # legacy call-site
        await agent.run(user_query=..., context={})  # new call-site
    """

    def __init__(self) -> None:
        self.mcp_url: str = getattr(settings, "CHEMICAL_MCP_URL", "http://localhost:8001/mcp")
        self._llm = _build_llm_client()
        logger.info("[ChemicalAgent] Initialized — MCP endpoint: %s", self.mcp_url)

    # ── Primary entry-point ──────────────────────────────────────────────────
    async def run(
        self,
        intent: str | None = None,
        entities: dict | None = None,
        *,
        user_query: str | None = None,
        context: dict | None = None,
    ) -> str:
        """Execute an LLM-driven MCP tool-calling workflow.

        Accepts BOTH the legacy ``(intent, entities)`` signature used by the
        Orchestrator AND the new ``(user_query, context)`` signature.
        """
        total_t0 = time.monotonic()
        query = self._resolve_query(intent, entities, user_query)
        logger.info("[ChemicalAgent] run() query=%r", query[:120])

        try:
            result = await self._run_llm_mcp_pipeline(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("[ChemicalAgent] Unexpected error: %s", exc, exc_info=True)
            result = messages.INTERNAL_ERROR

        logger.info(
            "[ChemicalAgent] run() complete total_latency=%.0fms",
            round((time.monotonic() - total_t0) * 1000, 2),
        )
        return result

    # ── Core pipeline ────────────────────────────────────────────────────────
    async def _run_llm_mcp_pipeline(self, user_query: str) -> str:
        """Open an MCP session and run the discover → select → execute → synthesise flow."""
        try:
            async with Client(self.mcp_url) as mcp_client:
                logger.info("[ChemicalAgent] MCP connected to %s", self.mcp_url)
                try:
                    return await self._run_session(mcp_client, user_query)
                except _PipelineStop as stop:
                    return stop.message

        except ExceptionGroup as exc_group:
            logger.error("[ChemicalAgent] MCP ExceptionGroup: %s", exc_group, exc_info=True)
            for i, inner in enumerate(exc_group.exceptions, 1):
                logger.error("[ChemicalAgent] Inner exception %d: %s", i, inner)
            return messages.MCP_UNREACHABLE

        except (ConnectionError, OSError, httpx.ConnectError) as exc:
            logger.error("[ChemicalAgent] MCP connection error: %s", exc)
            return messages.MCP_CONNECT_ERROR

        except httpx.TimeoutException as exc:
            logger.error("[ChemicalAgent] MCP HTTP timeout: %s", exc)
            return messages.MCP_HTTP_TIMEOUT

        except Exception as exc:  # noqa: BLE001
            logger.error("[ChemicalAgent] Unhandled error in MCP pipeline: %s", exc, exc_info=True)
            return messages.PIPELINE_ERROR

    async def _run_session(self, mcp_client: Client, user_query: str) -> str:
        """Pipeline steps inside an open MCP session; raises ``_PipelineStop`` to exit early."""
        openai_tools = await self._discover(mcp_client)
        tool_calls = await self._select_tools(user_query, openai_tools)
        valid_calls = self._validate_calls(tool_calls, openai_tools)
        tool_results = await self._execute_tools(mcp_client, valid_calls)
        return await self._synthesize(user_query, tool_results)

    # ── Pipeline steps ───────────────────────────────────────────────────────
    @staticmethod
    async def _discover(mcp_client: Client) -> list[dict]:
        try:
            tools = await asyncio.wait_for(_discover_tools(mcp_client), timeout=_DISCOVERY_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise _PipelineStop(messages.DISCOVERY_TIMEOUT) from None
        if not tools:
            raise _PipelineStop(messages.NO_TOOLS)
        return tools

    async def _select_tools(self, user_query: str, openai_tools: list[dict]) -> list[dict]:
        try:
            tool_calls = await asyncio.wait_for(
                _llm_tool_selection(self._llm, user_query, openai_tools),
                timeout=_SELECTION_TIMEOUT_S,
            )
        except _NoToolCallError as no_tool:
            # LLM answered in plain text (e.g. asked for clarification)
            logger.info("[ChemicalAgent] LLM responded without tool call: %r", no_tool.llm_text[:200])
            raise _PipelineStop(no_tool.llm_text) from None
        except asyncio.TimeoutError:
            raise _PipelineStop(messages.SELECTION_TIMEOUT) from None
        except ValueError as exc:
            logger.error("[ChemicalAgent] LLM tool selection error: %s", exc)
            raise _PipelineStop(messages.SELECTION_PARSE_ERROR) from None
        except Exception as exc:  # noqa: BLE001
            logger.error("[ChemicalAgent] LLM failure: %s", exc, exc_info=True)
            raise _PipelineStop(messages.SELECTION_FAILURE) from None

        logger.info(
            "[ChemicalAgent] LLM selected %d tool call(s): %s",
            len(tool_calls), [tc["name"] for tc in tool_calls],
        )
        return tool_calls

    @staticmethod
    def _validate_calls(tool_calls: list[dict], openai_tools: list[dict]) -> list[dict]:
        """Drop calls to tools the server did not advertise."""
        known = {t["function"]["name"] for t in openai_tools}
        valid = [tc for tc in tool_calls if tc["name"] in known]
        invalid = [tc for tc in tool_calls if tc["name"] not in known]
        if invalid:
            logger.warning(
                "[ChemicalAgent] LLM requested unknown tools: %s", [tc["name"] for tc in invalid]
            )
        if not valid:
            raise _PipelineStop(messages.NO_MATCHING_TOOL)
        return valid

    @staticmethod
    async def _execute_tools(mcp_client: Client, calls: list[dict]) -> list[_ToolResult]:
        """Run tool calls in parallel, each under its own timeout."""

        async def _timed_call(tc: dict) -> _ToolResult:
            try:
                return await asyncio.wait_for(
                    _execute_mcp_tool(mcp_client, tc["name"], tc["arguments"]),
                    timeout=_TOOL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return _ToolResult(
                    tc["name"], tc["arguments"],
                    f"[Timeout] Tool '{tc['name']}' exceeded {_TOOL_TIMEOUT_S:.0f}s execution limit.",
                    success=False,
                )

        results: list[_ToolResult] = await asyncio.gather(*[_timed_call(tc) for tc in calls])
        if all(not r.success for r in results):
            logger.error("[ChemicalAgent] All tool calls failed: %s", "; ".join(r.output for r in results))
            raise _PipelineStop(messages.ALL_TOOLS_FAILED)
        return results

    async def _synthesize(self, user_query: str, tool_results: list[_ToolResult]) -> str:
        """LLM synthesis, degrading to the raw tool output on timeout/error."""
        try:
            return await asyncio.wait_for(
                _llm_synthesize(self._llm, user_query, tool_results),
                timeout=_SYNTHESIS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("[ChemicalAgent] LLM synthesis timed out — returning raw output")
        except Exception as exc:  # noqa: BLE001
            logger.error("[ChemicalAgent] LLM synthesis error: %s", exc, exc_info=True)
        return "\n\n".join(f"**{r.tool_name}**:\n{r.output}" for r in tool_results)

    # ── Legacy query reconstruction (kept as a staticmethod for compatibility) ─
    _resolve_query = staticmethod(resolve_query)
