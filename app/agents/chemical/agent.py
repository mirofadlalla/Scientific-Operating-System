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
  • One MCP session per agent.run() call — avoids global state while
    minimising reconnects within a single request.
  • LLM is the sole routing / argument-generation mechanism — no keyword rules.
  • Errors are classified (connection / protocol / LLM / validation) and
    returned as clean user-facing messages; full tracebacks go to the logger.
  • Backward-compatible: run(intent, entities) still works because the
    orchestrator passes those; internally we reconstruct the natural-language
    query from them when called with the old signature.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any

import httpx
from mcp import Client
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CHEMICAL_AGENT_SYSTEM_PROMPT = """\
You are the Chemical Intelligence Agent of AILIXIR, an AI Scientific Operating System \
specializing in Drug Discovery and Cheminformatics.

You have access to a set of MCP scientific tools. You MUST use these tools whenever \
scientific computation or retrieval is required. NEVER fabricate or guess scientific results.

## Tool usage rules
- Use `calculate_admet` when the user asks about ADMET properties, toxicity, absorption, \
  distribution, metabolism, or excretion of a compound.
- Use `screen_drugs` when the user asks about drug repurposing, virtual screening, or \
  finding drug candidates for a disease.
- Use `chemical_similarity_search` when the user asks for structurally similar compounds, \
  molecular similarity, or chemical analogues.
- You MAY call multiple tools for a single request when the user explicitly asks for \
  multiple analyses (e.g. ADMET + similarity).
- If a required argument is missing (e.g. no SMILES, no disease name), ask the user \
  for it — do NOT invent or guess scientific inputs.
- Never bypass the MCP tools by making up scientific predictions.
- Always clearly label results as computational predictions, not experimental measurements.
- When synthesising results, preserve their scientific meaning and quantitative values exactly.
- If a tool returns an error, report it clearly to the user and explain what went wrong.

Respond in the same language the user used (Arabic or English).
"""

# ---------------------------------------------------------------------------
# Tool call result dataclass (lightweight)
# ---------------------------------------------------------------------------

class _ToolResult:
    """Holds the outcome of a single MCP tool execution."""
    def __init__(self, tool_name: str, arguments: dict, output: str, success: bool):
        self.tool_name = tool_name
        self.arguments = arguments
        self.output = output
        self.success = success


# ---------------------------------------------------------------------------
# MCP session helpers
# ---------------------------------------------------------------------------

async def _discover_tools(client: Client) -> list[dict]:
    """
    Fetch tool schemas from the MCP server and convert to OpenAI function
    definitions so the LLM can select and call them.
    """
    tools_response = await client.list_tools()
    openai_tools = []

    for tool in tools_response.tools:
        # tool.inputSchema is a JSON-schema dict
        schema = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
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


async def _execute_mcp_tool(
    client: Client,
    tool_name: str,
    arguments: dict,
) -> _ToolResult:
    """
    Call a single MCP tool and return a _ToolResult.
    Classifies errors so the caller can decide on retry vs. user message.
    """
    t0 = time.monotonic()
    # Sanitise for logging (do not log SMILES of clinical compounds at DEBUG in prod)
    safe_args = {k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v)
                 for k, v in arguments.items()}
    logger.info(
        "[ChemicalAgent] Calling MCP tool=%s args=%s", tool_name, safe_args
    )

    try:
        result = await client.call_tool(tool_name, arguments)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info(
            "[ChemicalAgent] Tool=%s SUCCESS latency=%.0fms", tool_name, latency_ms
        )

        # Extract text from result
        if result.structured_content:
            output = str(result.structured_content)
        elif result.content:
            texts = [item.text for item in result.content if hasattr(item, "text")]
            output = "\n".join(texts) if texts else "[Tool returned empty content]"
        else:
            output = "[Tool returned no content]"

        return _ToolResult(tool_name, arguments, output, success=True)

    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.error(
            "[ChemicalAgent] Tool=%s FAILED latency=%.0fms error=%s",
            tool_name, latency_ms, exc,
            exc_info=True,
        )
        return _ToolResult(
            tool_name, arguments,
            f"[Tool error — {tool_name}]: {type(exc).__name__}: {exc}",
            success=False,
        )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _build_llm_client() -> AsyncOpenAI:
    """
    Build the async OpenAI-compatible client pointing at Groq.
    Reuses the project's existing LLM configuration.
    """
    return AsyncOpenAI(
        base_url=settings.GROQ_BASE_URL,
        api_key=settings.GROQ_API_KEY,
    )


async def _llm_tool_selection(
    llm: AsyncOpenAI,
    user_query: str,
    openai_tools: list[dict],
) -> list[dict]:
    """
    Ask the LLM which MCP tools to call and with what arguments.
    Returns a list of {"name": str, "arguments": dict} dicts.
    Raises ValueError for LLM / parse failures.
    """
    t0 = time.monotonic()
    logger.info("[ChemicalAgent] LLM tool selection for query=%r", user_query[:100])

    response = await llm.chat.completions.create(
        model=settings.ROUTING_MODEL,
        messages=[
            {"role": "system", "content": CHEMICAL_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        tools=openai_tools,
        tool_choice="auto",
        temperature=0.0,
        timeout=30.0,
    )

    llm_latency_ms = round((time.monotonic() - t0) * 1000, 2)
    msg = response.choices[0].message
    logger.info(
        "[ChemicalAgent] LLM tool selection done latency=%.0fms tool_calls=%d",
        llm_latency_ms, len(msg.tool_calls or []),
    )

    if not msg.tool_calls:
        # LLM chose not to call a tool — surface its text response
        raise _NoToolCallError(msg.content or "")

    calls = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM produced malformed JSON for tool {tc.function.name!r}: {exc}"
            ) from exc
        calls.append({"name": tc.function.name, "arguments": args})

    return calls


async def _llm_synthesize(
    llm: AsyncOpenAI,
    user_query: str,
    tool_results: list[_ToolResult],
) -> str:
    """
    Ask the LLM to synthesise the raw MCP tool outputs into a clean response.
    """
    results_block = "\n\n".join(
        f"### {r.tool_name} result\n{r.output}"
        for r in tool_results
    )

    t0 = time.monotonic()
    response = await llm.chat.completions.create(
        model=settings.REASONING_MODEL,
        messages=[
            {"role": "system", "content": CHEMICAL_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_query}\n\n"
                    f"Scientific tool outputs:\n{results_block}\n\n"
                    "Please synthesise a clear, professional scientific answer "
                    "based strictly on the tool outputs above. "
                    "Label all results as computational predictions. "
                    "Do not add data that isn't in the tool outputs."
                ),
            },
        ],
        temperature=0.3,
        timeout=60.0,
    )

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info("[ChemicalAgent] LLM synthesis done latency=%.0fms", latency_ms)
    return response.choices[0].message.content or "[No synthesis generated]"


# ---------------------------------------------------------------------------
# Sentinel for "LLM chose not to use tools"
# ---------------------------------------------------------------------------

class _NoToolCallError(Exception):
    """Raised when the LLM responds with text instead of a tool call."""
    def __init__(self, llm_text: str):
        super().__init__(llm_text)
        self.llm_text = llm_text


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------

class ChemicalAgent:
    """
    LLM-driven Chemical Agent.

    Public interface (backward-compatible with existing Orchestrator):

        await agent.run(intent, entities)          # legacy call-site
        await agent.run(user_query=..., context={})  # new call-site
    """

    def __init__(self):
        self.mcp_url: str = settings.__dict__.get(
            "CHEMICAL_MCP_URL",
            "http://localhost:8001/mcp",
        )
        # Attempt to read a dedicated MCP URL config if it was added later
        if hasattr(settings, "CHEMICAL_MCP_URL"):
            self.mcp_url = settings.CHEMICAL_MCP_URL  # type: ignore[attr-defined]

        self._llm = _build_llm_client()
        logger.info("[ChemicalAgent] Initialized — MCP endpoint: %s", self.mcp_url)

    # ------------------------------------------------------------------
    # Primary entry-point
    # ------------------------------------------------------------------

    async def run(
        self,
        intent: str | None = None,
        entities: dict | None = None,
        *,
        user_query: str | None = None,
        context: dict | None = None,
    ) -> str:
        """
        Execute an LLM-driven MCP tool-calling workflow.

        Accepts BOTH the legacy (intent, entities) signature used by the
        Orchestrator AND the new (user_query, context) signature.
        """
        total_t0 = time.monotonic()

        # ── Reconstruct natural-language query from legacy args if needed ──
        query = self._resolve_query(intent, entities, user_query)
        logger.info("[ChemicalAgent] run() query=%r", query[:120])

        try:
            result = await self._run_llm_mcp_pipeline(query)
        except Exception as exc:
            logger.error("[ChemicalAgent] Unexpected error: %s", exc, exc_info=True)
            result = (
                "I encountered an unexpected internal error while processing "
                "your chemical query. Please try again or contact support."
            )

        total_ms = round((time.monotonic() - total_t0) * 1000, 2)
        logger.info(
            "[ChemicalAgent] run() complete total_latency=%.0fms", total_ms
        )
        return result

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _run_llm_mcp_pipeline(self, user_query: str) -> str:
        """
        Full pipeline:
          1. Open MCP session + discover tools
          2. LLM selects tool(s) + generates arguments
          3. Execute tool(s) in parallel
          4. LLM synthesises final answer
        """
        # ── 1. Connect and discover tools ──────────────────────────────
        try:
            async with Client(self.mcp_url) as mcp_client:
                logger.info("[ChemicalAgent] MCP connected to %s", self.mcp_url)

                try:
                    openai_tools = await asyncio.wait_for(
                        _discover_tools(mcp_client),
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    return (
                        "The chemical analysis service timed out during tool discovery. "
                        "Please try again in a moment."
                    )

                if not openai_tools:
                    return (
                        "No chemical analysis tools are currently available. "
                        "Please contact support."
                    )

                # ── 2. LLM tool selection ───────────────────────────────
                try:
                    tool_calls = await asyncio.wait_for(
                        _llm_tool_selection(self._llm, user_query, openai_tools),
                        timeout=35.0,
                    )
                except _NoToolCallError as no_tool_exc:
                    # LLM responded in plain text (e.g. asked for clarification)
                    logger.info(
                        "[ChemicalAgent] LLM responded without tool call: %r",
                        no_tool_exc.llm_text[:200],
                    )
                    return no_tool_exc.llm_text
                except asyncio.TimeoutError:
                    return (
                        "The AI reasoning engine timed out while analysing your request. "
                        "Please try again."
                    )
                except ValueError as exc:
                    logger.error("[ChemicalAgent] LLM tool selection error: %s", exc)
                    return (
                        "I had trouble understanding how to process your request. "
                        "Could you rephrase it?"
                    )
                except Exception as exc:
                    logger.error(
                        "[ChemicalAgent] LLM failure: %s", exc, exc_info=True
                    )
                    return (
                        "The AI reasoning engine encountered an error. "
                        "Please try again or check your query."
                    )

                logger.info(
                    "[ChemicalAgent] LLM selected %d tool call(s): %s",
                    len(tool_calls),
                    [tc["name"] for tc in tool_calls],
                )

                # ── 3. Validate tool names ─────────────────────────────
                known_tool_names = {t["function"]["name"] for t in openai_tools}
                valid_calls, invalid_calls = [], []
                for tc in tool_calls:
                    (valid_calls if tc["name"] in known_tool_names else invalid_calls).append(tc)

                if invalid_calls:
                    logger.warning(
                        "[ChemicalAgent] LLM requested unknown tools: %s",
                        [tc["name"] for tc in invalid_calls],
                    )

                if not valid_calls:
                    return (
                        "I couldn't match your request to any available chemical analysis tool. "
                        "Please rephrase your question or specify the analysis you need."
                    )

                # ── 4. Execute tools in parallel ───────────────────────
                async def _timed_call(tc: dict) -> _ToolResult:
                    try:
                        return await asyncio.wait_for(
                            _execute_mcp_tool(mcp_client, tc["name"], tc["arguments"]),
                            timeout=90.0,
                        )
                    except asyncio.TimeoutError:
                        return _ToolResult(
                            tc["name"], tc["arguments"],
                            f"[Timeout] Tool '{tc['name']}' exceeded 90s execution limit.",
                            success=False,
                        )

                tool_results: list[_ToolResult] = await asyncio.gather(
                    *[_timed_call(tc) for tc in valid_calls]
                )

                all_failed = all(not r.success for r in tool_results)
                if all_failed:
                    error_msgs = "; ".join(r.output for r in tool_results)
                    logger.error(
                        "[ChemicalAgent] All tool calls failed: %s", error_msgs
                    )
                    return (
                        "All requested chemical analyses encountered errors. "
                        "The underlying services may be temporarily unavailable. "
                        "Please try again shortly."
                    )

                # ── 5. LLM synthesis ───────────────────────────────────
                if len(tool_results) == 1 and tool_results[0].success:
                    # Single successful result: still synthesise for polish
                    pass  # fall through to synthesis

                try:
                    synthesis = await asyncio.wait_for(
                        _llm_synthesize(self._llm, user_query, tool_results),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    # Fall back to raw tool output
                    logger.warning("[ChemicalAgent] LLM synthesis timed out — returning raw output")
                    synthesis = "\n\n".join(
                        f"**{r.tool_name}**:\n{r.output}" for r in tool_results
                    )
                except Exception as exc:
                    logger.error(
                        "[ChemicalAgent] LLM synthesis error: %s", exc, exc_info=True
                    )
                    synthesis = "\n\n".join(
                        f"**{r.tool_name}**:\n{r.output}" for r in tool_results
                    )

                return synthesis

        # ── MCP connection failures ─────────────────────────────────────
        except ExceptionGroup as exc_group:
            logger.error(
                "[ChemicalAgent] MCP ExceptionGroup: %s",
                exc_group,
                exc_info=True,
            )
            for i, inner in enumerate(exc_group.exceptions, 1):
                logger.error("[ChemicalAgent] Inner exception %d: %s", i, inner)
            return (
                "The chemical analysis service is currently unreachable "
                "(connection error). Please try again shortly."
            )

        except (ConnectionError, OSError, httpx.ConnectError) as exc:
            logger.error("[ChemicalAgent] MCP connection error: %s", exc)
            return (
                "Unable to connect to the chemical analysis service. "
                "Please verify the service is running and try again."
            )

        except httpx.TimeoutException as exc:
            logger.error("[ChemicalAgent] MCP HTTP timeout: %s", exc)
            return (
                "The chemical analysis service timed out. "
                "Please try again in a moment."
            )

        except Exception as exc:
            logger.error(
                "[ChemicalAgent] Unhandled error in MCP pipeline: %s",
                exc,
                exc_info=True,
            )
            return (
                "An unexpected error occurred in the chemical analysis pipeline. "
                "Please try again or contact support."
            )

    # ------------------------------------------------------------------
    # Legacy query reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_query(
        intent: str | None,
        entities: dict | None,
        user_query: str | None,
    ) -> str:
        """
        Reconstruct a natural-language query from the legacy (intent, entities)
        call-site used by the Orchestrator, or pass through user_query directly.
        """
        if user_query:
            return user_query

        entities = entities or {}
        intent = (intent or "").lower()
        smiles = entities.get("smiles") or entities.get("compound") or ""
        disease = entities.get("disease") or ""

        # Build a descriptive query the LLM can reason about
        if "admet" in intent or "toxicity" in intent or "absorption" in intent:
            if smiles:
                return f"Calculate the ADMET properties for the compound with SMILES: {smiles}"
            return "Calculate ADMET properties"

        if any(k in intent for k in ("repurpos", "screen", "drug")):
            if disease:
                return f"Find drug repurposing candidates for {disease}"
            return "Screen drugs for repurposing"

        if any(k in intent for k in ("similar", "rag", "chemical_similarity")):
            explain = "explain" in intent or "detailed" in intent
            if smiles:
                suffix = " and explain their structural relevance" if explain else ""
                return f"Find compounds structurally similar to SMILES {smiles}{suffix}"
            return "Find similar chemical compounds"

        # Generic fallback — include all available context
        parts = []
        if smiles:
            parts.append(f"compound SMILES {smiles}")
        if disease:
            parts.append(f"disease: {disease}")
        context_str = "; ".join(parts)
        if intent and context_str:
            return f"{intent} — {context_str}"
        if intent:
            return intent
        if context_str:
            return f"Analyse {context_str}"
        return "Perform a chemical analysis"


# ---------------------------------------------------------------------------
# Standalone integration test
# ---------------------------------------------------------------------------

async def _main():
    """Quick end-to-end smoke test — requires the MCP server to be running."""
    import os

    logging.basicConfig(level=logging.INFO)
    agent = ChemicalAgent()

    tests = [
        # (label, kwargs)
        ("ADMET (legacy)", dict(intent="admet", entities={"smiles": "CCO"})),
        ("Drug repurposing (legacy)", dict(intent="repurposing", entities={"disease": "Alzheimer's disease"})),
        ("Similarity (legacy)", dict(intent="similarity", entities={"smiles": "CCO"})),
        ("ADMET (new API)", dict(user_query="Calculate the ADMET properties of ethanol (CCO)")),
        ("Multi-tool", dict(user_query="Analyse CCO for ADMET and find structurally similar compounds")),
        ("Missing SMILES", dict(user_query="Calculate ADMET properties")),
        ("Missing disease", dict(user_query="Screen drugs for repurposing")),
    ]

    for label, kwargs in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print("="*60)
        result = await agent.run(**kwargs)
        print(result[:500])

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(_main())