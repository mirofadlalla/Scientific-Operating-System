"""LLM helpers: client construction, tool selection, answer synthesis."""

from __future__ import annotations

import json
import logging
import time

from openai import AsyncOpenAI

from app.config import groq_llm_key, settings

from .models import _NoToolCallError, _ToolResult
from .prompts import CHEMICAL_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _build_llm_client() -> AsyncOpenAI:
    """Async OpenAI-compatible client for Groq (``GROQ_LLM_API_KEY`` → ``GROQ_API_KEY``)."""
    return AsyncOpenAI(base_url=settings.GROQ_BASE_URL, api_key=groq_llm_key())


async def _llm_tool_selection(
    llm: AsyncOpenAI,
    user_query: str,
    openai_tools: list[dict],
) -> list[dict]:
    """Ask the LLM which MCP tools to call.

    Returns a list of ``{"name": str, "arguments": dict}``.
    Raises:
        _NoToolCallError: the LLM answered in plain text.
        ValueError: the LLM produced malformed tool-call JSON.
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

    msg = response.choices[0].message
    logger.info(
        "[ChemicalAgent] LLM tool selection done latency=%.0fms tool_calls=%d",
        round((time.monotonic() - t0) * 1000, 2), len(msg.tool_calls or []),
    )

    if not msg.tool_calls:
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
    """Ask the LLM to synthesise raw MCP tool outputs into a clean response."""
    results_block = "\n\n".join(f"### {r.tool_name} result\n{r.output}" for r in tool_results)

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

    logger.info(
        "[ChemicalAgent] LLM synthesis done latency=%.0fms",
        round((time.monotonic() - t0) * 1000, 2),
    )
    return response.choices[0].message.content or "[No synthesis generated]"
