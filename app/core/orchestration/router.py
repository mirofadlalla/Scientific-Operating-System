"""``route_and_stream``: the orchestration entry point.

Flow:  greeting fast-path → composite split → route → out-of-domain refusal →
       (RAG | expert agents → LLM synthesis).

Performance design:
  * Greetings: 1 LLM call (streaming response).
  * Scientific: 1 combined routing call + agent + synthesis = 3 calls total.
  * Out-of-domain: instant rejection from the combined routing output.
  * Composite: 1 extra detection call, then each sub-question is processed
    independently, with memory written once for the original question.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from app import monitoring
from app.config import settings
from app.core.agent_dispatch import build_agent_context, select_agents
from app.core.chem_text import pubchem_image_url
from app.core.deps import short_memory
from app.core.greeting import should_skip_orchestrator

from .agents_runner import run_agents, run_rag, wants_rag
from .composite import section_header, split_composite_question
from .language import is_arabic
from .memory import remember_turn
from .prompts import (
    GREETING_SYSTEM_PROMPT,
    build_synthesis_system_prompt,
    composite_header,
    out_of_domain_refusal,
)
from .routing import classify_route
from .streaming import StreamStats, record_usage, stream_completion

_GREETING_HISTORY = 5
_SYNTHESIS_HISTORY = 12
_REFUSAL_WORD_DELAY_S = 0.02
_RAG_WORD_DELAY_S = 0.008


async def _stream_words(text: str, delay: float) -> AsyncIterator[str]:
    """Stream an already-complete answer word by word (typing effect)."""
    for word in text.split(" "):
        yield word + " "
        await asyncio.sleep(delay)


async def _greeting_reply(text_input: str, session_id: str, store_memory: bool) -> AsyncIterator[str]:
    """Fast path: casual messages skip all routing."""
    messages = [{"role": "system", "content": GREETING_SYSTEM_PROMPT}]
    messages.extend(short_memory.get_history(session_id, limit=_GREETING_HISTORY))
    messages.append({"role": "user", "content": text_input})

    stats = StreamStats()
    async for token in stream_completion(settings.ROUTING_MODEL, messages, 0.7, stats):
        yield token

    monitoring.record_agent_call("APP_AGENT", "APP_HELP", 0, success=True)
    record_usage(settings.ROUTING_MODEL, messages, stats)
    if store_memory:
        remember_turn(session_id, text_input, stats.full_reply, intent="APP_HELP", agent="APP_AGENT")


async def _composite_reply(
    text_input: str,
    sub_questions: list[str],
    session_id: str,
    user_id: str,
    store_memory: bool,
    include_images: bool,
) -> AsyncIterator[str]:
    """Answer each sub-question in turn; memory is written once for the original."""
    header = composite_header(len(sub_questions), is_arabic(text_input))
    yield header
    combined_answer = header

    for idx, sub_q in enumerate(sub_questions, start=1):
        heading = section_header(idx, sub_q)
        yield heading
        combined_answer += heading

        part_answer = ""
        async for token in route_and_stream(
            sub_q, session_id, user_id,
            allow_split=False, store_memory=False, include_images=include_images,
        ):
            yield token
            part_answer += token

        combined_answer += part_answer + "\n"
        yield "\n"

    if store_memory:
        remember_turn(session_id, text_input, combined_answer, intent="COMPOSITE", agent="MULTI")


async def route_and_stream(
    text_input: str,
    session_id: str,
    user_id: str,
    allow_split: bool = True,
    store_memory: bool = True,
    include_images: bool = True,
) -> AsyncIterator[str]:
    """Route a query through the expert agents and yield text tokens of the answer."""

    # ── Fast path: greetings skip ALL routing ────────────────────────────────
    if should_skip_orchestrator(text_input):
        async for token in _greeting_reply(text_input, session_id, store_memory):
            yield token
        return

    # ── Composite question detection ─────────────────────────────────────────
    if allow_split:
        sub_questions = await split_composite_question(text_input)
        if len(sub_questions) > 1:
            async for token in _composite_reply(
                text_input, sub_questions, session_id, user_id, store_memory, include_images,
            ):
                yield token
            return

    # ── Combined routing + domain classification (single LLM call) ───────────
    route = await classify_route(text_input, session_id)
    intent, target_agent, entities = route.intent, route.target_agent, route.entities

    # ── Out-of-domain ────────────────────────────────────────────────────────
    if route.is_out_of_domain:
        monitoring.record_out_of_domain(route.out_of_domain_reason)
        refusal = out_of_domain_refusal(is_arabic(text_input))
        async for token in _stream_words(refusal, _REFUSAL_WORD_DELAY_S):
            yield token
        remember_turn(session_id, text_input, refusal, intent=intent, agent=target_agent, long_term=False)
        return

    # ── Structure visualisation image (text channel only) ────────────────────
    if include_images:
        chem_identifier = entities.get("smiles") or entities.get("compound")
        if chem_identifier:
            img_url = pubchem_image_url(chem_identifier)
            if img_url:
                yield f"\n\n![Molecular Structure of {chem_identifier}]({img_url})\n\n"

    # Agents are chosen (and receive ``intent``) BEFORE any RAG fallback rewrites it.
    agent_keys = select_agents(intent, target_agent, entities)
    routed_intent = intent

    # ── RAG agent ────────────────────────────────────────────────────────────
    rag_output = ""
    if wants_rag(intent, target_agent):
        rag = await run_rag(text_input, intent, target_agent)
        rag_output, intent, target_agent = rag.output, rag.intent, rag.target_agent

    chemical_output, medical_output = await run_agents(agent_keys, routed_intent, entities, text_input)

    # ── RAG direct stream (no re-synthesis needed) ───────────────────────────
    if rag_output:
        async for token in _stream_words(rag_output, _RAG_WORD_DELAY_S):
            yield token
        if store_memory:
            remember_turn(session_id, text_input, rag_output, intent=intent, agent="RAG_AGENT")
        return

    # ── Synthesis via LLM ────────────────────────────────────────────────────
    agent_raw_output = build_agent_context(chemical_output, medical_output, intent)
    messages = [{"role": "system", "content": build_synthesis_system_prompt(is_arabic(text_input))}]
    messages.extend(short_memory.get_history(session_id, limit=_SYNTHESIS_HISTORY))
    messages.append({
        "role": "user",
        "content": f'User Input Question: "{text_input}"\nRetrieved Lab Data: "{agent_raw_output}"',
    })

    stats = StreamStats()
    async for token in stream_completion(settings.REASONING_MODEL, messages, 0.3, stats):
        yield token

    monitoring.record_agent_call(
        agent=target_agent,
        intent=intent,
        latency_ms=round((time.time() - stats.iteration_started_at) * 1000, 2),
        success=True,
    )
    record_usage(settings.REASONING_MODEL, messages, stats)
    if store_memory:
        remember_turn(session_id, text_input, stats.full_reply, intent=intent, agent=target_agent)
