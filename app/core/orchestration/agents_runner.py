"""Execution of the expert agents selected for a routed query."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.core.deps import chemical_agent, medical_agent, rag_agent

logger = logging.getLogger(__name__)

_NO_INFO_PHRASES = (
    "does not contain information", "not in the documentation",
    "not available in the documentation", "not available in the provided",
    "not mentioned in the documentation", "not covered in the documentation",
    "knowledge base is currently", "documentation does not",
    "cannot find", "no information",
)


@dataclass
class RagOutcome:
    """Result of the RAG step; ``output`` is empty when it fell back to the app agent."""

    output: str
    intent: str
    target_agent: str


def wants_rag(intent: str, target_agent: str) -> bool:
    return intent == "APP_SUPPORT_RAG" or target_agent == "RAG_AGENT"


async def run_rag(text_input: str, intent: str, target_agent: str) -> RagOutcome:
    """Query the knowledge base; fall back to APP_AGENT when it has nothing relevant."""
    try:
        output = await rag_agent.run(text_input)
        if any(p in output.lower() for p in _NO_INFO_PHRASES):
            logger.info("[RAG] No relevant docs — falling back to APP_AGENT.")
            return RagOutcome("", "APP_HELP", "APP_AGENT")
        return RagOutcome(output, intent, target_agent)
    except Exception as exc:  # noqa: BLE001
        return RagOutcome(f"[RAG Agent Error]: {exc}", intent, target_agent)


async def run_agents(
    keys: list[str], intent: str, entities: dict, text_input: str,
) -> tuple[str, str]:
    """Run the agents named in ``keys`` in parallel → ``(chemical_output, medical_output)``.

    ``intent`` must be the intent the keys were selected for (i.e. *before* any RAG
    fallback rewrote it) — agents receive it verbatim. Agent failures are reported
    as ``[<Agent> Agent Error]: …`` strings, never raised.
    """
    tasks, mapping = [], []
    for key in keys:
        if key == "CHEMICAL":
            tasks.append(chemical_agent.run(intent, entities, user_query=text_input))
            mapping.append("CHEMICAL")
        elif key == "MEDICAL":
            tasks.append(medical_agent.run(intent, entities, user_query=text_input))
            mapping.append("MEDICAL")

    outputs = {"CHEMICAL": "", "MEDICAL": ""}
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, res in zip(mapping, results):
            label = "Chemical" if name == "CHEMICAL" else "Medical"
            outputs[name] = f"[{label} Agent Error]: {res}" if isinstance(res, Exception) else str(res)
    return outputs["CHEMICAL"], outputs["MEDICAL"]
