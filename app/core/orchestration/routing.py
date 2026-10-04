"""Single-call intent routing with a rule-based fallback classifier."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.core.deps import client, orchestrator, short_memory

from .prompts import COMBINED_ORCHESTRATOR_PROMPT, COMBINED_ORCHESTRATOR_PROMPT_MINIMAL

logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 10


@dataclass
class RouteDecision:
    """Where a query should go."""

    intent: str = "APP_HELP"
    target_agent: str = "APP_AGENT"
    entities: dict = field(default_factory=dict)
    out_of_domain_reason: str = ""

    @property
    def is_out_of_domain(self) -> bool:
        return self.intent == "OUT_OF_DOMAIN" or self.target_agent == "NONE"


def _fallback_decision(text_input: str) -> RouteDecision:
    """Rule-based classification used when the routing LLM call fails."""
    classification = orchestrator.classify_intent(text_input)
    intent_raw = (classification.get("intent") or "").lower()
    entities = classification.get("entities") or {}
    if intent_raw == "chemical":
        return RouteDecision("CHEMICAL_SIMILARITY", "CHEMICAL_AGENT", entities)
    if intent_raw == "medical":
        return RouteDecision("BIOMEDICAL_MECHANISM", "MEDICAL_AGENT", entities)
    return RouteDecision("APP_HELP", "APP_AGENT", entities)


async def classify_route(text_input: str, session_id: str) -> RouteDecision:
    """Route + domain-classify ``text_input`` in one LLM call.

    Retries once on ``json_validate_failed`` (Groq rejects the prompt when the
    model produces malformed JSON) before falling back to the rule-based classifier.
    """
    messages = [{"role": "system", "content": COMBINED_ORCHESTRATOR_PROMPT}]
    messages.extend(short_memory.get_history(session_id, limit=_HISTORY_LIMIT))
    messages.append({"role": "user", "content": text_input})

    last_exc: Exception | None = None
    for attempt in range(2):  # attempt 0 = normal, attempt 1 = retry
        try:
            response = await client.chat.completions.create(
                model=settings.ROUTING_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            output = json.loads(raw)
            return RouteDecision(
                intent=output.get("intent", "APP_HELP"),
                target_agent=output.get("target_agent", "APP_AGENT"),
                entities=output.get("entities") or {},
                out_of_domain_reason=output.get("out_of_domain_reason", ""),
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_str = str(exc)
            if "json_validate_failed" in err_str and attempt == 0:
                # Groq rejected the JSON output — retry once with an even simpler
                # system prompt that omits the entities field entirely.
                logger.warning(
                    "Orchestrator routing json_validate_failed on attempt 1 — retrying with minimal prompt"
                )
                messages[0] = {"role": "system", "content": COMBINED_ORCHESTRATOR_PROMPT_MINIMAL}
                continue
            break  # non-retryable error or second failure

    logger.warning(f"Orchestrator routing failed: {last_exc}. Using fallback classifier.")
    return _fallback_decision(text_input)
