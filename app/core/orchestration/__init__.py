"""Central query routing and streaming (formerly ``app/core/orchestration.py``).

Modules:
    prompts        routing / composite / conversational prompts + canned replies
    routing        single-call intent classification (+ rule-based fallback)
    composite      multi-question detection and section formatting
    agents_runner  RAG step and parallel chemical/medical agent execution
    streaming      streaming completions with TTFT/TPS metrics
    memory         short-/long-term persistence of each exchange
    router         ``route_and_stream`` — wires the steps together

All previously importable names are re-exported here.
"""

from app.core.greeting import is_general_greeting, should_skip_orchestrator

from .composite import split_composite_question
from .prompts import COMBINED_ORCHESTRATOR_PROMPT, COMPOSITE_DETECTION_PROMPT
from .router import route_and_stream

__all__ = [
    "COMBINED_ORCHESTRATOR_PROMPT",
    "COMPOSITE_DETECTION_PROMPT",
    "is_general_greeting",
    "route_and_stream",
    "should_skip_orchestrator",
    "split_composite_question",
]
