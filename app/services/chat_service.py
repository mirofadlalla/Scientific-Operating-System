"""
app.services.chat_service
~~~~~~~~~~~~~~~~~~~~~~~~~
Text chat: stream the orchestrator's reply token by token.
"""
import logging
from typing import AsyncIterator

from app.core.orchestration import route_and_stream

logger = logging.getLogger(__name__)


async def stream_reply(text_input: str, session_id: str, user_id: str) -> AsyncIterator[str]:
    """Yield reply tokens; a failure mid-stream is emitted as a trailing error line."""
    try:
        async for token in route_and_stream(text_input, session_id, user_id):
            yield token
    except Exception as exc:
        logger.error(f"[STREAM CRASH]: {exc}")
        yield f"\n[Stream Error]: {exc}"
