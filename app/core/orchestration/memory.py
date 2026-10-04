"""Conversation-memory persistence (short-term always, long-term best-effort)."""

from __future__ import annotations

import logging

import app.core.state as state
from app.core.deps import short_memory

logger = logging.getLogger(__name__)


def remember_turn(
    session_id: str,
    user_text: str,
    reply: str,
    *,
    intent: str,
    agent: str,
    long_term: bool = True,
) -> None:
    """Record one user/assistant exchange.

    Long-term storage failures are logged and swallowed — they must never break a response.
    """
    short_memory.add_message(session_id, "user", user_text)
    short_memory.add_message(session_id, "assistant", reply)
    if not long_term or state.long_memory is None:
        return
    try:
        state.long_memory.add_entry(
            session_id,
            f"User: {user_text}\nAssistant: {reply}",
            metadata={"intent": intent, "agent": agent},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"long_memory.add_entry failed: {exc}")
