"""Structured diagnostics for the voice pipeline."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import settings

logger = logging.getLogger("app.audio.voice")


def voice_log(event: str, **kwargs: Any) -> None:
    """Emit a structured diagnostic entry — only when VOICE_DEBUG=true."""
    if not settings.VOICE_DEBUG:
        return
    entry = {"event": event, "ts": round(time.time(), 3), **kwargs}
    logger.info("[VOICE_DEBUG] %s", json.dumps(entry, ensure_ascii=False))
