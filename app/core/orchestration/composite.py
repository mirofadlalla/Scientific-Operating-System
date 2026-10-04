"""Detection and splitting of multi-part user questions."""

from __future__ import annotations

import json
import logging

from app.config import settings
from app.core.deps import client

from .prompts import COMPOSITE_DETECTION_PROMPT

logger = logging.getLogger(__name__)

SECTION_DIVIDER = "\n\n" + "─" * 52 + "\n"
_NUMBER_GLYPHS = ["❶", "❷", "❸", "❹", "❺", "❻", "❼", "❽", "❾", "❿"]


def section_header(index: int, question: str) -> str:
    """Numbered header for the ``index``-th (1-based) sub-question."""
    num = _NUMBER_GLYPHS[index - 1] if index <= len(_NUMBER_GLYPHS) else f"{index}."
    return f"{SECTION_DIVIDER}**{num} {question}**\n\n"


async def split_composite_question(text: str) -> list[str]:
    """Split ``text`` into sub-questions if it holds several distinct ones.

    Returns ``[text]`` unchanged for single questions or on any detection failure.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.ROUTING_MODEL,
            messages=[
                {"role": "system", "content": COMPOSITE_DETECTION_PROMPT},
                {"role": "user",   "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        parts = [q.strip() for q in data.get("sub_questions", []) if q.strip()]
        if data.get("is_composite") and len(parts) > 1:
            logger.info(f"[Composite] Split into {len(parts)} sub-questions: {parts}")
            return parts
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[Composite] Detection failed: {exc} — treating as single question")
    return [text]
