"""Constants shared across the audio pipeline."""

from __future__ import annotations

import re

MIN_AUDIO_BYTES = 1024
"""Payloads smaller than this are rejected before hitting the STT API."""

WHISPER_HALLUCINATION_BLOCKLIST = frozenset({
    "thanks for watching",
    "thank you for watching",
    "thanks for watching.",
    "thank you for watching.",
    "ترجمة نانسي قنقر",
    "اشتركوا في القناة",
})

WHISPER_PROMPT = (
    "محادثة علمية باللغة العربية والإنجليزية: أدوية، مركبات كيميائية، أحياء، "
    "جينات، مسارات بيولوجية، وبحث علمي. "
    "Scientific queries in Arabic (العربية) and English: drug discovery, chemistry, "
    "ADMET, biology, medicine, SMILES, molecular research."
)

ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")


def contains_arabic(text: str) -> bool:
    """Return ``True`` if ``text`` contains any Arabic-script character."""
    return bool(ARABIC_CHARS.search(text))
