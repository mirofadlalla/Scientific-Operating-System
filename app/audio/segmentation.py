"""Sentence splitting and batching for incremental TTS."""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\u060C\u061F])\s+|\n+")
# \u060C = Arabic comma ،   \u061F = Arabic question mark ؟
_MIN_FRAGMENT_CHARS = 6  # low because Arabic text is denser per character


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for incremental TTS.

    Handles English (. ! ?) and Arabic (. ، ؟ !) sentence endings, markdown
    bullets / numbered lists and newlines. Fragments shorter than 6 characters
    are merged into the previous sentence. Returns non-empty stripped strings.
    """
    sentences = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]

    merged: list[str] = []
    for s in sentences:
        if merged and len(s) < _MIN_FRAGMENT_CHARS:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def batch_sentences_for_tts(sentences: list[str], min_chars: int = 120) -> list[str]:
    """Group sentences so each batch has at least ``min_chars`` characters.

    Reduces TTS API calls dramatically, e.g. 15 short sentences → 3-4 batches.
    """
    batches: list[str] = []
    current = ""
    for s in sentences:
        current = f"{current} {s}" if current else s
        if len(current) >= min_chars:
            batches.append(current)
            current = ""
    if current:
        batches.append(current)
    return batches
