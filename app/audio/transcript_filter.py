"""Filtering of Whisper output: unreliable segments and known hallucinations."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .constants import WHISPER_HALLUCINATION_BLOCKLIST, WHISPER_PROMPT

_NO_SPEECH_HARD_LIMIT = 0.6
_NO_SPEECH_SOFT_LIMIT = 0.3
_LOGPROB_FLOOR = -1.0


def _normalize_transcript_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" .!?,،؟")
    return cleaned.casefold()


_NORMALIZED_BLOCKLIST = frozenset(_normalize_transcript_line(i) for i in WHISPER_HALLUCINATION_BLOCKLIST)
_NORMALIZED_PROMPT = _normalize_transcript_line(WHISPER_PROMPT)
_PROMPT_ECHO_MIN_WORDS = 4


def _is_prompt_echo(normalized: str) -> bool:
    """True when Whisper just parroted its own bias prompt back.

    On near-silent / echo-only audio Whisper often returns (part of) the ``prompt``
    text. Left in, it reaches the LLM as if the user had said it and the agent
    answers a question nobody asked.
    """
    return (
        len(normalized.split()) >= _PROMPT_ECHO_MIN_WORDS
        and normalized in _NORMALIZED_PROMPT
    )


def _segment_field(segment: Any, name: str, default: float = 0.0) -> float:
    value = segment.get(name, default) if isinstance(segment, dict) else getattr(segment, name, default)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _segment_text(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("text") or "")
    return str(getattr(segment, "text", "") or "")


def _is_unreliable_segment(segment: Any) -> bool:
    no_speech_prob = _segment_field(segment, "no_speech_prob", 0.0)
    avg_logprob = _segment_field(segment, "avg_logprob", 0.0)
    if no_speech_prob > _NO_SPEECH_HARD_LIMIT:
        return True
    return avg_logprob < _LOGPROB_FLOOR and no_speech_prob > _NO_SPEECH_SOFT_LIMIT


def _result_segments(result: Any) -> list[Any]:
    if result is None or isinstance(result, str):
        return []
    if isinstance(result, dict):
        segments = result.get("segments") or []
    else:
        segments = getattr(result, "segments", None) or []
    return list(segments) if isinstance(segments, Iterable) else []


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return str(getattr(result, "text", "") or "")


def filter_whisper_hallucinations(result: Any) -> str:
    """Drop silent/noisy Whisper segments and known hallucination phrases."""
    segments = _result_segments(result)
    if segments:
        kept = [_segment_text(seg) for seg in segments if not _is_unreliable_segment(seg)]
        text = " ".join(part.strip() for part in kept if part and part.strip())
    else:
        text = _result_text(result)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    normalized = _normalize_transcript_line(text)
    if normalized in _NORMALIZED_BLOCKLIST or _is_prompt_echo(normalized):
        return ""
    return text
