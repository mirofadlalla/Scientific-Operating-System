"""Audio pipeline exceptions."""

from __future__ import annotations

from .constants import MIN_AUDIO_BYTES


class AudioTooShortError(ValueError):
    """Raised when an audio payload is too small to transcribe."""

    def __init__(self, size: int, minimum: int = MIN_AUDIO_BYTES) -> None:
        self.size = size
        self.minimum = minimum
        super().__init__(f"audio_too_short:{size}<{minimum}")
