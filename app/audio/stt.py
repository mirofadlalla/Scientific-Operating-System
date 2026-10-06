"""Speech-to-Text (Groq Whisper) behaviour for :class:AudioProcessor."""

from __future__ import annotations

import io
import logging
import time

from openai import AsyncOpenAI

from app.config import settings

from .constants import MIN_AUDIO_BYTES, WHISPER_PROMPT
from .diagnostics import voice_log
from .errors import AudioTooShortError
from .formats import align_container_header, detect_format
from .transcript_filter import filter_whisper_hallucinations

logger = logging.getLogger(__name__)


class SpeechToTextMixin:
    """Requires self.groq_client (an AsyncOpenAI pointed at Groq)."""

    groq_client: AsyncOpenAI

    @staticmethod
    def _detect_format(audio_bytes: bytes) -> str:
        """Detect the audio container from magic bytes (see :func:.formats.detect_format)."""
        return detect_format(audio_bytes)

    async def transcribe_audio(self, audio_file: bytes, audio_format: str = "webm") -> str:
        """Transcribe audio with Groq whisper-large-v3-turbo.

        Works with any format Whisper supports (webm, mp4, wav, mp3, m4a…).

        Args:
            audio_file: raw audio bytes.
            audio_format: container hint (auto-detected from magic bytes when possible).

        Raises:
            ValueError: empty buffer, or transcription failure.
            AudioTooShortError: payload below MIN_AUDIO_BYTES.
        """
        if not audio_file:
            raise ValueError("Empty audio buffer — nothing to transcribe")
        if len(audio_file) < MIN_AUDIO_BYTES:
            raise AudioTooShortError(len(audio_file))

        audio_file = align_container_header(audio_file, audio_format)

        detected = self._detect_format(audio_file)
        effective_format = detected or audio_format
        if detected and detected != audio_format:
            logger.info(
                "[STT] Format override: told '%s' but magic bytes say '%s' — using '%s'",
                audio_format, detected, detected,
            )

        try:
            audio_stream = io.BytesIO(audio_file)
            audio_stream.name = f"recording.{effective_format}"

            stt_start = time.time()
            logger.info("[STT] Sending %s bytes as '%s' to Whisper…", f"{len(audio_file):,}", effective_format)

            transcript = await self.groq_client.audio.transcriptions.create(
                model=settings.GROQ_WHISPER_MODEL,
                file=audio_stream,
                response_format="verbose_json",
                prompt=WHISPER_PROMPT,
            )

            result_text = filter_whisper_hallucinations(transcript)
            stt_ms = round((time.time() - stt_start) * 1000, 1)
            logger.info('[STT OK] %s bytes → "%s" (%sms)', f"{len(audio_file):,}", result_text[:80], stt_ms)
            voice_log("stt_completed", audio_bytes=len(audio_file), format=effective_format,
                      latency_ms=stt_ms, transcript_preview=result_text[:60])
            return result_text

        except Exception as exc:
            logger.error("[STT FAIL] %s bytes (%s): %r", f"{len(audio_file):,}", effective_format, exc)
            raise ValueError(f"Transcription failed: {exc}") from exc

    async def transcribe_chunks(self, chunks: list[bytes], audio_format: str = "webm") -> str:
        """Concatenate audio chunks and transcribe as a single request."""
        combined = b"".join(chunks)
        logger.info("[STT] Assembled %d chunk(s) → %s bytes total", len(chunks), f"{len(combined):,}")
        voice_log("stt_chunks_assembled", chunk_count=len(chunks), total_bytes=len(combined))
        return await self.transcribe_audio(combined, audio_format)
