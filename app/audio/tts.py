"""Text-to-Speech (Groq Orpheus → OpenAI fallback) behaviour for :class:`AudioProcessor`."""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from app.config import settings

from .constants import contains_arabic
from .diagnostics import voice_log
from .segmentation import batch_sentences_for_tts, split_sentences
from .transliteration import _transliterate_for_arabic_tts

logger = logging.getLogger(__name__)

_TTS_BATCH_MIN_CHARS = 250


async def _read_audio(response: Any) -> bytes:
    """Extract audio bytes from the several response shapes the SDKs return."""
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "read"):
        return await response.read()
    return response


class TextToSpeechMixin:
    """Requires ``self.groq_tts_client`` and optionally ``self.openai_client``."""

    groq_tts_client: AsyncOpenAI
    openai_client: Optional[AsyncOpenAI]

    async def synthesize_speech(self, text: str, voice: str = "auto") -> bytes:
        """Synthesise ``text`` with Groq Orpheus, falling back to OpenAI TTS if configured.

        Language is auto-detected from the text.

        Args:
            text: text to synthesise.
            voice: voice name; ``'auto'`` picks ``abdullah`` (Arabic) / ``hannah`` (English).

        Returns:
            WAV audio bytes.

        Raises:
            ValueError: when every configured provider fails.
        """
        is_arabic = contains_arabic(text)
        arabic_model = getattr(settings, "GROQ_TTS_MODEL_ARABIC", "canopylabs/orpheus-arabic-saudi")
        english_model = getattr(settings, "GROQ_TTS_MODEL_ENGLISH", "canopylabs/orpheus-v1-english")
        model = arabic_model if is_arabic else english_model
        selected_voice = voice if voice != "auto" else ("abdullah" if is_arabic else "hannah")

        tts_start = time.time()
        try:
            response = await self.groq_tts_client.audio.speech.create(
                model=model, voice=selected_voice, response_format="wav", input=text,
            )
            audio_bytes = await _read_audio(response)

            tts_ms = round((time.time() - tts_start) * 1000, 1)
            logger.info("[TTS OK] %s (%s) -> %s bytes audio (%sms)",
                        model, selected_voice, f"{len(audio_bytes):,}", tts_ms)
            voice_log("tts_completed", model=model, voice=selected_voice,
                      audio_bytes=len(audio_bytes), latency_ms=tts_ms)
            return audio_bytes

        except Exception as exc:
            logger.info("[TTS INFO] Groq TTS unavailable (%s) — trying OpenAI TTS if configured…", exc)
            fallback = await self._synthesize_openai(text, voice)
            if fallback is not None:
                return fallback
            raise ValueError(f"Speech synthesis failed: {exc}") from exc

    async def _synthesize_openai(self, text: str, voice: str) -> Optional[bytes]:
        """OpenAI TTS fallback; returns ``None`` if unconfigured or failing."""
        if not self.openai_client:
            return None
        try:
            response = await self.openai_client.audio.speech.create(
                model=getattr(settings, "OPENAI_TTS_MODEL", "tts-1"),
                voice="nova" if voice == "auto" else voice,
                response_format="wav",
                input=text,
            )
            return await _read_audio(response)
        except Exception as oa_err:
            logger.error("[TTS FAIL] OpenAI fallback error: %s", oa_err)
            return None

    async def synthesize_speech_chunked(self, text: str, voice: str = "auto") -> AsyncIterator[bytes]:
        """Batched sentence-chunked TTS: yields WAV audio per batch of ~250+ chars.

        Reduces TTS API calls (e.g. 15 sentences → 3-4 batches) while still
        streaming progressively. Arabic text gets brand-name transliteration
        (AI-Lixir → اي ليكسر). A failing batch is skipped, never ending the stream.
        """
        sentences = split_sentences(text)
        if not sentences:
            return

        batches = batch_sentences_for_tts(sentences, min_chars=_TTS_BATCH_MIN_CHARS)
        voice_log("tts_chunked_start", sentence_count=len(sentences),
                  batch_count=len(batches), text_preview=text[:80])

        is_arabic_voice = contains_arabic(text)  # same logic as synthesize_speech

        for idx, batch_text in enumerate(batches):
            if not batch_text.strip():
                continue
            tts_text = _transliterate_for_arabic_tts(batch_text) if is_arabic_voice else batch_text
            try:
                audio = await self.synthesize_speech(tts_text, voice)
                voice_log("tts_chunk_ready", chunk_index=idx, text_len=len(tts_text), audio_bytes=len(audio))
                yield audio
            except Exception as exc:
                logger.warning("[TTS batch %d] Failed: %s", idx, exc)
                continue
