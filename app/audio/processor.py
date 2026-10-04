"""AudioProcessor: Groq Whisper STT + Orpheus/OpenAI TTS for the scientific agents."""

from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from app.config import groq_stt_key, groq_tts_key, settings

from .stt import SpeechToTextMixin
from .tts import TextToSpeechMixin


class AudioProcessor(SpeechToTextMixin, TextToSpeechMixin):
    """Owns the API clients; STT/TTS behaviour comes from the mixins."""

    def __init__(self) -> None:
        # STT client — GROQ_STT_API_KEY (falls back to GROQ_API_KEY)
        self.groq_client = AsyncOpenAI(base_url=settings.GROQ_BASE_URL, api_key=groq_stt_key())
        # TTS client — GROQ_TTS_API_KEY (falls back to GROQ_API_KEY)
        self.groq_tts_client = AsyncOpenAI(base_url=settings.GROQ_BASE_URL, api_key=groq_tts_key())
        # Optional: OpenAI for high-quality TTS (nova, alloy, shimmer…)
        self.openai_client: Optional[AsyncOpenAI] = None
        if getattr(settings, "OPENAI_API_KEY", None):
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Convenience pipelines ────────────────────────────────────────────────
    async def process_voice_input(self, audio_file: bytes, audio_format: str = "webm") -> str:
        """Full pipeline: voice → text."""
        return await self.transcribe_audio(audio_file, audio_format)

    async def process_voice_output(self, agent_response: str, voice: str = "nova") -> bytes:
        """Full pipeline: text → audio."""
        return await self.synthesize_speech(agent_response, voice)


audio_processor = AudioProcessor()
