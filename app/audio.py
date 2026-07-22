"""
Audio Processing Module - STT (Groq Whisper) and TTS capabilities for agents
- STT: Groq whisper-large-v3-turbo (uses existing GROQ_API_KEY, no OpenAI needed)
- TTS: OpenAI TTS when available, browser SpeechSynthesis as fallback
- Streaming: Chunked audio transcription for WebSocket voice channel
"""
import sys
import io as _io
# Force stdout/stderr to UTF-8 on Windows so Unicode in log messages never crashes خلي الـ stdout يكتب UTF-8.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import io
import asyncio
import struct
import math
from pathlib import Path
from openai import AsyncOpenAI
from app.config import settings


class AudioProcessor:
    """Handles Speech-to-Text (Groq Whisper) and Text-to-Speech for scientific agents"""

    def __init__(self):
        # Primary client — Groq (handles both LLM and Whisper STT)
        self.groq_client = AsyncOpenAI(
            base_url=settings.GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY
        )

        # Optional: OpenAI for high-quality TTS (nova, alloy, shimmer…)
        self.openai_client = None
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # ──────────────────────────────────────────────────────────────────────────
    # STT  —  Groq Whisper (free, fast, no OpenAI key required)
    # ──────────────────────────────────────────────────────────────────────────

    async def transcribe_audio(self, audio_file: bytes, audio_format: str = "webm") -> str:
        """
        Speech-to-Text using Groq whisper-large-v3-turbo.
        Works with any audio format supported by Whisper (webm, mp4, wav, mp3, m4a…).

        Args:
            audio_file: Raw audio bytes
            audio_format: Container format string

        Returns:
            Transcribed text string
        """
        if not audio_file:
            raise ValueError("Empty audio buffer — nothing to transcribe")

        try:
            audio_stream = io.BytesIO(audio_file) # Groq API عايز File => لكن إحنا عندنا Bytes => فنحوله لملف في الذاكرة.

            # Whisper needs a filename so it can detect the codec
            audio_stream.name = f"recording.{audio_format}" # Whisper بيعرف نوع الملف من الامتداد.

            transcript = await self.groq_client.audio.transcriptions.create(
                model=settings.GROQ_WHISPER_MODEL,
                file=audio_stream,
                response_format="text",
                prompt="Scientific query in English or Arabic (العربية). Chemistry, biology, compound, SMILES, medicine, research.",
            )

            # Groq returns plain text when response_format="text"
            result_text = transcript if isinstance(transcript, str) else transcript.text
            print(f"[STT OK] {len(audio_file):,} bytes -> \"{result_text[:80]}\"")
            return result_text.strip()

        except Exception as exc:
            print(f"[STT FAIL] Groq Whisper error: {repr(exc)}")
            raise ValueError(f"Transcription failed: {exc}") from exc

    async def transcribe_chunks(self, chunks: list[bytes], audio_format: str = "webm") -> str:
        """Concatenate audio chunks and transcribe as a single request."""
        combined = b"".join(chunks)
        return await self.transcribe_audio(combined, audio_format)

    # ──────────────────────────────────────────────────────────────────────────
    # TTS  —  OpenAI (high-quality) with graceful degradation
    # ──────────────────────────────────────────────────────────────────────────

    async def synthesize_speech(self, text: str, voice: str = "auto") -> bytes:
        """
        Text-to-Speech using Groq API with Orpheus models.
        Auto-detects language or defaults based on text content.

        Args:
            text: Text to synthesize
            voice: Voice parameter (defaults: 'abdullah' for Arabic, 'hannah' for English)

        Returns:
            WAV/Audio bytes
        """
        import re
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        model = "canopylabs/orpheus-arabic-saudi" if is_arabic else "canopylabs/orpheus-v1-english"
        selected_voice = voice if voice != "auto" else ("abdullah" if is_arabic else "hannah")

        try:
            response = await self.groq_client.audio.speech.create(
                model=model,
                voice=selected_voice,
                response_format="wav",
                input=text,
            )
            # Response in OpenAI/Groq SDK supports response.content or streaming/bytes
            if hasattr(response, "content"):
                audio_bytes = response.content
            elif hasattr(response, "read"):
                audio_bytes = await response.read()
            else:
                audio_bytes = response

            print(f"[TTS OK] {model} ({selected_voice}) -> {len(audio_bytes):,} bytes audio")
            return audio_bytes

        except Exception as exc:
            print(f"[TTS FAIL] Groq TTS error: {repr(exc)}")
            raise ValueError(f"Speech synthesis failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # VAD  —  Energy-based Voice Activity Detection (no extra libraries)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rms(pcm_bytes: bytes, sample_width: int = 2) -> float:
        """
        Compute RMS energy of raw PCM bytes.
        sample_width=2 means 16-bit samples (standard for WebAudio ScriptProcessor).
        """
        if len(pcm_bytes) < sample_width:
            return 0.0
        num_samples = len(pcm_bytes) // sample_width
        fmt = f"<{num_samples}h"  # little-endian 16-bit signed
        try:
            samples = struct.unpack(fmt, pcm_bytes[:num_samples * sample_width]) 
            # pcm is [-200 300 1000 ...] but it comes with bytes so we use struct.unpack to convert it to integers
            rms = math.sqrt(sum(s * s for s in samples) / num_samples) # الـ RMS يمثل متوسط طاقة الإشارة الصوتية. كلما ارتفع، كان الصوت أعلى.
            return rms
        except struct.error:
            return 0.0
        '''
         ليه بنحسب RMS؟
        الصوت عبارة عن موجة.
        لو الميكروفون ساكت:

        1 / -2 / 3 / -1 / 0

        الـ RMS هيبقى صغير جداً.

        لكن لو حد بيتكلم:
        300 / 800 / 1500 / 700 / 900
        الـ RMS هيكبر.

        فهو مقياس لشدة الصوت بغض النظر عن الإشارة الموجبة أو السالبة.
        '''

    @staticmethod
    def is_speech(rms: float, threshold: float = 1200.0) -> bool:
        """Simple energy threshold VAD."""
        return rms > threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience pipelines
    # ──────────────────────────────────────────────────────────────────────────

    async def process_voice_input(self, audio_file: bytes, audio_format: str = "webm") -> str:
        """Full pipeline: voice → text"""
        return await self.transcribe_audio(audio_file, audio_format)

    async def process_voice_output(self, agent_response: str, voice: str = "nova") -> bytes:
        """Full pipeline: text → audio"""
        return await self.synthesize_speech(agent_response, voice)


# Singleton
audio_processor = AudioProcessor()
