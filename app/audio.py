"""
Audio Processing Module - STT (Groq Whisper) and TTS capabilities for agents
- STT: Groq whisper-large-v3-turbo (uses existing GROQ_API_KEY, no OpenAI needed)
- TTS: Groq Orpheus Arabic/English with OpenAI TTS fallback
- Streaming: Chunked audio transcription for WebSocket voice channel
- Sentence-chunked TTS: yields audio per sentence for low-latency playback
"""
import sys
import io as _io
# Force stdout/stderr to UTF-8 on Windows so Unicode in log messages never crashes خلي الـ stdout يكتب UTF-8.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import io
import re
import asyncio
import struct
import math
import time
import json
import logging
from typing import AsyncIterator
from pathlib import Path
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)


def voice_log(event: str, **kwargs):
    """Structured diagnostic log for voice pipeline — only emits when VOICE_DEBUG=true."""
    if not settings.VOICE_DEBUG:
        return
    entry = {"event": event, "ts": round(time.time(), 3), **kwargs}
    print(f"[VOICE_DEBUG] {json.dumps(entry, ensure_ascii=False)}")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-sized chunks for incremental TTS.

    Handles:
      - English sentence endings (. ! ?)
      - Arabic sentence endings (. ، ؟ !)
      - Markdown bullets / numbered lists (treated as boundaries)
      - Newlines as boundaries
    Returns non-empty stripped strings only.
    """
    # Split on sentence-ending punctuation followed by whitespace, or on newlines
    # \u060C = Arabic comma ،   \u061F = Arabic question mark ؟
    parts = re.split(r'(?<=[.!?\u060C\u061F])\s+|\n+', text)
    sentences = [p.strip() for p in parts if p.strip()]

    # Merge very short fragments (< 6 chars) with previous sentence
    # Threshold is kept low because Arabic text is denser per character
    merged = []
    for s in sentences:
        if merged and len(s) < 6:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def batch_sentences_for_tts(sentences: list[str], min_chars: int = 120) -> list[str]:
    """
    Batch sentences into groups so each group has at least `min_chars` characters.
    This reduces the number of TTS API calls dramatically.
    E.g. 15 short sentences → 3-4 batches of ~120+ chars each.
    """
    if not sentences:
        return []
    batches = []
    current = ""
    for s in sentences:
        if current:
            current = current + " " + s
        else:
            current = s
        if len(current) >= min_chars:
            batches.append(current)
            current = ""
    if current:
        batches.append(current)
    return batches


# Brand names / proper nouns to transliterate for Arabic TTS
_ARABIC_TRANSLITERATIONS = {
    "AI-Lixir": "اي ليكسر",
    "Ai-Lixir": "اي ليكسر",
    "ai-lixir": "اي ليكسر",
    "ADMET": "ايه دي ام اي تي",
    "SMILES": "سمايلز",
}


def _transliterate_for_arabic_tts(text: str) -> str:
    """Replace English brand names with Arabic phonetic equivalents for TTS."""
    for eng, ar in _ARABIC_TRANSLITERATIONS.items():
        text = text.replace(eng, ar)
    return text


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

    @staticmethod
    def _detect_format(audio_bytes: bytes) -> str:
        """
        Detect actual audio container format from magic bytes.
        This lets us correct the format label when the browser sends a different
        container than expected (e.g. Firefox sends OGG, Safari sends MP4).
        """
        if audio_bytes[:4] == b'OggS':
            return 'ogg'
        if audio_bytes[:4] == b'fLaC':
            return 'flac'
        if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        if audio_bytes[:3] == b'ID3' or (len(audio_bytes) >= 2 and audio_bytes[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2')):
            return 'mp3'
        if audio_bytes[4:8] in (b'ftyp', b'mdat', b'moov') if len(audio_bytes) >= 8 else False:
            return 'mp4'
        if audio_bytes[:4] == b'\x1aE\xdf\xa3':  # EBML header — WebM / MKV
            return 'webm'
        return ''  # unknown — leave as caller-provided

    async def transcribe_audio(self, audio_file: bytes, audio_format: str = "webm") -> str:
        """
        Speech-to-Text using Groq whisper-large-v3-turbo.
        Works with any audio format supported by Whisper (webm, mp4, wav, mp3, m4a…).

        Args:
            audio_file: Raw audio bytes
            audio_format: Container format hint (auto-detected from magic bytes when possible)

        Returns:
            Transcribed text string
        """
        if not audio_file:
            raise ValueError("Empty audio buffer — nothing to transcribe")

        # Minimum sanity check — reject obviously corrupt/empty payloads
        MIN_AUDIO_BYTES = 1024  # ~1 KB; anything smaller is almost certainly unusable
        if len(audio_file) < MIN_AUDIO_BYTES:
            raise ValueError(
                f"Audio too short ({len(audio_file)} bytes) — please speak for at least 1 second"
            )

        # Ensure WebM / audio containers start at their magic header
        # (prevents 400 invalid media file if stray cluster bytes were prepended)
        ebml_pos = audio_file.find(b'\x1aE\xdf\xa3')
        if ebml_pos > 0:
            print(f"[STT] Slicing {ebml_pos} stray leading bytes to align WebM EBML header")
            audio_file = audio_file[ebml_pos:]
        elif ebml_pos == -1 and audio_format == "webm":
            riff_pos = audio_file.find(b'RIFF')
            if riff_pos > 0:
                audio_file = audio_file[riff_pos:]
            ogg_pos = audio_file.find(b'OggS')
            if ogg_pos > 0:
                audio_file = audio_file[ogg_pos:]

        # Auto-detect format from magic bytes; fall back to caller-provided hint
        detected = self._detect_format(audio_file)
        effective_format = detected if detected else audio_format
        if detected and detected != audio_format:
            print(f"[STT] Format override: told '{audio_format}' but magic bytes say '{detected}' — using '{detected}'")

        try:
            audio_stream = io.BytesIO(audio_file)
            audio_stream.name = f"recording.{effective_format}"

            stt_start = time.time()
            print(f"[STT] Sending {len(audio_file):,} bytes as '{effective_format}' to Whisper…")

            whisper_prompt = (
                "محادثة علمية باللغة العربية والإنجليزية: أدوية، مركبات كيميائية، أحياء، "
                "جينات، مسارات بيولوجية، وبحث علمي. "
                "Scientific queries in Arabic (العربية) and English: drug discovery, chemistry, "
                "ADMET, biology, medicine, SMILES, molecular research."
            )

            transcript = await self.groq_client.audio.transcriptions.create(
                model=settings.GROQ_WHISPER_MODEL,
                file=audio_stream,
                response_format="text",
                prompt=whisper_prompt,
            )

            # Groq returns plain text when response_format="text"
            result_text = transcript if isinstance(transcript, str) else transcript.text
            stt_ms = round((time.time() - stt_start) * 1000, 1)
            print(f"[STT OK] {len(audio_file):,} bytes → \"{result_text[:80]}\" ({stt_ms}ms)")
            voice_log("stt_completed", audio_bytes=len(audio_file), format=effective_format,
                       latency_ms=stt_ms, transcript_preview=result_text[:60])
            return result_text.strip()

        except Exception as exc:
            print(f"[STT FAIL] {len(audio_file):,} bytes ({effective_format}): {repr(exc)}")
            raise ValueError(f"Transcription failed: {exc}") from exc

    async def transcribe_chunks(self, chunks: list[bytes], audio_format: str = "webm") -> str:
        """Concatenate audio chunks and transcribe as a single request."""
        combined = b"".join(chunks)
        print(f"[STT] Assembled {len(chunks)} chunk(s) → {len(combined):,} bytes total")
        voice_log("stt_chunks_assembled", chunk_count=len(chunks), total_bytes=len(combined))
        return await self.transcribe_audio(combined, audio_format)


    # ──────────────────────────────────────────────────────────────────────────
    # TTS  —  Groq Orpheus with OpenAI fallback and browser Web Speech fallback
    # ──────────────────────────────────────────────────────────────────────────

    async def synthesize_speech(self, text: str, voice: str = "auto") -> bytes:
        """
        Text-to-Speech using Groq API with Orpheus models (with OpenAI TTS fallback).
        Auto-detects language or defaults based on text content.

        Args:
            text: Text to synthesize
            voice: Voice parameter (defaults: 'abdullah' for Arabic, 'hannah' for English)

        Returns:
            WAV/Audio bytes
        """
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        model = "canopylabs/orpheus-arabic-saudi" if is_arabic else "canopylabs/orpheus-v1-english"
        selected_voice = voice if voice != "auto" else ("abdullah" if is_arabic else "hannah")

        tts_start = time.time()
        try:
            response = await self.groq_client.audio.speech.create(
                model=model,
                voice=selected_voice,
                response_format="wav",
                input=text,
            )
            if hasattr(response, "content"):
                audio_bytes = response.content
            elif hasattr(response, "read"):
                audio_bytes = await response.read()
            else:
                audio_bytes = response

            tts_ms = round((time.time() - tts_start) * 1000, 1)
            print(f"[TTS OK] {model} ({selected_voice}) -> {len(audio_bytes):,} bytes audio ({tts_ms}ms)")
            voice_log("tts_completed", model=model, voice=selected_voice,
                       audio_bytes=len(audio_bytes), latency_ms=tts_ms)
            return audio_bytes

        except Exception as exc:
            print(f"[TTS INFO] Groq TTS unavailable ({exc}) — trying OpenAI TTS if configured…")
            if self.openai_client:
                try:
                    oa_voice = "nova" if voice == "auto" else voice
                    oa_response = await self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice=oa_voice,
                        response_format="mp3",
                        input=text,
                    )
                    if hasattr(oa_response, "content"):
                        return oa_response.content
                    elif hasattr(oa_response, "read"):
                        return await oa_response.read()
                    return oa_response
                except Exception as oa_err:
                    print(f"[TTS FAIL] OpenAI fallback error: {oa_err}")
            raise ValueError(f"Speech synthesis failed: {exc}") from exc

    async def synthesize_speech_chunked(
        self, text: str, voice: str = "auto"
    ) -> AsyncIterator[bytes]:
        """
        Batched sentence-chunked TTS: splits text into sentences, batches them
        into groups of ~120+ chars, and yields WAV audio per batch.
        This reduces TTS API calls (e.g. 15 sentences → 3-4 batches) while
        still streaming audio progressively.

        For Arabic voices, applies brand-name transliteration (AI-Lixir → اي ليكسر).

        Yields:
            bytes — WAV audio for each batch
        """
        sentences = split_sentences(text)
        if not sentences:
            return

        # Batch sentences into groups to reduce TTS API calls
        batches = batch_sentences_for_tts(sentences, min_chars=120)

        voice_log("tts_chunked_start", sentence_count=len(sentences),
                   batch_count=len(batches), text_preview=text[:80])

        # Determine if Arabic voice will be used (same logic as synthesize_speech)
        is_arabic_voice = bool(re.search(r'[\u0600-\u06FF]', text))

        for idx, batch_text in enumerate(batches):
            if not batch_text.strip():
                continue
            # Transliterate brand names for Arabic TTS
            tts_text = _transliterate_for_arabic_tts(batch_text) if is_arabic_voice else batch_text
            try:
                audio = await self.synthesize_speech(tts_text, voice)
                voice_log("tts_chunk_ready", chunk_index=idx,
                           text_len=len(tts_text), audio_bytes=len(audio))
                yield audio
            except Exception as exc:
                logger.warning(f"[TTS batch {idx}] Failed: {exc}")
                # Continue with remaining batches — don't break the stream
                continue

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
