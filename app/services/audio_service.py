"""
app.services.audio_service
~~~~~~~~~~~~~~~~~~~~~~~~~~
Speech-to-text, text-to-speech and the STT → agent → TTS pipeline.
"""
import logging
from typing import Optional, Tuple

from app.audio import audio_processor
from app.config import settings
from app.core.exceptions import BadRequestError, InternalError
from app.core.orchestration import route_and_stream
from app.core.text_cleaning import clean_for_tts

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {"webm", "mp4", "wav", "mp3", "m4a", "ogg", "flac"}


async def transcribe(audio_bytes: bytes, filename: Optional[str], audio_format: str) -> dict:
    if not audio_bytes:
        raise BadRequestError("Empty audio file")

    if filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            audio_format = ext

    text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
    return {
        "status": "success",
        "transcribed_text": text,
        "audio_format": audio_format,
        "model": settings.GROQ_WHISPER_MODEL,
    }


async def synthesize(text: str, voice: str) -> bytes:
    if not text or not text.strip():
        raise BadRequestError("Text cannot be empty")
    tts_text = clean_for_tts(text)
    if not tts_text:
        raise BadRequestError("Text contains no speakable content after cleaning")
    return await audio_processor.synthesize_speech(tts_text, voice)


async def agent_voice(
    audio_bytes: bytes,
    session_id: str,
    user_id: str,
    audio_format: str,
    voice: str,
) -> Tuple[bytes, str]:
    """Run STT → agent → TTS. Returns (wav_bytes, agent_text)."""
    if not audio_bytes:
        raise BadRequestError("Empty audio file")

    user_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
    logger.info(f"[Audio Agent] Transcribed: {user_text}")

    full_response = ""
    async for token in route_and_stream(user_text, session_id, user_id, include_images=False):
        full_response += token

    if not full_response:
        raise InternalError("Failed to generate response")

    tts_text = clean_for_tts(full_response)
    if not tts_text:
        raise InternalError("Agent response contains no speakable content")

    response_audio = await audio_processor.synthesize_speech(tts_text, voice)
    return response_audio, full_response
