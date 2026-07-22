"""
app.api.v1.audio
~~~~~~~~~~~~~~~~
POST /api/v1/audio/transcribe   — Speech-to-Text (Groq Whisper)
POST /api/v1/audio/synthesize   — Text-to-Speech (OpenAI TTS)
POST /api/v1/audio/agent-voice  — Full voice-to-voice pipeline (STT → Agent → TTS)
"""
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.audio import audio_processor
from app.config import settings
from app.core.orchestration import route_and_stream
from app.schemas.audio import AudioSynthesizeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_format: str = "webm",
):
    """
    Speech-to-Text using Groq whisper-large-v3-turbo.
    Accepts webm, mp4, wav, mp3, m4a, ogg, flac.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext in {"webm", "mp4", "wav", "mp3", "m4a", "ogg", "flac"}:
                audio_format = ext

        transcribed_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
        return {
            "status": "success",
            "transcribed_text": transcribed_text,
            "audio_format": audio_format,
            "model": settings.GROQ_WHISPER_MODEL,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")


@router.post("/synthesize")
async def synthesize_speech(request: AudioSynthesizeRequest):
    """Text-to-Speech using Groq Orpheus TTS."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        audio_bytes = await audio_processor.synthesize_speech(request.text, request.voice)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")


@router.post("/agent-voice")
async def agent_voice_interaction(
    file: UploadFile = File(...),
    session_id: str = Form(default="default_session"),
    user_id: str = Form(default="default_user"),
    audio_format: str = Form(default="webm"),
    voice: str = Form(default="auto"),
):
    """HTTP voice-to-voice pipeline: STT → Agent → TTS."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        user_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
        logger.info(f"[Audio Agent] Transcribed: {user_text}")

        full_response = ""
        async for token in route_and_stream(user_text, session_id, user_id):
            full_response += token

        if not full_response:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        response_audio = await audio_processor.synthesize_speech(full_response, voice)
        return StreamingResponse(
            iter([response_audio]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=agent_response.wav",
                "X-Agent-Text": full_response[:200],
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {exc}")
