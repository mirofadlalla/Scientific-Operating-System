"""
app.api.v1.audio
~~~~~~~~~~~~~~~~
POST /api/v1/audio/transcribe   — Speech-to-Text (Groq Whisper)
POST /api/v1/audio/synthesize   — Text-to-Speech
POST /api/v1/audio/agent-voice  — Full voice-to-voice pipeline (STT → Agent → TTS)
"""
from fastapi import APIRouter, File, Form, UploadFile

from app.controllers import audio_controller
from app.schemas.audio import AudioSynthesizeRequest

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
    return await audio_controller.transcribe(file, audio_format)


@router.post("/synthesize")
async def synthesize_speech(request: AudioSynthesizeRequest):
    """Text-to-Speech using Groq Orpheus TTS."""
    return await audio_controller.synthesize(request)


@router.post("/agent-voice")
async def agent_voice_interaction(
    file: UploadFile = File(...),
    session_id: str = Form(default="default_session"),
    user_id: str = Form(default="default_user"),
    audio_format: str = Form(default="webm"),
    voice: str = Form(default="auto"),
):
    """HTTP voice-to-voice pipeline: STT → Agent → TTS."""
    return await audio_controller.agent_voice(file, session_id, user_id, audio_format, voice)
