"""app.controllers.audio_controller"""
import urllib.parse

from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from app.core.exceptions import AppError, InternalError
from app.schemas.audio import AudioSynthesizeRequest
from app.services import audio_service


async def transcribe(file: UploadFile, audio_format: str) -> dict:
    try:
        audio_bytes = await file.read()
        return await audio_service.transcribe(audio_bytes, file.filename, audio_format)
    except AppError:
        raise
    except Exception as exc:
        raise InternalError(f"Transcription failed: {exc}")


async def synthesize(request: AudioSynthesizeRequest) -> StreamingResponse:
    try:
        audio_bytes = await audio_service.synthesize(request.text, request.voice)
    except AppError:
        raise
    except Exception as exc:
        raise InternalError(f"Speech synthesis failed: {exc}")
    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


async def agent_voice(
    file: UploadFile,
    session_id: str,
    user_id: str,
    audio_format: str,
    voice: str,
) -> StreamingResponse:
    try:
        audio_bytes = await file.read()
        response_audio, full_response = await audio_service.agent_voice(
            audio_bytes, session_id, user_id, audio_format, voice
        )
    except AppError:
        raise
    except Exception as exc:
        raise InternalError(f"Voice interaction failed: {exc}")
    return StreamingResponse(
        iter([response_audio]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=agent_response.wav",
            "X-Agent-Text": urllib.parse.quote(full_response[:200]),
        },
    )
