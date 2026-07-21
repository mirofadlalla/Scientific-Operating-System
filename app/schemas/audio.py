"""Pydantic request / response schemas for the audio endpoints."""
from pydantic import BaseModel


class AudioTranscribeRequest(BaseModel):
    audio_format: str = "webm"


class AudioSynthesizeRequest(BaseModel):
    text: str
    voice: str = "nova"


class AudioAgentRequest(BaseModel):
    session_id: str
    user_id: str
    audio_format: str = "webm"
    voice: str = "nova"
