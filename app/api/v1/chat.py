"""
app.api.v1.chat
~~~~~~~~~~~~~~~
POST /api/v1/orchestrate  — Streaming text chat
WS   /api/v1/ws/voice     — Real-time bi-directional voice channel
"""
import base64
import json
import logging
import traceback
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.audio import audio_processor
from app.core.orchestration import route_and_stream
from app.schemas.chat import UserQuery

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# ──────────────────────────────────────────────────────────────────────────────
# Voice session state (local to this module — only used by the WS handler)
# ──────────────────────────────────────────────────────────────────────────────

class VoiceSession:
    """Tracks state for a single WebSocket voice session."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.audio_chunks: List[bytes] = []
        self.is_speaking = False       # VAD: user is currently speaking
        self.ai_streaming = False      # AI is currently streaming a response
        self.interrupted = False       # User interrupted AI mid-stream
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 8     # ~800 ms of silence before auto-stop


_active_voice_sessions: dict[str, VoiceSession] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/orchestrate")
async def process_user_input(query: UserQuery):
    """
    Stream an AI response token-by-token.

    The response is plain text streamed via `text/plain` — the frontend
    appends each received chunk to the message bubble in real time.
    """
    async def streamer():
        try:
            async for token in route_and_stream(
                query.text_input, query.session_id, query.user_id
            ):
                yield token
        except Exception as exc:
            logger.error(f"[STREAM CRASH]: {exc}")
            yield f"\n[Stream Error]: {exc}"

    return StreamingResponse(streamer(), media_type="text/plain")


@router.websocket("/ws/voice")
async def websocket_voice_channel(
    websocket: WebSocket,
    session_id: str = "ws_session",
):
    """
    Real-time bi-directional voice channel.

    Client → Server messages (JSON):
        {"type": "audio_chunk", "data": "<base64 audio>", "format": "webm"}
        {"type": "audio_end"}           — user finished speaking
        {"type": "interrupt"}           — interrupt current AI response
        {"type": "vad_energy", "rms": 342.5}  — client-side VAD reading

    Server → Client messages (JSON):
        {"type": "vad_status", "speaking": true/false}
        {"type": "transcript",  "text": "...", "final": true/false}
        {"type": "ai_token",    "token": "...", "done": false}
        {"type": "ai_done"}
        {"type": "error",       "message": "..."}
        {"type": "interrupted"}
    """
    await websocket.accept()
    session = VoiceSession(websocket, session_id)
    _active_voice_sessions[session_id] = session
    logger.info(f"[WS] Voice session opened: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Audio chunk accumulation ──────────────────────────────────────
            if msg_type == "audio_chunk":
                chunk_b64 = msg.get("data", "")
                if chunk_b64:
                    session.audio_chunks.append(base64.b64decode(chunk_b64))
                # Interrupt AI if it's currently streaming
                if session.ai_streaming:
                    session.interrupted = True
                    await websocket.send_text(json.dumps({"type": "interrupted"}))

            # ── Client-side VAD energy ────────────────────────────────────────
            elif msg_type == "vad_energy":
                rms = float(msg.get("rms", 0))
                speaking = audio_processor.is_speech(rms)
                if speaking != session.is_speaking:
                    session.is_speaking = speaking
                    await websocket.send_text(json.dumps({
                        "type": "vad_status",
                        "speaking": speaking,
                        "rms": rms,
                    }))

            # ── User finished speaking → transcribe + respond ─────────────────
            elif msg_type == "audio_end":
                if not session.audio_chunks:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": "No audio received",
                    }))
                    continue

                audio_format = msg.get("format", "webm")
                try:
                    transcript = await audio_processor.transcribe_chunks(
                        session.audio_chunks, audio_format
                    )
                    session.audio_chunks = []
                    await websocket.send_text(json.dumps({
                        "type": "transcript", "text": transcript, "final": True,
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": f"Transcription failed: {exc}",
                    }))
                    session.audio_chunks = []
                    continue

                session.ai_streaming = True
                session.interrupted = False
                try:
                    async for token in route_and_stream(transcript, session_id, "ws_user"):
                        if session.interrupted:
                            break
                        await websocket.send_text(json.dumps({
                            "type": "ai_token", "token": token, "done": False,
                        }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": f"Agent error: {exc}",
                    }))
                finally:
                    session.ai_streaming = False
                    await websocket.send_text(json.dumps({"type": "ai_done"}))

            # ── Interrupt signal ──────────────────────────────────────────────
            elif msg_type == "interrupt":
                session.interrupted = True
                session.audio_chunks = []
                await websocket.send_text(json.dumps({"type": "interrupted"}))

    except WebSocketDisconnect:
        logger.info(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        logger.error(f"[WS] Session error: {exc}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        _active_voice_sessions.pop(session_id, None)
        logger.info(f"[WS] Session cleaned up: {session_id}")
