"""
app.controllers.voice_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WebSocket voice channel: accept the connection, read client messages,
dispatch each to a handler, and clean up on exit. The pipeline itself lives
in app.services.voice_service.

Client → Server messages (JSON text frames):
    {"type": "audio_chunk", "data": "<base64 audio>", "format": "webm"}
    {"type": "audio_end"}                 — user finished speaking
    {"type": "interrupt"}                 — interrupt current AI response
    {"type": "ping"}                      — keepalive

Server → Client messages:
    JSON text frames:
        {"type": "transcript",  "text": "...", "final": true/false}
        {"type": "ai_start"} | {"type": "ai_token", "token": "...", "done": false}
        {"type": "ai_done"}  | {"type": "interrupted"}  | {"type": "pong"}
        {"type": "error", "message": "..."} | {"type": "status", "status": "..."}
    Binary frames:
        Raw WAV audio bytes (one frame per TTS batch)
"""
import asyncio
import base64
import json
import logging
import traceback

from fastapi import WebSocket, WebSocketDisconnect

import app.core.state as state
from app.audio import voice_log
from app.services import voice_service
from app.services.voice_session import VoiceSession

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_SECONDS = 300.0   # tablets throttle timers; 5 min is safe with 10 s pings


# ── Message handlers ──────────────────────────────────────────────────────────

async def _on_audio_chunk(session: VoiceSession, msg: dict) -> None:
    chunk_b64 = msg.get("data", "")
    if chunk_b64:
        session.audio_chunks.append(base64.b64decode(chunk_b64))

    # User started speaking while the AI is replying → barge-in.
    if session.ai_streaming and session.current_task and not session.interrupted:
        session.interrupted = True
        await session.send_json({"type": "interrupted"})
        session.current_task.cancel()


async def _on_audio_end(session: VoiceSession, msg: dict) -> None:
    if session.current_task and not session.current_task.done():
        session.interrupted = True
        await voice_service.cancel_current_task(session)

    audio_format = msg.get("format", "webm")
    turn_id = session.new_turn()
    voice_log("audio_end_received", session_id=session.session_id,
              turn_id=turn_id, chunk_count=len(session.audio_chunks))

    # Background task keeps the receive loop responsive to interrupt/ping/new audio.
    session.current_task = asyncio.create_task(voice_service.process_turn(session, audio_format))


async def _on_interrupt(session: VoiceSession, msg: dict) -> None:
    session.interrupted = True
    session.audio_chunks = []
    await voice_service.cancel_current_task(session)
    await session.send_json({"type": "interrupted"})


async def _on_ping(session: VoiceSession, msg: dict) -> None:
    await session.send_json({"type": "pong"})


_HANDLERS = {
    "audio_chunk": _on_audio_chunk,
    "audio_end": _on_audio_end,
    "interrupt": _on_interrupt,
    "ping": _on_ping,
}


# ── Entry point ───────────────────────────────────────────────────────────────

async def handle_voice_channel(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = VoiceSession(websocket, session_id)
    state.active_voice_sessions[session_id] = session
    logger.info(f"[WS] Voice session opened: {session_id}")

    heartbeat_task = asyncio.create_task(voice_service.heartbeat_loop(session))

    try:
        while True:
            try:
                # Use receive() instead of receive_text() so that unexpected
                # binary frames from tablets don't crash the loop.
                frame = await asyncio.wait_for(websocket.receive(), timeout=IDLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.info(f"[WS] Session {session_id} timed out ({IDLE_TIMEOUT_SECONDS:.0f}s no message)")
                break

            # Text frames carry JSON control messages
            if "text" in frame:
                raw = frame["text"]
            elif "bytes" in frame:
                # Unexpected raw binary from tablet — ignore gracefully
                logger.debug(f"[WS] Received unexpected binary frame ({len(frame['bytes'])} bytes), ignoring")
                continue
            else:
                # WebSocketDisconnect or close frame
                raise WebSocketDisconnect()

            msg = json.loads(raw)
            handler = _HANDLERS.get(msg.get("type"))
            if handler:
                await handler(session, msg)

    except WebSocketDisconnect:
        logger.info(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        logger.error(f"[WS] Session error: {exc}")
        traceback.print_exc()
        await session.send_json({"type": "error", "message": str(exc)})
    finally:
        session._closed = True
        await voice_service.cancel_current_task(session)
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        state.active_voice_sessions.pop(session_id, None)
        logger.info(f"[WS] Session cleaned up: {session_id}")
