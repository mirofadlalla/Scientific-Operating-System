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
    {"type": "client_info", "vad": "silero"|"energy", ...} — which VAD the browser uses (logged)
    {"type": "ping"}                      — keepalive

Server → Client messages:
    JSON text frames:
        {"type": "transcript",  "text": "...", "final": true/false}
        {"type": "ai_start"} | {"type": "ai_token", "token": "...", "done": false}
        {"type": "ai_done"}  | {"type": "interrupted"}  | {"type": "pong"}
        {"type": "error", "message": "..."} | {"type": "status", "status": "..."}
    Binary frames:
        Raw WAV audio bytes (one frame per TTS batch)

                        Client connects
                            ↓
                        websocket.accept()
                            ↓
                        Create VoiceSession
                            ↓
                        Start heartbeat
                            ↓
                    ┌─────────────────────┐
                    │   while True        │
                    │                     │
                    │   receive messa     │
                    │       ↓             │
                    │   identify te       │
                    │       ↓             │
                    │   call handle       │
                    └─────────────────────┘
                            ↓
                        disconnect / timeout / error
                            ↓
                        finally
                            ↓
                        cancel AI task
                            ↓
                        cancel heartbeat
                            ↓
                        remove session
                            ↓
                        cleanup complete
"""
import asyncio
import base64
import binascii
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
MAX_UTTERANCE_BYTES = 25 * 1024 * 1024   # hard cap on one buffered utterance


def _turn_in_flight(session: VoiceSession) -> bool:
    """True while a turn (STT / LLM / TTS) is running and has not been interrupted."""
    task = session.current_task
    return task is not None and not task.done() and not session.interrupted


# ── Message handlers ──────────────────────────────────────────────────────────

async def _on_audio_chunk(session: VoiceSession, msg: dict) -> None:
    # A turn is in flight (STT / LLM / TTS) and the user has not interrupted it.
    # Mic audio arriving now is NOT a barge-in: speaker echo looks identical to
    # speech, and the old "any chunk cancels the answer" rule is what produced
    # the spurious "Interrupted" mid-answer. Real barge-in is explicit:
    #   - {"type": "interrupt"}  (client confirmed the user is speaking), or
    #   - a finished utterance ("audio_end") that carries audio of its own.
    # Drop the chunk so it cannot leak into the next turn's buffer.

    if _turn_in_flight(session):
        voice_log("audio_chunk_ignored_during_turn", session_id=session.session_id,
                  turn_id=session.turn_id)
        return

    chunk_b64 = msg.get("data", "")
    if not chunk_b64:
        return

    try:
        chunk = base64.b64decode(chunk_b64)
    except (binascii.Error, ValueError):
        logger.warning(f"[VOICE] session={session.session_id} dropped malformed base64 audio chunk")
        return

    # Bound the buffer: a client that streams forever without "audio_end" must not
    # be able to grow server memory without limit.
    if session.audio_bytes + len(chunk) > MAX_UTTERANCE_BYTES:
        voice_log("audio_buffer_overflow", session_id=session.session_id, turn_id=session.turn_id)
        session.audio_chunks = []
        return

    session.audio_chunks.append(chunk)


async def _on_audio_end(session: VoiceSession, msg: dict) -> None:
    if _turn_in_flight(session):
        # "audio_end" with no preceding "interrupt" while an answer is being produced
        # is speaker echo / background noise picked up by the client VAD. Cancelling
        # the turn here is what cut answers off when the user had not said anything
        # (and then failed with "No audio received", because the chunk had already
        # been dropped above). Keep the current answer going and discard the noise.
        voice_log("audio_end_ignored_during_turn", session_id=session.session_id,
                  turn_id=session.turn_id)
        session.audio_chunks = []
        return

    if not session.audio_chunks:
        # Nothing was recorded (e.g. chunks dropped by a reconnect): tell the client
        # to resume listening instead of surfacing a scary "No audio received" error.
        await session.send_json({"type": "status", "status": "No speech detected", "speak": False})
        await session.send_json({"type": "ai_done"})
        return

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


async def _on_client_info(session: VoiceSession, msg: dict) -> None:
    """Browser reports which VAD it is using (Silero vs energy fallback)."""
    vad = msg.get("vad")
    session.client_vad = vad
    if vad == "silero":
        logger.info(f"[VOICE] session={session.session_id} client VAD=SILERO "
                    f"source={msg.get('source')} model={msg.get('model')}")
    else:
        logger.warning(f"[VOICE] session={session.session_id} client VAD=ENERGY-FALLBACK "
                       f"(Silero did not load) reason={msg.get('reason')}")


async def _on_ping(session: VoiceSession, msg: dict) -> None:
    await session.send_json({"type": "pong"})


_HANDLERS = {
    "audio_chunk": _on_audio_chunk,
    "audio_end": _on_audio_end,
    "interrupt": _on_interrupt,
    "client_info": _on_client_info,
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
                frame = await asyncio.wait_for(websocket.receive(), timeout=IDLE_TIMEOUT_SECONDS) # استنى رسالة جديدة من الـ WebSocket، لكن بحد أقصى IDLE_TIMEOUT_SECONDS
            except asyncio.TimeoutError:
                logger.info(f"[WS] Session {session_id} timed out ({IDLE_TIMEOUT_SECONDS:.0f}s no message)")
                break

            # Text frames carry JSON control messages الفرونت بيبعت سترينج جيشون {"type": "audio_end"}
            if "text" in frame:
                raw = frame["text"]
            elif "bytes" in frame:
                # Unexpected raw binary from tablet — ignore gracefully
                logger.debug(f"[WS] Received unexpected binary frame ({len(frame['bytes'])} bytes), ignoring")
                continue
            else:
                # WebSocketDisconnect or close frame
                raise WebSocketDisconnect()

            msg = json.loads(raw) # تحويل سترينج الجيشون إلى dict
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
