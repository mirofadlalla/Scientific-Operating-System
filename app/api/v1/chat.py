"""
app.api.v1.chat
~~~~~~~~~~~~~~~
POST /api/v1/orchestrate  — Streaming text chat
WS   /api/v1/ws/voice     — Real-time bi-directional voice channel

Voice pipeline architecture (fixed):
  - Concurrent receive loop: always listening for client messages (interrupt, ping)
  - Processing runs in a background asyncio.Task that can be cancelled
  - TTS audio sent as binary WebSocket frames (not base64 JSON)
  - Sentence-chunked TTS for low time-to-first-audio
  - Safe send helpers that catch ConnectionClosed
  - Heartbeat ping/pong to keep connection alive through proxies
  - Structured timing logs per turn (VOICE_DEBUG)
"""
import asyncio
import base64
import json
import logging
import time
import traceback
import uuid
from typing import List, Optional

from app import monitoring

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

from app.audio import audio_processor, voice_log
from app.config import settings
from app.core.orchestration import route_and_stream
from app.schemas.chat import UserQuery

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# ──────────────────────────────────────────────────────────────────────────────
# Voice session state
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
        self.current_task: Optional[asyncio.Task] = None  # Active processing task
        self.turn_id: str = ""         # Unique ID per voice turn
        self.current_turn_seq: int = 0  # Sequence number to reject stray chunks from prior turns
        self._closed = False           # Track if we've detected a close

    def new_turn(self) -> str:
        """Generate a new turn ID for tracing."""
        self.turn_id = uuid.uuid4().hex[:8]
        self.interrupted = False
        self.ai_streaming = False
        return self.turn_id


_active_voice_sessions: dict[str, VoiceSession] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Safe WebSocket send helpers — never crash on closed connections
# ──────────────────────────────────────────────────────────────────────────────

async def _safe_send_text(ws: WebSocket, data: str, session: VoiceSession) -> bool:
    """Send a text frame. Returns False if the connection is closed."""
    if session._closed:
        return False
    try:
        if ws.client_state == WebSocketState.DISCONNECTED:
            session._closed = True
            return False
        await ws.send_text(data)
        return True
    except Exception:
        session._closed = True
        return False


async def _safe_send_bytes(ws: WebSocket, data: bytes, session: VoiceSession) -> bool:
    """Send a binary frame. Returns False if the connection is closed."""
    if session._closed:
        return False
    try:
        if ws.client_state == WebSocketState.DISCONNECTED:
            session._closed = True
            return False
        await ws.send_bytes(data)
        return True
    except Exception:
        session._closed = True
        return False


async def _safe_send_json(ws: WebSocket, obj: dict, session: VoiceSession) -> bool:
    """Send a JSON-encoded text frame. Returns False if the connection is closed."""
    return await _safe_send_text(ws, json.dumps(obj), session)


# ──────────────────────────────────────────────────────────────────────────────
# Binary audio frame chunking
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_CHUNK_SIZE = 32 * 1024  # 32 KB per binary frame


async def _send_audio_binary(ws: WebSocket, audio_bytes: bytes, session: VoiceSession) -> bool:
    """
    Send TTS audio as binary WebSocket frames in chunks.
    Returns False if the connection closes mid-send.
    """
    offset = 0
    while offset < len(audio_bytes):
        if session.interrupted or session._closed:
            return False
        chunk = audio_bytes[offset:offset + AUDIO_CHUNK_SIZE]
        ok = await _safe_send_bytes(ws, chunk, session)
        if not ok:
            return False
        offset += AUDIO_CHUNK_SIZE
    return True


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


# ──────────────────────────────────────────────────────────────────────────────
# Voice turn processor — runs as a cancellable asyncio.Task
# ──────────────────────────────────────────────────────────────────────────────

async def _process_voice_turn(session: VoiceSession, audio_format: str):
    """
    Full voice turn pipeline: STT → orchestrate → LLM stream → TTS → send audio.

    Runs as a background task so the receive loop stays responsive to
    interrupt / ping / new audio messages.
    """
    ws = session.ws
    turn_id = session.turn_id
    turn_start = time.time()

    chunks = session.audio_chunks[:]
    session.audio_chunks = []

    if not chunks:
        await _safe_send_json(ws, {"type": "error", "message": "No audio received"}, session)
        return

    voice_log("turn_start", session_id=session.session_id, turn_id=turn_id,
              chunk_count=len(chunks))

    # ── STT ──────────────────────────────────────────────────────────────
    await _safe_send_json(ws, {"type": "status", "status": "Transcribing..."}, session)
    stt_start = time.time()

    try:
        transcript = await audio_processor.transcribe_chunks(chunks, audio_format)
    except Exception as exc:
        await _safe_send_json(ws, {"type": "error", "message": f"Transcription failed: {exc}"}, session)
        return

    stt_ms = round((time.time() - stt_start) * 1000, 1)
    voice_log("stt_done", turn_id=turn_id, latency_ms=stt_ms, transcript=transcript[:60])

    # Send transcript
    ok = await _safe_send_json(ws, {
        "type": "transcript", "text": transcript, "final": True,
    }, session)
    if not ok:
        return

    await _safe_send_json(ws, {"type": "thought", "text": "Now retrieving information…"}, session)

    # Record STT metrics (non-blocking)
    try:
        monitoring.record_tokens(
            model=settings.GROQ_WHISPER_MODEL,
            prompt_tokens=100,
            completion_tokens=len(transcript) // 4,
            ttft_ms=round(stt_ms, 2),
        )
    except Exception:
        pass

    # Check for interruption after STT
    if session.interrupted:
        voice_log("turn_interrupted_after_stt", turn_id=turn_id)
        return

    # ── LLM streaming ────────────────────────────────────────────────────
    await _safe_send_json(ws, {"type": "thought", "text": "Generating answer…"}, session)
    await _safe_send_json(ws, {"type": "ai_start"}, session)


    session.ai_streaming = True
    full_reply = ""
    llm_start = time.time()
    first_token_time = None

    try:
        async for token in route_and_stream(transcript, session.session_id, "ws_user"):
            if session.interrupted:
                voice_log("turn_interrupted_during_llm", turn_id=turn_id,
                          tokens_so_far=len(full_reply))
                break
            full_reply += token
            if first_token_time is None:
                first_token_time = time.time()
            await _safe_send_json(ws, {
                "type": "ai_token", "token": token, "done": False,
            }, session)
    except asyncio.CancelledError:
        voice_log("turn_cancelled_during_llm", turn_id=turn_id)
        raise
    except Exception as exc:
        await _safe_send_json(ws, {"type": "error", "message": f"Agent error: {exc}"}, session)
        session.ai_streaming = False
        await _safe_send_json(ws, {"type": "ai_done"}, session)
        return

    llm_ms = round((time.time() - llm_start) * 1000, 1)
    ttft_ms = round((first_token_time - llm_start) * 1000, 1) if first_token_time else None
    voice_log("llm_done", turn_id=turn_id, latency_ms=llm_ms, ttft_ms=ttft_ms,
              reply_len=len(full_reply))

    # ── TTS (batched sentence-chunked, progressive streaming) ───────────
    if not session.interrupted and full_reply.strip():
        await _safe_send_json(ws, {"type": "status", "status": "Synthesizing voice..."}, session)
        tts_start = time.time()
        audio_bytes_total = 0
        chunk_idx = 0

        try:
            async for audio_chunk in audio_processor.synthesize_speech_chunked(
                full_reply, voice="auto"
            ):
                if session.interrupted or session._closed:
                    voice_log("tts_interrupted", turn_id=turn_id)
                    break
                audio_bytes_total += len(audio_chunk)
                # Send each TTS batch as a single binary frame so the client
                # can start playing it immediately (progressive playback)
                ok = await _safe_send_bytes(ws, audio_chunk, session)
                if not ok:
                    break
                chunk_idx += 1
        except asyncio.CancelledError:
            voice_log("turn_cancelled_during_tts", turn_id=turn_id)
            raise
        except Exception as tts_err:
            logger.warning(f"[WS TTS Error] turn={turn_id}: {tts_err}")

        tts_ms = round((time.time() - tts_start) * 1000, 1)
        voice_log("tts_send_done", turn_id=turn_id, latency_ms=tts_ms,
                  audio_bytes=audio_bytes_total, chunks_sent=chunk_idx)

    # ── Finalize ─────────────────────────────────────────────────────────
    session.ai_streaming = False
    await _safe_send_json(ws, {"type": "ai_done"}, session)

    total_ms = round((time.time() - turn_start) * 1000, 1)
    voice_log("turn_complete", turn_id=turn_id, total_ms=total_ms, stt_ms=stt_ms,
              llm_ms=llm_ms, reply_len=len(full_reply))
    print(f"[VOICE] Turn {turn_id} complete: STT={stt_ms}ms LLM={llm_ms}ms total={total_ms}ms")


# ──────────────────────────────────────────────────────────────────────────────
# Heartbeat — keeps the WebSocket alive through reverse proxies
# ──────────────────────────────────────────────────────────────────────────────

async def _heartbeat_loop(ws: WebSocket, session: VoiceSession, interval: float = 15.0):
    """Send a ping JSON every `interval` seconds to keep the connection alive."""
    try:
        while not session._closed:
            await asyncio.sleep(interval)
            if session._closed:
                break
            ok = await _safe_send_json(ws, {"type": "pong"}, session)
            if not ok:
                break
    except asyncio.CancelledError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket voice channel
# ──────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/voice")
async def websocket_voice_channel(
    websocket: WebSocket,
    session_id: str = "ws_session",
):
    """
    Real-time bi-directional voice channel.

    Client → Server messages (JSON text frames):
        {"type": "audio_chunk", "data": "<base64 audio>", "format": "webm"}
        {"type": "audio_end"}           — user finished speaking
        {"type": "interrupt"}           — interrupt current AI response
        {"type": "vad_energy", "rms": 342.5}  — client-side VAD reading
        {"type": "ping"}               — keepalive

    Server → Client messages:
        JSON text frames:
            {"type": "vad_status", "speaking": true/false}
            {"type": "transcript",  "text": "...", "final": true/false}
            {"type": "ai_start"}                    — AI response beginning
            {"type": "ai_token",    "token": "...", "done": false}
            {"type": "ai_done"}
            {"type": "error",       "message": "..."}
            {"type": "interrupted"}
            {"type": "status",      "status": "..."}
            {"type": "pong"}
        Binary frames:
            Raw WAV audio bytes (sent in 32KB chunks)
    """
    await websocket.accept()
    session = VoiceSession(websocket, session_id)
    _active_voice_sessions[session_id] = session
    logger.info(f"[WS] Voice session opened: {session_id}")

    # Start heartbeat to keep connection alive through HF Spaces proxy
    heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket, session))

    try:
        while True:
            # Receive with a timeout to detect stale connections
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
            except asyncio.TimeoutError:
                # No message for 2 minutes — connection is likely dead
                logger.info(f"[WS] Session {session_id} timed out (120s no message)")
                break

            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Audio chunk accumulation ──────────────────────────────────────
            if msg_type == "audio_chunk":
                chunk_b64 = msg.get("data", "")
                if chunk_b64:
                    session.audio_chunks.append(base64.b64decode(chunk_b64))

                # If AI is streaming and user starts speaking, interrupt
                if session.ai_streaming and session.current_task and not session.interrupted:
                    session.interrupted = True
                    await _safe_send_json(websocket, {"type": "interrupted"}, session)
                    # Cancel the processing task
                    session.current_task.cancel()

            # ── Client-side VAD energy ────────────────────────────────────────
            elif msg_type == "vad_energy":
                rms = float(msg.get("rms", 0))
                speaking = audio_processor.is_speech(rms, threshold=18.0) if rms <= 255.0 else audio_processor.is_speech(rms, threshold=1200.0)
                if speaking != session.is_speaking:
                    session.is_speaking = speaking
                    await _safe_send_json(websocket, {
                        "type": "vad_status",
                        "speaking": speaking,
                        "rms": rms,
                    }, session)

            # ── User finished speaking → start processing in background ──────
            elif msg_type == "audio_end":
                # Increment turn sequence counter so any trailing in-flight chunks are dropped
                session.current_turn_seq += 1

                # Cancel any in-flight processing from a previous turn
                if session.current_task and not session.current_task.done():
                    session.interrupted = True
                    session.current_task.cancel()
                    try:
                        await session.current_task
                    except (asyncio.CancelledError, Exception):
                        pass

                audio_format = msg.get("format", "webm")
                turn_id = session.new_turn()

                voice_log("audio_end_received", session_id=session_id,
                          turn_id=turn_id, chunk_count=len(session.audio_chunks))

                # Launch processing as a background task so the receive loop
                # stays responsive to interrupt / ping / new audio
                session.current_task = asyncio.create_task(
                    _process_voice_turn(session, audio_format)
                )

            # ── Interrupt signal ──────────────────────────────────────────────
            elif msg_type == "interrupt":
                session.interrupted = True
                session.audio_chunks = []
                if session.current_task and not session.current_task.done():
                    session.current_task.cancel()
                    try:
                        await session.current_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await _safe_send_json(websocket, {"type": "interrupted"}, session)

            # ── Keepalive ping ────────────────────────────────────────────────
            elif msg_type == "ping":
                await _safe_send_json(websocket, {"type": "pong"}, session)

    except WebSocketDisconnect:
        logger.info(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        logger.error(f"[WS] Session error: {exc}")
        traceback.print_exc()
        await _safe_send_json(websocket, {"type": "error", "message": str(exc)}, session)
    finally:
        session._closed = True
        # Cancel any in-flight processing
        if session.current_task and not session.current_task.done():
            session.current_task.cancel()
            try:
                await session.current_task
            except (asyncio.CancelledError, Exception):
                pass
        # Cancel heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        _active_voice_sessions.pop(session_id, None)
        logger.info(f"[WS] Session cleaned up: {session_id}")
