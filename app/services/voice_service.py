"""
app.services.voice_service
~~~~~~~~~~~~~~~~~~~~~~~~~~
Voice-turn pipeline (STT → orchestrator stream → TTS),
heartbeat, and task cancellation for the WebSocket voice channel.
"""
import asyncio
import logging
import time

from app import monitoring
from app.audio import AudioTooShortError, audio_processor, voice_log
from app.config import settings
from app.core.orchestration import route_and_stream
from app.core.text_cleaning import clean_for_tts
from app.services.voice_session import VoiceSession

logger = logging.getLogger(__name__)


async def cancel_current_task(session: VoiceSession) -> None:
    """Cancel the in-flight turn (if any) and wait for it to unwind."""
    task = session.current_task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def heartbeat_loop(session: VoiceSession, interval: float = 15.0) -> None:
    """Send a pong every `interval` seconds to keep the socket alive through proxies."""
    try:
        while not session._closed:
            await asyncio.sleep(interval)
            if session._closed:
                break
            if not await session.send_json({"type": "pong"}):
                break
    except asyncio.CancelledError:
        pass


async def process_turn(session: VoiceSession, audio_format: str) -> None:
    """
    Full voice turn: STT → orchestrate → LLM stream → TTS → send audio.

    Runs as a background task so the receive loop stays responsive to
    interrupt / ping / new-audio messages.
    """
    turn_id = session.turn_id
    turn_start = time.time()

    chunks = session.audio_chunks[:]
    session.audio_chunks = []

    if not chunks:
        await session.send_json({"type": "error", "message": "No audio received"})
        return

    voice_log("turn_start", session_id=session.session_id, turn_id=turn_id, chunk_count=len(chunks))

    # ── STT ──────────────────────────────────────────────────────────────────
    await session.send_json({"type": "status", "status": "Transcribing..."})
    stt_start = time.time()

    try:
        transcript = await audio_processor.transcribe_chunks(chunks, audio_format)
    except AudioTooShortError:
        await session.send_json({"type": "status", "status": "No speech detected"})
        await session.send_json({"type": "ai_done"})
        return
    except Exception as exc:
        await session.send_json({"type": "error", "message": f"Transcription failed: {exc}"})
        return

    stt_ms = round((time.time() - stt_start) * 1000, 1)
    voice_log("stt_done", turn_id=turn_id, latency_ms=stt_ms, transcript=transcript[:60])

    if not transcript.strip():
        await session.send_json({"type": "status", "status": "No speech detected"})
        await session.send_json({"type": "ai_done"})
        return

    if not await session.send_json({"type": "transcript", "text": transcript, "final": True}):
        return

    await session.send_json({"type": "thought", "text": "Now retrieving information…"})

    try:
        monitoring.record_tokens(
            model=settings.GROQ_WHISPER_MODEL,
            prompt_tokens=100,
            completion_tokens=len(transcript) // 4,
            ttft_ms=round(stt_ms, 2),
        )
    except Exception:
        pass

    if session.interrupted:
        voice_log("turn_interrupted_after_stt", turn_id=turn_id)
        return

    # ── LLM streaming ────────────────────────────────────────────────────────
    await session.send_json({"type": "thought", "text": "Generating answer…"})
    await session.send_json({"type": "ai_start"})

    session.ai_streaming = True
    full_reply = ""
    llm_start = time.time()
    first_token_time = None

    try:
        async for token in route_and_stream(transcript, session.session_id, "ws_user", include_images=False):
            if session.interrupted:
                voice_log("turn_interrupted_during_llm", turn_id=turn_id, tokens_so_far=len(full_reply))
                break
            full_reply += token
            if first_token_time is None:
                first_token_time = time.time()
            await session.send_json({"type": "ai_token", "token": token, "done": False})
    except asyncio.CancelledError:
        voice_log("turn_cancelled_during_llm", turn_id=turn_id)
        raise
    except Exception as exc:
        await session.send_json({"type": "error", "message": f"Agent error: {exc}"})
        session.ai_streaming = False
        await session.send_json({"type": "ai_done"})
        return

    llm_ms = round((time.time() - llm_start) * 1000, 1)
    ttft_ms = round((first_token_time - llm_start) * 1000, 1) if first_token_time else None
    voice_log("llm_done", turn_id=turn_id, latency_ms=llm_ms, ttft_ms=ttft_ms, reply_len=len(full_reply))

    # ── TTS (sentence-chunked, progressive streaming) ────────────────────────
    tts_text = clean_for_tts(full_reply)
    if not session.interrupted and tts_text:
        await session.send_json({"type": "status", "status": "Synthesizing voice..."})
        tts_start = time.time()
        audio_bytes_total = 0
        chunk_idx = 0

        try:
            async for audio_chunk in audio_processor.synthesize_speech_chunked(tts_text, voice="auto"):
                if session.interrupted or session._closed:
                    voice_log("tts_interrupted", turn_id=turn_id)
                    break
                audio_bytes_total += len(audio_chunk)
                # One binary frame per TTS batch so the client can start playback immediately.
                if not await session.send_bytes(audio_chunk):
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

    # ── Finalize ─────────────────────────────────────────────────────────────
    session.ai_streaming = False
    await session.send_json({"type": "ai_done"})

    total_ms = round((time.time() - turn_start) * 1000, 1)
    voice_log("turn_complete", turn_id=turn_id, total_ms=total_ms, stt_ms=stt_ms,
              llm_ms=llm_ms, reply_len=len(full_reply))
    print(f"[VOICE] Turn {turn_id} complete: STT={stt_ms}ms LLM={llm_ms}ms total={total_ms}ms")
