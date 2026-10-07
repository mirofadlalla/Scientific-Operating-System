"""
app.services.voice_service
~~~~~~~~~~~~~~~~~~~~~~~~~~
Voice-turn pipeline (STT → orchestrator stream → TTS),
heartbeat, and task cancellation for the WebSocket voice channel.

TTS is now pipelined with LLM streaming: sentence chunks are synthesized
concurrently with ongoing token generation, so the client hears the first
audio well before the LLM has finished producing the full reply.

Pipeline
--------
LLM stream                   TTS worker
──────────────────────────   ─────────────────────────────────────────
token → token_buf            ← waits on _tts_queue
when sentence boundary        dequeues cleaned chunk
  + len ≥ MIN_FLUSH_CHARS  →  synthesize_speech(chunk) → send_bytes
flush chunk to _tts_queue     loop until sentinel (None) received
after LLM done: flush tail
  + enqueue sentinel (None)
"""
import asyncio
import logging
import re
import time
from contextlib import aclosing

from app import monitoring
from app.audio import AudioTooShortError, _transliterate_for_arabic_tts, audio_processor, voice_log
from app.audio.segmentation import split_for_orpheus
from app.config import settings
from app.core.orchestration import route_and_stream
from app.core.text_cleaning import clean_for_tts
from app.services.voice_session import VoiceSession

logger = logging.getLogger(__name__)

# ── Sentence-flush tuning constants ──────────────────────────────────────────
# Flush a TTS chunk when the buffer ends at a sentence boundary AND has at
# least this many characters.  Smaller → lower TTFA (time-to-first-audio)
# but more API round-trips.  250 chars ≈ 2-3 sentences, ~1.5 s of speech.
MIN_FLUSH_CHARS = 120

# Regex: ends with sentence-terminal punctuation + whitespace or end-of-string
_SENTENCE_END_RE = re.compile(r'[.!?\u060C\u061F]\s*$')


FIRST_FLUSH_CHARS = 40  # first chunk goes out early so speech starts quickly


def _should_flush(buf: str, first: bool = False) -> bool:
    """True when buf ends at a sentence boundary and is long enough.

    The very first chunk of a reply uses a smaller threshold (lower time-to-first-audio).
    """
    limit = FIRST_FLUSH_CHARS if first else MIN_FLUSH_CHARS
    return len(buf) >= limit and bool(_SENTENCE_END_RE.search(buf))


async def _send_status(session: VoiceSession, status: str) -> bool:
    """UI-only status line (shown as text, never synthesized)."""
    return await session.send_json({"type": "status", "status": status, "speak": False})


async def _send_thought(session: VoiceSession, text: str) -> bool:
    """UI-only progress line such as "Generating answer…" (never synthesized)."""
    return await session.send_json({"type": "thought", "text": text, "speak": False})


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

    TTS is pipelined: sentence chunks are synthesized concurrently with the
    ongoing LLM token stream, drastically reducing time-to-first-audio.
    """
    turn_id = session.turn_id
    turn_start = time.time()

    chunks = session.audio_chunks[:]
    session.audio_chunks = []

    if not chunks:
        await _send_status(session, "No speech detected")
        await session.send_json({"type": "ai_done"})
        return

    voice_log("turn_start", session_id=session.session_id, turn_id=turn_id, chunk_count=len(chunks))

    # ── STT ──────────────────────────────────────────────────────────────────
    await _send_status(session, "Transcribing...")
    stt_start = time.time()

    try:
        transcript = await audio_processor.transcribe_chunks(chunks, audio_format)
    except AudioTooShortError:
        await _send_status(session, "No speech detected")
        await session.send_json({"type": "ai_done"})
        return
    except Exception as exc:
        await session.send_json({"type": "error", "message": f"Transcription failed: {exc}"})
        return

    stt_ms = round((time.time() - stt_start) * 1000, 1)
    voice_log("stt_done", turn_id=turn_id, latency_ms=stt_ms, transcript=transcript[:60])

    if not transcript.strip():
        await _send_status(session, "No speech detected")
        await session.send_json({"type": "ai_done"})
        return

    if not await session.send_json({"type": "transcript", "text": transcript, "final": True}):
        return

    await _send_thought(session, "Now retrieving information…")

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

    # ── LLM streaming + concurrent TTS pipeline ──────────────────────────────
    await _send_thought(session, "Generating answer…")
    await session.send_json({"type": "ai_start"})

    session.ai_streaming = True
    full_reply = ""
    llm_start = time.time()
    first_token_time = None

    # Bounded queue: the TTS worker reads chunks from here.
    # Sentinel value None signals the worker to stop.
    _tts_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)

    audio_bytes_total = 0
    chunk_idx = 0
    tts_start = time.time()
    first_audio_sent = False

    # ── TTS worker coroutine (runs concurrently with LLM streaming) ──────────
    async def _drain_queue() -> None:
        """Consume the queue up to the sentinel so the producer never blocks on put()."""
        while await _tts_queue.get() is not None:
            pass

    async def _tts_worker() -> None:
        nonlocal audio_bytes_total, chunk_idx, first_audio_sent
        tts_error_sent = False

        while True:
            chunk = await _tts_queue.get()
            if chunk is None:           # sentinel — LLM is done
                break

            if session.interrupted or session._closed:
                voice_log("tts_interrupted", turn_id=turn_id)
                await _drain_queue()
                return

            clean = clean_for_tts(chunk)
            if not clean:
                continue

            if not first_audio_sent:
                # First audio is ready — tell the client
                await _send_status(session, "Speaking…")
                first_audio_sent = True

            try:
                is_arabic = bool(re.search(r'[\u0600-\u06FF]', clean))
                tts_input = _transliterate_for_arabic_tts(clean) if is_arabic else clean
                # Groq Orpheus accepts max 200 chars per request -> sub-split.
                for piece in split_for_orpheus(tts_input):
                    audio = await audio_processor.synthesize_speech(piece, voice="auto")
                    if session.interrupted or session._closed:
                        voice_log("tts_interrupted", turn_id=turn_id)
                        await _drain_queue()
                        return
                    audio_bytes_total += len(audio)
                    if not await session.send_bytes(audio):
                        await _drain_queue()
                        return
                    chunk_idx += 1
                    voice_log("tts_chunk_sent", turn_id=turn_id, chunk_idx=chunk_idx,
                              text_len=len(piece), audio_bytes=len(audio))
            except asyncio.CancelledError:
                raise
            except Exception as tts_err:
                logger.error(f"[WS TTS Error] turn={turn_id} chunk={chunk_idx}: {tts_err}")
                if not tts_error_sent:
                    tts_error_sent = True
                    await session.send_json({"type": "error", "message": f"TTS failed: {tts_err}"})

    # ── LLM producer coroutine ────────────────────────────────────────────────
    async def _llm_producer() -> None:
        nonlocal full_reply, first_token_time

        token_buf = ""
        flushed_any = False
        try:
            # aclosing(): leaving the loop early (interrupt / cancel) closes the
            # generator immediately, which stops the upstream LLM stream instead of
            # leaving it running (and billing tokens) until garbage collection.
            async with aclosing(route_and_stream(
                transcript, session.session_id, "ws_user", include_images=False
            )) as reply_stream:
                async for token in reply_stream:
                    if session.interrupted:
                        voice_log("turn_interrupted_during_llm", turn_id=turn_id,
                                  tokens_so_far=len(full_reply))
                        break

                    full_reply += token
                    token_buf  += token

                    if first_token_time is None:
                        first_token_time = time.time()

                    # Stream token to UI immediately (raw markdown — not cleaned)
                    await session.send_json({"type": "ai_token", "token": token, "done": False})

                    # Flush when we have a complete sentence of sufficient length
                    if _should_flush(token_buf, first=not flushed_any):
                        await _tts_queue.put(token_buf)
                        token_buf = ""
                        flushed_any = True

        except asyncio.CancelledError:
            voice_log("turn_cancelled_during_llm", turn_id=turn_id)
            raise
        except Exception as exc:
            await session.send_json({"type": "error", "message": f"Agent error: {exc}"})

        # Flush whatever remains in the buffer (tail of the response) — unless the
        # user interrupted, in which case the tail must not be spoken.
        if token_buf.strip() and not session.interrupted:
            await _tts_queue.put(token_buf)

        # Signal worker that the stream is exhausted
        await _tts_queue.put(None)

    # ── Run both concurrently ─────────────────────────────────────────────────
    try:
        await asyncio.gather(_llm_producer(), _tts_worker())
    except asyncio.CancelledError:
        voice_log("turn_cancelled_during_pipeline", turn_id=turn_id)
        raise

    llm_ms  = round((time.time() - llm_start)  * 1000, 1)
    tts_ms  = round((time.time() - tts_start)   * 1000, 1)
    ttft_ms = round((first_token_time - llm_start) * 1000, 1) if first_token_time else None

    voice_log("llm_done",  turn_id=turn_id, latency_ms=llm_ms,
              ttft_ms=ttft_ms, reply_len=len(full_reply))
    voice_log("tts_send_done", turn_id=turn_id, latency_ms=tts_ms,
              audio_bytes=audio_bytes_total, chunks_sent=chunk_idx)

    try:
        monitoring.record_agent_call(
            agent="VOICE",
            intent="VOICE_TURN",
            latency_ms=round((time.time() - llm_start) * 1000, 2),
            success=True,
        )
        monitoring.record_tokens(
            model=settings.REASONING_MODEL,
            prompt_tokens=len(transcript) // 4,
            completion_tokens=len(full_reply) // 4,
            ttft_ms=ttft_ms,
        )
    except Exception:
        pass

    # ── Finalize ─────────────────────────────────────────────────────────────
    session.ai_streaming = False
    await session.send_json({"type": "ai_done"})

    total_ms = round((time.time() - turn_start) * 1000, 1)
    voice_log("turn_complete", turn_id=turn_id, total_ms=total_ms, stt_ms=stt_ms,
              llm_ms=llm_ms, reply_len=len(full_reply))
    print(f"[VOICE] Turn {turn_id} complete: STT={stt_ms}ms LLM={llm_ms}ms "
          f"TTS_chunks={chunk_idx} total={total_ms}ms")
