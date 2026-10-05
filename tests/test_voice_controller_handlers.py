"""
Unit tests for the WebSocket voice-channel message handlers.

Regression coverage for the spurious "Interrupted" bug: mic audio chunks that
arrive while the AI is replying (e.g. speaker echo) must NOT cancel the answer.
Real barge-in is an explicit {"type": "interrupt"} message or a finished
utterance ("audio_end").
"""
import asyncio
import base64
import logging

from app.controllers import voice_controller as vc
from app.services.voice_session import VoiceSession


class FakeWS:
    client_state = None

    def __init__(self):
        self.sent = []

    async def send_text(self, data):
        self.sent.append(data)

    async def send_bytes(self, data):
        pass


def _chunk(payload: bytes = b"abc") -> dict:
    return {"type": "audio_chunk", "data": base64.b64encode(payload).decode()}


def test_idle_audio_chunk_is_buffered():
    async def run():
        session = VoiceSession(FakeWS(), "t_idle")
        await vc._on_audio_chunk(session, _chunk())
        assert len(session.audio_chunks) == 1

    asyncio.run(run())


def test_audio_chunk_during_turn_does_not_interrupt():
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_turn")
        session.current_task = asyncio.create_task(asyncio.sleep(30))
        session.ai_streaming = True
        await asyncio.sleep(0)

        await vc._on_audio_chunk(session, _chunk())

        assert session.audio_chunks == []            # dropped, not buffered
        assert not session.interrupted
        assert not session.current_task.done()       # answer keeps going
        assert ws.sent == []                         # no "interrupted" sent
        session.current_task.cancel()

    asyncio.run(run())


def test_explicit_interrupt_cancels_and_then_accepts_chunks():
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_int")
        session.current_task = asyncio.create_task(asyncio.sleep(30))
        session.ai_streaming = True
        await asyncio.sleep(0)

        await vc._on_interrupt(session, {"type": "interrupt"})
        assert session.interrupted and session.current_task.cancelled()
        assert any('"interrupted"' in m for m in ws.sent)

        await vc._on_audio_chunk(session, _chunk())
        assert len(session.audio_chunks) == 1        # new utterance is buffered

    asyncio.run(run())


def test_client_info_logs_which_vad_is_used(caplog):
    async def run():
        session = VoiceSession(FakeWS(), "t_vad")
        await vc._on_client_info(session, {"vad": "silero", "source": "local", "model": "v5"})
        assert session.client_vad == "silero"
        await vc._on_client_info(session, {"vad": "energy", "reason": "local: failed to fetch"})
        assert session.client_vad == "energy"

    with caplog.at_level(logging.INFO, logger=vc.logger.name):
        asyncio.run(run())
    text = caplog.text
    assert "client VAD=SILERO" in text
    assert "client VAD=ENERGY-FALLBACK" in text and "failed to fetch" in text
