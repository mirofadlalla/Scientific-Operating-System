"""
Pipeline-level regression tests for app.services.voice_service.process_turn.

* A dead socket / interrupted TTS worker must never leave the LLM producer blocked
  on a full queue (the turn used to hang until something cancelled it).
* After an interrupt the unspoken tail of the reply must not be sent to TTS.
"""
import asyncio

import pytest

from app.services import voice_service
from app.services.voice_session import VoiceSession


class FakeWS:
    client_state = None

    def __init__(self, fail_bytes: bool = False):
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.fail_bytes = fail_bytes

    async def send_text(self, data):
        self.sent_text.append(data)

    async def send_bytes(self, data):
        if self.fail_bytes:
            raise RuntimeError("socket closed")
        self.sent_bytes.append(data)


def _long_reply(sentences: int):
    async def gen(*args, **kwargs):
        for i in range(sentences):
            yield f"This is sentence number {i} of a fairly long spoken answer. "
    return gen


@pytest.fixture
def patched(monkeypatch):
    calls = {"tts": 0}

    async def fake_transcribe(chunks, fmt):
        return "what is aspirin"

    async def fake_tts(text, voice="auto"):
        calls["tts"] += 1
        return b"RIFFfake"

    monkeypatch.setattr(voice_service.audio_processor, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(voice_service.audio_processor, "synthesize_speech", fake_tts)
    return calls


def test_dead_socket_does_not_deadlock_the_llm_producer(monkeypatch, patched):
    monkeypatch.setattr(voice_service, "route_and_stream", _long_reply(60))

    async def run():
        session = VoiceSession(FakeWS(fail_bytes=True), "t_dead")
        session.audio_chunks = [b"x" * 2048]
        session.new_turn()
        await asyncio.wait_for(voice_service.process_turn(session, "wav"), timeout=5)

    asyncio.run(run())


def test_interrupt_during_tts_stops_further_synthesis(monkeypatch, patched):
    monkeypatch.setattr(voice_service, "route_and_stream", _long_reply(60))

    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_int_tts")
        session.audio_chunks = [b"x" * 2048]
        session.new_turn()

        real_tts = voice_service.audio_processor.synthesize_speech

        async def tts_then_interrupt(text, voice="auto"):
            audio = await real_tts(text, voice)
            session.interrupted = True          # user barges in after the first clip
            return audio

        monkeypatch.setattr(voice_service.audio_processor, "synthesize_speech", tts_then_interrupt)
        await asyncio.wait_for(voice_service.process_turn(session, "wav"), timeout=5)

        assert patched["tts"] == 1              # nothing spoken after the interrupt
        assert len(ws.sent_bytes) == 0          # the interrupted clip itself is not sent

    asyncio.run(run())


def test_empty_buffer_reports_no_speech(patched):
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_empty")
        await voice_service.process_turn(session, "wav")
        assert any('"No speech detected"' in m for m in ws.sent_text)
        assert any('"ai_done"' in m for m in ws.sent_text)
        assert not any('"error"' in m for m in ws.sent_text)

    asyncio.run(run())
