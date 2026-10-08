"""
Unit tests for the WebSocket voice-channel message handlers.

Regression coverage for the spurious "Interrupted" bug: mic audio chunks that
arrive while the AI is replying (e.g. speaker echo) must NOT cancel the answer.
Real barge-in is an explicit {"type": "interrupt"} message. A bare "audio_end"
while an answer is being produced is echo/noise and must not cancel it either.
"""
import asyncio
import base64
import json
import logging

from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

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


def test_client_info_sends_vad_status_ack():
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_vad_ack")
        await vc._on_client_info(session, {"vad": "silero", "source": "local", "model": "v5"})
        sent = [json.loads(m) for m in ws.sent]
        assert sent == [{
            "type": "vad_status",
            "vad": "silero",
            "ok": True,
            "speaking": False,
        }]

    asyncio.run(run())


class SequenceWS(FakeWS):
    def __init__(self, frames):
        super().__init__()
        self._frames = list(frames)
        self.client_state = WebSocketState.CONNECTED

    async def accept(self):
        pass

    async def receive(self):
        if not self._frames:
            self.client_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect()
        return self._frames.pop(0)


def test_invalid_json_payload_does_not_crash_voice_loop(caplog, monkeypatch):
    async def noop_heartbeat(session, interval=15.0):
        return

    monkeypatch.setattr(vc.voice_service, "heartbeat_loop", noop_heartbeat)

    async def run():
        ws = SequenceWS([
            {"text": "not-json{"},
            {"text": json.dumps({"type": "ping"})},
        ])
        await vc.handle_voice_channel(ws, "t_bad_json")
        sent = [json.loads(m) for m in ws.sent]
        assert any(m.get("type") == "pong" for m in sent)

    with caplog.at_level(logging.WARNING, logger=vc.logger.name):
        asyncio.run(run())
    assert "invalid JSON" in caplog.text


def test_audio_end_during_turn_does_not_cancel_the_answer():
    """Regression: echo-triggered audio_end used to cancel the turn and then fail
    with "No audio received" although the user never spoke."""
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_end_turn")
        session.current_task = asyncio.create_task(asyncio.sleep(30))
        session.ai_streaming = True
        session.audio_chunks = [b"stale-echo"]
        await asyncio.sleep(0)
        turn_before = session.turn_id

        await vc._on_audio_end(session, {"type": "audio_end", "format": "wav"})

        assert not session.current_task.done()       # answer keeps going
        assert not session.interrupted
        assert session.audio_chunks == []            # echo discarded
        assert session.turn_id == turn_before        # no new turn started
        assert ws.sent == []                         # no "interrupted", no error
        session.current_task.cancel()

    asyncio.run(run())


def test_audio_end_with_nothing_buffered_resumes_listening_without_error():
    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_end_empty")

        await vc._on_audio_end(session, {"type": "audio_end", "format": "wav"})

        sent = [json.loads(m) for m in ws.sent]
        assert [m["type"] for m in sent] == ["status", "ai_done"]
        assert sent[0]["status"] == "No speech detected"
        assert session.current_task is None

    asyncio.run(run())


def test_barge_in_flow_interrupt_then_utterance_starts_a_new_turn(monkeypatch):
    """interrupt -> audio_chunk -> audio_end is the one supported barge-in sequence."""
    started = []

    async def fake_process_turn(session, audio_format):
        started.append((session.turn_id, list(session.audio_chunks), audio_format))

    monkeypatch.setattr(vc.voice_service, "process_turn", fake_process_turn)

    async def run():
        ws = FakeWS()
        session = VoiceSession(ws, "t_barge")
        session.current_task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)

        await vc._on_interrupt(session, {"type": "interrupt"})
        await vc._on_audio_chunk(session, _chunk(b"hello"))
        await vc._on_audio_end(session, {"type": "audio_end", "format": "wav"})
        await session.current_task

        assert len(started) == 1
        assert started[0][1] == [b"hello"] and started[0][2] == "wav"
        assert session.interrupted is False          # reset by new_turn()

    asyncio.run(run())


def test_malformed_base64_chunk_is_dropped_not_fatal():
    async def run():
        session = VoiceSession(FakeWS(), "t_b64")
        await vc._on_audio_chunk(session, {"type": "audio_chunk", "data": "***not base64***"})
        assert session.audio_chunks == []

    asyncio.run(run())


def test_oversized_utterance_is_discarded(monkeypatch):
    monkeypatch.setattr(vc, "MAX_UTTERANCE_BYTES", 10)

    async def run():
        session = VoiceSession(FakeWS(), "t_big")
        await vc._on_audio_chunk(session, _chunk(b"12345678"))
        assert session.audio_bytes == 8
        await vc._on_audio_chunk(session, _chunk(b"12345678"))   # would exceed the cap
        assert session.audio_chunks == []

    asyncio.run(run())
