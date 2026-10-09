"""
Regression tests for three confirmed voice-channel reliability bugs:

1. process_turn: an unexpected exception mid-turn left the client without
   "ai_done" (UI stuck on "Processing response…", mic never re-armed) and could
   leak the sibling gather() task. CancelledError must still propagate.
2. align_container_header: scanning the whole payload for the WebM EBML magic
   could slice a valid WAV whose PCM samples happened to contain those 4 bytes.
3. handle_voice_channel cleanup: an old connection with the same session_id
   popped the NEW connection's entry from state.active_voice_sessions.
"""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocketDisconnect  # noqa: F401  (documented dependency of the controller)

import app.core.state as state
from app.audio.formats import align_container_header, detect_format
from app.controllers import voice_controller as vc
from app.services import voice_service
from app.services.voice_session import VoiceSession

EBML = b"\x1aE\xdf\xa3"


# ── helpers ──────────────────────────────────────────────────────────────────

class FakeWS:
    client_state = None

    def __init__(self):
        self.sent = []

    async def send_text(self, data):
        self.sent.append(json.loads(data))

    async def send_bytes(self, data):
        self.sent.append({"type": "<bytes>"})

    def types(self):
        return [m["type"] for m in self.sent]


async def _tokens(*_a, **_k):
    for t in ["Hello there, this is a fairly long first sentence for tts. ", "Second one. "]:
        yield t


async def _stalls_after_first_token(*_a, **_k):
    yield "Hello there, this is a fairly long first sentence for tts. "
    await asyncio.sleep(60)


def _new_session(ws):
    s = VoiceSession(ws, "t_reliability")
    s.audio_chunks = [b"x" * 3000]
    s.new_turn()
    return s


WAV_HEADER = b"RIFF" + (0).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 24


# ── 1. process_turn ──────────────────────────────────────────────────────────

def test_unexpected_turn_exception_sends_error_then_ai_done_and_leaks_nothing():
    async def run():
        ws = FakeWS()
        session = _new_session(ws)
        with patch("app.services.voice_service.audio_processor.transcribe_chunks",
                   AsyncMock(return_value="what is aspirin")), \
             patch("app.services.voice_service.route_and_stream", _tokens), \
             patch("app.services.voice_service.audio_processor.synthesize_speech",
                   AsyncMock(return_value=b"RIFFxxxxWAVE" + b"0" * 64)), \
             patch("app.services.voice_service.clean_for_tts", side_effect=RuntimeError("boom")):
            await voice_service.process_turn(session, "wav")   # must NOT raise

        types = ws.types()
        assert "error" in types and types[-1] == "ai_done"
        assert types.index("error") < len(types) - 1
        assert session.ai_streaming is False
        # the llm-producer sibling must not be left blocked on the bounded queue
        await asyncio.sleep(0)
        leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert leftovers == []

    asyncio.run(run())


def test_transcription_failure_sends_error_then_ai_done():
    async def run():
        ws = FakeWS()
        session = _new_session(ws)
        with patch("app.services.voice_service.audio_processor.transcribe_chunks",
                   AsyncMock(side_effect=ValueError("Transcription failed: 400"))):
            await voice_service.process_turn(session, "wav")
        types = ws.types()
        assert "error" in types and types[-1] == "ai_done"

    asyncio.run(run())


def test_cancellation_still_propagates_and_sends_no_ai_done():
    async def run():
        ws = FakeWS()
        session = _new_session(ws)
        with patch("app.services.voice_service.audio_processor.transcribe_chunks",
                   AsyncMock(return_value="what is aspirin")), \
             patch("app.services.voice_service.route_and_stream", _stalls_after_first_token), \
             patch("app.services.voice_service.audio_processor.synthesize_speech",
                   AsyncMock(return_value=b"RIFFxxxxWAVE" + b"0" * 64)):
            task = asyncio.create_task(voice_service.process_turn(session, "wav"))
            for _ in range(200):                       # wait until streaming started
                if "ai_token" in ws.types():
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert "ai_done" not in ws.types()
        assert "error" not in ws.types()

    asyncio.run(run())


# ── 2. align_container_header ────────────────────────────────────────────────

@pytest.mark.parametrize("hint", ["wav", "webm"])
def test_wav_payload_containing_ebml_bytes_is_not_sliced(hint):
    wav = WAV_HEADER + b"\x01\x02" * 50 + EBML + b"\x03\x04" * 50
    assert detect_format(wav) == "wav"
    assert align_container_header(wav, hint) == wav


def test_ogg_and_mp4_payloads_containing_ebml_bytes_are_not_sliced():
    ogg = b"OggS" + b"\x00" * 20 + EBML + b"tail"
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20 + EBML + b"tail"
    assert align_container_header(ogg, "webm") == ogg
    assert align_container_header(mp4, "mp4") == mp4


def test_genuine_stray_prefix_before_ebml_is_still_stripped():
    assert align_container_header(b"junk" + EBML + b"data", "webm") == EBML + b"data"


# ── 3. active_voice_sessions cleanup race ────────────────────────────────────

class LiveFakeWS:
    """Just enough WebSocket for handle_voice_channel: accept/receive/send."""
    client_state = None

    def __init__(self):
        self.q = asyncio.Queue()

    async def accept(self):
        pass

    async def receive(self):
        return await self.q.get()

    async def send_text(self, data):
        pass

    async def send_bytes(self, data):
        pass

    def disconnect(self):
        self.q.put_nowait({"type": "websocket.disconnect", "code": 1000})


def test_old_connection_cleanup_does_not_remove_newer_session():
    async def run():
        state.active_voice_sessions.pop("dup", None)
        ws1, ws2 = LiveFakeWS(), LiveFakeWS()
        h1 = asyncio.create_task(vc.handle_voice_channel(ws1, "dup"))
        await asyncio.sleep(0.05)
        h2 = asyncio.create_task(vc.handle_voice_channel(ws2, "dup"))
        await asyncio.sleep(0.05)
        newer = state.active_voice_sessions["dup"]
        assert newer.ws is ws2

        ws1.disconnect()                     # the OLD socket goes away
        await asyncio.wait_for(h1, 5)
        assert state.active_voice_sessions.get("dup") is newer   # race: used to be None

        ws2.disconnect()
        await asyncio.wait_for(h2, 5)
        assert "dup" not in state.active_voice_sessions           # normal cleanup intact

    asyncio.run(run())
