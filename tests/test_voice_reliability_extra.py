"""
Extra regression coverage for the voice-channel reliability fixes. Complements
tests/test_reliability_regressions.py (does not replace it):

* sibling-task leak when the TTS worker dies while the LLM producer is blocked
  on the bounded queue (needs a LONG reply to reproduce),
* exactly one ai_done on success / empty-transcript / failure paths,
* ordering error -> ai_done over a real WebSocket round trip,
* EBML bytes inside FLAC / MP3 payloads, stray-RIFF-before-webm still sliced,
* Whisper really receives the untouched WAV bytes.
"""
import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.audio.formats import align_container_header, detect_format
from app.services import voice_service
from app.services.voice_session import VoiceSession

EBML = b"\x1aE\xdf\xa3"


# ── align_container_header ───────────────────────────────────────────────────

@pytest.mark.parametrize("prefix,fmt", [
    (b"fLaC\x00\x00", "flac"),
    (b"ID3\x03\x00", "mp3"),
    (b"\xff\xfb\x90\x00", "mp3"),
])
def test_flac_and_mp3_payloads_with_embedded_ebml_are_not_sliced(prefix, fmt):
    audio = prefix + b"\x00" * 200 + EBML + b"\x00" * 200
    assert detect_format(audio) == fmt
    assert align_container_header(audio, fmt) == audio
    assert align_container_header(audio, "webm") == audio


def test_stray_prefix_before_riff_with_webm_label_is_still_sliced():
    wav = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVEfmt " + b"\x00" * 40
    assert align_container_header(b"xx" + wav, "webm") == wav


def test_wav_pcm_that_contains_ebml_bytes_reaches_whisper_intact():
    from app.audio import audio_processor

    pcm = b"\x01\x02" * 500 + EBML + b"\x03\x04" * 500
    wav = (b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
           + b"\x01\x00\x01\x00" + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
           + b"\x02\x00\x10\x00" + b"data" + len(pcm).to_bytes(4, "little") + pcm)
    seen = {}

    async def fake_create(**kwargs):
        seen["bytes"] = kwargs["file"].read()
        seen["name"] = kwargs["file"].name
        return {"text": "hello world", "segments": []}

    with patch.object(audio_processor.groq_client.audio.transcriptions, "create", fake_create):
        text = asyncio.run(audio_processor.transcribe_audio(wav, "wav"))
    assert text == "hello world"
    assert seen["bytes"] == wav and seen["name"].endswith(".wav")


# ── process_turn ─────────────────────────────────────────────────────────────

class CaptureWS:
    client_state = None

    def __init__(self):
        self.sent_text, self.sent_bytes = [], []

    async def send_text(self, data):
        self.sent_text.append(data)

    async def send_bytes(self, data):
        self.sent_bytes.append(data)

    def types(self):
        return [json.loads(m)["type"] for m in self.sent_text]


def _reply(n):
    async def gen(*a, **k):
        for i in range(n):
            yield f"This is sentence number {i} of a fairly long spoken answer. "
    return gen


@pytest.fixture
def stubs(monkeypatch):
    async def fake_transcribe(chunks, fmt):
        return "what is aspirin"

    async def fake_tts(text, voice="auto"):
        return b"RIFFfake"

    monkeypatch.setattr(voice_service.audio_processor, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(voice_service.audio_processor, "synthesize_speech", fake_tts)
    monkeypatch.setattr(voice_service, "route_and_stream", _reply(3))


def _session(ws, sid):
    s = VoiceSession(ws, sid)
    s.audio_chunks = [b"x" * 3000]
    s.new_turn()
    return s


def test_dead_tts_worker_does_not_leak_the_blocked_llm_producer(stubs, monkeypatch):
    """60 long sentences fill the bounded queue (maxsize 8): without cancelling the
    sibling, the producer stays blocked on put() forever after the worker dies."""
    monkeypatch.setattr(voice_service, "route_and_stream", _reply(60))
    monkeypatch.setattr(voice_service, "clean_for_tts",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    ws = CaptureWS()
    session = _session(ws, "t_leak")

    async def run():
        await voice_service.process_turn(session, "wav")
        await asyncio.sleep(0.05)
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    assert asyncio.run(run()) == []
    types = ws.types()
    assert "error" in types and types[-1] == "ai_done" and types.count("ai_done") == 1


def test_stt_failure_ends_with_error_then_exactly_one_ai_done(stubs, monkeypatch):
    async def bad_stt(chunks, fmt):
        raise ValueError("Transcription failed: nope")

    monkeypatch.setattr(voice_service.audio_processor, "transcribe_chunks", bad_stt)
    ws = CaptureWS()
    asyncio.run(voice_service.process_turn(_session(ws, "t_stt"), "wav"))
    types = ws.types()
    assert types.index("error") < types.index("ai_done") and types.count("ai_done") == 1


def test_successful_turn_sends_exactly_one_ai_done(stubs):
    ws = CaptureWS()
    asyncio.run(voice_service.process_turn(_session(ws, "t_ok"), "wav"))
    types = ws.types()
    assert types.count("ai_done") == 1 and "error" not in types and types[-1] == "ai_done"
    assert ws.sent_bytes


def test_empty_transcript_sends_exactly_one_ai_done(stubs, monkeypatch):
    async def empty(chunks, fmt):
        return "   "

    monkeypatch.setattr(voice_service.audio_processor, "transcribe_chunks", empty)
    ws = CaptureWS()
    asyncio.run(voice_service.process_turn(_session(ws, "t_empty"), "wav"))
    assert ws.types().count("ai_done") == 1


def test_websocket_round_trip_error_precedes_ai_done(stubs, monkeypatch):
    from app.main import app

    monkeypatch.setattr(voice_service, "clean_for_tts",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    with TestClient(app).websocket_connect("/api/v1/ws/voice?session_id=t_extra_e2e") as ws:
        ws.send_text(json.dumps({"type": "audio_chunk", "format": "wav",
                                 "data": base64.b64encode(b"x" * 3000).decode()}))
        ws.send_text(json.dumps({"type": "audio_end", "format": "wav"}))
        seen = []
        while True:
            seen.append(json.loads(ws.receive()["text"])["type"])
            if seen[-1] == "ai_done":
                break
            assert len(seen) < 60, seen
    assert "error" in seen and seen.index("error") < seen.index("ai_done")
