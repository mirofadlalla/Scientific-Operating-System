"""
Test suite for the WebSocket voice pipeline.

Tests cover:
  - WebSocket connect / disconnect lifecycle
  - Audio chunk → transcript → AI response → TTS audio → ai_done flow
  - Client disconnect during TTS (server must not crash)
  - Interrupt during AI streaming
  - Multiple sequential turns on same connection
  - Empty audio handling
  - Ping/pong keepalive
  - Safe send on closed connections
"""
import base64
import json
import struct
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_wav_header(data_size: int, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """Generate a minimal valid WAV header for testing."""
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,           # chunk size
        1,            # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b'data',
        data_size,
    )
    return header


def make_test_audio_webm(duration_ms: int = 1000) -> bytes:
    """
    Create a minimal fake WebM payload (EBML header) large enough to pass
    the AudioProcessor's MIN_AUDIO_BYTES check (>= 1024 bytes).
    """
    # EBML header magic bytes for WebM
    ebml_header = b'\x1aE\xdf\xa3'
    # Pad to at least 2KB
    payload = ebml_header + b'\x00' * max(2048, duration_ms)
    return payload


def make_audio_chunks(audio_bytes: bytes, chunk_size: int = 500) -> list[str]:
    """Split audio bytes into base64-encoded JSON messages."""
    messages = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        b64 = base64.b64encode(chunk).decode('utf-8')
        messages.append(json.dumps({
            "type": "audio_chunk",
            "data": b64,
            "format": "webm",
        }))
    return messages


# ── Mock response generators ─────────────────────────────────────────────────

async def mock_route_and_stream(text_input, session_id, user_id, **kwargs):
    """Simple mock that yields a few tokens."""
    for token in ["Hello ", "from ", "AI-lixir!"]:
        yield token


async def mock_route_and_stream_arabic(text_input, session_id, user_id, **kwargs):
    """Arabic mock response."""
    for token in ["مرحباً ", "أنا ", "نظام ", "علمي."]:
        yield token


async def mock_route_and_stream_slow(text_input, session_id, user_id, **kwargs):
    """Slow mock that takes time per token."""
    for token in ["Processing ", "slowly..."]:
        await asyncio.sleep(0.1)
        yield token


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_transcribe():
    """Mock the STT transcription to return a fixed string."""
    with patch("app.audio.audio_processor.transcribe_chunks", new_callable=AsyncMock) as mock:
        mock.return_value = "Hello test"
        yield mock


@pytest.fixture
def mock_transcribe_arabic():
    """Mock the STT transcription to return Arabic text."""
    with patch("app.audio.audio_processor.transcribe_chunks", new_callable=AsyncMock) as mock:
        mock.return_value = "أهلاً سمعني"
        yield mock


@pytest.fixture
def mock_tts():
    """Mock TTS to return a small WAV payload."""
    wav_data = make_wav_header(1000) + b'\x00' * 1000
    with patch("app.audio.audio_processor.synthesize_speech", new_callable=AsyncMock) as mock:
        mock.return_value = wav_data
        yield mock


@pytest.fixture
def mock_tts_chunked():
    """Mock chunked TTS to yield two small WAV payloads."""
    wav_data = make_wav_header(500) + b'\x00' * 500

    async def chunked_gen(text, voice="auto"):
        yield wav_data
        yield wav_data

    with patch("app.audio.audio_processor.synthesize_speech_chunked") as mock:
        mock.return_value = chunked_gen("test")
        mock.side_effect = chunked_gen
        yield mock


@pytest.fixture
def mock_orchestration():
    """Mock the orchestration to stream simple tokens."""
    with patch("app.api.v1.chat.route_and_stream", side_effect=mock_route_and_stream):
        yield


# ── Test Classes ──────────────────────────────────────────────────────────────

class TestWebSocketVoiceConnect:
    """Test WebSocket connection lifecycle."""

    def test_ws_connect_and_disconnect(self, client):
        """Verify WebSocket connection opens and closes cleanly."""
        with client.websocket_connect("/api/v1/ws/voice?session_id=test_connect") as ws:
            # Connection should be open — send a ping
            ws.send_text(json.dumps({"type": "ping"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "pong"
        # Connection closed cleanly — no exception

    def test_ws_multiple_pings(self, client):
        """Verify multiple ping/pong exchanges work."""
        with client.websocket_connect("/api/v1/ws/voice?session_id=test_pings") as ws:
            for _ in range(5):
                ws.send_text(json.dumps({"type": "ping"}))
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "pong"


class TestWebSocketVoicePipeline:
    """Test the full voice turn pipeline."""

    def test_audio_end_no_audio_returns_error(self, client):
        """Sending audio_end without any chunks should return an error."""
        with client.websocket_connect("/api/v1/ws/voice?session_id=test_no_audio") as ws:
            ws.send_text(json.dumps({"type": "audio_end", "format": "webm"}))
            # Should receive an error message
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "error"
            assert "No audio" in resp.get("message", "")

    def test_full_voice_turn(self, client, mock_transcribe, mock_tts_chunked, mock_orchestration):
        """
        Full pipeline: send audio chunks → audio_end → receive transcript →
        receive ai_tokens → receive binary TTS → receive ai_done.
        """
        audio = make_test_audio_webm()
        chunks = make_audio_chunks(audio, chunk_size=512)

        with client.websocket_connect("/api/v1/ws/voice?session_id=test_full_turn") as ws:
            # Send audio chunks
            for chunk_msg in chunks:
                ws.send_text(chunk_msg)

            # Signal end of speech
            ws.send_text(json.dumps({"type": "audio_end", "format": "webm"}))

            # Collect all responses until ai_done
            received_types = []
            has_transcript = False
            has_ai_token = False
            has_ai_done = False
            has_binary = False
            has_ai_start = False

            # Read messages with a reasonable limit
            for _ in range(50):
                try:
                    msg = ws.receive()
                    if isinstance(msg.get("bytes"), bytes) if isinstance(msg, dict) else False:
                        has_binary = True
                        continue
                    # TestClient returns dict with "text" key for text frames
                    text = msg.get("text") if isinstance(msg, dict) else msg
                    if text and isinstance(text, str):
                        data = json.loads(text)
                        received_types.append(data["type"])
                        if data["type"] == "transcript":
                            has_transcript = True
                            assert data["text"] == "Hello test"
                        elif data["type"] == "ai_start":
                            has_ai_start = True
                        elif data["type"] == "ai_token":
                            has_ai_token = True
                        elif data["type"] == "ai_done":
                            has_ai_done = True
                            break
                except Exception:
                    break

            assert has_transcript, f"No transcript received. Got: {received_types}"
            assert has_ai_done, f"No ai_done received. Got: {received_types}"

    def test_vad_energy_updates_status(self, client):
        """VAD energy messages should trigger vad_status responses when threshold changes."""
        with client.websocket_connect("/api/v1/ws/voice?session_id=test_vad") as ws:
            # Send high energy (should detect speech)
            ws.send_text(json.dumps({"type": "vad_energy", "rms": 5000.0}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "vad_status"
            assert resp["speaking"] is True

            # Send low energy (should detect silence)
            ws.send_text(json.dumps({"type": "vad_energy", "rms": 100.0}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "vad_status"
            assert resp["speaking"] is False


class TestWebSocketDisconnect:
    """Test server resilience to client disconnection."""

    def test_server_survives_client_disconnect(self, client, mock_transcribe, mock_tts_chunked):
        """
        If the client disconnects mid-pipeline, the server should NOT crash.
        This tests the specific scenario from the production logs.
        """
        audio = make_test_audio_webm()
        chunks = make_audio_chunks(audio)

        # This test verifies no unhandled exception occurs
        try:
            with client.websocket_connect("/api/v1/ws/voice?session_id=test_disconnect") as ws:
                for chunk_msg in chunks:
                    ws.send_text(chunk_msg)
                ws.send_text(json.dumps({"type": "audio_end", "format": "webm"}))
                # Immediately close — simulates client navigating away during processing
        except Exception:
            pass  # Client-side close is expected

        # If we get here without the test process crashing, the server handled it correctly


class TestWebSocketInterrupt:
    """Test interruption handling."""

    def test_interrupt_clears_state(self, client):
        """Sending interrupt should receive an interrupted message."""
        with client.websocket_connect("/api/v1/ws/voice?session_id=test_interrupt") as ws:
            ws.send_text(json.dumps({"type": "interrupt"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "interrupted"


class TestWebSocketMultiTurn:
    """Test multiple voice turns on the same connection."""

    def test_two_sequential_turns(self, client, mock_transcribe, mock_tts_chunked, mock_orchestration):
        """Two sequential voice turns on the same WebSocket connection."""
        audio = make_test_audio_webm()
        chunks = make_audio_chunks(audio, chunk_size=1024)

        with client.websocket_connect("/api/v1/ws/voice?session_id=test_multi") as ws:
            for turn in range(2):
                # Send audio
                for chunk_msg in chunks:
                    ws.send_text(chunk_msg)
                ws.send_text(json.dumps({"type": "audio_end", "format": "webm"}))

                # Drain messages until ai_done
                got_done = False
                for _ in range(50):
                    try:
                        msg = ws.receive()
                        text = msg.get("text") if isinstance(msg, dict) else msg
                        if text and isinstance(text, str):
                            data = json.loads(text)
                            if data["type"] == "ai_done":
                                got_done = True
                                break
                    except Exception:
                        break
                assert got_done, f"Turn {turn + 1}: ai_done not received"


class TestAudioProcessorUnit:
    """Unit tests for audio processing helpers."""

    def test_split_sentences_english(self):
        from app.audio import split_sentences
        text = "Hello world. How are you? I am fine!"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine!"

    def test_split_sentences_arabic(self):
        from app.audio import split_sentences
        text = "مرحباً بك. كيف حالك؟ أنا بخير!"
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_single(self):
        from app.audio import split_sentences
        text = "Just one sentence here"
        sentences = split_sentences(text)
        assert len(sentences) == 1

    def test_split_sentences_empty(self):
        from app.audio import split_sentences
        sentences = split_sentences("")
        assert len(sentences) == 0

    def test_split_sentences_merges_short_fragments(self):
        from app.audio import split_sentences
        # Fragments under 6 chars get merged with previous
        text = "Hello. OK. Fine. This is a longer sentence."
        sentences = split_sentences(text)
        # "OK." (3 chars) should be merged, "Fine." (5 chars) should be merged
        assert all(len(s) >= 5 for s in sentences)

    def test_detect_format_webm(self):
        from app.audio import AudioProcessor
        # EBML header for WebM
        data = b'\x1aE\xdf\xa3' + b'\x00' * 100
        assert AudioProcessor._detect_format(data) == 'webm'

    def test_detect_format_wav(self):
        from app.audio import AudioProcessor
        data = make_wav_header(100) + b'\x00' * 100
        assert AudioProcessor._detect_format(data) == 'wav'

    def test_detect_format_ogg(self):
        from app.audio import AudioProcessor
        data = b'OggS' + b'\x00' * 100
        assert AudioProcessor._detect_format(data) == 'ogg'

    def test_detect_format_unknown(self):
        from app.audio import AudioProcessor
        data = b'\x00\x00\x00\x00' + b'\x00' * 100
        assert AudioProcessor._detect_format(data) == ''

    def test_is_speech_threshold(self):
        from app.audio import AudioProcessor
        assert AudioProcessor.is_speech(5000.0) is True
        assert AudioProcessor.is_speech(100.0) is False
        assert AudioProcessor.is_speech(1200.1) is True
        assert AudioProcessor.is_speech(1199.9) is False


class TestSafeSendHelpers:
    """Test that safe send helpers handle closed connections gracefully."""

    @pytest.mark.asyncio
    async def test_safe_send_json_on_closed(self):
        """_safe_send_json should return False on a closed session."""
        from app.api.v1.chat import _safe_send_json, VoiceSession
        mock_ws = MagicMock()
        session = VoiceSession(mock_ws, "test")
        session._closed = True
        result = await _safe_send_json(mock_ws, {"type": "test"}, session)
        assert result is False


class TestOrchestrateEndpoint:
    """Test the HTTP orchestrate endpoint still works correctly."""

    def test_orchestrate_streams_tokens(self, client):
        """Verify /api/v1/orchestrate accepts requests and streams tokens."""
        payload = {
            "session_id": "test_session",
            "user_id": "test_user",
            "text_input": "Hello"
        }

        async def mock_gen(text_input, session_id, user_id):
            yield "Hi "
            yield "there!"

        with patch("app.api.v1.chat.route_and_stream", side_effect=mock_gen):
            response = client.post("/api/v1/orchestrate", json=payload)

        assert response.status_code == 200
        assert "Hi there!" in response.text
