"""STT/TTS behaviour of the modular ``app.audio`` package (API clients faked)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app.audio import AudioProcessor, AudioTooShortError, batch_sentences_for_tts
from app.audio.formats import align_container_header, detect_format
from app.audio.segmentation import split_sentences

EBML = b"\x1aE\xdf\xa3"


class _Recorder:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.kwargs = result, exc, None

    async def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.result


def _processor(stt=None, tts=None, openai=None) -> AudioProcessor:
    p = AudioProcessor.__new__(AudioProcessor)
    p.groq_client = NS(audio=NS(transcriptions=stt))
    p.groq_tts_client = NS(audio=NS(speech=tts))
    p.openai_client = NS(audio=NS(speech=openai)) if openai else None
    return p


def test_align_strips_stray_bytes_before_ebml():
    assert align_container_header(b"junk" + EBML + b"data", "webm") == EBML + b"data"


def test_align_finds_riff_then_ogg_for_unknown_webm():
    assert align_container_header(b"xxRIFFab", "webm") == b"RIFFab"
    assert align_container_header(b"RIFFxxOggSzz", "webm") == b"OggSzz"
    assert align_container_header(b"xxRIFFab", "wav") == b"xxRIFFab"  # only for webm hint


def test_detect_format_mp4_requires_8_bytes():
    assert detect_format(b"\0\0\0\0ftyp") == "mp4"
    assert detect_format(b"\0\0\0") == ""


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_and_short():
    p = _processor(stt=_Recorder())
    with pytest.raises(ValueError, match="Empty"):
        await p.transcribe_audio(b"")
    with pytest.raises(AudioTooShortError):
        await p.transcribe_audio(b"x" * 10)


@pytest.mark.asyncio
async def test_transcribe_overrides_format_and_filters_hallucinations():
    api = _Recorder(result={"text": "Thanks for watching", "segments": []})
    p = _processor(stt=api)
    assert await p.transcribe_audio(b"OggS" + b"\0" * 2000, "webm") == ""
    assert api.kwargs["file"].name == "recording.ogg"
    assert api.kwargs["response_format"] == "verbose_json"


@pytest.mark.asyncio
async def test_transcribe_wraps_api_errors():
    p = _processor(stt=_Recorder(exc=RuntimeError("down")))
    with pytest.raises(ValueError, match="Transcription failed: down"):
        await p.transcribe_audio(b"\0" * 2000)


@pytest.mark.asyncio
async def test_tts_picks_arabic_model_and_voice():
    api = _Recorder(result=NS(content=b"WAV"))
    out = await _processor(tts=api).synthesize_speech("مرحبا")
    assert out == b"WAV"
    assert api.kwargs["voice"] == "abdullah" and "arabic" in api.kwargs["model"]


@pytest.mark.asyncio
async def test_tts_falls_back_to_openai_then_raises():
    groq = _Recorder(exc=RuntimeError("groq down"))
    assert await _processor(tts=groq, openai=_Recorder(result=NS(content=b"OA"))).synthesize_speech("hi") == b"OA"
    with pytest.raises(ValueError, match="Speech synthesis failed: groq down"):
        await _processor(tts=groq, openai=_Recorder(exc=RuntimeError("oa down"))).synthesize_speech("hi")
    with pytest.raises(ValueError):
        await _processor(tts=groq).synthesize_speech("hi")


@pytest.mark.asyncio
async def test_chunked_tts_skips_failed_batch_and_uses_patched_synth():
    p = _processor()
    calls = []

    async def fake(text, voice):
        calls.append(text)
        if "FAIL" in text:
            raise ValueError("x")
        return text.encode()

    p.synthesize_speech = fake  # chunked TTS must go through the instance method
    text = " ".join(f"Sentence number {i} is reasonably long here." for i in range(20))
    text = text.replace("number 9 ", "FAIL 9 ")
    chunks = [c async for c in p.synthesize_speech_chunked(text)]
    assert len(chunks) == len(calls) - 1 and all(b"FAIL" not in c for c in chunks)


def test_segmentation_helpers():
    assert split_sentences("Hello. OK. Fine. A longer sentence.")[0].startswith("Hello.")
    assert batch_sentences_for_tts([]) == []
    assert batch_sentences_for_tts(["a" * 130, "b"], 120) == ["a" * 130, "b"]
