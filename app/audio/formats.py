"""Audio container detection and header alignment (magic-byte based)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EBML = b"\x1aE\xdf\xa3"  # WebM / MKV header
_MP3_SYNC = (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


def detect_format(audio_bytes: bytes) -> str:
    """Detect the real container from magic bytes ('' if unknown).

    Corrects the label when the browser sends a different container than
    expected (Firefox → OGG, Safari → MP4).
    """
    if audio_bytes[:4] == b"OggS":
        return "ogg"
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "wav"
    if audio_bytes[:3] == b"ID3" or (len(audio_bytes) >= 2 and audio_bytes[:2] in _MP3_SYNC):
        return "mp3"
    if len(audio_bytes) >= 8 and audio_bytes[4:8] in (b"ftyp", b"mdat", b"moov"):
        return "mp4"
    if audio_bytes[:4] == _EBML:
        return "webm"
    return ""


def align_container_header(audio: bytes, audio_format: str) -> bytes:
    """Strip stray leading bytes so the container starts at its magic header.

    Prevents 400 invalid media file when cluster bytes were prepended.
    """
    ebml_pos = audio.find(_EBML)
    if ebml_pos > 0:
        logger.info("[STT] Slicing %d stray leading bytes to align WebM EBML header", ebml_pos)
        return audio[ebml_pos:]

    if ebml_pos == -1 and audio_format == "webm":
        riff_pos = audio.find(b"RIFF")
        if riff_pos > 0:
            audio = audio[riff_pos:]
        ogg_pos = audio.find(b"OggS")
        if ogg_pos > 0:
            audio = audio[ogg_pos:]
    return audio
