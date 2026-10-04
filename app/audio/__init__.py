"""Audio processing: STT (Groq Whisper) and TTS for the voice channel.

Formerly the single ``app/audio.py`` module. Every public (and previously
imported private) name is re-exported here, so ``from app.audio import …`` and
``patch("app.audio.audio_processor.<method>")`` keep working unchanged.

Layout:
    constants / errors / console / diagnostics   shared plumbing
    formats                                      container detection & alignment
    transcript_filter                            Whisper hallucination filtering
    segmentation / transliteration               text prep for TTS
    stt / tts                                    behaviour mixins
    processor                                    ``AudioProcessor`` + ``audio_processor`` singleton
"""

from .console import ensure_utf8_console

ensure_utf8_console()  # preserved import-time behaviour of the old module

from .constants import MIN_AUDIO_BYTES, WHISPER_HALLUCINATION_BLOCKLIST  # noqa: E402
from .diagnostics import voice_log  # noqa: E402
from .errors import AudioTooShortError  # noqa: E402
from .processor import AudioProcessor, audio_processor  # noqa: E402
from .segmentation import batch_sentences_for_tts, split_sentences  # noqa: E402
from .transcript_filter import filter_whisper_hallucinations  # noqa: E402
from .transliteration import _transliterate_for_arabic_tts  # noqa: E402

__all__ = [
    "MIN_AUDIO_BYTES",
    "WHISPER_HALLUCINATION_BLOCKLIST",
    "AudioProcessor",
    "AudioTooShortError",
    "_transliterate_for_arabic_tts",
    "audio_processor",
    "batch_sentences_for_tts",
    "filter_whisper_hallucinations",
    "split_sentences",
    "voice_log",
]
