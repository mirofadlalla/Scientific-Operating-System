"""Brand-name transliteration so Arabic TTS pronounces English terms correctly."""

from __future__ import annotations

_ARABIC_TRANSLITERATIONS = {
    "AI-Lixir": "اي ليكسر",
    "Ai-Lixir": "اي ليكسر",
    "ai-lixir": "اي ليكسر",
    "ADMET": "ايه دي ام اي تي",
    "SMILES": "سمايلز",
}


def _transliterate_for_arabic_tts(text: str) -> str:
    """Replace English brand names with Arabic phonetic equivalents for TTS."""
    for eng, ar in _ARABIC_TRANSLITERATIONS.items():
        text = text.replace(eng, ar)
    return text
