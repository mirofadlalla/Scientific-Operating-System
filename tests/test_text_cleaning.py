"""
tests/test_text_cleaning.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for app.core.text_cleaning.clean_for_tts.
No network I/O required.
"""
import pytest
from app.core.text_cleaning import clean_for_tts


# ---------------------------------------------------------------------------
# Parametrized cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected_substring, absent_substring", [
    # Markdown image removed
    (
        "Here is a structure\n\n![Molecular Structure](https://pubchem.example.com/PNG)\n\nand text.",
        "and text",          # kept
        "pubchem.example",   # removed
    ),
    # Link [label](url) -> label only
    (
        "See [aspirin](https://en.wikipedia.org/wiki/Aspirin) for details.",
        "aspirin",
        "wikipedia.org",
    ),
    # **bold** markers stripped
    (
        "This is **very important** text.",
        "very important",
        "**",
    ),
    # # Title heading stripped
    (
        "# Section Title\n\nSome content here.",
        "Section Title",
        "#",
    ),
    # Emoji removed (general emoji block)
    (
        "Hello 🧬 world 🤖 science.",
        "Hello world science",   # whitespace is collapsed to single space
        "\U0001F9EC",
    ),
    # Circled-number glyph ❶ (U+2776) removed
    (
        "❶ First point and ❷ second point.",
        "First point",
        "❶",
    ),
    # Divider run removed
    (
        "Start\n\n─────────────────────\n\nEnd",
        "Start",
        "──",
    ),
    # Arabic text and ؟ preserved untouched
    (
        "ما هي آلية عمل الميتفورمين؟",
        "ما هي آلية عمل الميتفورمين؟",
        None,
    ),
    # SMILES string dropped (6+ chars, SMILES chars, structural mark)
    (
        "The compound CC(=O)Oc1ccccc1C(=O)O is aspirin.",
        "aspirin",
        "CC(=O)",
    ),
    # Fenced code block dropped
    (
        "Result:\n```python\nprint('hello')\n```\nDone.",
        "Done",
        "print",
    ),
])
def test_clean_cases(raw, expected_substring, absent_substring):
    result = clean_for_tts(raw)
    assert expected_substring in result, (
        f"Expected {expected_substring!r} in result but got: {result!r}"
    )
    if absent_substring is not None:
        assert absent_substring not in result, (
            f"Expected {absent_substring!r} NOT in result but got: {result!r}"
        )


def test_idempotent():
    """clean_for_tts(clean_for_tts(x)) == clean_for_tts(x) for a complex string."""
    raw = (
        "## Title\n\n"
        "Hello **world** 🧬, here is [a link](https://example.com).\n\n"
        "![img](https://x.com/img.png)\n\n"
        "```python\ncode here\n```\n\n"
        "SMILES: CC(=O)Oc1ccccc1C(=O)O\n\n"
        "❶ First ❷ Second\n\n"
        "─────────────────────\n\n"
        "Normal text."
    )
    once  = clean_for_tts(raw)
    twice = clean_for_tts(once)
    assert once == twice, f"Not idempotent:\nonce={once!r}\ntwice={twice!r}"


def test_empty_result():
    """A string made only of markdown/emoji/SMILES should return ''."""
    raw = "![img](https://x.com/img.png) 🧬 CC(=O)Oc1ccccc1C(=O)O ─────────"
    result = clean_for_tts(raw)
    assert result == "", f"Expected '' but got: {result!r}"


def test_bare_url_removed():
    result = clean_for_tts("Visit https://example.com/path?a=1 for more.")
    assert "https" not in result
    assert "for more" in result


def test_table_removed():
    raw = "| Col A | Col B |\n|-------|-------|\n| val1  | val2  |\n\nAfter table."
    result = clean_for_tts(raw)
    assert "After table" in result
    assert "|" not in result


def test_inline_code_removed():
    result = clean_for_tts("Use the `calculate_admet` function.")
    assert "calculate_admet" not in result
    assert "Use the" in result


def test_arabic_arabic_punctuation_kept():
    """Arabic text including ، and ؟ must survive."""
    arabic = "السلام عليكم، كيف حالك؟"
    result = clean_for_tts(arabic)
    assert "السلام عليكم" in result
    assert "؟" in result
    assert "،" in result
