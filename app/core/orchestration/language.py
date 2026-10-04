"""Tiny language helper (Arabic-script detection)."""

from __future__ import annotations

import re

_ARABIC = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text: str) -> bool:
    """``True`` if ``text`` contains any Arabic-script character."""
    return bool(_ARABIC.search(text))
