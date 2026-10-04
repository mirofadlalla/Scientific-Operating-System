"""Console encoding safety."""

from __future__ import annotations

import sys


def ensure_utf8_console() -> None:
    """Force stdout/stderr to UTF-8 so Unicode (Arabic) log text never crashes on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
