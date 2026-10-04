"""Formatting helpers for MCP tool output."""

from __future__ import annotations

from typing import Any


def fmt(val: Any, decimals: int = 4) -> str:
    """Safely format numeric values (``None`` → ``"N/A"``)."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val) if val is not None else "N/A"
