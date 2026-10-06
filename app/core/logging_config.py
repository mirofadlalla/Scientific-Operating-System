"""Application-wide logging setup.

Uvicorn only configures its own loggers, so without this module every
logger.info in the app (STT/TTS timings, RAG lifecycle, agent traces) would
be silently dropped. configure_logging is idempotent and respects any
handlers already installed by the host (e.g. pytest or a custom dictConfig).
"""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "weaviate")


def configure_logging(level: str | None = None) -> None:
    """Install a root handler (once) at LOG_LEVEL (default INFO)."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=resolved, format=_FORMAT)
    else:
        root.setLevel(resolved)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
