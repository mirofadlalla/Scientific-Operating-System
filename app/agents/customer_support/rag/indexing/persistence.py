"""Where the fallback (non-Weaviate) index is persisted on disk.

Resolution is lazy and cached: nothing touches the filesystem at import time.
"""

from __future__ import annotations

import logging
import pathlib
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# …/customer_support/rag  (this file lives in rag/indexing/)
RAG_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
_HF_PERSISTENT = pathlib.Path("/data/rag_index")  # Hugging Face Spaces persistent volume
_LOCAL_FALLBACK = RAG_ROOT / "storage" / "rag_index"


@lru_cache(maxsize=1)
def get_persist_dir() -> str:
    """Return /data/rag_index when writable, else the package-local storage dir."""
    try:
        _HF_PERSISTENT.parent.mkdir(parents=True, exist_ok=True)
        probe = _HF_PERSISTENT.parent / ".write_test"
        probe.touch()
        probe.unlink()
    except Exception:  # noqa: BLE001 - any failure means "not writable"
        logger.info("[Indexer] /data not writable — using local storage: %s", _LOCAL_FALLBACK)
        return str(_LOCAL_FALLBACK)
    logger.info("[Indexer] Using HF persistent storage: %s", _HF_PERSISTENT)
    return str(_HF_PERSISTENT)


def persist_index(index: Any) -> str:
    """Persist index to disk and return the directory used."""
    import os

    persist_dir = get_persist_dir()
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    return persist_dir


def __getattr__(name: str) -> Any:  # backward-compat: lazy PERSIST_DIR
    if name == "PERSIST_DIR":
        return get_persist_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
