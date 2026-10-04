"""Vector index management (Weaviate with disk / in-memory fallbacks)."""

from typing import Any

from .compat import apply_weaviate_compat
from .connection import connect_weaviate
from .manager import VectorIndexManager
from .persistence import get_persist_dir, persist_index

__all__ = [
    "VectorIndexManager",
    "apply_weaviate_compat",
    "connect_weaviate",
    "get_persist_dir",
    "persist_index",
]


def __getattr__(name: str) -> Any:  # backward-compat: lazy ``PERSIST_DIR``
    if name == "PERSIST_DIR":
        return get_persist_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
