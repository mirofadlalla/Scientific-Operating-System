"""Health probe for the Weaviate-backed knowledge base."""

from __future__ import annotations

import logging
from typing import Tuple

from .indexing import connect_weaviate

logger = logging.getLogger(__name__)


def probe_weaviate(index_name: str) -> Tuple[bool, int]:
    """Return (connected, object_count) for index_name.

    object_count is 0 when unreachable and -1 when the collection
    does not exist yet. Blocking: call from an executor in async code.
    """
    try:
        client = connect_weaviate()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[RAGStatus] Weaviate unreachable: %s", exc)
        return False, 0

    try:
        connected = bool(client.is_ready())
        try:
            count = client.collections.get(index_name).aggregate.over_all().total_count
        except Exception:  # noqa: BLE001 - collection may not exist yet
            count = -1
        return connected, count
    finally:
        client.close()
