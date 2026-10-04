"""Single place that knows how to open a Weaviate connection."""

from __future__ import annotations

import weaviate

from ..config import WEAVIATE_GRPC_PORT, WEAVIATE_HOST, WEAVIATE_PORT


def connect_weaviate() -> weaviate.WeaviateClient:
    """Connect to the configured local/remote Weaviate instance.

    Raises whatever the weaviate client raises if the instance is unreachable;
    callers decide how to degrade.
    """
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
