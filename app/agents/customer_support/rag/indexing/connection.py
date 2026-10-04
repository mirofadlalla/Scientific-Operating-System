"""Single place that knows how to open a Weaviate connection."""

from __future__ import annotations

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

from ..config import WEAVIATE_GRPC_PORT, WEAVIATE_HOST, WEAVIATE_PORT

# How long (seconds) to wait for Weaviate's HTTP readiness probe during
# connect_to_local().  The default (2 s) is fine for production; we keep it
# explicit so CI doesn't stall when no Weaviate sidecar is running.
_CONNECT_TIMEOUT_S: int = 3


def connect_weaviate() -> weaviate.WeaviateClient:
    """Connect to the configured local/remote Weaviate instance.

    Uses skip_init_checks=True so the gRPC/HTTP readiness probes are NOT
    executed at construction time.  The probes block for up to
    ``_CONNECT_TIMEOUT_S`` seconds per attempt *with retries* — in
    environments that have no Weaviate service (e.g. CI smoke tests) this
    caused the entire container to stall before uvicorn could bind port 7860.

    Callers (VectorIndexManager.__init__, probe_weaviate) already catch any
    exception raised here and degrade gracefully.
    """
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        skip_init_checks=True,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=_CONNECT_TIMEOUT_S, query=30, insert=90),
        ),
    )
