"""
app.core.middleware
~~~~~~~~~~~~~~~~~~~
FastAPI middleware classes:
  - ReadinessMiddleware : blocks requests while the RAG engine is loading
  - MonitoringMiddleware: records per-request latency + status for /metrics
"""
import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

from app import monitoring
from app.agents.customer_support.agent import rag_state

logger = logging.getLogger(__name__)


class ReadinessMiddleware(BaseHTTPMiddleware):
    """
    Returns 503 for all API routes until the RAG engine finishes loading.
    Health / docs / status paths always pass through immediately.
    """
    _ALWAYS_ALLOW = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        # v1 equivalents
        "/api/v1/rag/status",
        "/api/v1/metrics",
        "/api/v1/metrics/requests",
    }

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self._ALWAYS_ALLOW or path.startswith("/api/v1/rag/ingest/status"):
            return await call_next(request)

        if not rag_state["ready"]:
            return JSONResponse(
                status_code=503,
                content={"detail": "System is initializing — please wait a few seconds and try again."},
            )

        return await call_next(request)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Records latency + HTTP status for every request so /metrics stays live."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            monitoring.record_error(request.url.path, str(exc))
            raise
        finally:
            latency_ms = round((time.time() - start) * 1000, 2)
            monitoring.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                latency_ms=latency_ms,
            )
