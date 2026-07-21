"""
app.api.v1.monitoring
~~~~~~~~~~~~~~~~~~~~~
GET /api/v1/metrics
GET /api/v1/metrics/requests
"""
from fastapi import APIRouter

from app import monitoring

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


@router.get("")
async def get_metrics():
    """Full system metrics snapshot — consumed by the dashboard."""
    return monitoring.get_snapshot()


@router.get("/requests")
async def get_recent_requests(limit: int = 50):
    """Return the last N request log entries."""
    return monitoring.get_recent_requests(limit=min(limit, 200))
