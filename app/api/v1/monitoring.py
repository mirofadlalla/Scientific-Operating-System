from fastapi import APIRouter

from app.controllers import monitoring_controller

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


@router.get("")
async def get_metrics():
    """Full system metrics snapshot — consumed by the dashboard."""
    return monitoring_controller.metrics()


@router.get("/requests")
async def get_recent_requests(limit: int = 50):
    """Return the last N request log entries."""
    return monitoring_controller.recent_requests(limit)
