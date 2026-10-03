"""app.services.monitoring_service — read access to runtime metrics."""
from app import monitoring

MAX_RECENT_REQUESTS = 200


def get_metrics() -> dict:
    return monitoring.get_snapshot()


def get_recent_requests(limit: int = 50) -> list:
    return monitoring.get_recent_requests(limit=min(limit, MAX_RECENT_REQUESTS))
