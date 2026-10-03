"""app.controllers.monitoring_controller"""
from app.services import monitoring_service


def metrics() -> dict:
    return monitoring_service.get_metrics()


def recent_requests(limit: int) -> list:
    return monitoring_service.get_recent_requests(limit)
