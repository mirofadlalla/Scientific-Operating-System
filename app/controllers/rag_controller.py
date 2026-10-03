"""app.controllers.rag_controller"""
import logging

from fastapi import UploadFile

from app.core.exceptions import AppError, InternalError
from app.services import rag_service

logger = logging.getLogger(__name__)


async def ingest(file: UploadFile, strategy: str) -> dict:
    rag_service.validate_upload(file.filename)
    try:
        content = await file.read()
        return await rag_service.submit_ingest(file.filename, content, strategy)
    except AppError:
        raise
    except Exception as exc:
        logger.error(f"[/rag/ingest] Unexpected error: {exc}")
        raise InternalError(f"Ingestion error: {exc}")


def ingest_status(job_id: str) -> dict:
    return rag_service.get_job_status(job_id)


async def knowledge_base_status() -> dict:
    try:
        return await rag_service.get_knowledge_base_status()
    except Exception as exc:
        logger.error(f"[/rag/status] Error: {exc}")
        raise InternalError(f"Status check failed: {exc}")
