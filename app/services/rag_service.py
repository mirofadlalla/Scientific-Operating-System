"""
app.services.rag_service
~~~~~~~~~~~~~~~~~~~~~~~~
RAG knowledge-base ingestion: upload validation, job scheduling, the
read → chunk → embed → index → reload pipeline, and job status tracking.

`run_background_ingest_job` is enqueued on RQ by dotted path, so keep it
importable as app.services.rag_service.run_background_ingest_job.
"""
import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

import app.core.state as state
from app.core.deps import rag_agent
from app.core.exceptions import BadRequestError
from app.repositories.ingestion_job_repository import ingestion_job_repository as job_repo

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".md", ".txt"}


# ── Job status ────────────────────────────────────────────────────────────────

def update_job_status(
    job_id: str,
    status: str,
    message: str = "",
    extra_data: Optional[dict] = None,
) -> None:
    data: Dict[str, Any] = {
        "status":        status,
        "message":       message,
        "filename":      (extra_data or {}).get("filename", ""),
        "strategy":      (extra_data or {}).get("strategy", ""),
        "nodes_created": (extra_data or {}).get("nodes_created", 0),
        "index_name":    (extra_data or {}).get("index_name", ""),
        "error_message": (extra_data or {}).get("error_message", ""),
    }
    if extra_data:
        data.update(extra_data)
    job_repo.save(job_id, data)


def get_job_status(job_id: str) -> dict:
    return job_repo.get(job_id) or {"status": "unknown", "message": "Job not found"}


# ── Background pipeline ───────────────────────────────────────────────────────

async def run_background_ingest(job_id: str, filename: str, content: bytes, strategy: str) -> None:
    """Async ingestion pipeline: read → chunk → embed → index → reload."""

    def callback(step_name: str, step_msg: str) -> None:
        update_job_status(job_id, step_name, step_msg, {"filename": filename, "strategy": strategy})

    try:
        if not rag_agent._ready:
            callback("reading", "Warming up RAG environment…")
            await rag_agent._initialise()

        ingestion_svc = rag_agent.get_ingestion_service()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ingestion_svc.ingest_bytes(
                filename=filename,
                content=content,
                strategy=strategy,
                status_callback=callback,
            ),
        )

        if result.get("status") == "error":
            error_msg = result.get("message", "Unknown ingestion error.")
            update_job_status(job_id, "failed", f"Ingestion failed: {error_msg}", {
                "filename": filename, "strategy": strategy, "error_message": error_msg,
            })
            return

        update_job_status(job_id, "reloading", "Reloading query engine…", {
            "filename": filename, "strategy": strategy,
            "nodes_created": result.get("nodes_created"),
            "index_name":    result.get("index_name"),
        })
        await rag_agent.reload_engine()

        update_job_status(job_id, "completed", "Document ingested successfully.", {
            "filename":      filename,
            "strategy":      strategy,
            "nodes_created": result.get("nodes_created"),
            "index_name":    result.get("index_name"),
        })

    except Exception as exc:
        logger.error(f"[run_background_ingest] Unexpected error: {exc}")
        update_job_status(job_id, "failed", f"Ingestion failed: {exc}", {
            "filename": filename, "strategy": strategy, "error_message": str(exc),
        })


def run_background_ingest_job(job_id: str, filename: str, content: bytes, strategy: str) -> None:
    """Synchronous wrapper called by the RQ worker process."""
    try:
        asyncio.get_running_loop()
        asyncio.create_task(run_background_ingest(job_id, filename, content, strategy))
    except RuntimeError:
        asyncio.run(run_background_ingest(job_id, filename, content, strategy))


# ── Use cases ─────────────────────────────────────────────────────────────────

def validate_upload(filename: Optional[str]) -> None:
    if not filename:
        raise BadRequestError("No file provided.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise BadRequestError(
            f"Unsupported file type '{ext}'. Only {ALLOWED_EXTENSIONS} are accepted."
        )


async def submit_ingest(filename: str, content: bytes, strategy: str) -> dict:
    """Register a job and run it on RQ (if available) or as an asyncio task."""
    if not content:
        raise BadRequestError("Uploaded file is empty.")

    job_id = str(uuid.uuid4())
    update_job_status(job_id, "pending", "Job scheduled…", {
        "filename": filename, "strategy": strategy,
    })

    if state.rq_queue:
        state.rq_queue.enqueue(run_background_ingest_job, job_id, filename, content, strategy)
        message = "Ingestion job queued (running in background via Redis)."
    else:
        logger.info(f"Redis unavailable — scheduling ingestion as async task for job {job_id}")
        asyncio.create_task(run_background_ingest(job_id, filename, content, strategy))
        message = "Ingestion job scheduled (running in background)."

    return {
        "status":   "success",
        "job_id":   job_id,
        "filename": filename,
        "strategy": strategy,
        "message":  message,
    }


async def get_knowledge_base_status() -> dict:
    return await rag_agent.status()
