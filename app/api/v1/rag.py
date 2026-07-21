"""
app.api.v1.rag
~~~~~~~~~~~~~~
POST /api/v1/rag/ingest                  — Upload + ingest a document
GET  /api/v1/rag/ingest/status/{job_id}  — Poll background job status
GET  /api/v1/rag/status                  — Knowledge-base health check
"""
import asyncio
import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

import app.core.state as state
from app.core.auth import verify_token
from app.core.deps import rag_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["Knowledge Base"])


# ──────────────────────────────────────────────────────────────────────────────
# Job status helpers
# ──────────────────────────────────────────────────────────────────────────────

def update_job_status(
    job_id: str,
    status: str,
    message: str = "",
    extra_data: dict | None = None,
) -> None:
    """Write job state to in-process dict and optionally mirror to Redis."""
    data: dict = {
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

    state.ingestion_jobs[job_id] = data

    try:
        lm = state.long_memory
        if lm and hasattr(lm, "is_redis") and lm.is_redis:
            lm.redis_client.set(f"rag:job:{job_id}", json.dumps(data), ex=3600)
    except Exception as exc:
        logger.warning(f"Could not sync job status to Redis: {exc} — using in-memory storage.")


def get_job_status(job_id: str) -> dict:
    """Read job state from Redis (if available) or fall back to in-process dict."""
    try:
        lm = state.long_memory
        if lm and hasattr(lm, "is_redis") and lm.is_redis:
            val = lm.redis_client.get(f"rag:job:{job_id}")
            if val:
                return json.loads(val)
    except Exception as exc:
        logger.warning(f"Could not fetch job status from Redis: {exc}")
    return state.ingestion_jobs.get(job_id, {"status": "unknown", "message": "Job not found"})


# ──────────────────────────────────────────────────────────────────────────────
# Background ingestion pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def run_background_ingest(
    job_id: str,
    filename: str,
    content: bytes,
    strategy: str,
) -> None:
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


def run_background_ingest_job(
    job_id: str,
    filename: str,
    content: bytes,
    strategy: str,
) -> None:
    """
    Synchronous wrapper called by the RQ worker process.
    Bridges the sync RQ world into the async ingestion pipeline.
    """
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(run_background_ingest(job_id, filename, content, strategy))
    except RuntimeError:
        asyncio.run(run_background_ingest(job_id, filename, content, strategy))


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def rag_ingest(
    file: UploadFile = File(...),
    strategy: str = Form(default="markdown"),
    username: str = Depends(verify_token),   # ← requires valid JWT
):
    """
    Upload a Markdown (.md) or plain-text (.txt) file and ingest it into the
    vector store in the background.

    - **strategy**: `markdown` (default), `sentence`, or `token`.

    Returns a `job_id` you can poll at `/api/v1/rag/ingest/status/{job_id}`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_ext = {".md", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only {allowed_ext} are accepted.",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        job_id = str(uuid.uuid4())
        update_job_status(job_id, "pending", "Job scheduled…", {
            "filename": file.filename, "strategy": strategy,
        })

        if state.rq_queue:
            state.rq_queue.enqueue(
                run_background_ingest_job, job_id, file.filename, content, strategy,
            )
            message = "Ingestion job queued (running in background via Redis)."
        else:
            logger.info(f"Redis unavailable — scheduling ingestion as async task for job {job_id}")
            asyncio.create_task(run_background_ingest(job_id, file.filename, content, strategy))
            message = "Ingestion job scheduled (running in background)."

        return {
            "status":   "success",
            "job_id":   job_id,
            "filename": file.filename,
            "strategy": strategy,
            "message":  message,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[/rag/ingest] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")


@router.get("/ingest/status/{job_id}")
async def rag_ingest_status(job_id: str):
    """Poll the status of a background ingestion job."""
    return get_job_status(job_id)


@router.get("/status")
async def rag_status():
    """
    Health check for the RAG knowledge base.
    Returns Weaviate connectivity, index name, node count, and engine readiness.
    """
    try:
        return await rag_agent.status()
    except Exception as exc:
        logger.error(f"[/rag/status] Error: {exc}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {exc}")
