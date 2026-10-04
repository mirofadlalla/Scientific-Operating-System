from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.controllers import rag_controller
from app.core.auth import verify_token

router = APIRouter(prefix="/rag", tags=["Knowledge Base"])


@router.post("/ingest")
async def rag_ingest(
    file: UploadFile = File(...),
    strategy: str = Form(default="markdown"),
    username: str = Depends(verify_token),   # requires valid JWT
):
    """
    Upload a Markdown (.md) or plain-text (.txt) file and ingest it into the
    vector store in the background.

    - **strategy**: markdown (default), sentence, or token.

    Returns a job_id you can poll at /api/v1/rag/ingest/status/{job_id}.
    """
    return await rag_controller.ingest(file, strategy)


@router.get("/ingest/status/{job_id}")
async def rag_ingest_status(job_id: str):
    """Poll the status of a background ingestion job."""
    return rag_controller.ingest_status(job_id)


@router.get("/status")
async def rag_status():
    """
    Health check for the RAG knowledge base.
    Returns Weaviate connectivity, index name, node count, and engine readiness.
    """
    return await rag_controller.knowledge_base_status()
