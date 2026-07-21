"""
app.core.lifespan
~~~~~~~~~~~~~~~~~
FastAPI lifespan context manager — startup / shutdown hooks.

Responsibilities:
  1. Initialise Redis + RQ queue (graceful fallback to in-memory).
  2. Initialise long-term memory (Redis-backed or JSON-file).
  3. Spawn the RQ background worker process when Redis is available.
  4. Pre-warm the RAG engine in the background.
  5. Cleanly shut down the worker and close the Weaviate connection on exit.
"""
import asyncio
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.memory.long_term import LongTermMemory
from app import core  # noqa
import app.core.state as state
from app.core.deps import rag_agent

logger = logging.getLogger(__name__)


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _init_redis() -> None:
    """Try to connect to Redis; set state flags accordingly."""
    try:
        from redis import Redis
        from rq import Queue

        conn = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        conn.ping()
        state.redis_conn     = conn
        state.rq_queue       = Queue("default", connection=conn)
        state.redis_available = True
        logger.info(f"✅ Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except Exception as exc:
        state.redis_available = False
        state.redis_conn      = None
        state.rq_queue        = None
        logger.warning(f"⚠️  Redis unavailable: {exc}. Falling back to in-memory mode.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up & teardown hook for long-lived resources."""

    # 1. Redis (graceful fallback)
    _init_redis()

    # 2. Long-term memory
    if state.redis_available:
        try:
            state.long_memory = LongTermMemory(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
            )
            logger.info("✅ Long-term memory initialized (Redis-backed)")
        except Exception as exc:
            logger.warning(f"Redis long-term memory init failed: {exc}. Falling back to JSON file.")
            state.long_memory = LongTermMemory()
    else:
        state.long_memory = LongTermMemory()
        logger.info("✅ Long-term memory initialized (JSON file-backed — Redis not available)")

    # 3. MongoDB Atlas — ping to validate credentials on startup
    if settings.MONGODB_URI:
        from app.database.mongodb import ping as mongo_ping
        ok = await mongo_ping()
        if not ok:
            logger.warning(
                "⚠️  MongoDB Atlas ping failed — authentication will be unavailable."
            )
    else:
        logger.warning(
            "⚠️  MONGODB_URI not set — authentication endpoints are disabled."
        )

    # 4. RQ worker process
    if state.redis_available:
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            logger.info(f"Starting RQ worker → {redis_url}…")
            state.worker_process = subprocess.Popen(
                [sys.executable, "-m", "rq", "worker", "default", "--url", redis_url],
                env=env,
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("RQ worker spawned successfully.")
        except Exception as exc:
            logger.warning(f"Could not start RQ worker: {exc}")
    else:
        logger.info("RQ worker skipped (Redis not available)")

    # 5. Pre-warm RAG engine in the background
    asyncio.create_task(rag_agent._initialise())

    yield  # ── application runs ──────────────────────────────────────────────

    # 6. Shutdown
    if state.worker_process:
        try:
            logger.info("Terminating RQ worker…")
            state.worker_process.terminate()
            state.worker_process.wait(timeout=5)
            logger.info("RQ worker terminated.")
        except Exception as exc:
            logger.error(f"Error terminating RQ worker: {exc}")

    await rag_agent.close()

    # 7. Close MongoDB connection pool
    if settings.MONGODB_URI:
        from app.database.mongodb import close as mongo_close
        await mongo_close()
