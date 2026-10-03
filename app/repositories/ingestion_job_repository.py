"""
app.repositories.ingestion_job_repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Storage for RAG ingestion job records.

Writes to the in-process dict (app.core.state.ingestion_jobs) and mirrors to
Redis so the RQ worker process and the API process see the same state.
Zero business logic: callers decide what a record contains.
"""
import json
import logging
from typing import Any, Dict, Optional

import app.core.state as state

logger = logging.getLogger(__name__)

_KEY_PREFIX = "rag:job:"
_TTL_SECONDS = 3600


class IngestionJobRepository:
    def __init__(self) -> None:
        self._redis_client = None
        self._redis_failed = False

    def _get_redis(self):
        """Redis client from app state, or a direct connection when in the worker process."""
        if self._redis_failed:
            return None
        lm = getattr(state, "long_memory", None)
        if lm and getattr(lm, "is_redis", False) and lm.redis_client:
            return lm.redis_client
        if getattr(state, "redis_conn", None):
            return state.redis_conn
        if self._redis_client is None:
            try:
                import redis
                from app.config import settings
                client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                )
                client.ping()
                self._redis_client = client
            except Exception:
                self._redis_failed = True
                return None
        return self._redis_client

    def save(self, job_id: str, data: Dict[str, Any]) -> None:
        state.ingestion_jobs[job_id] = data
        try:
            r = self._get_redis()
            if r:
                r.set(f"{_KEY_PREFIX}{job_id}", json.dumps(data), ex=_TTL_SECONDS)
        except Exception as exc:
            logger.warning(f"Could not sync job status to Redis: {exc} — using in-memory storage.")

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = self._get_redis()
            if r:
                val = r.get(f"{_KEY_PREFIX}{job_id}")
                if val:
                    return json.loads(val)
        except Exception as exc:
            logger.warning(f"Could not fetch job status from Redis: {exc}")
        return state.ingestion_jobs.get(job_id)


ingestion_job_repository = IngestionJobRepository()
