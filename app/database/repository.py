"""
ProcessingJobRepository
========================
MongoDB persistence layer for ProcessingJob records.

Responsibilities
----------------
- save()          : Insert a new ProcessingJob document and return its id.
- get()           : Retrieve a ProcessingJob by job_id.
- update_status() : Atomically update status (and optionally any extra fields).

Design rules
------------
- Zero business logic.  This class only reads and writes.
- No validation rules, no routing decisions.
- Returns typed Pydantic models — callers never touch raw dicts.
- All operations are async (Motor).
"""

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import logger
from app.schemas.processing_job import ProcessingJob, ProcessingStatus

_COLLECTION = "processing_jobs"


class ProcessingJobRepository:
    """
    Usage::

        repo = ProcessingJobRepository(db)

        job_id = await repo.save(job)
        job    = await repo.get(job_id)
        await   repo.update_status(job_id, ProcessingStatus.COMPLETED)
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[_COLLECTION]

    #  #
    # Public interface                                                     #
    async def save(self, job: ProcessingJob) -> str:
        """
        Insert a new ProcessingJob document.

        Returns
        -------
        str
            The MongoDB-assigned ObjectId as a string.
        """
        doc = job.model_dump(mode="json")
        doc.pop("job_id", None)          # let MongoDB assign the _id

        result = await self._col.insert_one(doc)
        job_id = str(result.inserted_id)

        logger.info("ProcessingJob saved: job_id=%s status=%s", job_id, job.status)
        return job_id

    async def get(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Retrieve a ProcessingJob by its string id.

        Returns None when the document is not found.
        """
        try:
            oid = ObjectId(job_id)
        except Exception:
            logger.warning("get() called with invalid ObjectId: %s", job_id)
            return None

        doc = await self._col.find_one({"_id": oid})
        if doc is None:
            return None

        doc["job_id"] = str(doc.pop("_id"))
        return ProcessingJob.model_validate(doc)

    async def update_status(
        self,
        job_id: str,
        status: ProcessingStatus,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Atomically update the status field (and any additional fields).

        Parameters
        ----------
        job_id:
            Target document id.
        status:
            New status value.
        extra_fields:
            Optional dict of additional top-level fields to $set at the
            same time (e.g. ``{"extraction": {...}, "validation": {...}}``).
        """
        try:
            oid = ObjectId(job_id)
        except Exception:
            logger.warning(
                "update_status() called with invalid ObjectId: %s", job_id
            )
            return

        update: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if extra_fields:
            update.update(extra_fields)

        await self._col.update_one({"_id": oid}, {"$set": update})
        logger.info(
            "ProcessingJob updated: job_id=%s status=%s", job_id, status.value
        )

    async def list_jobs(self, skip: int = 0, limit: int = 20, filters: Optional[dict[str, Any]] = None) -> list[ProcessingJob]:
        query = filters or {}
        cursor = self._col.find(query).sort("created_at", -1).skip(skip).limit(limit)
        jobs = []
        async for doc in cursor:
            doc["job_id"] = str(doc.pop("_id"))
            jobs.append(ProcessingJob.model_validate(doc))
        return jobs

    async def delete_job(self, job_id: str) -> bool:
        try:
            oid = ObjectId(job_id)
        except Exception:
            return False
        result = await self._col.delete_one({"_id": oid})
        return result.deleted_count > 0

    async def get_statistics(self) -> dict[str, Any]:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_jobs": {"$sum": 1},
                    "approved_jobs": {"$sum": {"$cond": [{"$eq": ["$workflow_action", "APPROVE"]}, 1, 0]}},
                    "review_jobs": {"$sum": {"$cond": [{"$eq": ["$workflow_action", "ROUTE_TO_HUMAN_REVIEW"]}, 1, 0]}},
                    "rejected_jobs": {"$sum": {"$cond": [{"$eq": ["$workflow_action", "REJECT"]}, 1, 0]}},
                    "avg_confidence": {"$avg": "$validation.confidence_score"}
                }
            }
        ]
        stats_cursor = self._col.aggregate(pipeline)
        stats_list = await stats_cursor.to_list(length=1)
        if not stats_list:
            return {
                "total_jobs": 0,
                "approved_jobs": 0,
                "review_jobs": 0,
                "rejected_jobs": 0,
                "average_confidence": 0.0
            }
        stats = stats_list[0]
        return {
            "total_jobs": stats.get("total_jobs", 0),
            "approved_jobs": stats.get("approved_jobs", 0),
            "review_jobs": stats.get("review_jobs", 0),
            "rejected_jobs": stats.get("rejected_jobs", 0),
            "average_confidence": stats.get("avg_confidence", 0.0)
        }
