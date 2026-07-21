"""
app.database.mongodb
~~~~~~~~~~~~~~~~~~~~
Motor (async MongoDB) client for MongoDB Atlas.

Exposes:
  get_database()  — returns the app database handle
  ping()          — connectivity check (called from lifespan)
  close()         — closes the connection pool (called on shutdown)

MONGODB_URI and MONGODB_DB_NAME are read from environment via app.config.settings.
"""
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return the shared Motor client, creating it on first call."""
    global _client
    if _client is None:
        if not settings.MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI environment variable is not set. "
                "Add it to your HF Space secrets or .env file."
            )
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
        logger.info("MongoDB Atlas client initialised.")
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the application database handle."""
    return get_client()[settings.MONGODB_DB_NAME]


async def ping() -> bool:
    """
    Ping the MongoDB server.
    Returns True on success, False on connection failure.
    Called from the FastAPI lifespan to fail fast on startup.
    """
    try:
        await get_client().admin.command("ping")
        logger.info("✅ MongoDB Atlas ping successful (db=%s).", settings.MONGODB_DB_NAME)
        return True
    except Exception as exc:
        logger.error("❌ MongoDB Atlas ping failed: %s", exc)
        return False


async def close() -> None:
    """Close the Motor client connection pool (called on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB Atlas client closed.")
