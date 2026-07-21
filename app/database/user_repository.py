"""
app.database.user_repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User lookup against the MongoDB Atlas `users` collection.

Expected document shape in MongoDB:
  {
    "_id":      ObjectId(...),
    "username": "omar",
    "password": "$2b$12$..."   ← bcrypt hash
  }

The repository NEVER stores plaintext passwords.
Use the helper below to create users from the CLI:

    python -c "
    import asyncio, bcrypt
    from motor.motor_asyncio import AsyncIOMotorClient
    pw = bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode()
    print(pw)
    "
    # Then insert into Atlas:
    # db.users.insertOne({ username: 'omar', password: '<hash>' })
"""
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_COLLECTION = "users"


class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[_COLLECTION]

    async def find_by_username(self, username: str) -> Optional[dict]:
        """
        Return the raw user document for `username`, or None if not found.
        The returned dict will have at minimum: username, password (bcrypt hash).
        """
        doc = await self._col.find_one({"username": username})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
