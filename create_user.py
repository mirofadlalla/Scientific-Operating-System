"""Create (or reset the password of) a login user in MongoDB.

Matches what app.services.auth_service expects: collection users,
document {"username": str, "password": <bcrypt hash>}. Connection settings
come from the same MONGODB_URI / MONGODB_DB_NAME the app uses
(environment or .env).

Run from the repository root::

    python -m scripts.create_user --username omar            # prompts for password
    python -m scripts.create_user --username omar --update   # reset existing password

For automation, supply the password via the CREATE_USER_PASSWORD env var
(preferred over --password, which is visible in shell history / ps).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from datetime import datetime, timezone
from typing import Any

import bcrypt

COLLECTION = "users"
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt hard limit


class UserError(Exception):
    """A problem the operator can fix (bad input, duplicate user, ...)."""


def validate(username: str, password: str) -> str:
    """Return the normalised username or raise UserError."""
    username = username.strip()
    if not username:
        raise UserError("Username must not be empty.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise UserError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes (bcrypt limit).")
    return username


def hash_password(password: str) -> str:
    """bcrypt hash, as a str, in the same format auth_service.verify_password checks."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def upsert_user(db: Any, username: str, password: str, update: bool = False) -> str:
    """Insert a new user, or reset the password when update is true.

    Returns "created" or "updated". Raises UserError if the user already
    exists and update is false, or doesn't exist and update is true.
    """
    username = validate(username, password)
    col = db[COLLECTION]
    await col.create_index("username", unique=True)  # idempotent; prevents duplicates
    existing = await col.find_one({"username": username})
    hashed = hash_password(password)
    now = datetime.now(timezone.utc)

    if existing:
        if not update:
            raise UserError(f"User {username!r} already exists. Use --update to reset the password.")
        await col.update_one({"username": username}, {"$set": {"password": hashed, "updated_at": now}})
        return "updated"

    if update:
        raise UserError(f"User {username!r} does not exist; drop --update to create it.")
    await col.insert_one({"username": username, "password": hashed, "created_at": now})
    return "created"


def _read_password(args: argparse.Namespace) -> str:
    if args.password:
        print("warning: --password is visible in shell history; prefer the prompt or "
              "CREATE_USER_PASSWORD.", file=sys.stderr)
        return args.password
    env_password = os.getenv("CREATE_USER_PASSWORD")
    if env_password:
        return env_password
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Confirm password: "):
        raise UserError("Passwords do not match.")
    return first


async def _run(args: argparse.Namespace) -> int:
    from motor.motor_asyncio import AsyncIOMotorClient

    from app.config import settings

    if not settings.MONGODB_URI:
        raise UserError("MONGODB_URI is not set (environment or .env).")

    password = _read_password(args)
    client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=8000)
    try:
        result = await upsert_user(client[settings.MONGODB_DB_NAME], args.username, password, args.update)
    finally:
        client.close()
    print(f"User {args.username.strip()!r} {result} in database {settings.MONGODB_DB_NAME!r}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", help="avoid: visible in shell history (use the prompt or CREATE_USER_PASSWORD)")
    parser.add_argument("--update", action="store_true", help="reset the password of an existing user")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - connection/auth failures from pymongo
        print(f"error: could not reach MongoDB: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
