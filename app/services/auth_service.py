"""
app.services.auth_service
~~~~~~~~~~~~~~~~~~~~~~~~~
Credential checking and JWT creation/verification.

Passwords are bcrypt hashes stored in the MongoDB `users` collection.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import settings
from app.core.exceptions import ServiceUnavailableError, UnauthorizedError
from app.database.mongodb import get_database
from app.database.user_repository import UserRepository

logger = logging.getLogger(__name__)

try:
    from jose import JWTError, jwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    logger.warning("python-jose not installed — JWT auth will be unavailable.")

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed — password verification will be unavailable.")


def _require_deps() -> None:
    if not _JWT_AVAILABLE:
        raise ServiceUnavailableError("Auth service unavailable: python-jose not installed.")
    if not _BCRYPT_AVAILABLE:
        raise ServiceUnavailableError("Auth service unavailable: bcrypt not installed.")


def create_access_token(username: str) -> str:
    _require_deps()
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_password(plain: str, hashed: str) -> bool:
    _require_deps()
    return bcrypt.checkpw(plain.encode(), hashed.encode())


async def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Return the user document if credentials are valid, else None."""
    _require_deps()
    if not settings.MONGODB_URI:
        raise ServiceUnavailableError("Authentication is unavailable: MONGODB_URI is not configured.")
    repo = UserRepository(get_database())
    user = await repo.find_by_username(username)
    if user is None:
        return None
    if not verify_password(password, user.get("password", "")):
        return None
    return user


async def login(username: str, password: str) -> str:
    """Verify credentials and return a signed access token."""
    user = await authenticate_user(username, password)
    if user is None:
        logger.warning("Failed login attempt for username: %s", username)
        raise UnauthorizedError(
            "Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user["username"])
    logger.info("User logged in: %s", user["username"])
    return token


def verify_access_token(token: str) -> str:
    """Validate a JWT and return its subject (username)."""
    _require_deps()
    invalid = UnauthorizedError(
        "Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        raise invalid
    username: Optional[str] = payload.get("sub")
    if not username:
        raise invalid
    return username
