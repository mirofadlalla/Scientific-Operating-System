"""
app.core.auth
~~~~~~~~~~~~~
JWT-based authentication utilities.

Flow:
  1. Client POSTs username + password to POST /api/v1/auth/login
  2. Server verifies credentials against MongoDB users collection (bcrypt)
  3. Server returns a signed JWT access token (HS256, 1-hour expiry by default)
  4. Client sends token in Authorization: Bearer <token> header
  5. Protected endpoints use Depends(verify_token) to validate the token

Dependencies:
  pip install python-jose[cryptography] bcrypt motor
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.database.mongodb import get_database
from app.database.user_repository import UserRepository

logger = logging.getLogger(__name__)

# ── JWT helpers ───────────────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    logger.warning("python-jose not installed — JWT auth will be unavailable.")

# ── bcrypt helpers ────────────────────────────────────────────────────────────
try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed — password verification will be unavailable.")

_bearer = HTTPBearer(auto_error=True)


def _require_deps() -> None:
    if not _JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable: python-jose not installed.",
        )
    if not _BCRYPT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable: bcrypt not installed.",
        )


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(username: str) -> str:
    """Create a signed JWT access token for `username`."""
    _require_deps()
    expire = datetime.now(tz=timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# ── Password verification ─────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches the bcrypt `hashed` password."""
    _require_deps()
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Authenticate user against MongoDB ────────────────────────────────────────

async def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Look up `username` in MongoDB and verify the password.
    Returns the user document on success, None on failure.
    """
    _require_deps()
    if not settings.MONGODB_URI:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is unavailable: MONGODB_URI is not configured.",
        )
    db   = get_database()
    repo = UserRepository(db)
    user = await repo.find_by_username(username)
    if user is None:
        return None
    if not verify_password(password, user.get("password", "")):
        return None
    return user


# ── FastAPI dependency — protects endpoints ───────────────────────────────────

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    FastAPI dependency that validates the Bearer JWT token.

    Usage::

        @router.post("/rag/ingest")
        async def rag_ingest(..., username: str = Depends(verify_token)):
            ...

    Returns the authenticated username on success.
    Raises HTTP 401 on invalid / expired tokens.
    """
    _require_deps()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: Optional[str] = payload.get("sub")
        if not username:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
